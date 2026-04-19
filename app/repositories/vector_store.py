from datetime import datetime
from typing import List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.models.embeddings import get_embeddings
from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.models.models import Note, NoteChunk


async def add_note_with_chunks(
    session: AsyncSession,
    user_id: int,
    content: str,
    title: Optional[str] = None,
    event_date: Optional[str] = None,
    event_confidence: Optional[str] = None,
    event_reasoning: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Note:
    try:
        note = Note(
            user_id=user_id,
            title=title,
            content=content,
            event_date=event_date,
            event_confidence=event_confidence,
            event_reasoning=event_reasoning,
            metadata_=metadata or {},
        )
        session.add(note)
        await session.flush()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        text_to_embed = f"{title}\n{content}" if title else content
        text_chunks = splitter.split_text(text_to_embed)

        embeddings_model = get_embeddings()
        embeddings = embeddings_model.embed_documents(text_chunks)

        for i, (chunk_text, embedding) in enumerate(zip(text_chunks, embeddings)):
            chunk = NoteChunk(
                note_id=note.id,
                chunk_content=chunk_text,
                chunk_index=i,
                embedding=embedding,
            )
            session.add(chunk)

        await session.commit()
        await session.refresh(note)
        return note
    except Exception as exc:
        await session.rollback()
        raise VectorStoreError(f"Failed to store note and chunks: {exc}") from exc


async def search_notes_semantic(
    session: AsyncSession,
    query: str,
    user_id: int,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 4,
    threshold: Optional[float] = None,
    window_size: int = 1,
) -> List[Tuple[Note, str, float]]:
    try:
        if not query.strip():
            # If no query text, perform a standard metadata search
            stmt = select(Note).where(Note.user_id == user_id)
            if start_time:
                # Convert datetime to string for comparison with Note.event_date (String column)
                start_str = start_time.strftime("%Y-%m-%d") if isinstance(start_time, datetime) else start_time
                stmt = stmt.where(Note.event_date >= start_str)
            if end_time:
                end_str = end_time.strftime("%Y-%m-%d") if isinstance(end_time, datetime) else end_time
                stmt = stmt.where(Note.event_date <= end_str)
            stmt = stmt.order_by(Note.event_date.desc()).limit(limit)

            result = await session.execute(stmt)
            notes = result.scalars().all()
            return [(n, n.content, 0.0) for n in notes]

        embeddings_model = get_embeddings()
        query_embedding = embeddings_model.embed_query(query)

        distance = NoteChunk.embedding.cosine_distance(query_embedding).label(
            "distance"
        )

        stmt = (
            select(Note, NoteChunk.chunk_content, distance, NoteChunk.chunk_index)
            .join(NoteChunk, Note.id == NoteChunk.note_id)
            .where(Note.user_id == user_id)
        )

        if threshold is not None:
            stmt = stmt.where(distance <= threshold)

        if start_time:
            start_str = start_time.strftime("%Y-%m-%d") if isinstance(start_time, datetime) else start_time
            stmt = stmt.where(Note.event_date >= start_str)
        if end_time:
            end_str = end_time.strftime("%Y-%m-%d") if isinstance(end_time, datetime) else end_time
            stmt = stmt.where(Note.event_date <= end_str)

        stmt = stmt.order_by(distance).limit(limit)

        result = await session.execute(stmt)
        matches = result.all()

        if not matches:
            return []

        if window_size == 0:
            return [(m[0], m[1], m[2]) for m in matches]

        note_matches = {}
        note_objects = {}
        for note, content, dist, idx in matches:
            note_matches.setdefault(note.id, []).append((idx, dist))
            note_objects[note.id] = note

        note_ids = list(note_matches.keys())
        chunks_stmt = (
            select(NoteChunk)
            .where(NoteChunk.note_id.in_(note_ids))
            .order_by(NoteChunk.note_id, NoteChunk.chunk_index)
        )
        chunks_result = await session.execute(chunks_stmt)
        all_chunks = chunks_result.scalars().all()

        note_chunks_map = {}
        for chunk in all_chunks:
            note_chunks_map.setdefault(chunk.note_id, {})[chunk.chunk_index] = (
                chunk.chunk_content
            )

        final_results = []
        for note_id, matches_in_note in note_matches.items():
            note = note_objects[note_id]
            needed_indices = set()
            match_distances = {}
            for idx, dist in matches_in_note:
                for offset in range(-window_size, window_size + 1):
                    needed_indices.add(idx + offset)
                match_distances[idx] = dist

            sorted_indices = sorted(
                [i for i in needed_indices if i in note_chunks_map.get(note_id, {})]
            )

            if not sorted_indices:
                continue

            blocks = []
            current_block = [sorted_indices[0]]
            for i in range(1, len(sorted_indices)):
                if sorted_indices[i] == sorted_indices[i - 1] + 1:
                    current_block.append(sorted_indices[i])
                else:
                    blocks.append(current_block)
                    current_block = [sorted_indices[i]]
            blocks.append(current_block)

            for block in blocks:
                content = "\n".join([note_chunks_map[note_id][idx] for idx in block])
                block_dist = min(
                    [match_distances[idx] for idx in block if idx in match_distances],
                    default=min([d for i, d in matches_in_note]),
                )
                final_results.append((note, content, block_dist))

        final_results.sort(key=lambda x: x[2])
        return final_results
    except Exception as exc:
        raise VectorStoreError(f"Semantic search failed: {exc}") from exc


async def delete_note_from_store(
    session: AsyncSession,
    user_id: int,
    note_id: int,
) -> bool:
    try:
        await session.execute(delete(NoteChunk).where(NoteChunk.note_id == note_id))

        stmt = delete(Note).where(Note.id == note_id, Note.user_id == user_id)
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount > 0
    except Exception as exc:
        await session.rollback()
        raise VectorStoreError(f"Failed to delete note: {exc}") from exc


async def update_note_with_chunks(
    session: AsyncSession,
    user_id: int,
    note_id: int,
    title: Optional[str] = None,
    content: Optional[str] = None,
    event_date: Optional[str] = None,
    event_confidence: Optional[str] = None,
    event_reasoning: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[Note]:
    try:
        stmt = select(Note).where(Note.id == note_id, Note.user_id == user_id)
        result = await session.execute(stmt)
        note = result.scalar_one_or_none()

        if not note:
            return None

        if title is not None:
            note.title = title
        if event_date is not None:
            note.event_date = event_date
        if event_confidence is not None:
            note.event_confidence = event_confidence
        if event_reasoning is not None:
            note.event_reasoning = event_reasoning
        if metadata is not None:
            note.metadata_ = metadata

        content_changed = False
        if content is not None and content != note.content:
            note.content = content
            content_changed = True

        if content_changed:
            note.summary = None
            note.tags = []
            note.sentiment = None

            await session.execute(delete(NoteChunk).where(NoteChunk.note_id == note.id))

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            text_to_embed = (
                f"{note.title}\n{note.content}" if note.title else note.content
            )
            text_chunks = splitter.split_text(text_to_embed)

            embeddings_model = get_embeddings()
            embeddings = embeddings_model.embed_documents(text_chunks)

            for i, (chunk_text, embedding) in enumerate(zip(text_chunks, embeddings)):
                chunk = NoteChunk(
                    note_id=note.id,
                    chunk_content=chunk_text,
                    chunk_index=i,
                    embedding=embedding,
                )
                session.add(chunk)

        await session.commit()
        await session.refresh(note)
        return note
    except Exception as exc:
        await session.rollback()
        raise VectorStoreError(f"Failed to update note and chunks: {exc}") from exc
