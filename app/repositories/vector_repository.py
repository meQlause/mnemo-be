from typing import List, Optional, Tuple
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.models.embeddings import get_embeddings
from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.models.models import Note, NoteChunk
from app.schemas.note import VectorStoreSearchParams


async def _bulk_save_chunks(session: AsyncSession, chunks: List[NoteChunk]) -> None:
    """Standardized bulk insert for note chunks."""
    session.add_all(chunks)
    await session.flush()


async def _get_chunks_by_note_ids(
    session: AsyncSession, note_ids: List[int]
) -> List[NoteChunk]:
    """Fetches all chunks for a set of notes, ordered by index."""
    stmt = (
        select(NoteChunk)
        .where(NoteChunk.note_id.in_(note_ids))
        .order_by(NoteChunk.note_id, NoteChunk.chunk_index)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def _delete_chunks_for_note(session: AsyncSession, note_id: int) -> None:
    """Deletes all chunks for a specific note."""
    await session.execute(delete(NoteChunk).where(NoteChunk.note_id == note_id))
    await session.flush()


async def _run_vector_search_query(
    session: AsyncSession,
    user_id: int,
    query_embedding: List[float],
    limit: int,
    threshold: Optional[float] = None,
    start_date=None,
    end_date=None,
) -> List[Tuple[Note, str, float, int]]:
    """Executes the raw pgvector similarity search in the database."""
    distance = NoteChunk.embedding.cosine_distance(query_embedding).label("distance")

    stmt = (
        select(Note, NoteChunk.chunk_content, distance, NoteChunk.chunk_index)
        .join(NoteChunk, Note.id == NoteChunk.note_id)
        .where(Note.user_id == user_id)
    )

    if threshold is not None:
        stmt = stmt.where(distance <= threshold)

    if start_date:
        stmt = stmt.where(Note.event_date >= start_date)
    if end_date:
        stmt = stmt.where(Note.event_date <= end_date)

    stmt = stmt.order_by(distance).limit(limit)

    result = await session.execute(stmt)
    return result.all()


async def get_chunks_by_note_ids(
    session: AsyncSession, note_ids: List[int]
) -> List[NoteChunk]:
    """Public wrapper to fetch ordered chunks for the service layer."""
    return await _get_chunks_by_note_ids(session, note_ids)


async def add_note_chunks(
    session: AsyncSession,
    note_id: int,
    content: str,
    title: str | None = None,
) -> None:
    """Generates embeddings and stores semantic chunks for a given note."""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        text_to_embed = f"{title}\n{content}" if title else content
        text_chunks = splitter.split_text(text_to_embed)

        embeddings_model = get_embeddings()
        logger.bind(task="VEC").info(f"Generating embeddings for {len(text_chunks)} chunks for note {note_id}...")
        embeddings = embeddings_model.embed_documents(text_chunks)

        chunks = [
            NoteChunk(
                note_id=note_id,
                chunk_content=chunk_text,
                chunk_index=i,
                embedding=embedding,
            )
            for i, (chunk_text, embedding) in enumerate(zip(text_chunks, embeddings))
        ]
        
        await _bulk_save_chunks(session, chunks)
    except Exception as exc:
        raise VectorStoreError(f"Failed to generate and store chunks: {exc}") from exc


async def search_note_chunks_vector(
    session: AsyncSession,
    params: VectorStoreSearchParams,
) -> List[Tuple[Note, str, float, int]]:
    """Calculates query embeddings and performs vector similarity search."""
    embeddings_model = get_embeddings()
    query_embedding = embeddings_model.embed_query(params.query)

    logger.bind(task="VEC").info(f"Executing cosine search for: '{params.query}'")
    
    matches = await _run_vector_search_query(
        session=session,
        user_id=params.user_id,
        query_embedding=query_embedding,
        limit=params.limit,
        threshold=params.threshold,
        start_date=params.start_time,
        end_date=params.end_time
    )

    if matches:
        logger.bind(task="VEC").info(f"Top {len(matches)} matches found:")
        for note, content, distance, idx in matches:
            logger.bind(task="VEC").info(f"  - Note ID: {note.id}, Index: {idx}, Distance: {distance:.4f}, Content: {content[:50]}...")
    else:
        logger.bind(task="VEC").info("No matches found.")

    return matches


async def reconstruct_note_context(
    session: AsyncSession,
    matches: List[Tuple[Note, str, float, int]],
    window_size: int,
) -> List[Tuple[Note, str, float]]:
    """Groups matching chunks and fetches neighboring context window chunks."""
    note_matches = {}
    note_objects = {}
    for note, content, dist, idx in matches:
        note_matches.setdefault(note.id, []).append((idx, dist))
        note_objects[note.id] = note

    note_ids = list(note_matches.keys())
    all_chunks = await _get_chunks_by_note_ids(session, note_ids)

    note_chunks_map = {}
    for chunk in all_chunks:
        note_chunks_map.setdefault(chunk.note_id, {})[chunk.chunk_index] = chunk.chunk_content

    final_results = []
    for note_id, matches_in_note in note_matches.items():
        note = note_objects[note_id]
        needed_indices = set()
        match_distances = {}
        for idx, dist in matches_in_note:
            for offset in range(-window_size, window_size + 1):
                needed_indices.add(idx + offset)
            match_distances[idx] = dist

        sorted_indices = sorted([i for i in needed_indices if i in note_chunks_map.get(note_id, {})])

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
            block_dist = min([match_distances[idx] for idx in block if idx in match_distances], default=min([d for i, d in matches_in_note]))
            final_results.append((note, content, block_dist))

    final_results.sort(key=lambda x: x[2])
    return final_results


async def search_semantic(
    session: AsyncSession,
    params: VectorStoreSearchParams,
) -> List[Tuple[Note, str, float]]:
    """Performs semantic search across note chunks using vector similarity."""
    try:
        matches = await search_note_chunks_vector(session, params)

        if not matches:
            return []

        if params.window_size == 0:
            return [(m[0], m[1], m[2]) for m in matches]

        return await reconstruct_note_context(session, matches, params.window_size)
    except Exception as exc:
        raise VectorStoreError(f"Semantic search failed: {exc}") from exc


async def delete_chunks_by_note_id(session: AsyncSession, note_id: int) -> None:
    """Deletes all semantic chunks for a given note."""
    await _delete_chunks_for_note(session, note_id)
