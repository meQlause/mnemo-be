from typing import List, Tuple, Optional
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.models import Note, NoteChunk

async def bulk_save_chunks(session: AsyncSession, chunks: List[NoteChunk]) -> None:
    """Standardized bulk insert for note chunks."""
    session.add_all(chunks)
    await session.flush()

async def get_chunks_by_note_ids(session: AsyncSession, note_ids: List[int]) -> List[NoteChunk]:
    """Fetches all chunks for a set of notes, ordered by index."""
    stmt = (
        select(NoteChunk)
        .where(NoteChunk.note_id.in_(note_ids))
        .order_by(NoteChunk.note_id, NoteChunk.chunk_index)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())

async def delete_chunks_for_note(session: AsyncSession, note_id: int) -> None:
    """Deletes all chunks for a specific note."""
    await session.execute(delete(NoteChunk).where(NoteChunk.note_id == note_id))
    await session.flush()

async def run_vector_search_query(
    session: AsyncSession,
    user_id: int,
    query_embedding: List[float],
    limit: int,
    threshold: Optional[float] = None,
    start_date = None,
    end_date = None,
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
