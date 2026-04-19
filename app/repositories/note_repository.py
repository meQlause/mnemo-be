from datetime import datetime
from typing import List, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Note
from app.schemas.note import (
    NoteRecordCreateParams,
    NoteRecordSearchParams,
    NoteRecordUpdateParams,
)

async def create_note_record(
    session: AsyncSession,
    params: NoteRecordCreateParams,
) -> Note:
    """Creates a basic Note record in PostgreSQL."""
    note = Note(
        user_id=params.user_id,
        title=params.title,
        content=params.content,
        event_date=params.event_date,
        event_start_date=params.event_start_date,
        event_end_date=params.event_end_date,
        event_confidence=params.event_confidence,
        event_reasoning=params.event_reasoning,
        metadata_=params.metadata or {},
    )
    session.add(note)
    await session.flush()
    return note

async def get_note_by_id(
    session: AsyncSession, user_id: int, note_id: int
) -> Optional[Note]:
    """Retrieves a single note by ID, verifying ownership."""
    stmt = select(Note).where(Note.id == note_id, Note.user_id == user_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()

async def get_user_notes_list(session: AsyncSession, user_id: int) -> List[Note]:
    """Lists all notes belonging to a specific user."""
    stmt = select(Note).where(Note.user_id == user_id).order_by(Note.created_at.desc())
    result = await session.execute(stmt)
    return list(result.scalars().all())

async def delete_note_record(session: AsyncSession, user_id: int, note_id: int) -> bool:
    """Deletes a note record from the database."""
    stmt = delete(Note).where(Note.id == note_id, Note.user_id == user_id)
    result = await session.execute(stmt)
    return result.rowcount > 0

async def search_notes_metadata(
    session: AsyncSession,
    params: NoteRecordSearchParams,
) -> List[Note]:
    """Searches for notes using native date overlap logic."""
    stmt = select(Note).where(Note.user_id == params.user_id)
    
    if params.start_time:
        start_date = params.start_time.date() if isinstance(params.start_time, datetime) else params.start_time
        stmt = stmt.where(Note.event_date >= start_date)
    
    if params.end_time:
        end_date = params.end_time.date() if isinstance(params.end_time, datetime) else params.end_time
        stmt = stmt.where(Note.event_date <= end_date)

    stmt = stmt.order_by(Note.event_date.desc().nulls_last(), Note.created_at.desc()).limit(params.limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())

async def update_note_record(
    session: AsyncSession,
    params: NoteRecordUpdateParams,
) -> Optional[Note]:
    """Updates a note record using structured parameters."""
    note = await get_note_by_id(session, params.user_id, params.note_id)
    if not note:
        return None

    update_data = params.model_dump(exclude={"note_id", "user_id"}, exclude_unset=True)
    for key, value in update_data.items():
        if hasattr(note, key):
            setattr(note, key, value)

    await session.flush()
    return note

async def commit_session(session: AsyncSession):
    """Utility to commit the session."""
    await session.commit()

async def rollback_session(session: AsyncSession):
    """Utility to rollback the session."""
    await session.rollback()
