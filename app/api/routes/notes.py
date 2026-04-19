from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.db import get_db
from app.models.models import User
from app.schemas.note import (
    AnalyzeRequest,
    ChatRequest,
    GenerateTitleRequest,
    NoteAnalysisUpdate,
    NoteCreate,
    NoteResponse,
    NoteUpdate,
    NoteServiceSearchParams,
    NoteUpdateOrchestrationParams,
    NoteAnalysisOrchestrationParams,
)
from app.services.note_service import (
    analyze_note,
    chat_with_notes,
    create_note,
    delete_note,
    get_user_notes,
    save_note_analysis,
    search_notes,
    suggest_title,
    update_note,
    generate_random_note,
)

router = APIRouter()


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_note_endpoint(
    request: NoteCreate,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Creates a new note with AI-driven event extraction and vector embedding."""
    return StreamingResponse(
        create_note(session, current_user.id, request), media_type="text/event-stream"
    )


@router.get("/", response_model=List[NoteResponse])
async def list_notes_endpoint(
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Lists all notes belonging to the authenticated user."""
    return await get_user_notes(session, current_user.id)


@router.get("/search", response_model=List[NoteResponse])
async def search_notes_endpoint(
    query: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Performs a semantic search across user notes."""
    params = NoteServiceSearchParams(
        user_id=current_user.id,
        query=query,
        start_time=start_time,
        end_time=end_time
    )
    return await search_notes(session, params)


@router.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Initiates an AI chat session based on note context (RAG)."""
    return StreamingResponse(
        chat_with_notes(session, current_user.id, request),
        media_type="text/event-stream",
    )


@router.patch("/{note_id}", response_model=NoteResponse)
async def update_note_endpoint(
    note_id: int,
    request: NoteUpdate,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    params = NoteUpdateOrchestrationParams(
        user_id=current_user.id,
        note_id=note_id,
        request=request
    )
    note = await update_note(session, params)
    if not note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Note not found"
        )
    return note


@router.delete("/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_note_endpoint(
    note_id: int,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    success = await delete_note(session, current_user.id, note_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Note not found"
        )
    return


@router.post("/analyze")
async def analyze_endpoint(
    request: AnalyzeRequest,
    _: User = Depends(get_current_user),
):
    """Generates an AI analysis (summary, tags, sentiment) for a note."""
    return StreamingResponse(analyze_note(request), media_type="text/event-stream")


@router.post("/generate-title")
async def run_generate_title_chain_endpoint(
    request: GenerateTitleRequest,
    _: User = Depends(get_current_user),
):
    return StreamingResponse(suggest_title(request), media_type="text/event-stream")


@router.post("/{note_id}/analysis", response_model=NoteResponse)
async def save_note_analysis_endpoint(
    note_id: int,
    request: NoteAnalysisUpdate,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    params = NoteAnalysisOrchestrationParams(
        user_id=current_user.id,
        note_id=note_id,
        request=request
    )
    note = await save_note_analysis(session, params)
    if not note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Note not found"
        )
    return note


@router.post("/generate-random")
async def run_generate_random_note_endpoint(
    _: User = Depends(get_current_user),
):
    return StreamingResponse(generate_random_note(), media_type="text/event-stream")
