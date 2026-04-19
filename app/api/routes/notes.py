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
    """Creates a new note with AI-driven event extraction and vector embedding.

    This endpoint triggers a streaming response that provides status updates 
    as the AI parses the content, extracts event dates, and generates embeddings.

    Args:
        request: The note content and title.
        session: Database session dependency.
        current_user: The authenticated user creating the note.

    Returns:
        A StreamingResponse (SSE) with status updates and the final note data.
    """
    return StreamingResponse(
        create_note(session, current_user.id, request), media_type="text/event-stream"
    )


@router.get("/", response_model=List[NoteResponse])
async def list_notes_endpoint(
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Lists all notes belonging to the authenticated user.

    Args:
        session: Database session dependency.
        current_user: The authenticated user.

    Returns:
        A list of all user's notes, ordered by creation date.
    """
    return await get_user_notes(session, current_user.id)


@router.get("/search", response_model=List[NoteResponse])
async def search_notes_endpoint(
    query: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Performs a semantic search across user notes.

    Uses vector embeddings to find relevant notes based on the query string.
    Supports optional time-based filtering.

    Args:
        query: The search string.
        start_time: Optional start date for filtering.
        end_time: Optional end date for filtering.
        session: Database session dependency.
        current_user: The authenticated user performing the search.

    Returns:
        A list of matching notes with relevancy scores.
    """
    return await search_notes(
        session, current_user.id, query, start_time=start_time, end_time=end_time
    )


@router.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Initiates an AI chat session based on note context (RAG).

    Performs semantic retrieval of relevant note chunks and uses them as 
    context for the LLM to answer the user's question.

    Args:
        request: The user's question and conversation history.
        session: Database session dependency.
        current_user: The authenticated user chatting.

    Returns:
        A StreamingResponse (SSE) with the AI's answer chunks.
    """
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
    note = await update_note(session, current_user.id, note_id, request)
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
    """Generates an AI analysis (summary, tags, sentiment) for a note.

    Args:
        request: The note content to analyze.
        _: Authenticated user check.

    Returns:
        A StreamingResponse (SSE) with the analysis results.
    """
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
    note = await save_note_analysis(session, current_user.id, note_id, request)
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
