import asyncio
import json
from datetime import date
from typing import AsyncGenerator, List

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.chains.chain import (
    run_analyze_chain,
    run_chat_chain,
    run_extract_event_date_chain,
    run_generate_random_note_chain,
    run_generate_title_chain,
)
from app.core.config import settings
from app.repositories import note_repository, vector_repository
from app.schemas.note import (
    AnalyzeRequest,
    ChatRequest,
    GenerateTitleRequest,
    NoteAnalysisOrchestrationParams,
    NoteCreate,
    NoteRecordCreateParams,
    NoteRecordSearchParams,
    NoteRecordUpdateParams,
    NoteResponse,
    NoteServiceSearchParams,
    NoteUpdateOrchestrationParams,
    VectorStoreSearchParams,
)
from app.utils.date_utils import get_jakarta_today_str


def parse_ai_date_range(date_str: str | None) -> date | None:
    """Parses 'YYYY-MM-DD' or 'YYYY-MM-DD/YYYY-MM-DD' into a single date (start)."""
    if not date_str or not date_str.strip():
        return None

    try:
        if "/" in date_str:
            parts = date_str.split("/")
            return date.fromisoformat(parts[0].strip())
        else:
            return date.fromisoformat(date_str.strip())
    except (ValueError, IndexError):
        logger.bind(task="AI").warning(f"Failed to parse date string: {date_str}")
        return None


async def search_notes(
    session: AsyncSession,
    params: NoteServiceSearchParams,
) -> List[NoteResponse]:
    """Searches for notes using semantic search and applies temporal filters."""
    logger.bind(task="SEARCH").info(
        f"User {params.user_id} searching for: '{params.query}'"
    )

    if not params.query.strip():
        search_params = NoteRecordSearchParams(
            user_id=params.user_id,
            start_time=params.start_time,
            end_time=params.end_time,
            limit=params.limit,
        )
        notes = await note_repository.search_notes_metadata(session, search_params)
        return [NoteResponse.model_validate(n) for n in notes]

    vector_params = VectorStoreSearchParams(
        query=params.query,
        user_id=params.user_id,
        limit=params.limit,
        threshold=settings.SEMANTIC_THRESHOLD_UI,
        start_time=params.start_time,
        end_time=params.end_time,
    )
    matches = await vector_repository.search_semantic(
        session=session, params=vector_params
    )

    return [NoteResponse.model_validate(m[0]) for m in matches]


async def create_note(
    session: AsyncSession, user_id: int, request: NoteCreate
) -> AsyncGenerator[str, None]:
    """Creates a note, extracts events via AI, and generates vector chunks."""
    logger.bind(task="NOTE").info(f"Starting creation process for user {user_id}")
    yield "data: status: parsing\n\n"
    await asyncio.sleep(0.5)

    yield "data: status: extracting events\n\n"
    ref_date = get_jakarta_today_str()
    logger.bind(task="AI").info(f"Extracting events from content. Ref date: {ref_date}")
    extraction = await run_extract_event_date_chain(request.content, ref_date)

    event_date = parse_ai_date_range(extraction.get("event_date"))

    yield "data: status: saving\n\n"
    logger.bind(task="DB").info("Saving note record...")

    params = NoteRecordCreateParams(
        user_id=user_id,
        content=request.content,
        title=request.title,
        event_date=event_date,
        event_confidence=extraction.get("event_confidence", "LOW"),
        event_reasoning=extraction.get("event_reasoning"),
    )
    note = await note_repository.create_note_record(session=session, params=params)

    logger.bind(task="VEC").info(f"Generating embeddings for note {note.id}...")
    await vector_repository.add_note_chunks(
        session=session, note_id=note.id, content=note.content, title=note.title
    )

    await note_repository.commit_session(session)

    logger.bind(task="NOTE").success(
        f"Note '{request.title}' created successfully (ID: {note.id})"
    )
    resp = NoteResponse.model_validate(note)
    yield f"data: {resp.model_dump_json()}\n\n"


async def get_user_notes(session: AsyncSession, user_id: int) -> List[NoteResponse]:
    notes = await note_repository.get_user_notes_list(session, user_id)
    return [NoteResponse.model_validate(n) for n in notes]


async def chat_with_notes(
    session: AsyncSession, user_id: int, request: ChatRequest
) -> AsyncGenerator[str, None]:
    """Orchestrates a RAG-based chat session."""
    yield "data: status: searching notes\n\n"
    params = VectorStoreSearchParams(
        query=request.question,
        user_id=user_id,
        limit=settings.RETRIEVER_K,
        threshold=settings.SEMANTIC_THRESHOLD_CHAT,
        window_size=settings.WINDOW_SIZE,
    )
    results = await vector_repository.search_semantic(session=session, params=params)

    prev_context_content = None
    for msg in reversed(request.history):
        if msg.context_content:
            prev_context_content = msg.context_content
            break

    yield "data: status: building context\n\n"
    context_meta = [{"id": res[0].id, "title": res[0].title} for res in results]
    yield f"data: context: {json.dumps(context_meta)}\n\n"

    current_context_content = None
    if results:
        current_context_content = "\n\n---\n\n".join(
            f"Note Title: {res[0].title or 'Untitled'} (from {res[0].created_at.strftime('%Y-%m-%d')})\nContent: {res[1]}"
            for res in results
        )
        yield f"data: context_content: {json.dumps(current_context_content)}\n\n"
    elif prev_context_content:
        current_context_content = prev_context_content

    context_to_pass = (
        current_context_content if current_context_content else "No context"
    )
    is_followup = prev_context_content is not None

    yield "data: status: generating response\n\n"
    logger.bind(task="AI").info("Running RAG chat chain...")
    history_str = "\n".join(
        [f"{m.role.capitalize()}: {m.content}" for m in request.history]
    )

    async for chunk in run_chat_chain(
        context=context_to_pass,
        user_input=request.question,
        history=history_str,
        is_followup=is_followup,
    ):
        yield chunk


async def delete_note(session: AsyncSession, user_id: int, note_id: int) -> bool:
    await vector_repository.delete_chunks_by_note_id(session, note_id)
    success = await note_repository.delete_note_record(session, user_id, note_id)
    if success:
        await note_repository.commit_session(session)
    return success


async def update_note(
    session: AsyncSession, params: NoteUpdateOrchestrationParams
) -> NoteResponse | None:
    """Updates a note's record and regenerates embeddings if content changed."""
    current_note = await note_repository.get_note_by_id(
        session, params.user_id, params.note_id
    )
    if not current_note:
        return None

    content_changed = (
        params.request.content is not None
        and params.request.content != current_note.content
    )

    update_params = NoteRecordUpdateParams(
        note_id=params.note_id,
        user_id=params.user_id,
        title=params.request.title,
        content=params.request.content,
    )
    note = await note_repository.update_note_record(
        session=session, params=update_params
    )

    if content_changed:
        logger.bind(task="VEC").info(
            f"Content changed for note {params.note_id}, regenerating embeddings..."
        )
        await vector_repository.delete_chunks_by_note_id(session, params.note_id)
        await vector_repository.add_note_chunks(
            session=session, note_id=note.id, content=note.content, title=note.title
        )

    await note_repository.commit_session(session)
    return NoteResponse.model_validate(note)


async def analyze_note(request: AnalyzeRequest) -> AsyncGenerator[str, None]:
    async for chunk in run_analyze_chain(title=request.title, content=request.content):
        yield chunk


async def suggest_title(request: GenerateTitleRequest) -> AsyncGenerator[str, None]:
    async for chunk in run_generate_title_chain(request.content):
        yield chunk


async def save_note_analysis(
    session: AsyncSession, params: NoteAnalysisOrchestrationParams
) -> NoteResponse | None:
    """Saves AI-generated analysis metadata to the note record."""
    update_params = NoteRecordUpdateParams(
        note_id=params.note_id,
        user_id=params.user_id,
        summary=params.request.summary,
        tags=params.request.tags,
        sentiment=params.request.sentiment,
    )
    note = await note_repository.update_note_record(
        session=session, params=update_params
    )

    if not note:
        return None

    await note_repository.commit_session(session)
    return NoteResponse.model_validate(note)


async def generate_random_note() -> AsyncGenerator[str, None]:
    async for chunk in run_generate_random_note_chain():
        yield chunk
