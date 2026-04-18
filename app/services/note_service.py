import asyncio
import json
from typing import AsyncGenerator, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.chains.chain import (
    run_analyze_chain,
    run_chat_chain,
    run_generate_title_chain,
    run_generate_random_note_chain,
)
from app.core.config import settings
from app.models.models import Note
from app.repositories.vector_store import (
    add_note_with_chunks,
    delete_note_from_store,
    search_notes_semantic,
    update_note_with_chunks,
)
from app.schemas.note import (
    AnalyzeRequest,
    ChatRequest,
    GenerateTitleRequest,
    NoteAnalysisUpdate,
    NoteCreate,
    NoteResponse,
    NoteUpdate,
)


async def search_notes(
    session: AsyncSession, user_id: int, query: str, limit: int = 10
) -> List[NoteResponse]:
    matches = await search_notes_semantic(
        session=session,
        query=query,
        user_id=user_id,
        limit=limit,
        threshold=settings.SEMANTIC_THRESHOLD_UI,
    )

    return [NoteResponse.model_validate(m[0]) for m in matches]


async def create_note(
    session: AsyncSession, user_id: int, request: NoteCreate
) -> AsyncGenerator[str, None]:
    yield "data: status: parsing\n\n"
    await asyncio.sleep(0.5)

    yield "data: status: processing\n\n"
    await asyncio.sleep(0.5)

    note = await add_note_with_chunks(
        session=session, user_id=user_id, content=request.content, title=request.title
    )

    yield "data: status: saving\n\n"
    await asyncio.sleep(0.5)

    resp = NoteResponse.model_validate(note)
    yield f"data: {resp.model_dump_json()}\n\n"


async def get_user_notes(session: AsyncSession, user_id: int) -> List[NoteResponse]:
    stmt = select(Note).where(Note.user_id == user_id).order_by(Note.created_at.desc())
    result = await session.execute(stmt)
    notes = result.scalars().all()

    return [NoteResponse.model_validate(n) for n in notes]


async def chat_with_notes(
    session: AsyncSession, user_id: int, request: ChatRequest
) -> AsyncGenerator[str, None]:
    yield "data: status: searching notes\n\n"
    results = await search_notes_semantic(
        session=session,
        query=request.question,
        user_id=user_id,
        limit=settings.RETRIEVER_K,
        threshold=settings.SEMANTIC_THRESHOLD_CHAT,
        window_size=settings.WINDOW_SIZE,
    )

    # Check if we have any previous successul RAG context in history
    prev_context_content = None
    for msg in reversed(request.history):
        if msg.context_content:
            prev_context_content = msg.context_content
            break

    yield "data: status: building context\n\n"
    context_meta = [{"id": res[0].id, "title": res[0].title} for res in results]
    yield f"data: context: {json.dumps(context_meta)}\n\n"

    # Determine current context and if it's a follow-up
    current_context_content = None
    if results:
        current_context_content = "\n\n---\n\n".join(
            f"Note Title: {res[0].title or 'Untitled'} (from {res[0].created_at.strftime('%Y-%m-%d')})\nContent: {res[1]}"
            for res in results
        )
        # Send the content to the frontend so it can be saved in history
        yield f"data: context_content: {json.dumps(current_context_content)}\n\n"
    elif prev_context_content:
        # Reuse previous context if current search yields nothing
        current_context_content = prev_context_content
    
    # Logic flags for prompt selection
    context_to_pass = current_context_content if current_context_content else "No context"
    is_followup = prev_context_content is not None

    yield "data: status: generating response\n\n"

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
    return await delete_note_from_store(session, user_id, note_id)


async def update_note(
    session: AsyncSession, user_id: int, note_id: int, request: NoteUpdate
) -> NoteResponse | None:
    note = await update_note_with_chunks(
        session=session,
        user_id=user_id,
        note_id=note_id,
        title=request.title,
        content=request.content,
    )
    if not note:
        return None
    return NoteResponse.model_validate(note)


async def analyze_note(request: AnalyzeRequest) -> AsyncGenerator[str, None]:
    async for chunk in run_analyze_chain(title=request.title, content=request.content):
        yield chunk


async def suggest_title(request: GenerateTitleRequest) -> AsyncGenerator[str, None]:
    async for chunk in run_generate_title_chain(request.content):
        yield chunk


async def save_note_analysis(
    session: AsyncSession, user_id: int, note_id: int, request: NoteAnalysisUpdate
) -> NoteResponse | None:
    stmt = select(Note).where(Note.id == note_id, Note.user_id == user_id)
    result = await session.execute(stmt)
    note = result.scalar_one_or_none()

    if not note:
        return None

    note.summary = request.summary
    note.tags = request.tags
    note.sentiment = request.sentiment

    await session.commit()
    await session.refresh(note)

    return NoteResponse.model_validate(note)


async def generate_random_note() -> AsyncGenerator[str, None]:
    async for chunk in run_generate_random_note_chain():
        yield chunk
