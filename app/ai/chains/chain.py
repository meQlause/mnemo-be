from typing import AsyncGenerator
import json
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from app.ai.models.llm import get_llm, get_parse_llm
from app.schemas.note import EventExtraction
from app.ai.prompts.note_prompt import (
    analyze_prompt,
    no_context_prompt,
    rag_followup_prompt,
    rag_initial_prompt,
    title_prompt,
    random_note_prompt,
    extract_event_date_prompt,
)


async def run_generate_title_chain(text: str) -> AsyncGenerator[str, None]:
    llm = get_llm()
    chain = title_prompt | llm | StrOutputParser()
    try:
        result = await chain.ainvoke({"text": text})
        yield f"data: {result}\n\n"
    except Exception:
        yield "data: Untitled Note\n\n"


async def run_analyze_chain(
    title: str | None, content: str
) -> AsyncGenerator[str, None]:
    llm = get_llm()
    chain = analyze_prompt | llm | JsonOutputParser()

    try:
        async for chunk in chain.astream(
            {"title": title or "Untitled", "content": content}
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
    except Exception:
        yield f"data: {json.dumps({'summary': 'Analysis failed', 'tags': [], 'sentiment': 'Unknown'})}\n\n"


async def run_chat_chain(
    context: str, user_input: str, history: str, is_followup: bool
) -> AsyncGenerator[str, None]:
    llm = get_llm()

    # Selection logic based on context and follow-up status
    if context == "No context":
        prompt = no_context_prompt
    elif is_followup:
        prompt = rag_followup_prompt
    else:
        prompt = rag_initial_prompt

    chain = prompt | llm | StrOutputParser()

    try:
        async for chunk in chain.astream(
            {"context": context, "input": user_input, "history": history}
        ):
            yield f"data: {chunk}\n\n"
    except Exception as exc:
        yield f"data: Error: {exc}\n\n"


async def run_generate_random_note_chain() -> AsyncGenerator[str, None]:
    llm = get_llm()
    chain = random_note_prompt | llm | StrOutputParser()

    try:
        async for chunk in chain.astream({}):
            yield f"data: {chunk}\n\n"
    except Exception as exc:
        yield f"data: Error generating text.\n\n"


async def run_extract_event_date_chain(text: str, reference_date: str) -> dict:
    llm = get_parse_llm()
    # Use native structured output for high reliability
    structured_llm = llm.with_structured_output(EventExtraction)
    chain = extract_event_date_prompt | structured_llm

    try:
        # returns the Pydantic object directly
        result = await chain.ainvoke({"input": text, "reference_date": reference_date})
        return result.model_dump()
    except Exception as e:
        print(f"Extraction Error: {e}")
        return {"event_date": None, "original_expression": None, "confidence": 0.0}
