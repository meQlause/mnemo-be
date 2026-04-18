from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.GOOGLE_LLM_MODEL,
        temperature=0.3,
        google_api_key=settings.GOOGLE_API_KEY,
    )


@lru_cache(maxsize=1)
def get_parse_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.GOOGLE_PARSE_MODEL,
        temperature=0.0,
        google_api_key=settings.GOOGLE_API_KEY,
    )
