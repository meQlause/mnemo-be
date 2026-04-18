from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=settings.GOOGLE_EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        output_dimensionality=768,
    )
