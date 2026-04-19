from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GOOGLE_API_KEY: str = "placeholder"
    GOOGLE_LLM_MODEL: str = "gemini-3-flash-preview"
    GOOGLE_PARSE_MODEL: str = "gemini-2.5-pro"
    GOOGLE_EMBEDDING_MODEL: str = "gemini-embedding-001"

    POSTGRES_HOST: str = "127.0.0.1"
    POSTGRES_PORT: int = 8883
    POSTGRES_USER: str = "raguser"
    POSTGRES_PASSWORD: str = "ragpassword"
    POSTGRES_DB: str = "ragdb"
    
    MISTRAL_API_KEY: str = "placeholder"
    MISTRAL_PARSE_MODEL: str = "mistral-small-latest"

    COLLECTION_NAME: str = "rag_notes_store"
    CHUNK_SIZE: int = 768
    CHUNK_OVERLAP: int = 100
    RETRIEVER_K: int = 5
    SEMANTIC_THRESHOLD_UI: float = 0.4
    SEMANTIC_THRESHOLD_CHAT: float = 0.3
    WINDOW_SIZE: int = 1

    SECRET_KEY: str = "supersecretkey_please_change_in_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    REFRESH_TOKEN_COOKIE_NAME: str = "refresh_token"
    CSRF_TOKEN_COOKIE_NAME: str = "fastapi-csrf-token"
    SECRET_KEY_CSRF: str = "supersecretkey_please_change_in_production"
    COOKIE_SAMESITE: str = "strict"
    COOKIE_SECURE: bool = False

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
