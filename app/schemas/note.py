from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class EventExtraction(BaseModel):
    event_date: Optional[str] = Field(None, description="YYYY-MM-DD format")
    original_expression: Optional[str] = Field(None, description="The exact phrase found")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class NoteCreate(BaseModel):
    title: str = Field(
        default=None, max_length=255, description="Optional custom title"
    )
    content: str = Field(
        min_length=1, max_length=100000, description="The raw note text"
    )


class NoteUpdate(BaseModel):
    title: str | None = Field(
        default=None, max_length=255, description="Optional custom title"
    )
    content: str | None = Field(
        default=None, min_length=1, max_length=100000, description="The raw note text"
    )


class NoteResponse(BaseModel):
    id: int
    title: str | None
    content: str

    summary: Optional[str] = None
    tags: List[str] = []
    sentiment: Optional[str] = None
    occurrence_time: Optional[datetime] = None

    created_at: datetime
    updated_at: datetime

    @field_validator("tags", mode="before")
    @classmethod
    def validate_tags(cls, v):
        return v or []

    class Config:
        from_attributes = True


class NoteAnalysisUpdate(BaseModel):
    summary: str
    tags: List[str]
    sentiment: str


class ChatMessage(BaseModel):
    role: str
    content: str
    context_content: Optional[str] = None


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1, max_length=2000, description="Question asked to the AI"
    )
    history: List[ChatMessage] = []


class AnalyzeRequest(BaseModel):
    title: str | None = None
    content: str


class GenerateTitleRequest(BaseModel):
    content: str
