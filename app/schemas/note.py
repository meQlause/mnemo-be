from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class EventExtraction(BaseModel):
    event_date: Optional[str] = Field(None, description="YYYY-MM-DD or range format")
    event_confidence: str = Field("LOW", description="HIGH | MEDIUM | LOW")
    event_reasoning: Optional[str] = Field(None, description="Why this date was chosen")


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
    event_date: Optional[str] = None
    event_confidence: Optional[str] = None
    event_reasoning: Optional[str] = None

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
