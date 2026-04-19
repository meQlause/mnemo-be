from datetime import datetime, date
from app.utils.date_utils import get_jakarta_now
from typing import List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Date, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=get_jakarta_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=get_jakarta_now, onupdate=get_jakarta_now
    )

    notes: Mapped[List["Note"]] = relationship(
        "Note", back_populates="user", cascade="all, delete-orphan"
    )


class Note(Base):
    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id"), nullable=False
    )
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    event_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    event_start_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    event_end_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=get_jakarta_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=get_jakarta_now, onupdate=get_jakarta_now
    )
    
    event_confidence: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    event_reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list, nullable=True)
    sentiment: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    metadata_: Mapped[dict] = mapped_column(JSONB, default=dict)

    user: Mapped["User"] = relationship("User", back_populates="notes")
    chunks: Mapped[List["NoteChunk"]] = relationship(
        "NoteChunk", back_populates="note", cascade="all, delete-orphan"
    )


class NoteChunk(Base):
    __tablename__ = "note_chunks"

    __table_args__ = (
        Index(
            "idx_note_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("idx_note_chunks_chunk_index", "chunk_index"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    note_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("notes.id"), nullable=False
    )
    chunk_content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    embedding: Mapped[Vector] = mapped_column(Vector(768))

    note: Mapped["Note"] = relationship("Note", back_populates="chunks")
