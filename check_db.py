
import asyncio
from app.core.config import settings
from app.models.models import Note, NoteChunk
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

async def count():
    engine = create_async_engine(settings.postgres_dsn)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as s:
        notes = (await s.execute(select(Note))).scalars().all()
        chunks = (await s.execute(select(NoteChunk))).scalars().all()
        print(f"Notes: {len(notes)}, Chunks: {len(chunks)}")
        for n in notes:
            print(f"Note ID: {n.id}, Title: {n.title}")

if __name__ == "__main__":
    asyncio.run(count())
