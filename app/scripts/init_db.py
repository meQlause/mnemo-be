import asyncio
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sqlalchemy import text
from app.core.db import engine
from app.models.models import Base

async def init_db():
    print("Initializing database...")
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        print("Creating tables...")
        await conn.run_sync(Base.metadata.create_all)
    
    print("Database initialization complete!")
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(init_db())
