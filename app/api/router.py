from fastapi import APIRouter
from app.api.routes import notes, auth

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Auth"])
api_router.include_router(notes.router, prefix="/notes", tags=["Notes"])
