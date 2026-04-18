from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.db import get_db
from app.models.models import User
from app.schemas.auth import RefreshTokenRequest, Token, UserCreate, UserResponse
from app.services.auth_service import login_user, refresh_access_token, register_user

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_create: UserCreate, session: AsyncSession = Depends(get_db)):
    """
    Register a new user.
    """
    new_user = await register_user(session, user_create)
    return new_user


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_db),
):
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    # form_data.username is used as username_or_email
    token = await login_user(session, form_data.username, form_data.password)
    return token


@router.post("/refresh", response_model=Token)
async def refresh_token(
    request: RefreshTokenRequest, session: AsyncSession = Depends(get_db)
):
    """
    Refresh an expired access token using a valid refresh token.
    """
    token = await refresh_access_token(session, request.refresh_token)
    return token


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """
    Get the current authenticated user's profile.
    """
    return current_user
