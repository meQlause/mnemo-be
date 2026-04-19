from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi_csrf_protect import CsrfProtect
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.db import get_db
from app.models.models import User
from app.schemas.auth import Token, UserCreate, UserResponse
from app.services.auth_service import login_user, refresh_access_token, register_user

router = APIRouter()


@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def register(user_create: UserCreate, session: AsyncSession = Depends(get_db)):
    """Registers a new user in the system.

    Args:
        user_create: The user data including username, email, and password.
        session: Database session dependency.

    Returns:
        The newly created user profile.
    """
    new_user = await register_user(session, user_create)
    return new_user


@router.post("/login", response_model=Token)
async def login(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_db),
    csrf_protect: CsrfProtect = Depends(),
):
    """Authenticates a user and sets security cookies.

    Performs OAuth2 compatible token login. Sets refresh_token and 
    csrf_token in Secure/HttpOnly cookies for the session.

    Args:
        response: FastAPI response object used to set cookies.
        form_data: Standard OAuth2 password request form (username/password).
        session: Database session dependency.
        csrf_protect: CSRF protection dependency.

    Returns:
        The access token and token type.
    """
    token = await login_user(session, form_data.username, form_data.password)

    response.set_cookie(
        key=settings.REFRESH_TOKEN_COOKIE_NAME,
        value=token.refresh_token,
        httponly=True,
        samesite=settings.COOKIE_SAMESITE,
        secure=settings.COOKIE_SECURE,
        max_age=settings.REFRESH_TOKEN_EXPIRE_MINUTES * 60,
    )

    _, signed_token = csrf_protect.generate_csrf_tokens()
    csrf_protect.set_csrf_cookie(signed_token, response)

    return Token(access_token=token.access_token, token_type=token.token_type)


@router.post("/refresh", response_model=Token)
async def refresh_token(
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_db),
    csrf_protect: CsrfProtect = Depends(),
):
    """Refreshes an expired access token.

    Validates the refresh token from cookies and the CSRF token from headers.
    If valid, issues new access and refresh tokens.

    Args:
        request: The incoming request containing cookies and headers.
        response: FastAPI response object used to set new cookies.
        session: Database session dependency.
        csrf_protect: CSRF protection dependency.

    Returns:
        A new access token.

    Raises:
        HTTPException: If the refresh token is missing or CSRF/Auth validation fails.
    """

    refresh_token_value = request.cookies.get(settings.REFRESH_TOKEN_COOKIE_NAME)
    if not refresh_token_value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing",
        )

    token = await refresh_access_token(session, refresh_token_value)

    response.set_cookie(
        key=settings.REFRESH_TOKEN_COOKIE_NAME,
        value=token.refresh_token,
        httponly=True,
        samesite=settings.COOKIE_SAMESITE,
        secure=settings.COOKIE_SECURE,
        max_age=settings.REFRESH_TOKEN_EXPIRE_MINUTES * 60,
    )

    _, signed_token = csrf_protect.generate_csrf_tokens()
    csrf_protect.set_csrf_cookie(signed_token, response)

    return Token(
        access_token=token.access_token, token_type=token.token_type, refresh_token=None
    )


@router.post("/logout")
async def logout(response: Response):
    """Logs out the current user by clearing security cookies.

    Args:
        response: FastAPI response object used to delete cookies.

    Returns:
        A confirmation message.
    """
    response.delete_cookie(settings.REFRESH_TOKEN_COOKIE_NAME)
    response.delete_cookie(settings.CSRF_TOKEN_COOKIE_NAME)
    return {"detail": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """
    Get the current authenticated user's profile.
    """
    return current_user
