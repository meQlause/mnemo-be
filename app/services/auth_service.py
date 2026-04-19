from fastapi import HTTPException, status
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_password_hash,
    verify_password,
)
from app.models.models import User
from app.repositories import user_repository
from app.schemas.auth import Token, UserCreate, UserLoginParams, UserPersistenceParams


async def get_user_by_email_or_username(
    session: AsyncSession, username_or_email: str
) -> User | None:
    """Retrieves a user by their email or username."""
    return await user_repository.get_user_by_email_or_username(
        session, username_or_email
    )


async def register_user(session: AsyncSession, user_create: UserCreate) -> User:
    """Registers a new user in the database."""
    existing_user = await get_user_by_email_or_username(session, user_create.email)
    if existing_user:
        logger.bind(task="AUTH").warning(
            f"Registration failed: Email {user_create.email} already exists."
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    existing_user_name = await get_user_by_email_or_username(
        session, user_create.username
    )
    if existing_user_name:
        logger.bind(task="AUTH").warning(
            f"Registration failed: Username {user_create.username} already exists."
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken",
        )

    logger.bind(task="AUTH").info(f"Creating new user: {user_create.username}")
    hashed_password = get_password_hash(user_create.password)

    params = UserPersistenceParams(
        username=user_create.username,
        email=user_create.email,
        hashed_password=hashed_password,
    )
    return await user_repository.create_user(session=session, params=params)


async def authenticate_user(
    session: AsyncSession, params: UserLoginParams
) -> User | None:
    """Verifies user credentials."""
    user = await get_user_by_email_or_username(session, params.username_or_email)
    if not user:
        logger.bind(task="AUTH").warning(
            f"Authentication failed: User {params.username_or_email} not found."
        )
        return None
    if not verify_password(params.password, user.hashed_password):
        logger.bind(task="AUTH").warning(
            f"Authentication failed: Invalid password for {params.username_or_email}."
        )
        return None
    logger.bind(task="AUTH").success(
        f"User {params.username_or_email} authenticated successfully."
    )
    return user


async def login_user(session: AsyncSession, params: UserLoginParams) -> Token:
    """Authenticates a user and generates access/refresh tokens."""
    user = await authenticate_user(session, params)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(subject=user.id)
    refresh_token = create_refresh_token(subject=user.id)

    return Token(
        access_token=access_token, token_type="bearer", refresh_token=refresh_token
    )


async def refresh_access_token(session: AsyncSession, refresh_token: str) -> Token:
    """Validates a refresh token and generates a new token pair."""
    try:
        payload = decode_token(refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user_id_str = payload.get("sub")
        if user_id_str is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = int(user_id_str)

    user = await user_repository.get_user_by_id(session, user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(subject=user.id)
    new_refresh_token = create_refresh_token(subject=user.id)

    return Token(
        access_token=access_token, token_type="bearer", refresh_token=new_refresh_token
    )
