from fastapi import HTTPException, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_password_hash,
    verify_password,
)
from app.models.models import User
from app.schemas.auth import Token, UserCreate


async def get_user_by_email_or_username(
    session: AsyncSession, username_or_email: str
) -> User | None:
    stmt = select(User).where(
        or_(User.email == username_or_email, User.username == username_or_email)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def register_user(session: AsyncSession, user_create: UserCreate) -> User:
    """Registers a new user in the database.

    Args:
        session: Database session.
        user_create: Schema containing username, email, and password.

    Returns:
        The created User model instance.

    Raises:
        HTTPException: If the email or username is already taken.
    """
    existing_user = await get_user_by_email_or_username(session, user_create.email)
    if existing_user:
        logger.bind(task="AUTH").warning(f"Registration failed: Email {user_create.email} already exists.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    existing_user_name = await get_user_by_email_or_username(session, user_create.username)
    if existing_user_name:
        logger.bind(task="AUTH").warning(f"Registration failed: Username {user_create.username} already exists.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken",
        )

    logger.bind(task="AUTH").info(f"Creating new user: {user_create.username}")
    hashed_password = get_password_hash(user_create.password)
    new_user = User(
        username=user_create.username,
        email=user_create.email,
        hashed_password=hashed_password,
    )
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)
    return new_user


async def authenticate_user(
    session: AsyncSession, username_or_email: str, password: str
) -> User | None:
    user = await get_user_by_email_or_username(session, username_or_email)
    if not user:
        logger.bind(task="AUTH").warning(f"Authentication failed: User {username_or_email} not found.")
        return None
    if not verify_password(password, user.hashed_password):
        logger.bind(task="AUTH").warning(f"Authentication failed: Invalid password for {username_or_email}.")
        return None
    logger.bind(task="AUTH").success(f"User {username_or_email} authenticated successfully.")
    return user


async def login_user(session: AsyncSession, username_or_email: str, password: str) -> Token:
    """Authenticates a user and generates access/refresh tokens.

    Args:
        session: Database session.
        username_or_email: The identifier provided by the user.
        password: The plain text password.

    Returns:
        A Token schema containing the tokens.

    Raises:
        HTTPException: If authentication fails.
    """
    user = await authenticate_user(session, username_or_email, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(subject=user.id)
    refresh_token = create_refresh_token(subject=user.id)
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        refresh_token=refresh_token
    )


async def refresh_access_token(session: AsyncSession, refresh_token: str) -> Token:
    """Validates a refresh token and generates a new token pair.

    Args:
        session: Database session.
        refresh_token: The refresh token string from the cookie.

    Returns:
        A new Token schema with updated tokens.

    Raises:
        HTTPException: If the token is invalid, expired, or the user no longer exists.
    """
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
    
    stmt = select(User).where(User.id == user_id)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token = create_access_token(subject=user.id)
    new_refresh_token = create_refresh_token(subject=user.id)
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        refresh_token=new_refresh_token
    )
