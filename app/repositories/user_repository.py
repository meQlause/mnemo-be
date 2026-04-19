from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.models import User
from app.schemas.auth import UserPersistenceParams

async def get_user_by_email_or_username(
    session: AsyncSession, username_or_email: str
) -> User | None:
    """Retrieves a user by their email or username.

    Args:
        session: Database session.
        username_or_email: The identifier to search for.

    Returns:
        The User model instance or None if not found.
    """
    stmt = select(User).where(
        or_(User.email == username_or_email, User.username == username_or_email)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()

async def get_user_by_id(session: AsyncSession, user_id: int) -> User | None:
    """Retrieves a user by their primary ID.

    Args:
        session: Database session.
        user_id: The user's ID.

    Returns:
        The User model instance or None if not found.
    """
    stmt = select(User).where(User.id == user_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()

async def create_user(
    session: AsyncSession, 
    params: UserPersistenceParams
) -> User:
    """Creates a new user record in the database.

    Args:
        session: Database session.
        params: Object containing username, email, and hashed password.

    Returns:
        The newly created User model instance.
    """
    new_user = User(
        username=params.username,
        email=params.email,
        hashed_password=params.hashed_password,
    )
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)
    return new_user
