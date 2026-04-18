from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import ValidationError

from app.core.db import get_db
from app.core.security import decode_token
from app.models.models import User
from app.schemas.auth import TokenData

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme), session: AsyncSession = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
        user_id_str: str = payload.get("sub")
        token_type: str = payload.get("type")
        if user_id_str is None or token_type != "access":
            raise credentials_exception
        token_data = TokenData(sub=user_id_str)
    except (ValueError, ValidationError):
        raise credentials_exception

    user_id = int(token_data.sub)
    stmt = select(User).where(User.id == user_id)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
        
    return user
