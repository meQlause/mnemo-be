from datetime import datetime, timedelta, timezone
from typing import Any, Union

import jwt
from passlib.context import CryptContext

from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_token(
    subject: Union[str, Any], expires_delta: timedelta, token_type: str = "access"
) -> str:
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode = {"exp": expire, "sub": str(subject), "type": token_type}
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def create_access_token(
    subject: Union[str, Any], expires_delta: timedelta | None = None
) -> str:
    if expires_delta:
        expire_delta = expires_delta
    else:
        expire_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return create_token(subject, expire_delta, "access")


def create_refresh_token(
    subject: Union[str, Any], expires_delta: timedelta | None = None
) -> str:
    if expires_delta:
        expire_delta = expires_delta
    else:
        expire_delta = timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
    return create_token(subject, expire_delta, "refresh")


def decode_token(token: str) -> dict:
    try:
        decoded_data = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        return decoded_data
    except jwt.PyJWTError as exc:
        raise ValueError("Invalid credentials") from exc
