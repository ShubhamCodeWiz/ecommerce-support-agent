from datetime import datetime, timedelta, timezone
import jwt
from pwdlib import PasswordHash
from pydantic_ai_agent.config import settings

ALGORITHM = "HS256"

pwd_context = PasswordHash.recommended()


class TokenError(Exception):
    """Token missing/expired/tampered. Router layer isko 401 mein badlega."""


def create_access_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
    }
    return jwt.encode(payload, settings.secret_key.get_secret_value(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=[ALGORITHM],
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.PyJWTError as e:
        raise TokenError("Invalid or expired token") from e


def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)