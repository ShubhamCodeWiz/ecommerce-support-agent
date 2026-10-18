import asyncio
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic_ai_agent.db_setup import User
from pydantic_ai_agent.deps import get_db
from pydantic_ai_agent.schemas import Token
from pydantic_ai_agent.security import (
    create_access_token,
    verify_password,
    hash_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])

DUMMY_HASH = hash_password("not-a-real-password")


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_db),
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

    result = await session.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    hash_to_check = user.password_hash if user else DUMMY_HASH
    ok = await asyncio.to_thread(verify_password, form_data.password, hash_to_check)

    if user is None or not ok:
        raise credentials_exception

    return Token(
        access_token=create_access_token(user.id),
        token_type="bearer",
    )