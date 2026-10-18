import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic_ai_agent.config import settings
from pydantic_ai_agent.db_setup import User, DEV_PASSWORD
from pydantic_ai_agent.security import hash_password, create_access_token


# Sirf un 2 tests ke liye jahan non-dev password verify karna hai
@pytest.fixture
async def custom_user(db_session: AsyncSession) -> User:
    user = User(
        name="Tester",
        email="tester@example.com",
        password_hash=hash_password("pw-123"),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


# --- 1. Custom password se login success ---
async def test_login_success(client: AsyncClient, custom_user: User):
    response = await client.post(
        "/auth/login",
        data={"username": "tester@example.com", "password": "pw-123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


# --- 2. Bug Catcher: Custom user dev password se login na kar sake ---
async def test_login_fails_with_dev_password(client: AsyncClient, custom_user: User):
    response = await client.post(
        "/auth/login",
        data={"username": "tester@example.com", "password": DEV_PASSWORD},
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Incorrect email or password"}


# --- 3. Seeded user DEV_PASSWORD se login kar sake ---
async def test_seeded_user_login_success(client: AsyncClient):
    response = await client.post(
        "/auth/login",
        data={"username": "shubham@example.com", "password": DEV_PASSWORD},
    )
    assert response.status_code == 200
    assert "access_token" in response.json()


# --- 4. User Enumeration Guard ---
async def test_login_identical_error_for_user_enumeration(client: AsyncClient):
    res_fake_user = await client.post(
        "/auth/login",
        data={"username": "notfound@example.com", "password": "any-password"},
    )
    res_wrong_pw = await client.post(
        "/auth/login",
        data={"username": "shubham@example.com", "password": "wrong-password"},
    )

    assert res_fake_user.status_code == 401
    assert res_wrong_pw.status_code == 401
    assert res_fake_user.json() == res_wrong_pw.json()


# --- 5. /me endpoint security checks ---
async def test_me_endpoint_security(client: AsyncClient, monkeypatch: pytest.MonkeyPatch):
    # Case A: Bina token ke -> 401
    res_no_token = await client.get("/me")
    assert res_no_token.status_code == 401

    # Case B: Tampered signature -> 401
    valid_token = create_access_token(user_id=1)
    tampered_token = valid_token[:-1] + ("a" if valid_token[-1] != "a" else "b")
    res_tampered = await client.get(
        "/me",
        headers={"Authorization": f"Bearer {tampered_token}"},
    )
    assert res_tampered.status_code == 401

    # Case C: Expired token (monkeypatch se past token banaya) -> 401
    monkeypatch.setattr(settings, "access_token_expire_minutes", -10)
    expired_token = create_access_token(user_id=1)
    monkeypatch.undo()  # Wapas 30 minutes kar diya

    res_expired = await client.get(
        "/me",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert res_expired.status_code == 401

    # Case D: Deleted / Non-existent user -> 401
    ghost_token = create_access_token(user_id=9999)
    res_ghost = await client.get(
        "/me",
        headers={"Authorization": f"Bearer {ghost_token}"},
    )
    assert res_ghost.status_code == 401

    # Case E: Valid token -> 200 OK aur user data
    res_valid = await client.get(
        "/me",
        headers={"Authorization": f"Bearer {valid_token}"},
    )
    assert res_valid.status_code == 200
    assert res_valid.json()["id"] == 1
    assert res_valid.json()["email"] == "shubham@example.com"