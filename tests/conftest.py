"""
Pytest configuration for the e-commerce agent test suite.

IMPORTANT: the env override below must run at module level (not inside a
function/fixture) and BEFORE db_setup is imported. Python caches modules on
first import, so if db_setup's `engine` gets created against the real
DATABASE_URL once, no later os.environ change will undo that.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Point at the test database BEFORE db_setup (or anything importing it) loads.
os.environ["DATABASE_URL"] = os.environ["TEST_DATABASE_URL"]

import pytest
from pydantic_ai_agent.db_setup import init_db, AsyncSessionLocal, Order

from httpx import ASGITransport, AsyncClient
from pydantic_ai_agent.main import app


@pytest.fixture
async def client():
    """FastAPI ke endpoints ko bina server start kiye memory me test karne ke liye."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def db_session():
    """
    Fresh DB state before every single test:
    - drops + recreates tables
    - seeds the default users (Shubham id=1, Madhuri id=2) and their orders

    No cleanup needed after yield: the NEXT test's db_session call will do
    init_db() again, which drops everything fresh. This keeps every test
    independent of execution order.
    """
    await init_db()
    async with AsyncSessionLocal() as session:
        yield session


@pytest.fixture
async def add_extra_order(db_session):
    """
    Helper fixture for tests that need a DB state beyond the default seed
    (e.g. giving Shubham a second refundable order, or a third user).
    Use this INSIDE a specific test rather than permanently changing
    db_setup.py's seed data, so other tests aren't affected.

    Usage in a test:
        async def test_x(db_session, add_extra_order):
            order = await add_extra_order(user_id=1, product_name="Keyboard", status="Processing")
    """
    async def _add(user_id: int, product_name: str, status: str) -> Order:
        order = Order(user_id=user_id, product_name=product_name, status=status)
        db_session.add(order)
        await db_session.commit()
        await db_session.refresh(order)
        return order
    return _add