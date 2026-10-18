import asyncio
import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from pydantic_ai_agent.config import settings
from pydantic_ai_agent.enums import ConversationStatus
from pydantic_ai_agent.security import hash_password

engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


def _assert_safe_to_wipe() -> None:
    db_name = make_url(settings.database_url).database or ""
    if "test" not in db_name:
        raise RuntimeError(f"Refusing to drop tables on non-test DB '{db_name}'")


# 1. User Table
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)


# 2. Order Table
class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"))
    product_name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)


# 3. Ticket Table
# NOTE: composite UniqueConstraint order_id=NULL wale general tickets ko dedupe
# NAHI karta (Postgres mein NULL != NULL). Intentional hai, agent file ka TODO dekh.
class Ticket(Base):
    __tablename__ = "tickets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"))
    order_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("orders.id"), nullable=True)
    reason: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)

    __table_args__ = (UniqueConstraint("user_id", "order_id", name="uq_ticket_user_order"),)


# 4. Conversation Table
class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    messages: Mapped[list] = mapped_column(JSONB, default=list)
    status: Mapped[ConversationStatus] = mapped_column(
        SAEnum(
            ConversationStatus,
            native_enum=False,
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        default=ConversationStatus.IDLE,
    )
    pending_tool_call_ids: Mapped[list] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (Index("ix_conversation_user_updated", "user_id", "updated_at"),)


DEV_PASSWORD = "test-password"  # sirf development ke liye
DEV_PASSWORD_HASH = hash_password(DEV_PASSWORD)  # import time pe ek baar


async def init_db():
    _assert_safe_to_wipe()  # prod DB pe drop_all se bachata hai
    async with engine.begin() as conn:
        print("Puraani tables drop kar rahe hain (fresh start)...")
        await conn.run_sync(Base.metadata.drop_all)
        print("Nayi tables create kar rahe hain...")
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        shubham = User(name="Shubham", email="shubham@example.com", password_hash=DEV_PASSWORD_HASH)
        madhuri = User(name="Madhuri", email="madhuri@example.com", password_hash=DEV_PASSWORD_HASH)
        session.add_all([shubham, madhuri])
        await session.flush()

        session.add_all([
            Order(user_id=shubham.id, product_name="MacBook Pro", status="Shipped"),
            Order(user_id=shubham.id, product_name="Wireless Mouse", status="Processing"),
            Order(user_id=madhuri.id, product_name="Smartphone", status="Delivered"),
            Order(user_id=shubham.id, product_name="Keyboard", status="Delivered"),
        ])
        await session.commit()
        print("✅ Database setup complete and seed data inserted!")


if __name__ == "__main__":
    print(Base.metadata.tables.keys())
    asyncio.run(init_db())