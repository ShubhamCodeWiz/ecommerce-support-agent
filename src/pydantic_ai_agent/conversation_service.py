import uuid
from datetime import timedelta

from sqlalchemy import and_, func, or_, update
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic_ai_agent.db_setup import Conversation
from pydantic_ai_agent.enums import ConversationStatus

# Agar conversation itni der se "processing" mein hai, toh maan lete hain
# ki server beech mein mar gaya tha, aur use dobara claim karne dete hain.
STALE_TIMEOUT_MINUTES = 3


async def claim(
    session: AsyncSession,
    conv_id: uuid.UUID,
    user_id: int,
    expected_status: ConversationStatus,
) -> list | None:
    """Atomic claim: ek hi UPDATE mein check bhi aur set bhi.

    Kaamyab hone par purani messages history return hoti hai.
    None ka matlab: row nahi hai, kisi aur user ki hai, ya status galat hai.
    Caller ko 404 aur 409 mein farak khud nikalna hai.
    """
    stale_cutoff = func.now() - timedelta(minutes=STALE_TIMEOUT_MINUTES)

    stmt = (
        update(Conversation)
        .where(
            Conversation.id == conv_id,
            Conversation.user_id == user_id,
            or_(
                Conversation.status == expected_status,
                and_(
                    Conversation.status == ConversationStatus.PROCESSING,
                    Conversation.updated_at < stale_cutoff,
                ),
            ),
        )
        .values(status=ConversationStatus.PROCESSING, updated_at=func.now())
        .returning(Conversation.messages)
    )

    result = await session.execute(stmt)
    messages = result.scalar_one_or_none()
    await session.commit()  # claim turant commit, taaki lock LLM call tak na rahe
    return messages


async def release(
    session: AsyncSession,
    conv_id: uuid.UUID,
    user_id: int,
    new_status: ConversationStatus,
    messages: list | None = None,
    pending_tool_call_ids: list | None = None,
) -> None:
    """Conversation ko 'processing' se nikaal kar new_status par set karta hai.

    messages / pending_tool_call_ids None ho toh DB mein wahi purani value rehti hai.
    """
    values: dict = {"status": new_status, "updated_at": func.now()}
    if messages is not None:
        values["messages"] = messages
    if pending_tool_call_ids is not None:
        values["pending_tool_call_ids"] = pending_tool_call_ids

    stmt = (
        update(Conversation)
        .where(Conversation.id == conv_id, Conversation.user_id == user_id)
        .values(**values)
    )
    await session.execute(stmt)
    await session.commit()