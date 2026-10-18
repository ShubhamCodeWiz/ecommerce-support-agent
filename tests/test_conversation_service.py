import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from pydantic_ai_agent.conversation_service import claim, release
from pydantic_ai_agent.db_setup import AsyncSessionLocal, Conversation
from pydantic_ai_agent.enums import ConversationStatus

SHUBHAM = 1  # db_session fixture ka seed: Shubham id=1, Madhuri id=2
MADHURI = 2


async def _make_conversation(**kwargs) -> Conversation:
    """Alag session mein row banata hai, taaki test ka session saaf rahe."""
    conv = Conversation(user_id=kwargs.pop("user_id", SHUBHAM), **kwargs)
    async with AsyncSessionLocal() as s:
        s.add(conv)
        await s.commit()
    return conv


async def test_concurrent_claims_only_one_wins(db_session):
    conv = await _make_conversation(status=ConversationStatus.IDLE, messages=[{"role": "test"}])

    async def attempt():
        async with AsyncSessionLocal() as own_session:  # har claim ka alag session
            return await claim(own_session, conv.id, SHUBHAM, ConversationStatus.IDLE)

    results = await asyncio.gather(attempt(), attempt())

    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    assert winners[0] == [{"role": "test"}]


async def test_enum_stored_as_lowercase_value_in_db(db_session):
    conv = await _make_conversation(status=ConversationStatus.AWAITING_APPROVAL)

    raw = await db_session.execute(
        text("SELECT status FROM conversations WHERE id = :cid"), {"cid": conv.id}
    )
    assert raw.scalar_one() == "awaiting_approval"


async def test_claim_release_lifecycle(db_session):
    conv = await _make_conversation(status=ConversationStatus.IDLE)

    # idle -> processing
    assert await claim(db_session, conv.id, SHUBHAM, ConversationStatus.IDLE) == []

    # processing -> awaiting_approval, history aur pending ids ke saath
    history = [{"kind": "request"}, {"kind": "response"}]
    await release(
        db_session, conv.id, SHUBHAM, ConversationStatus.AWAITING_APPROVAL,
        messages=history, pending_tool_call_ids=["call_123"],
    )

    # galat expected status se claim nahi hona chahiye
    assert await claim(db_session, conv.id, SHUBHAM, ConversationStatus.IDLE) is None
    # sahi status se hona chahiye, aur saved history milni chahiye
    assert await claim(db_session, conv.id, SHUBHAM, ConversationStatus.AWAITING_APPROVAL) == history


async def test_release_without_messages_keeps_existing_history(db_session):
    conv = await _make_conversation(
        status=ConversationStatus.PROCESSING, messages=[{"keep": "me"}], pending_tool_call_ids=["x"]
    )

    await release(db_session, conv.id, SHUBHAM, ConversationStatus.IDLE)  # None, None

    row = await db_session.execute(
        text("SELECT messages, pending_tool_call_ids FROM conversations WHERE id = :cid"),
        {"cid": conv.id},
    )
    messages, pending = row.one()
    assert messages == [{"keep": "me"}]
    assert pending == ["x"]


async def test_other_users_cannot_claim(db_session):
    conv = await _make_conversation(status=ConversationStatus.IDLE)

    assert await claim(db_session, conv.id, MADHURI, ConversationStatus.IDLE) is None
    # aur asli owner ka claim abhi bhi chalta hai (Madhuri ne kuch bigada nahi)
    assert await claim(db_session, conv.id, SHUBHAM, ConversationStatus.IDLE) == []


async def test_stale_processing_can_be_reclaimed(db_session):
    five_min_ago = datetime.now(timezone.utc) - timedelta(minutes=5)
    conv = await _make_conversation(
        status=ConversationStatus.PROCESSING,
        messages=[{"role": "stuck_chat"}],
        updated_at=five_min_ago,
    )

    assert await claim(db_session, conv.id, SHUBHAM, ConversationStatus.IDLE) == [{"role": "stuck_chat"}]


async def test_fresh_processing_cannot_be_reclaimed(db_session):
    # upar wale test ka ulta. Isi se pata chalta hai ki stale check sach mein kaam kar raha hai.
    conv = await _make_conversation(status=ConversationStatus.PROCESSING, messages=[{"role": "busy"}])

    assert await claim(db_session, conv.id, SHUBHAM, ConversationStatus.IDLE) is None