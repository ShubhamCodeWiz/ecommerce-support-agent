import uuid

import logfire
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic_ai import DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.usage import UsageLimits
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic_ai_agent.agent import DBDependencies, agent  # <- apne agent file ka asli naam daal
from pydantic_ai_agent.conversation_service import claim, release
from pydantic_ai_agent.db_setup import Conversation, User
from pydantic_ai_agent.deps import get_current_user, get_db
from pydantic_ai_agent.schemas import (
    ApprovalRequest,
    ChatRequest,
    ChatResponse,
    ConversationStatus,
    ToolApprovalItem,
)
from pydantic_ai.settings import ModelSettings


router = APIRouter(prefix="/chat", tags=["chat"])

USAGE_LIMITS = UsageLimits(request_limit=5)
MODEL_SETTINGS: ModelSettings = {"temperature": 0.0}


async def _create_conversation(session: AsyncSession, user_id: int) -> uuid.UUID:
    conv = Conversation(
        user_id=user_id,
        status=ConversationStatus.IDLE,
        messages=[],
        pending_tool_call_ids=[],
    )
    session.add(conv)
    await session.commit()
    return conv.id


async def _exists_for_user(session: AsyncSession, conv_id: uuid.UUID, user_id: int) -> bool:
    stmt = select(Conversation.id).where(
        Conversation.id == conv_id, Conversation.user_id == user_id
    )
    return (await session.execute(stmt)).scalar_one_or_none() is not None


async def _load_pending_ids(session: AsyncSession, conv_id: uuid.UUID, user_id: int) -> list[str]:
    stmt = select(Conversation.pending_tool_call_ids).where(
        Conversation.id == conv_id, Conversation.user_id == user_id
    )
    return (await session.execute(stmt)).scalar_one_or_none() or []


async def _claim_or_raise(
    session: AsyncSession,
    conv_id: uuid.UUID,
    user_id: int,
    expected: ConversationStatus,
) -> list:
    messages = await claim(session, conv_id, user_id, expected)
    if messages is None:
        # claim fail: ya toh row hai hi nahi / kisi aur ki hai (404), ya state galat hai (409)
        if await _exists_for_user(session, conv_id, user_id):
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "Conversation is busy or in a different state (e.g. waiting for approval).",
            )
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found.")
    return messages


async def _run_and_release(
    session: AsyncSession,
    conv_id: uuid.UUID,
    user_id: int,
    raw_history: list,
    restore_status: ConversationStatus,
    *,
    prompt: str | None = None,
    deferred: DeferredToolResults | None = None,
) -> ChatResponse:
    """Agent chalata hai, phir conversation release karta hai.
    Fail hone par status purane state pe wapas, history save nahi hoti."""
    deps = DBDependencies(session=session, user_id=user_id)

    try:
        history = ModelMessagesTypeAdapter.validate_python(raw_history)
        result = await agent.run(
            prompt,
            deps=deps,
            message_history=history,
            deferred_tool_results=deferred,
            model_settings=MODEL_SETTINGS,
            usage_limits=USAGE_LIMITS,
        )
    except Exception as e:
        await session.rollback()
        logfire.error("agent run failed", conversation_id=str(conv_id), user_id=user_id, error=str(e))
        await release(session, conv_id, user_id, restore_status)  # messages/pending untouched
        # raw error client ko mat bhej, andar ki details leak hoti hain
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "The assistant is temporarily unavailable. Please try again.",
        )

    new_messages = ModelMessagesTypeAdapter.dump_python(result.all_messages(), mode="json")

    if isinstance(result.output, DeferredToolRequests):
        calls = result.output.approvals
        await release(
            session,
            conv_id,
            user_id,
            ConversationStatus.AWAITING_APPROVAL,
            messages=new_messages,
            pending_tool_call_ids=[c.tool_call_id for c in calls],
        )
        return ChatResponse(
            conversation_id=conv_id,
            pending_approvals=[
                ToolApprovalItem(
                    tool_call_id=c.tool_call_id,
                    tool_name=c.tool_name,
                    args=c.args_as_dict(),
                )
                for c in calls
            ],
        )

    await release(
        session,
        conv_id,
        user_id,
        ConversationStatus.IDLE,
        messages=new_messages,
        pending_tool_call_ids=[],
    )
    return ChatResponse(conversation_id=conv_id, reply=result.output)


@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    user_id = current_user.id  # poora ORM object agent tak nahi jata, sirf id

    conv_id = body.conversation_id or await _create_conversation(session, user_id)
    raw_history = await _claim_or_raise(session, conv_id, user_id, ConversationStatus.IDLE)

    # Stale-recovery guard: idle conversation ke pending ids hamesha khali hote hain.
    # Agar nahi hain, toh ye ek crashed approvals-run thi, wapas awaiting_approval mein bhej.
    if await _load_pending_ids(session, conv_id, user_id):
        await release(session, conv_id, user_id, ConversationStatus.AWAITING_APPROVAL)
        raise HTTPException(status.HTTP_409_CONFLICT, "Conversation is waiting for approval.")

    return await _run_and_release(
        session, conv_id, user_id, raw_history, ConversationStatus.IDLE, prompt=body.message
    )


@router.post("/{conversation_id}/approvals", response_model=ChatResponse)
async def submit_approvals(
    conversation_id: uuid.UUID,
    body: ApprovalRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    user_id = current_user.id

    raw_history = await _claim_or_raise(
        session, conversation_id, user_id, ConversationStatus.AWAITING_APPROVAL
    )
    pending = await _load_pending_ids(session, conversation_id, user_id)

    # Pending khali hai matlab ye crashed /chat run thi, approval ka koi matlab nahi
    if not pending:
        await release(session, conversation_id, user_id, ConversationStatus.IDLE)
        raise HTTPException(status.HTTP_409_CONFLICT, "Nothing is waiting for approval.")

    # Exact match: na kam, na zyada
    if {d.tool_call_id for d in body.decisions} != set(pending):
        await release(session, conversation_id, user_id, ConversationStatus.AWAITING_APPROVAL)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Decisions must cover exactly the pending tool calls.",
        )

    deferred = DeferredToolResults()
    for d in body.decisions:
        deferred.approvals[d.tool_call_id] = (
            True if d.approved else ToolDenied("The customer did not approve this action.")
        )

    return await _run_and_release(
        session,
        conversation_id,
        user_id,
        raw_history,
        ConversationStatus.AWAITING_APPROVAL,
        deferred=deferred,
    )