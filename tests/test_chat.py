import uuid
from types import SimpleNamespace

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from sqlalchemy import select

from pydantic_ai_agent.agent import agent  # <- apni agent file ka asli naam
from pydantic_ai_agent.db_setup import AsyncSessionLocal, Conversation, Order
from pydantic_ai_agent.enums import ConversationStatus
from pydantic_ai_agent.security import create_access_token

SHUBHAM = 1  # seed: Shubham id=1, Madhuri id=2
MADHURI = 2


# ---------------------------------------------------------------------------
# Fake LLMs: deterministic, koi real Groq call nahi
# ---------------------------------------------------------------------------
def plain_model(messages, info):
    return ModelResponse(parts=[TextPart("Hello from the test model.")])


def failing_model(messages, info):
    raise RuntimeError("LLM exploded: secret-internal-detail")


def refund_model(order_id: int):
    """Pehle request_refund tool call karta hai; tool result aane par text se jawab deta hai."""

    def fn(messages, info):
        last = messages[-1]
        if isinstance(last, ModelRequest):
            for part in last.parts:
                if isinstance(part, ToolReturnPart):
                    return ModelResponse(parts=[TextPart(f"Tool result: {part.content}")])
        return ModelResponse(
            parts=[ToolCallPart(tool_name="request_refund", args={"order_id": order_id})]
        )

    return fn


def use_model(fn):
    return agent.override(model=FunctionModel(fn))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def auth(user_id: int) -> dict:
    return {"Authorization": f"Bearer {create_access_token(user_id)}"}


async def conv_state(conv_id) -> SimpleNamespace:
    """DB se taaza state, alag session se (koi stale cache nahi)."""
    async with AsyncSessionLocal() as s:
        row = (
            await s.execute(
                select(
                    Conversation.status,
                    Conversation.messages,
                    Conversation.pending_tool_call_ids,
                ).where(Conversation.id == uuid.UUID(str(conv_id)))
            )
        ).one()
    return SimpleNamespace(status=row.status, messages=row.messages, pending=row.pending_tool_call_ids)


async def order_id_of(product_name: str) -> int:
    async with AsyncSessionLocal() as s:
        return (await s.execute(select(Order.id).where(Order.product_name == product_name))).scalar_one()


async def order_status(order_id: int) -> str:
    async with AsyncSessionLocal() as s:
        return (await s.execute(select(Order.status).where(Order.id == order_id))).scalar_one()


async def start_refund_chat(client, user_id: int, order_id: int) -> dict:
    """Refund request bhejta hai; approval pe ruki hui response ka JSON return karta hai."""
    res = await client.post(
        "/chat", json={"message": "I want a refund"}, headers=auth(user_id)
    )
    assert res.status_code == 200, res.text
    return res.json()


def approve_body(tool_call_id: str, approved: bool) -> dict:
    return {"decisions": [{"tool_call_id": tool_call_id, "approved": approved}]}


# ---------------------------------------------------------------------------
# Auth + validation
# ---------------------------------------------------------------------------
async def test_chat_endpoints_require_auth(client, db_session):
    assert (await client.post("/chat", json={"message": "hi"})).status_code == 401
    fake = uuid.uuid4()
    res = await client.post(f"/chat/{fake}/approvals", json=approve_body("x", True))
    assert res.status_code == 401


async def test_message_validation(client, db_session):
    for bad in ["", "   ", "x" * 501]:
        res = await client.post("/chat", json={"message": bad}, headers=auth(SHUBHAM))
        assert res.status_code == 422, bad


# ---------------------------------------------------------------------------
# Normal chat + persistence
# ---------------------------------------------------------------------------
async def test_new_chat_returns_reply_and_persists(client, db_session):
    with use_model(plain_model):
        res = await client.post("/chat", json={"message": "hi"}, headers=auth(SHUBHAM))

    assert res.status_code == 200
    body = res.json()
    assert body["reply"] == "Hello from the test model."
    assert body["pending_approvals"] == []

    state = await conv_state(body["conversation_id"])
    assert state.status == ConversationStatus.IDLE
    assert len(state.messages) >= 2  # request + response
    assert state.pending == []


async def test_conversation_history_grows_across_turns(client, db_session):
    with use_model(plain_model):
        first = (await client.post("/chat", json={"message": "hi"}, headers=auth(SHUBHAM))).json()
        n1 = len((await conv_state(first["conversation_id"])).messages)

        second = await client.post(
            "/chat",
            json={"conversation_id": first["conversation_id"], "message": "and again"},
            headers=auth(SHUBHAM),
        )

    assert second.status_code == 200
    assert second.json()["conversation_id"] == first["conversation_id"]
    n2 = len((await conv_state(first["conversation_id"])).messages)
    assert n2 > n1


async def test_response_never_leaks_raw_history(client, db_session):
    with use_model(plain_model):
        res = await client.post("/chat", json={"message": "hi"}, headers=auth(SHUBHAM))
    assert set(res.json().keys()) == {"conversation_id", "reply", "pending_approvals"}


# ---------------------------------------------------------------------------
# Ownership: 404 rule
# ---------------------------------------------------------------------------
async def test_unknown_conversation_is_404(client, db_session):
    res = await client.post(
        "/chat",
        json={"conversation_id": str(uuid.uuid4()), "message": "hi"},
        headers=auth(SHUBHAM),
    )
    assert res.status_code == 404


async def test_other_users_conversation_is_404_not_403(client, db_session):
    with use_model(plain_model):
        conv_id = (await client.post("/chat", json={"message": "hi"}, headers=auth(SHUBHAM))).json()[
            "conversation_id"
        ]
        before = await conv_state(conv_id)

        res = await client.post(
            "/chat",
            json={"conversation_id": conv_id, "message": "I am Madhuri"},
            headers=auth(MADHURI),
        )
        res_appr = await client.post(
            f"/chat/{conv_id}/approvals", json=approve_body("x", True), headers=auth(MADHURI)
        )

    assert res.status_code == 404
    assert res_appr.status_code == 404
    after = await conv_state(conv_id)
    assert after.status == ConversationStatus.IDLE
    assert len(after.messages) == len(before.messages)  # Madhuri ne kuch nahi badla


# ---------------------------------------------------------------------------
# Human-in-the-loop flow
# ---------------------------------------------------------------------------
async def test_refund_pauses_for_approval(client, db_session):
    mouse = await order_id_of("Wireless Mouse")  # Shubham, Processing -> refundable

    with use_model(refund_model(mouse)):
        body = await start_refund_chat(client, SHUBHAM, mouse)

    assert body["reply"] is None
    assert len(body["pending_approvals"]) == 1
    item = body["pending_approvals"][0]
    assert item["tool_name"] == "request_refund"
    assert item["args"] == {"order_id": mouse}

    state = await conv_state(body["conversation_id"])
    assert state.status == ConversationStatus.AWAITING_APPROVAL
    assert state.pending == [item["tool_call_id"]]
    assert await order_status(mouse) == "Processing"  # abhi kuch commit nahi hua


async def test_new_message_while_awaiting_approval_is_409(client, db_session):
    mouse = await order_id_of("Wireless Mouse")
    with use_model(refund_model(mouse)):
        body = await start_refund_chat(client, SHUBHAM, mouse)
        res = await client.post(
            "/chat",
            json={"conversation_id": body["conversation_id"], "message": "hello?"},
            headers=auth(SHUBHAM),
        )

    assert res.status_code == 409
    state = await conv_state(body["conversation_id"])
    assert state.status == ConversationStatus.AWAITING_APPROVAL  # state na bigda


async def test_wrong_or_extra_tool_call_ids_are_422_and_state_recovers(client, db_session):
    mouse = await order_id_of("Wireless Mouse")
    with use_model(refund_model(mouse)):
        body = await start_refund_chat(client, SHUBHAM, mouse)
        conv_id = body["conversation_id"]
        real_id = body["pending_approvals"][0]["tool_call_id"]
        url = f"/chat/{conv_id}/approvals"

        # 1. bilkul galat id
        res = await client.post(url, json=approve_body("made-up-id", True), headers=auth(SHUBHAM))
        assert res.status_code == 422

        # 2. sahi id + ek extra
        extra = {
            "decisions": [
                {"tool_call_id": real_id, "approved": True},
                {"tool_call_id": "extra-id", "approved": True},
            ]
        }
        res = await client.post(url, json=extra, headers=auth(SHUBHAM))
        assert res.status_code == 422

        # 3. duplicate id (schema validator)
        dup = {
            "decisions": [
                {"tool_call_id": real_id, "approved": True},
                {"tool_call_id": real_id, "approved": False},
            ]
        }
        res = await client.post(url, json=dup, headers=auth(SHUBHAM))
        assert res.status_code == 422

        # reject hone ke baad bhi conversation awaiting_approval mein hai, aur
        # asli approval ab bhi kaam karta hai
        assert (await conv_state(conv_id)).status == ConversationStatus.AWAITING_APPROVAL
        ok = await client.post(url, json=approve_body(real_id, True), headers=auth(SHUBHAM))
        assert ok.status_code == 200


async def test_approve_commits_refund_in_db(client, db_session):
    mouse = await order_id_of("Wireless Mouse")
    with use_model(refund_model(mouse)):
        body = await start_refund_chat(client, SHUBHAM, mouse)
        res = await client.post(
            f"/chat/{body['conversation_id']}/approvals",
            json=approve_body(body["pending_approvals"][0]["tool_call_id"], True),
            headers=auth(SHUBHAM),
        )

    assert res.status_code == 200, res.text
    assert "submitted" in res.json()["reply"]
    assert res.json()["pending_approvals"] == []

    # asli proof: DB mein effect
    assert await order_status(mouse) == "requested_refund"
    state = await conv_state(body["conversation_id"])
    assert state.status == ConversationStatus.IDLE
    assert state.pending == []


async def test_deny_leaves_order_unchanged(client, db_session):
    mouse = await order_id_of("Wireless Mouse")
    with use_model(refund_model(mouse)):
        body = await start_refund_chat(client, SHUBHAM, mouse)
        res = await client.post(
            f"/chat/{body['conversation_id']}/approvals",
            json=approve_body(body["pending_approvals"][0]["tool_call_id"], False),
            headers=auth(SHUBHAM),
        )

    assert res.status_code == 200, res.text
    assert "did not approve" in res.json()["reply"]
    assert await order_status(mouse) == "Processing"  # sabse zaroori assert
    state = await conv_state(body["conversation_id"])
    assert state.status == ConversationStatus.IDLE


async def test_double_approval_second_is_409(client, db_session):
    mouse = await order_id_of("Wireless Mouse")
    with use_model(refund_model(mouse)):
        body = await start_refund_chat(client, SHUBHAM, mouse)
        url = f"/chat/{body['conversation_id']}/approvals"
        payload = approve_body(body["pending_approvals"][0]["tool_call_id"], True)

        first = await client.post(url, json=payload, headers=auth(SHUBHAM))
        second = await client.post(url, json=payload, headers=auth(SHUBHAM))

    assert first.status_code == 200
    assert second.status_code == 409
    assert await order_status(mouse) == "requested_refund"


async def test_other_user_cannot_approve_my_refund(client, db_session):
    mouse = await order_id_of("Wireless Mouse")
    with use_model(refund_model(mouse)):
        body = await start_refund_chat(client, SHUBHAM, mouse)
        res = await client.post(
            f"/chat/{body['conversation_id']}/approvals",
            json=approve_body(body["pending_approvals"][0]["tool_call_id"], True),
            headers=auth(MADHURI),
        )

    assert res.status_code == 404
    assert await order_status(mouse) == "Processing"
    assert (await conv_state(body["conversation_id"])).status == ConversationStatus.AWAITING_APPROVAL


async def test_agent_cannot_refund_another_users_order(client, db_session):
    """Tool-level IDOR HTTP ke through: Madhuri ka LLM Shubham ka order refund karna chahe."""
    mouse = await order_id_of("Wireless Mouse")  # Shubham ka
    with use_model(refund_model(mouse)):
        res = await client.post("/chat", json={"message": "refund"}, headers=auth(MADHURI))

    assert res.status_code == 200
    body = res.json()
    assert body["pending_approvals"] == []  # approval tak pahunchna hi nahi chahiye
    assert "not found" in body["reply"].lower()
    assert await order_status(mouse) == "Processing"


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------
async def test_llm_failure_returns_502_and_restores_state(client, db_session):
    with use_model(failing_model):
        res = await client.post("/chat", json={"message": "hi"}, headers=auth(SHUBHAM))

    assert res.status_code == 502
    assert "secret-internal-detail" not in res.text  # andar ki details leak nahi

    async with AsyncSessionLocal() as s:
        conv = (
            await s.execute(select(Conversation).where(Conversation.user_id == SHUBHAM))
        ).scalar_one()
    assert conv.status == ConversationStatus.IDLE  # processing mein atki nahi
    assert conv.messages == []  # adhuri history save nahi hui