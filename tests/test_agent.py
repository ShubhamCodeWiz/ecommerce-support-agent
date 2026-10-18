"""
Test suite for multi_trip_agent.

Convention: test_<behavior>_<condition>, not test_<tool_name>.
Every test follows Arrange (fixture gives fresh seeded DB) / Act (agent.run) / Assert.

model_settings={'temperature': 0.0} is used everywhere to keep output deterministic
enough for keyword/behavior assertions to be reliable, not flaky.

ASSERTION RULE (read this before adding a new test):
  Prefer get_tool_returns(messages, "<tool_name>") over the model's final
  result.output wherever a tool was actually called. Tool-return strings are
  OURS - we wrote the exact text in multi_trip_agent.py, so they're
  deterministic. result.output is the model's PARAPHRASE of that, which has
  repeatedly changed wording across runs (see the fragility map at the
  bottom of this file for concrete examples that bit us before).
  Only use result.output when:
    (a) no tool was called at all (e.g. the off-topic decline), or
    (b) as a SECOND, defense-in-depth check that the model didn't leak
        something in its own words even though the tool correctly said
        "not found" (see test_no_cross_user_data_leak for the pattern).
"""
import re

import pytest
from pydantic_ai import DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.messages import ModelResponse, ModelRequest, ToolCallPart, ToolReturnPart
from pydantic_ai.usage import UsageLimits
from sqlalchemy import select

from pydantic_ai_agent.agent import agent, DBDependencies
from pydantic_ai_agent.db_setup import Order, Ticket


def make_deps(session, user_id: int) -> DBDependencies:
    return DBDependencies(session=session, user_id=user_id)


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace (handles the \u202f narrow-no-break-space
    some model replies use around bold text/numbers)."""
    return re.sub(r"\s+", " ", text.lower())


def expect_text_output(result) -> str:
    """
    Narrow result.output to plain str before normalizing it.

    WHY THIS EXISTS (don't remove it / don't make normalize() accept
    DeferredToolRequests again): with Milestone 1's output_type=[str,
    DeferredToolRequests], result.output can legitimately be a
    DeferredToolRequests if a test forgets to resolve a pending approval.
    Silently stringifying that object (an earlier version of this file did
    this inside normalize()) lets a test pass or fail for the WRONG reason -
    it happened once: a test matched text against the literal Python repr
    of a DeferredToolRequests instead of against a real reply, because the
    approval was never actually resolved. This assert fails loudly and
    immediately instead, with a message that tells you what to fix.
    """
    assert isinstance(result.output, str), (
        f"Expected a plain text reply but got {type(result.output).__name__} - "
        "did you forget to call run_and_resolve_approval()?"
    )
    return normalize(result.output)


def get_tool_calls(messages) -> list[ToolCallPart]:
    """Every ToolCallPart across all ModelResponse messages in a run."""
    calls = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    calls.append(part)
    return calls


def get_tool_returns(messages, tool_name):
    """Every raw string a specific tool actually returned, across a run -
    OUR deterministic text, not the model's paraphrase of it."""
    returns = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in getattr(msg, "parts", []):
                if isinstance(part, ToolReturnPart) and part.tool_name == tool_name:
                    returns.append(part.content)
    return returns


def called_tool(messages, tool_name: str) -> bool:
    return any(c.tool_name == tool_name for c in get_tool_calls(messages))


def get_usage_limits() -> UsageLimits:
    return UsageLimits(request_limit=5)


async def run_and_resolve_approval(prompt, deps, *, approve: bool):
    """
    THE answer to "how do you test an approval pause without a real human
    typing into input()?": you build the exact same DeferredToolResults
    object main()'s resume-loop would have built from that keystroke, and
    hand it to a second agent.run() yourself. approve=True stands in for a
    human typing "y"; approve=False stands in for "n". Mirrors main()'s
    resume loop - if that loop's resolution logic changes, mirror it here.
    """
    result = await agent.run(
        prompt,
        deps=deps,
        model_settings={"temperature": 0.0},
        usage_limits=get_usage_limits(),
    )
    while isinstance(result.output, DeferredToolRequests):
        deferred = DeferredToolResults()
        for call in result.output.approvals:
            # Uses the SAME wording main() uses on a real "n" - keeps this
            # test representative of production, not an arbitrary sentinel.
            deferred.approvals[call.tool_call_id] = (
                True if approve else ToolDenied("The customer did not approve this action.")
            )
        result = await agent.run(
            deps=deps,
            message_history=result.all_messages(),
            deferred_tool_results=deferred,
            model_settings={"temperature": 0.0},
            usage_limits=get_usage_limits(),
        )
    return result


# ---------------------------------------------------------------------------
# get_my_orders
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "user_id,expected_products,forbidden_products",
    [
        (1, ["MacBook Pro", "Wireless Mouse", "Keyboard"], ["Smartphone"]),
        (2, ["Smartphone"], ["MacBook Pro", "Wireless Mouse", "Keyboard"]),
    ],
)
async def test_get_my_orders_returns_only_own_orders(
    db_session, user_id, expected_products, forbidden_products
):
    deps = make_deps(db_session, user_id)
    result = await agent.run(
        "show me my orders",
        deps=deps,
        model_settings={"temperature": 0.0},
        usage_limits=get_usage_limits(),
    )
    messages = result.all_messages()

    # Primary: the tool's own deterministic return.
    tool_text = " ".join(get_tool_returns(messages, "get_my_orders")).lower()
    for product in expected_products:
        assert product.lower() in tool_text
    for product in forbidden_products:
        assert product.lower() not in tool_text

    # Secondary leak-defense: the model shouldn't invent another user's
    # product in its own words either, even though the tool never gave it.
    output = expect_text_output(result)
    for product in forbidden_products:
        assert product.lower() not in output


# ---------------------------------------------------------------------------
# IDOR - get_order_details / check_refund_eligibility
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "user_id,other_users_order_id",
    [
        (2, 1),  # Madhuri asking for Shubham's MacBook Pro
        (1, 3),  # Shubham asking for Madhuri's Smartphone
    ],
)
async def test_no_cross_user_data_leak(db_session, user_id, other_users_order_id):
    deps = make_deps(db_session, user_id)
    result = await agent.run(
        f"show me details of order {other_users_order_id}",
        deps=deps,
        model_settings={"temperature": 0.0},
        usage_limits=get_usage_limits(),
    )
    messages = result.all_messages()
    assert "Order not found." in get_tool_returns(messages, "get_order_details")

    output = expect_text_output(result)  # leak-defense layer
    assert "macbook" not in output
    assert "mouse" not in output
    assert "smartphone" not in output


@pytest.mark.parametrize(
    "user_id,other_users_order_id,forbidden_products",
    [
        (2, 1, ["macbook", "mouse"]),
        (1, 3, ["smartphone"]),
    ],
)
async def test_no_cross_user_refund(db_session, user_id, other_users_order_id, forbidden_products):
    deps = make_deps(db_session, user_id)
    result = await agent.run(
        f"show me refund eligibility for order {other_users_order_id}",
        deps=deps,
        model_settings={"temperature": 0.0},
        usage_limits=get_usage_limits(),
    )
    messages = result.all_messages()
    assert "Order not found." in get_tool_returns(messages, "check_refund_eligibility")

    output = expect_text_output(result)  # leak-defense layer
    for product in forbidden_products:
        assert product not in output


@pytest.mark.parametrize(
    "user_id,order_id,expected_eligibility",
    [
        (1, 1, "not_eligible"),  # MacBook Pro, Shipped
        (1, 2, "eligible"),      # Wireless Mouse, Processing
        (2, 3, "eligible"),      # Smartphone, Delivered
    ],
)
async def test_refund_eligibility_matches_actual_status(
    db_session, user_id, order_id, expected_eligibility
):
    """
    "eligible" is a substring of "not eligible" - both branches are checked
    explicitly so a wrong verdict can't accidentally pass via substring match.
    """
    deps = make_deps(db_session, user_id)
    result = await agent.run(
        f"show me refund eligibility for order {order_id}",
        deps=deps,
        model_settings={"temperature": 0.0},
        usage_limits=get_usage_limits(),
    )
    returns = get_tool_returns(result.all_messages(), "check_refund_eligibility")
    assert returns, "check_refund_eligibility was not called"
    tool_text = returns[0].lower()

    if expected_eligibility == "eligible":
        assert "eligible" in tool_text and "not eligible" not in tool_text
    else:
        assert "not eligible" in tool_text


# ---------------------------------------------------------------------------
# Refund flow: order-inference when the customer doesn't give an order id
# ---------------------------------------------------------------------------
async def test_refund_flow_infers_order_when_single_order_exists(db_session):
    """
    "I want a refund" with no order id, from a user with exactly ONE order
    -> get_my_orders, then request_refund with THAT real id (not guessed).
    This now goes through request_refund (not check_refund_eligibility) per
    the current system prompt, so it pauses for approval - resolved here.
    Madhuri (user_id=2) has exactly one seeded order: Smartphone, id=3.
    """
    deps = make_deps(db_session, 2)
    result = await run_and_resolve_approval("I want a refund", deps, approve=True)

    messages = result.all_messages()
    assert called_tool(messages, "get_my_orders")
    refund_calls = [c for c in get_tool_calls(messages) if c.tool_name == "request_refund"]
    assert refund_calls, "request_refund was never called"
    assert "3" in str(refund_calls[0].args)  # Madhuri's real order id, not guessed

    order = (await db_session.execute(select(Order).where(Order.id == 3))).scalar_one_or_none()
    assert order.status == "requested_refund"


async def test_refund_flow_asks_when_multiple_orders_exist(db_session):
    """Shubham (user_id=1, 3 orders) asking for a refund with no order id
    -> agent must ask which one, never guess by calling request_refund."""
    deps = make_deps(db_session, 1)
    result = await agent.run(
        "I want a refund",
        deps=deps,
        model_settings={"temperature": 0.0},
        usage_limits=get_usage_limits(),
    )
    messages = result.all_messages()
    assert called_tool(messages, "get_my_orders")
    assert not called_tool(messages, "request_refund")


# ---------------------------------------------------------------------------
# Off-topic / guardrail - the ONE place a final-output check is primary,
# because no tool is called at all here, so there's no tool-return to check.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "user_id,off_topic_question",
    [
        (1, "solve 2 + 2"),
        (2, "write a python loop"),
    ],
)
async def test_off_topic_request_declines_without_tools(db_session, user_id, off_topic_question):
    deps = make_deps(db_session, user_id)
    result = await agent.run(
        off_topic_question,
        deps=deps,
        model_settings={"temperature": 0.0},
        usage_limits=get_usage_limits(),
    )
    output = expect_text_output(result)
    assert "e-commerce assistant" in output
    assert len(get_tool_calls(result.all_messages())) == 0


@pytest.mark.parametrize(
    "user_id,non_existent_order_id",
    [
        (1, 999),
        (2, 888),
    ],
)
async def test_order_details_nonexistent_id(db_session, user_id, non_existent_order_id):
    deps = make_deps(db_session, user_id)
    result = await agent.run(
        f"show me details of order {non_existent_order_id}",
        deps=deps,
        model_settings={"temperature": 0.0},
        usage_limits=get_usage_limits(),
    )
    messages = result.all_messages()
    assert "Order not found." in get_tool_returns(messages, "get_order_details")


# ---------------------------------------------------------------------------
# Escalation - capfire is Logfire's OWN testing fixture, not caplog. caplog
# only captures stdlib `logging`; Logfire routes through OpenTelemetry spans.
# ---------------------------------------------------------------------------
async def test_escalation_logs_user_id(db_session, capfire):
    deps = make_deps(db_session, 1)
    result = await agent.run(
        "I am angry, I want to speak to a human",
        deps=deps,
        model_settings={"temperature": 0.0},
        usage_limits=get_usage_limits(),
    )
    returns = get_tool_returns(result.all_messages(), "escalate_to_human")
    assert any("ticket" in r.lower() for r in returns)

    spans = capfire.exporter.exported_spans_as_dict()
    escalation_spans = [s for s in spans if "escalation" in s.get("name", "").lower()]
    assert escalation_spans, "no escalation log/span was captured"
    attributes = escalation_spans[0].get("attributes", {})
    assert attributes.get("user_id") == 1


async def test_escalation_creates_ticket(db_session):
    """Escalating creates a Ticket row linked to the right user."""
    deps = make_deps(db_session, 2)
    result = await agent.run(
        "I am angry, I want to speak to a human regarding order id 3",
        deps=deps,
        model_settings={"temperature": 0.0},
        usage_limits=get_usage_limits(),
    )
    returns = get_tool_returns(result.all_messages(), "escalate_to_human")
    assert any("ticket" in r.lower() for r in returns)

    ticket = (
        await db_session.execute(select(Ticket).where(Ticket.user_id == 2))
    ).scalar_one_or_none()
    assert ticket is not None, "No ticket was created for user_id=2"


@pytest.mark.parametrize(
    "user_id,other_users_order_id",
    [
        (2, 1),  # Madhuri referencing Shubham's MacBook Pro
        (1, 3),  # Shubham referencing Madhuri's Smartphone
    ],
)
async def test_no_cross_user_escalation(db_session, user_id, other_users_order_id):
    """
    REGRESSION TEST for the ownership check in escalate_to_human: a user
    should not be able to open a ticket referencing an order that isn't
    theirs. Mirrors test_no_cross_user_data_leak / test_no_cross_user_refund.
    """
    deps = make_deps(db_session, user_id)
    result = await agent.run(
        f"I want to escalate to a human regarding order id {other_users_order_id}",
        deps=deps,
        model_settings={"temperature": 0.0},
        usage_limits=get_usage_limits(),
    )
    messages = result.all_messages()
    assert "Order not found." in get_tool_returns(messages, "escalate_to_human")

    tickets = (
        await db_session.execute(
            select(Ticket).where(
                (Ticket.user_id == user_id) & (Ticket.order_id == other_users_order_id)
            )
        )
    ).scalars().all()
    assert len(tickets) == 0, "No ticket should exist referencing someone else's order"


async def test_duplicate_escalation_does_not_create_multiple_tickets(db_session):
    """Escalating twice for the same user+order creates only ONE ticket."""
    deps = make_deps(db_session, 1)
    order_id = 2

    await agent.run(
        f"I want to escalate to a human regarding order id {order_id}",
        deps=deps,
        model_settings={"temperature": 0.0},
        usage_limits=get_usage_limits(),
    )
    result = await agent.run(
        f"I want to escalate to a human regarding order id {order_id}",
        deps=deps,
        model_settings={"temperature": 0.0},
        usage_limits=get_usage_limits(),
    )
    returns = get_tool_returns(result.all_messages(), "escalate_to_human")
    assert any("already exists" in r.lower() for r in returns)

    tickets = (
        await db_session.execute(
            select(Ticket).where((Ticket.user_id == 1) & (Ticket.order_id == order_id))
        )
    ).scalars().all()
    assert len(tickets) == 1, "Multiple tickets were created for the same user and order"


async def test_escalation_flow_creates_ticket_with_two_different_orders(db_session):
    """
    REGRESSION TEST for the Ticket schema bug: user_id and order_id used to
    be unique=True SEPARATELY instead of as a composite, so a single user
    could only ever have ONE ticket in the table's entire history across
    ALL orders. Proves it's fixed - escalating about two different orders
    must both succeed.

    NOTE: both order ids below belong to Shubham (user_id=1) - an earlier
    version of this test used order_id=3, which is Madhuri's, not Shubham's.
    It still passed, because escalate_to_human does NOT check order
    ownership (unlike get_order_details/check_refund_eligibility/
    request_refund, which all call _get_owned_order). That's a real gap -
    see the fragility map at the bottom of this file.
    """
    deps = make_deps(db_session, 1)
    order_id_1, order_id_2 = 1, 2

    await agent.run(
        f"I want to escalate to a human regarding order id {order_id_1}",
        deps=deps,
        model_settings={"temperature": 0.0},
        usage_limits=get_usage_limits(),
    )
    await agent.run(
        f"I want to escalate to a human regarding order id {order_id_2}",
        deps=deps,
        model_settings={"temperature": 0.0},
        usage_limits=get_usage_limits(),
    )

    tickets = (
        await db_session.execute(
            select(Ticket).where(
                (Ticket.user_id == 1)
                & ((Ticket.order_id == order_id_1) | (Ticket.order_id == order_id_2))
            )
        )
    ).scalars().all()
    assert len(tickets) == 2, "Tickets were not created for both different orders"


async def test_escalation_flow_with_no_order_id_creates_ticket(db_session):
    """
    REGRESSION TEST for the nullable order_id bug: order_id used to be
    NOT NULL even though escalate_to_human allows order_id=None for a
    general complaint - inserting with order_id=None used to raise an
    IntegrityError. Proves it's fixed.
    """
    deps = make_deps(db_session, 1)
    await agent.run(
        "I want to escalate to a human",
        deps=deps,
        model_settings={"temperature": 0.0},
        usage_limits=get_usage_limits(),
    )

    tickets = (
        await db_session.execute(select(Ticket).where(Ticket.user_id == 1))
    ).scalars().all()
    assert len(tickets) == 1, "No ticket was created when no order ID was provided"


# ---------------------------------------------------------------------------
# request_refund
# ---------------------------------------------------------------------------
async def test_request_refund_for_nonexistent_order(db_session):
    """Ownership check happens before the approval gate, so this resolves
    to plain text without ever pausing - nothing to resolve here."""
    deps = make_deps(db_session, 1)
    result = await agent.run(
        "I want to request a refund for order id 9999",
        deps=deps,
        model_settings={"temperature": 0.0},
        usage_limits=get_usage_limits(),
    )
    assert not isinstance(result.output, DeferredToolRequests)
    assert "Order not found." in get_tool_returns(result.all_messages(), "request_refund")


async def test_refund_request_for_ineligible_order(db_session):
    """Eligibility check also happens before the approval gate."""
    deps = make_deps(db_session, 1)
    result = await agent.run(
        "I want to request a refund for order id 1",  # MacBook Pro, Shipped
        deps=deps,
        model_settings={"temperature": 0.0},
        usage_limits=get_usage_limits(),
    )
    assert not isinstance(result.output, DeferredToolRequests)
    returns = get_tool_returns(result.all_messages(), "request_refund")
    assert any("not eligible for refund" in r for r in returns)


async def test_duplicate_refund_request_prevention(db_session):
    """
    Approving a refund once sets status to requested_refund; asking again
    for the SAME order short-circuits on the duplicate check (which fires
    before the approval gate) and needs no second approval.
    """
    deps = make_deps(db_session, 1)
    order_id = 2  # Wireless Mouse, Processing -> eligible

    first = await run_and_resolve_approval(
        f"I want to request a refund for order id {order_id}", deps, approve=True
    )
    first_returns = get_tool_returns(first.all_messages(), "request_refund")
    assert any("submitted" in r.lower() for r in first_returns)

    second = await run_and_resolve_approval(
        f"I want to request a refund for order id {order_id}", deps, approve=True
    )
    second_returns = get_tool_returns(second.all_messages(), "request_refund")
    assert any("already been requested" in r.lower() for r in second_returns)
    # the duplicate check fires before the approval gate, so there should be
    # no second pause to resolve:
    assert not isinstance(second.output, DeferredToolRequests)


async def test_refund_request_flow_db_status_change(db_session):
    """An APPROVED refund request actually updates the order's status in the DB."""
    deps = make_deps(db_session, 1)
    order_id = 2  # Wireless Mouse, Processing -> eligible

    await run_and_resolve_approval(
        f"I want to request a refund for order id {order_id}", deps, approve=True
    )

    order = (
        await db_session.execute(select(Order).where(Order.id == order_id))
    ).scalar_one_or_none()
    assert order is not None
    assert order.status == "requested_refund"


async def test_request_refund_with_denied_approval(db_session):
    """
    CONFIRMED (via `pytest -s -k denied_approval`): a ToolDenied's message
    reaches the model as the tool's literal return content, unchanged -
    get_tool_returns() picks it up reliably, same as any other tool return.

    Two guarantees checked:
    1. The DB is NOT mutated on denial (the real security property).
    2. The denial reason reaches the model as a normal tool return (UX -
       the customer's final reply should be grounded in a real reason,
       not an LLM-invented one).
    """
    deps = make_deps(db_session, 1)
    order_id = 2  # eligible, but we'll deny it

    result = await run_and_resolve_approval(
        f"I want to request a refund for order id {order_id}", deps, approve=False
    )
    assert isinstance(result.output, str), "expected the run to resolve to text after denial"

    order = (
        await db_session.execute(select(Order).where(Order.id == order_id))
    ).scalar_one_or_none()
    assert order.status != "requested_refund"

    returns = get_tool_returns(result.all_messages(), "request_refund")
    assert any("did not approve" in r.lower() for r in returns)


# ---------------------------------------------------------------------------
# FRAGILITY MAP - what future changes break what, and why.
# Read this before changing db_setup.py, the system prompt, or any tool.
# ---------------------------------------------------------------------------
#
# 1. db_setup.py seed data (order of User/Order creation, product names,
#    statuses) - almost EVERY test in this file hardcodes an order_id,
#    user_id, or product name (e.g. order_id=2 is assumed to be Shubham's
#    Wireless Mouse, Processing). IDs are assigned by INSERTION ORDER, not
#    by variable name - this already bit us once (order3/order4 were
#    declared out of insertion order). If you ever reorder or add seed
#    rows, expect most tests to break and need their hardcoded IDs updated.
#
# 2. Exact tool-return wording in multi_trip_agent.py - changing any of
#    these literal strings breaks the matching test(s):
#      "Order not found."                 -> IDOR tests, nonexistent-id tests
#      "eligible for refund." / "not eligible for refund."
#                                          -> test_refund_eligibility_matches_actual_status
#      "already exists"                   -> test_duplicate_escalation_does_not_create_multiple_tickets
#      "already been requested"           -> test_duplicate_refund_request_prevention
#      "submitted"                        -> test_duplicate_refund_request_prevention (first call)
#      "Ticket" (in escalate_to_human's return)
#                                          -> test_escalation_logs_user_id, test_escalation_creates_ticket
#    This is a deliberate, cheap coupling - these are OUR strings, so
#    updating a test when we deliberately change one is a 1-line fix, not a
#    flaky-test hunt.
#
# 3. REFUNDABLE_STATUSES set - changing which statuses are refundable
#    breaks test_refund_eligibility_matches_actual_status and any test
#    that assumes order 2 (Processing) or order 3 (Delivered) is eligible.
#
# 4. Check ORDER inside request_refund (ownership -> duplicate ->
#    eligibility -> approval) - tests like test_request_refund_for_
#    nonexistent_order and test_refund_request_for_ineligible_order
#    assume those checks resolve BEFORE the approval gate, so they expect
#    a plain string with no DeferredToolRequests pause. If the approval
#    check is ever moved earlier in that function, those tests will need
#    run_and_resolve_approval() too.
#
# 5. escalate_to_human NOW verifies order ownership (added after this file
#    was first written) - test_no_cross_user_escalation covers it, mirroring
#    the same pattern as get_order_details/check_refund_eligibility. If this
#    check is ever removed, that test will fail and point straight back here.
#
# 6. pydantic-ai version upgrades - DeferredToolRequests/DeferredToolResults/
#    ToolDenied are a newer, fast-moving part of the API. If
#    run_and_resolve_approval() or the resume-call shape in main() ever
#    throws after an upgrade, re-check https://ai.pydantic.dev/deferred-tools
#    before assuming the test (or the agent code) is wrong.
#
# 7. System prompt rule 2/3 wording (when to call get_my_orders vs.
#    request_refund) - test_refund_flow_infers_order_when_single_order_exists
#    and test_refund_flow_asks_when_multiple_orders_exist depend on the
#    MODEL's tool-choice behavior, which is prompt-driven and only
#    probabilistically stable even at temperature=0.0 across model/provider
#    updates (Groq could update the underlying model weights without
#    warning). These two are the least deterministic tests in this file.
# ---------------------------------------------------------------------------