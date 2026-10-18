import asyncio
from dotenv import load_dotenv
from pydantic_ai import (
    Agent,
    ApprovalRequired,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolDenied,
)
from pydantic_ai.messages import ModelMessagesTypeAdapter

load_dotenv()

agent = Agent(
    "groq:openai/gpt-oss-120b",
    output_type=[str, DeferredToolRequests],
    system_prompt=(
        "You are a support agent. Use refund_tool to process refunds "
        "for valid customer requests. If a tool call is denied, "
        "explain politely to the customer and do not retry."
    ),
)


@agent.tool
def refund_tool(ctx: RunContext, order_id: str) -> str:
    """Process a refund for the given order ID."""
    if not ctx.tool_call_approved:
        raise ApprovalRequired
    return f"Refund processed for order {order_id}."


async def resume(stored_json: bytes, call_id: str, approve: bool):
    history = ModelMessagesTypeAdapter.validate_json(stored_json)

    results = DeferredToolResults()
    if approve:
        results.approvals[call_id] = True
    else:
        results.approvals[call_id] = ToolDenied(
            "Refund rejected by manager: order is outside the refund window."
        )

    return await agent.run(
        message_history=history,
        deferred_tool_results=results,
    )


async def main():
    print("--- PHASE 1: first request ---")
    result = await agent.run(
        "Refund order ORD-123, customer received a damaged item"
    )

    if not isinstance(result.output, DeferredToolRequests):
        print("Model did NOT call the tool. Plain output:")
        print(result.output)
        return

    call_id = result.output.approvals[0].tool_call_id
    print(f"Paused. Waiting for approval on call_id={call_id}")

    print("\n--- PHASE 2: save to 'DB' ---")
    stored_json = ModelMessagesTypeAdapter.dump_json(result.all_messages())
    print(f"Saved {len(stored_json)} bytes")

    print("\n--- PHASE 3A: human APPROVES ---")
    approved = await resume(stored_json, call_id, approve=True)
    print(approved.output)

    print("\n--- PHASE 3B: human DENIES (same saved state) ---")
    denied = await resume(stored_json, call_id, approve=False)
    print(denied.output)

    print("\n--- Debug: what the model saw on denial ---")
    for msg in denied.all_messages():
        print(msg)


if __name__ == "__main__":
    asyncio.run(main())