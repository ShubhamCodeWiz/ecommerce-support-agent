# E-commerce Support Agent

> A customer-support AI agent built with **Pydantic AI + FastAPI + PostgreSQL**. It only ever touches the logged-in customer's data, refunds need an explicit confirmation before anything changes, and conversation state survives pauses, crashes and concurrent requests.

<!-- Add a 60-90 second screen recording or GIF here. It is the most valuable part of this README. -->
![Demo](<<VERIFY: link to your demo GIF or video>>)

## What it does

A customer logs in and chats with an agent that can:

- list their orders and show one order's details
- check whether an order is eligible for a refund
- **request a refund**: the run pauses, the customer confirms or denies, and only then does the order move to `requested_refund` for a human agent to process
- open a support ticket (one open ticket per customer and order; a general complaint needs no order)
- refuse off-topic requests without calling any tool

## What makes it more than a chatbot with tools

| Concern | How it is handled |
|---|---|
| Whose data? | Identity comes from the JWT. No tool takes a `user_id`, and every query filters by it (IDOR-tested in both directions). |
| Business rules | Refund eligibility is plain code (`REFUNDABLE_STATUSES`). The model chooses tools and talks, it does not decide. |
| Destructive actions | `request_refund` raises `ApprovalRequired`. The paused run is stored in PostgreSQL and resumed only through `POST /chat/{id}/approvals`. |
| Approval integrity | Decisions must match the stored pending tool-call ids exactly. On resume the tool re-checks ownership and status before writing. |
| Concurrency | A conversation is claimed with one atomic `UPDATE ... RETURNING`. A parallel request gets `409`; a conversation stuck in `processing` for 3 minutes can be reclaimed. |
| Failures | A failed LLM call returns a generic `502`, restores the previous conversation state and saves no history. Raw errors never reach the client. |
| Cost control | `UsageLimits(request_limit=5)` on every run; messages are limited to 500 characters. |
| Auth hardening | PyJWT with required `exp`/`iat`/`sub`, Argon2 password hashing (`pwdlib`) off the event loop, identical error for unknown email and wrong password, a dummy hash to even out timing. |

## Architecture

```mermaid
sequenceDiagram
    participant C as Customer
    participant API as FastAPI
    participant A as Pydantic AI agent
    participant DB as PostgreSQL

    C->>API: POST /auth/login
    API-->>C: JWT
    C->>API: POST /chat "I want a refund for my mouse" + Bearer JWT
    API->>DB: claim conversation (atomic UPDATE)
    API->>A: run(history, deps = session + user_id from JWT)
    A->>DB: request_refund: ownership, duplicate, eligibility checks
    A-->>API: paused (ApprovalRequired)
    API->>DB: save history + pending tool_call_ids, status = awaiting_approval
    API-->>C: pending_approvals
    C->>API: POST /chat/{id}/approvals (approved: true)
    API->>DB: claim, ids must match pending exactly
    API->>A: resume with the decision
    A->>DB: re-check order, set status = requested_refund
    A-->>C: confirmation message
```

Conversation lifecycle:

```mermaid
stateDiagram-v2
    [*] --> idle: conversation created
    idle --> processing: POST /chat (claim)
    processing --> idle: reply saved
    processing --> awaiting_approval: refund needs confirmation
    awaiting_approval --> processing: POST approvals (claim)
```

If the LLM call fails, the conversation goes back to the state it was in and the history is not saved.

## API

| Method and path | Purpose |
|---|---|
| `POST /auth/login` | OAuth2 password form (`username` = email) and returns a JWT |
| `GET /me` | The current user |
| `POST /chat` | `{ "message": "...", "conversation_id": optional }` returns `reply` or `pending_approvals` |
| `POST /chat/{conversation_id}/approvals` | `{ "decisions": [{ "tool_call_id": "...", "approved": true }] }` |

## Tech stack

Python 3.12, FastAPI, Pydantic AI, SQLAlchemy 2 (async) with asyncpg and PostgreSQL (JSONB for stored history), PyJWT, pwdlib (Argon2), pydantic-settings, Logfire, pytest, pytest-asyncio, httpx. Packaged with `uv`. Model: Groq `openai/gpt-oss-120b`.

## Getting started

```bash
git clone https://github.com/ShubhamCodeWiz/ecommerce-support-agent.git
cd ecommerce-support-agent
uv sync
cp .env.example .env     # <<VERIFY: add .env.example with fake values to the repo>>
```

Required environment variables:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Async URL, e.g. `postgresql+asyncpg://user:pass@localhost/shop_test` |
| `TEST_DATABASE_URL` | Database used by the test suite |
| `SECRET_KEY` | JWT signing key, at least 32 characters |
| `GROQ_API_KEY` | LLM provider key |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Optional, default 30 |
| `LOGFIRE_TOKEN` | Optional, sends traces to Logfire |

Create the tables and demo data, then start the API:

```bash
uv run python -m pydantic_ai_agent.db_setup
uv run uvicorn pydantic_ai_agent.main:app --reload
```

Note: the seed script drops and recreates all tables, so it refuses to run unless the database name contains `test`.

Try it:

```bash
# demo users: shubham@example.com and madhuri@example.com, password: test-password
TOKEN=$(curl -s -X POST localhost:8000/auth/login \
  -d "username=shubham@example.com&password=test-password" | jq -r .access_token)

curl -s -X POST localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"message": "I want a refund for my wireless mouse"}'
```

The response contains a `conversation_id` and `pending_approvals`. Confirm with `POST /chat/{conversation_id}/approvals`.

Interactive docs are at `http://localhost:8000/docs`.

## Tests

```bash
uv run pytest
```

55 test functions in two groups:

- **36 deterministic tests, no LLM call.** Auth and token handling, password hashing, the conversation claim/release logic including a concurrent-claim race and stale-lease recovery, and the whole HTTP flow with a scripted `FunctionModel`: approval pause, wrong or extra tool-call ids, double approval, another user approving my refund, LLM failure restoring state.
- **19 live behavioural tests (`tests/test_agent.py`)** against the real model: no cross-user leaks, refund eligibility, asking which order when there are several, declining off-topic requests, escalation tickets, approval and denial. They assert on tool calls and tool-return text, not on the model's paraphrase.

The live tests need `GROQ_API_KEY`, cost API calls and can vary slightly between model versions.

## What I learned

- My first version let the model pass `user_id` into a tool, which is an authorization hole. Identity now comes from the token and is never a tool argument.
- `"eligible" in output` also passes for "not eligible", so tests assert both branches.
- I prototyped the approval flow in `poc_history.py` before building it into the API.
- A check-then-update on conversation state is a race. A single `UPDATE ... RETURNING` fixes it.

<<VERIFY: keep only the points that are true for you>>

## Built with AI assistance

Much of the code was drafted with an LLM. I read it, ran it, tested it and changed it. <<VERIFY: keep this note and make sure it is true>>

## Limitations and roadmap

- Refunds stop at `requested_refund`; there is no payment or ticketing integration. The customer confirms the action, there is no separate manager approval.
- No streaming responses yet.
- No rate limiting on login or on chat across conversations.
- Order status is a plain string; an Enum would be safer.
- Tables are created with `create_all`; no migrations yet.
- No Docker Compose or CI yet.
- Next: evals with `pydantic-evals`, agentic RAG over refund and shipping policy documents, and exposing the order tools through an MCP server.

## Project layout

```text
src/pydantic_ai_agent/
  main.py                  FastAPI app
  agent.py                 agent, tools, system prompt
  routers/auth.py          login
  routers/chat.py          chat and approvals endpoints
  conversation_service.py  atomic claim and release of conversations
  security.py              JWT and password hashing
  deps.py                  database session and current-user dependencies
  db_setup.py              models, seed data, guarded reset
  schemas.py, enums.py, config.py
tests/                     deterministic and live behavioural tests
poc_history.py             design spike for the approval flow
```
