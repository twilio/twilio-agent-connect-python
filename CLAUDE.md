# CLAUDE.md

## Project Overview

Twilio Agent Connect (TAC) is a Python SDK — middleware (not an agent runtime) that enables LLM applications (OpenAI Agents SDK, Bedrock, LangChain, etc.) to use Twilio primitives: Conversation Memory for memory, Conversation Orchestrator for conversations, ConversationRelay for voice.

## Development Commands

```bash
make sync              # Install dependencies (uses uv)
make dev-setup         # Full dev setup with pre-commit hooks
make format            # Format with ruff
make lint              # Lint check only
make type-check        # mypy strict mode
make test              # Run pytest
make check             # All checks (lint + type-check + test)

uv run pytest tests/test_tac.py                      # Single test file
uv run pytest tests/test_tac.py::test_function_name  # Single test
```

When creating PRs, read and fill in `.github/PULL_REQUEST_TEMPLATE.md`.

## Package Structure

```
src/tac/
├── core/           # TAC class, TACConfig, context models
├── context/        # API clients: MemoryClient, ConversationClient, KnowledgeClient
├── models/         # Pydantic models (memory, conversation, session, voice, knowledge, intelligence)
├── channels/       # Communication channels (base, sms, rcs, whatsapp, chat, messaging, voice/)
│   └── voice/      # Voice channel (channel.py, twiml.py, config.py)
├── intelligence/   # Conversation Intelligence webhook processing
├── tools/          # LLM tool integration (@function_tool decorator, TACTool)
├── adapters/       # Runtime adapters (OpenAI memory injection, prompt builder)
└── server/         # Optional TACFastAPIServer (FastAPI-based, install with tac[server])
```

Tests are in `tests/` — one test file per module (e.g., `test_tac.py`, `test_sms_channel.py`, `test_rcs_channel.py`, `test_whatsapp_channel.py`, `test_chat_channel.py`, `test_voice_channel.py`).

## Code Conventions

- **Python 3.10+**: Use built-in generics (`list[str]`, `dict[str, Any]`) and union syntax (`X | None`, `X | Y`) instead of `typing.List`, `typing.Dict`, `typing.Optional`, `typing.Union`
- **mypy strict**: All functions need type hints, no incomplete defs
- **Pydantic v2**: Use `Field(alias=...)` for API name mapping, `model_config = {"populate_by_name": True}`, `.model_dump(by_alias=True, exclude_none=True)` for API payloads
- **ruff**: Line length 100, black-compatible formatting
- **Lint rules**: pycodestyle (E/W), pyflakes (F), isort (I), flake8-bugbear (B), flake8-comprehensions (C4), pyupgrade (UP)
- **Per-file ignores**: Examples allow E402 and E501

## Key Architecture Concepts

- **Channel-based**: Messaging channels (SMS, RCS, WhatsApp, Chat) and Voice channel process Twilio webhooks, manage conversation lifecycle, and trigger `on_message_ready` / `on_conversation_ended` callbacks
- **Callback responses**: Callbacks return `str` (auto-sent) or `None` (manual `channel.send_response()`)
- **Memory modes**: Channels support three memory retrieval modes:
  - `"never"` (default): No automatic memory retrieval
  - `"always"`: Fetch memory on every message with the user's query string for semantic search
  - `"once"`: Fetch once with empty query, cache it. Cache invalidated on INACTIVE (when Orchestrator updates memory). Uses `cache_lock` to coordinate concurrent async access
- **Memory fallback**: `TAC.retrieve_memory()` tries Conversation Memory first, gracefully falls back to Conversation Orchestrator's `list_communications()` on any failure
- **Profile resolution**: Automatic profile lookup by phone/email if `profile_id` not present in webhook
- **Memory auto-init**: Memory client is always initialized from Conversation Orchestrator configuration's `memory_store_id`
- **Auth**: All API clients use HTTP Basic Auth (API Key as username, API Token as password)
- **BaseAPIClient**: All API clients (ConversationClient, MemoryClient, KnowledgeClient) inherit from `BaseAPIClient`, which provides shared HTTP client configuration, authentication, and User-Agent header management following Twilio SDK conventions
- **Horizontal scaling limitation**: Channels track active conversations in instance-local memory (`self._conversations`). Works perfectly for single-instance deployments. In multi-instance deployments behind a load balancer, webhooks may route to a different instance than the one that handled the connection/message, preventing proper conversation cleanup and causing memory leaks. Recommended solutions: sticky sessions (route by conversation_id) or shared state store (Redis/database).
- **ConversationRelay-only mode**: When `conversation_configuration_id` is omitted from TACConfig, TAC runs with just the Voice channel (messaging channels raise at construction), `TAC.retrieve_memory()` returns an empty `TACMemoryResponse`, and the ConversationRelay callback handles session cleanup. Use `tac.is_orchestrator_enabled()` to check mode at runtime.

## OpenAI Adapter

The OpenAI adapter (`src/tac/adapters/openai/adapter.py`) supports both Chat Completions and Responses APIs for automatic memory injection:

**Chat Completions API**:
- Injects memory as system message at start of messages array
- Example: `client.chat.completions.create(model="gpt-5.4-mini", messages=[...])`

**Responses API**:
- Injects memory by prepending to instructions parameter
- Example: `client.responses.create(model="gpt-5.4-mini", instructions="...", input=[...])`

Both APIs are fully supported with sync/async variants and streaming support.

## Dependencies

- **Core**: `pydantic>=2`, `httpx>=0.27`, `twilio>=9.8.3`
- **Server** (optional): `fastapi`, `uvicorn`, `python-multipart` — install with `pip install tac[server]`
- **Dev**: `pytest`, `ruff`, `mypy`
- **Examples**: `openai`, `openai-agents`, `langchain-core`, `langchain-openai`, `boto3`, `strands-agents`, `python-dotenv`
