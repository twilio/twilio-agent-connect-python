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
- **Memory modes**:
  - `"never"` (default): No automatic memory retrieval
  - `"always"`: Fetch memory on every message with the user's query string for semantic search
  - `"once"` (**Voice only**): Fetch once with empty query, cache it on the session. Invalidated on INACTIVE; uses `cache_lock` for concurrent async access. Needs a session outliving one request, which only voice has — on messaging channels it is deprecated (removed in 3.0): `MessagingChannel.__init__` warns and runs it as `"always"`.
  - `memory_config.fetch_profile_traits=False` drops the `get_profile` call when prompts don't use `build_profile_prompt()`.
- **Memory fallback**: `TAC.retrieve_memory()` tries Conversation Memory first, gracefully falls back to Conversation Orchestrator's `list_communications()` on any failure
- **Profile resolution**: Automatic profile lookup by phone/email if `profile_id` not present in webhook
- **Memory auto-init**: Memory client is always initialized from Conversation Orchestrator configuration's `memory_store_id`
- **Auth**: All API clients use HTTP Basic Auth (API Key as username, API Token as password)
- **BaseAPIClient**: All API clients (ConversationClient, MemoryClient, KnowledgeClient) inherit from `BaseAPIClient`, which provides shared HTTP client configuration, authentication, and User-Agent header management following Twilio SDK conventions
- **Horizontal scaling**: N replicas behind a load balancer, **no shared datastore**. See `docs/deployment.md`.
  - **Messaging is stateless** — `MessagingChannel` has no session store; each webhook derives a `ConversationSession` from the payload, config, and one `list_participants` call it needs anyway. The one thing kept between webhooks is a bounded (24 h sliding TTL, 10k entries), best-effort per-instance cache of each conversation's `session.metadata`, so it carries across turns on the same instance as on `main`; `ConversationSession.conversation_metadata()` reads the CO-stored metadata (written at outbound initiation) on any instance. The idempotency LRU is per-process, so a retry landing elsewhere is reprocessed: `on_message_ready` handlers must be idempotent.
  - **Voice is instance-local**, correctly: a WebSocket pins a call to one process. Safe because teardown is guaranteed — `_release_session()` runs on every disconnect path and fires `on_call_ended` (invariant: `assert_no_residual_state` in `tests/voice_invariants.py`). `VoiceChannel.aclose()` drains at shutdown, wired by `TACFastAPIServer`.
  - **Out-of-band call webhooks** (status/AMD/recording, `<Connect action>`) carry only a `CallSid`. Set `TACConfig.instance_public_domain` and TAC mints every callback URL against it; otherwise route by `CallSid` at the balancer, or use `VoiceChannel.resolve_conversation_session_by_call_sid()` (orchestrated ConversationRelay only), which falls back to an identity-only session rebuilt from CO.
  - **Per-call Media Streams `session_config` requires `instance_public_domain` with more than one instance.** The config (from `on_inbound_call_session_config` or outbound `session_config`) is kept in the memory of the instance that served the TwiML or placed the call; only a token rides the TwiML. `MediaStreamsOpenAIProvider._claim_session_config` warns when a stream's token isn't held locally, and the call falls back to `default_session_config`. Deferred fix (G2): re-derive inbound configs from TwiML fields forwarded as stream parameters.
  - **`on_conversation_ended`** fires on whichever replica gets CO's CLOSED webhook — both families rebuild the session from CO. A CLOSED for a call still live on that replica fires from a snapshot and leaves the session for teardown, so `on_call_ended` still fires. `on_call_ended` is the only hook carrying live in-memory state (transcript, metadata).
  - **Lifecycle analytics** ("Conversation Started"/"Conversation Ended") come from CO webhooks wherever a CO conversation exists: `PARTICIPANT_ADDED` for the customer, and `CONVERSATION_UPDATED`/CLOSED with `duration_ms` = CO's `updatedAt − createdAt`. Each channel reports only its own channel type (`BaseChannel._track_conversation_started` / `_track_conversation_ended`), so one webhook handed to every channel counts once per channel (at-least-once across webhook retries and replicas, like the callbacks). The CLOSED participant lookup is shared via `TAC._list_participants_shared` (in-flight + 60 s), and is skipped when analytics is off and no `on_conversation_ended` is registered. Relay-only and Media Streams voice report at session start/teardown instead. On chat, where customers stay UNKNOWN (no customer reconciliation), each UNKNOWN non-agent participant counts as a start, so a conversation with several such participants over-counts.
- **ConversationRelay-only mode**: When `conversation_configuration_id` is omitted from TACConfig, TAC runs with just the Voice channel (messaging channels raise at construction), `TAC.retrieve_memory()` returns an empty `TACMemoryResponse`, and the ConversationRelay callback handles session cleanup. Use `tac.is_orchestrator_enabled()` to check mode at runtime.
- **Multi-sender support**: Each channel has a singular default sender plus a plural allowlist (`phone_number`/`phone_numbers`, `rcs_sender_id`/`rcs_sender_ids`, `whatsapp_number`/`whatsapp_numbers`). Inbound derives the agent address from the webhook's recipient (SMS/RCS/WhatsApp) or the dialed number (voice), validates it against the allowlist. Messaging (SMS/RCS/WhatsApp) drops messages addressed to an unconfigured number; voice falls back to the default sender (with a warning) for a call to an unrecognized dialed number. Outbound selects the sender via an optional `from_` (defaulting to the channel's default sender), and digital handoff sends From the session's active agent number.

## OpenAI Adapter

The OpenAI adapter (`src/tac/adapters/openai/adapter.py`) supports both Chat Completions and Responses APIs for automatic memory injection:

**Chat Completions API**:
- Injects memory as system message at start of messages array
- Example: `client.chat.completions.create(model="gpt-5.4-mini", messages=[...])`

**Responses API**:
- Injects memory by prepending to instructions parameter
- Example: `client.responses.create(model="gpt-5.4-mini", instructions="...", input=[...])`

Both APIs are fully supported with sync/async variants and streaming support.

## Documentation

- The full public API reference is published at https://twilio.github.io/twilio-agent-connect-python/ — consult it for the surface exposed to package consumers.
- API docs are generated from docstrings by MkDocs Material + mkdocstrings, so **docstrings are published documentation** — write them for both source readers and the rendered site.
- Use **Markdown** in docstrings and `Field(description=...)` text: `**bold**` for emphasis/headers, `- ` bullet lists, backticks for code/identifiers, and fenced code blocks for examples. These render on the docs site.
- Keep Markdown blocks well-formed: leave a blank line before a bullet list or fenced block, and indent consistently, so mkdocstrings parses them correctly.
- Internal-facing methods are hidden from the docs via the mkdocstrings `filters` list in `mkdocs.yml`. When you add, rename, or change the visibility of an internal-facing API method (one that consumers shouldn't call), update that `filters` list so the published reference stays accurate.

## Dependencies

- **Core**: `pydantic>=2`, `httpx>=0.27`, `twilio>=9.8.3`, `segment-analytics-python>=2.3`
- **Server** (optional): `fastapi`, `uvicorn`, `python-multipart` — install with `pip install tac[server]`
- **Dev**: `pytest`, `ruff`, `mypy`, `openai`, `openai-agents`
