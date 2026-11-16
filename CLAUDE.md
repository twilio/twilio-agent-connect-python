# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Keep it accurate: when a change alters anything described here (behavior, commands, layout, CI), update this file in the same PR.

If a PR, including later pushes to it, conflicts with anything in this file, the PR must also update this file to match before it merges.

## Project Overview

Twilio Agent Connect (TAC) is a Python SDK, published to PyPI as `twilio-agent-connect` and imported as `tac`. It is middleware, not an agent runtime: it lets LLM applications (OpenAI Agents SDK, Bedrock, LangChain, etc.) use Twilio primitives: Conversation Memory for memory, Conversation Orchestrator for conversations, ConversationRelay and Media Streams for voice.

**Key Architecture**: This is the reference SDK. [twilio-agent-connect-typescript](https://github.com/twilio/twilio-agent-connect-typescript) is kept at feature parity with it, and the partner packages (AWS, Microsoft, Google) depend on it from PyPI.

## Development Commands

```bash
make sync              # Install dependencies (uv sync --all-extras --all-groups)
make dev-setup         # Full dev setup (sync + pre-commit hooks)
make format            # Format with ruff
make lint              # Lint check only (ruff check + ruff format --check)
make type-check        # mypy (disallow_untyped_defs, pydantic plugin)
make test              # Run pytest with coverage
make check             # All checks (lint + type-check + test)
make build             # Build the wheel and sdist (uv build)
make setup             # Start the Twilio setup wizard on port 8080
make docs              # Build the docs site into site/
make docs-serve        # Serve docs locally with live reload
make docs-versions     # List versioned docs published to gh-pages

uv run pytest tests/test_tac.py                      # Single test file
uv run pytest tests/test_tac.py::test_function_name  # Single test
```

Lint and format run ruff over the whole repo; mypy covers `src/tac` and `getting_started`; pytest covers `tests/`.

## Package Structure

```
src/tac/
├── __init__.py          # Public exports: TAC, TACConfig, PartnerConnector, get_logger, mask_* helpers
├── core/                # TAC class, TACConfig, logging, analytics (Segment telemetry)
├── context/             # API clients (BaseAPIClient, ConversationClient, MemoryClient, KnowledgeClient) and PartnerConnector
├── models/              # Pydantic models (memory, conversation, session, voice, knowledge, intelligence, handoff, outbound, ...)
├── session/             # Session state and thread-safe session storage
├── channels/            # Messaging channels (sms, rcs, whatsapp, chat on a shared messaging.py) and WebSocket helpers
│   └── voice/           # VoiceChannel, VoiceProvider base, TwiML
│       ├── conversation_relay/  # ConversationRelay provider
│       └── media_streams/       # Media Streams providers (openai_realtime, gpt_live, shared base)
├── intelligence/        # Conversation Intelligence operator result processing
├── tools/               # LLM tools (@function_tool, TACTool) plus built-in memory, knowledge and handoff tools
├── adapters/            # MemoryPromptBuilder, AdapterOptions, OpenAI client wrapper (adapters/openai/)
├── server/              # Optional TACFastAPIServer and Twilio signature validation (requires server extra)
└── utils/               # Conversation metadata, expiring dict, PII redaction, timestamps

tests/                   # Flat pytest suite, organised by feature (channels, scaling, relay-only mode, ...)
getting_started/         # Runnable examples (examples/features, examples/partners) and the twilio_setup wizard
docs/                    # MkDocs sources for the published API reference
```

## Code Conventions

- **Python 3.10+**: `requires-python` is `>=3.10,<4`. Use built-in generics (`list[str]`, `dict[str, Any]`) and union syntax (`X | None`) instead of `typing.List`, `typing.Optional`, `typing.Union`.
- **mypy**: `disallow_untyped_defs`, `disallow_incomplete_defs`, `warn_return_any`, `warn_unreachable` and `strict_equality` are on (not `strict = true`), with the `pydantic.mypy` plugin. Every function needs full type hints.
- **ruff**: Line length 100, target `py310`, rule sets `E W F I B C4 UP N`, isort ordering with `tac` as first-party and `combine-as-imports`, double quotes, `docstring-code-format`.
- **Pydantic v2**: Use `Field(alias=...)` for API name mapping, `model_config = {"populate_by_name": True}`, and `.model_dump(by_alias=True, exclude_none=True)` for API payloads.
- **Pre-commit**: `.pre-commit-config.yaml` runs ruff, ruff-format and mypy.

## Dependencies

### Core Dependency

```toml
dependencies = [
    "pydantic>=2.0.0,<3",
    "httpx>=0.27.0,<1",
    "twilio>=9.8.3,<10",
    "segment-analytics-python>=2.3,<3",
]
```

### Optional Extras

- `server` - `TACFastAPIServer` (`fastapi`, `uvicorn[standard]`, `python-multipart`). Install with `pip install "twilio-agent-connect[server]"`.
- `openai-realtime` - OpenAI Realtime Media Streams provider (`websockets`)
- `gpt-live` - GPT Live Media Streams provider (`websockets`)
- Dependency groups (not extras): `dev` (pytest, pytest-asyncio, pytest-cov, ruff, mypy), `examples` (openai, openai-agents, boto3, strands-agents, ...) and `docs`. `dev` and `examples` are installed by default.

## Key Architecture Concepts

- **Channel-based**: Messaging channels (SMS, RCS, WhatsApp, Chat) and the Voice channel process Twilio webhooks, manage conversation lifecycle, and trigger `on_message_ready` / `on_conversation_ended` callbacks
- **Callback responses**: Callbacks return `str` (auto-sent) or `None` (manual `channel.send_response()`)
- **Memory modes**:
  - `"never"` (default): No automatic memory retrieval
  - `"always"`: Fetch memory on every message with the user's query string for semantic search
  - `"once"` (**Voice only**): Fetch once with empty query, cache it on the session. Invalidated on INACTIVE; uses `cache_lock` for concurrent async access. Needs a session outliving one request, which only voice has — on messaging channels it is deprecated (removed in 3.0): `MessagingChannel.__init__` warns and runs it as `"always"`.
  - `memory_config.fetch_profile_traits=False` drops the `get_profile` call when prompts don't use `build_profile_prompt()`.
- **Memory fallback**: `TAC.retrieve_memory()` tries Conversation Memory first, gracefully falls back to Conversation Orchestrator's `list_communications()` on any failure
- **Profile resolution**: Automatic profile lookup by phone/email if `profile_id` not present in webhook
- **Client init**: `ConversationClient` and `MemoryClient` are created only when `conversation_configuration_id` is set; the Memory store comes from that Conversation Orchestrator configuration's `memory_store_id`. `KnowledgeClient` is created only when `knowledge_base_id` is set. `conversation_intelligence_config` requires Conversation Orchestrator.
- **Auth**: All API clients use HTTP Basic Auth (API Key as username, API Token as password)
- **BaseAPIClient** (`context/base.py`): `ConversationClient`, `MemoryClient` and `KnowledgeClient` inherit from it for shared HTTP client configuration, authentication, and User-Agent header management following Twilio SDK conventions. Partner packages identify themselves in the User-Agent through `PartnerConnector`.
- **Horizontal scaling**: N replicas behind a load balancer, **no shared datastore**.
  - **Messaging is stateless** — `MessagingChannel` has no session store; each webhook derives a `ConversationSession` from the payload, config, and one `list_participants` call it needs anyway. Nothing is kept between webhooks: a session lasts one turn, so `session.metadata` doesn't carry to the next message (apps keep cross-turn state themselves, keyed by `conversation_id`), and a turn whose `list_participants` fails is dropped (`on_error` with `dropped_inbound`). `ConversationSession.conversation_metadata()` reads the CO-stored metadata (written at outbound initiation) on any instance. The idempotency LRU is per-process, so a retry landing elsewhere is reprocessed and can send a second reply. `TACFastAPIServer` acks before the handler runs, so retries come only from failures (lost ack, crashed replica) and are rare; the guide treats this as acceptable, not as a handler requirement.
  - **Voice is instance-local**, correctly: a WebSocket pins a call to one process. Safe because teardown is guaranteed — `_release_session()` runs on every disconnect path and fires `on_call_ended` (invariant: `assert_no_residual_state` in `tests/voice_invariants.py`). Shutdown drain is the platform's job (readiness, `preStop`, termination grace period): uvicorn closes live WebSockets with 1012 on SIGTERM, and teardown fires `on_call_ended`.
  - **Out-of-band call webhooks** (status/AMD/recording, `<Connect action>`) carry only a `CallSid`. Set `TACConfig.instance_public_domain` and TAC mints the URLs it controls against it: the WebSocket and `<Connect action>` URLs always, status/AMD/recording callbacks only on calls TAC places (`_apply_call_event_callbacks` runs on outbound only; inbound call events come from the number's own config). Otherwise route by `CallSid` at the balancer, or use `VoiceChannel.resolve_conversation_session_by_call_sid()` (orchestrated ConversationRelay only), which falls back to an identity-only session rebuilt from CO.
  - **Per-call Media Streams `session_config` requires `instance_public_domain` with more than one instance.** The config (from `on_inbound_call_session_config` or outbound `session_config`) is kept in the memory of the instance that served the TwiML or placed the call; only a token rides the TwiML. `MediaStreamsOpenAIProvider._claim_session_config` warns when a stream's token isn't held locally, and the call falls back to `default_session_config`.
  - **`on_conversation_ended`** fires on whichever replica gets CO's CLOSED webhook. Messaging rebuilds the session from CO; its CO metadata is read via `conversation_metadata()`. Orchestrated voice keeps nothing after teardown: CLOSED gets a session rebuilt from CO (identity only) on every instance, the one that held the call included, and `end_call()` leaves `on_conversation_ended` to that CLOSED like any hangup. A CLOSED for a call still live on that replica fires from a snapshot and leaves the session for teardown (`_closed_while_live`), so `on_call_ended` still fires. `on_call_ended` is the only hook guaranteed the live session (transcript, metadata).
  - **Lifecycle analytics** ("Conversation Started"/"Conversation Ended") come from CO webhooks wherever a CO conversation exists: `PARTICIPANT_ADDED` for the customer, and `CONVERSATION_UPDATED`/CLOSED with `duration_ms` = CO's `updatedAt − createdAt`. Each channel reports only its own channel type (`BaseChannel._track_conversation_started` / `_track_conversation_ended`), so one webhook handed to every channel counts once per channel (at-least-once across webhook retries and replicas, like the callbacks). The CLOSED participant lookup is shared via `TAC._list_participants_shared` (in-flight + 60 s), and is skipped when analytics is off and no `on_conversation_ended` is registered. Relay-only and Media Streams voice report at session start/teardown instead. On chat, where customers stay UNKNOWN (no customer reconciliation), each UNKNOWN non-agent participant counts as a start, so a conversation with several such participants over-counts. `TAC_ANALYTICS_DISABLED=true` turns analytics off.
- **ConversationRelay-only mode**: When `conversation_configuration_id` is omitted from TACConfig, TAC runs with just the Voice channel (messaging channels raise at construction), `TAC.retrieve_memory()` returns an empty `TACMemoryResponse`, and the ConversationRelay callback handles session cleanup. `tac.is_orchestrator_enabled()` reports the mode (it is hidden from the published reference).
- **Multi-sender support**: Each channel has a singular default sender plus a plural allowlist (`phone_number`/`phone_numbers`, `rcs_sender_id`/`rcs_sender_ids`, `whatsapp_number`/`whatsapp_numbers`). Inbound derives the agent address from the webhook's recipient (SMS/RCS/WhatsApp) or the dialed number (voice), validates it against the allowlist. Messaging (SMS/RCS/WhatsApp) drops messages addressed to an unconfigured number; voice falls back to the default sender (with a warning) for a call to an unrecognized dialed number. Outbound selects the sender via an optional `from_` (defaulting to the channel's default sender), and digital handoff sends From the session's active agent number.

## OpenAI Adapter

`with_tac_memory()` (`src/tac/adapters/openai/`) wraps an `OpenAI` or `AsyncOpenAI` client and injects memory built by `MemoryPromptBuilder`:

- **Chat Completions** (`chat.completions.create` / `.stream`): memory is inserted as a system message at the start of `messages`.
- **Responses** (`responses.create` only): memory is prepended to `instructions`. Other Responses methods pass through without injection.

Both have sync and async variants. Every other client attribute passes through to the wrapped client.

## Testing

Tests live in `tests/` and use pytest. pytest-asyncio runs in strict mode, so async tests need `@pytest.mark.asyncio`. `--strict-markers` is on: register new markers in `pyproject.toml`. Tests should:
- Mock the Twilio API clients rather than calling Twilio. `tests/conftest.py` auto-mocks `ConversationClient.get_configuration` (opt out with `@pytest.mark.no_auto_mock`) and sets `TAC_ANALYTICS_DISABLED=true`.
- Call `assert_no_residual_state` (`tests/voice_invariants.py`) after voice tests that end a call.

CI (`.github/workflows/ci.yml`) runs lockfile hygiene, ruff, mypy, tests on Python 3.10–3.14, and a package build that installs the wheel and imports `tac`. Installs use `uv sync --frozen`, so commit `uv.lock` with any dependency change.

## Documentation

- The public API reference is published at https://twilio.github.io/twilio-agent-connect-python/, versioned with `mike` and deployed from `.github/workflows/docs.yml` on each GitHub release.
- API docs are generated from docstrings by MkDocs Material + mkdocstrings (Google docstring style), so **docstrings are published documentation**. Write them for both source readers and the rendered site.
- Use **Markdown** in docstrings and `Field(description=...)` text (rendered via `griffe_pydantic`): `**bold**`, `- ` bullet lists, backticks for identifiers, and fenced code blocks for examples, with a blank line before a list or fenced block.
- `docs/api/*.md` targets individual modules. When you add a public module, add it to the matching docs page.
- Internal-facing methods are hidden via the mkdocstrings `filters` list in `mkdocs.yml`. When you add, rename, or change the visibility of an internal-facing method (one consumers shouldn't call), update that list.
- `docs/index.md` is a snippet include of `README.md`. Edit `README.md`, never the stub.
- Verify docs changes with `make docs`.

## Releases, Skills and PRs

- `/create-release-pr` (`.claude/skills/create-release-pr/`) bumps the version in `pyproject.toml`, refreshes `uv.lock` and opens a release PR. `.github/workflows/deploy.yml` publishes to PyPI when a GitHub release is published, after checking the `vX.Y.Z` tag matches `pyproject.toml`.
- `/sync-to-ts-sdk` (`.claude/skills/sync-to-ts-sdk/`) ports a merged change to twilio-agent-connect-typescript. Behavior changes should land in both SDKs.
- When creating PRs, read and fill in `.github/PULL_REQUEST_TEMPLATE.md`.

## Common Pitfalls

1. **Install name is `twilio-agent-connect`** - the import name is `tac`, but there is no `tac` package on PyPI, so `pip install tac[server]` fails.
2. **Messaging sessions last one turn** - don't keep cross-turn state in `session.metadata`; key it by `conversation_id` yourself.
3. **Messaging needs Conversation Orchestrator** - in ConversationRelay-only mode, messaging channels raise and memory is empty.
4. **`"once"` memory is voice-only** - on messaging channels it warns and runs as `"always"`.
5. **Keep the TypeScript SDK in sync** - a public behavior change here needs a matching change there.

## Related Documentation

- TAC TypeScript SDK: [twilio-agent-connect-typescript](https://github.com/twilio/twilio-agent-connect-typescript)
- Partner packages: [twilio-agent-connect-aws](https://github.com/twilio/twilio-agent-connect-aws), [twilio-agent-connect-microsoft](https://github.com/twilio/twilio-agent-connect-microsoft), [twilio-agent-connect-google](https://github.com/twilio/twilio-agent-connect-google)
- Product docs: [twilio.com/docs/conversations/agent-connect](https://www.twilio.com/docs/conversations/agent-connect)
- Examples: `getting_started/`
- Published API reference: [twilio.github.io/twilio-agent-connect-python](https://twilio.github.io/twilio-agent-connect-python/)
