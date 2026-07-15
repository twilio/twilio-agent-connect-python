# Voice Channel Observability Demo (Langfuse)

Local Langfuse stack for viewing the OpenTelemetry spans TAC's two voice
paths emit for each call — `RealtimeVoiceChannel` (speech-to-speech) and
`VoiceChannel` (ConversationRelay) — so you can compare connect/response
latency, tool calls, and barge-ins between them side by side. See
`src/tac/core/tracing.py` for the exporter setup.

## Quick Start

### 1. Start Langfuse

```bash
cd getting_started/examples/observability-demo
docker compose up -d
```

Wait ~30-60s for services to become healthy (`docker compose ps`).

### 2. Configure Langfuse

1. Open http://localhost:3001
2. Create an account and project
3. Go to **Settings → API Keys**
4. Copy the keys and add to `getting_started/examples/.env`:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3001
```

Without `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` set, spans just print to
stdout instead — useful if you don't want to run this stack at all.

### 3. Run a voice example

Run either, or both to compare (both default to port 8000, so if running both
at once, override one — e.g. `PORT=8001` for the ConversationRelay server):

```bash
cd getting_started/examples
# .env needs Twilio + OpenAI + Langfuse credentials, and
# TWILIO_VOICE_PUBLIC_DOMAIN set to your ngrok domain.
uv run python features/voice_realtime.py       # RealtimeVoiceChannel (speech-to-speech)
uv run python features/voice_cascaded.py       # VoiceChannel (ConversationRelay)
```

`voice_cascaded.py` is purpose-built as the comparison counterpart to
`voice_realtime.py` — same greeting, tone, tool, and model tier, no memory on
either side — so the only real variable between their traces is the voice
architecture itself.

Expose each with ngrok (`ngrok http --url=<your-domain> <port>`) and point the
Twilio number you're testing at that server's TwiML endpoint.

### 4. Make a call and check Langfuse

Call the number, talk to the assistant, hang up. In Langfuse
(http://localhost:3001), open **Traces** and find the trace for that call:

```
Realtime Voice Call                          (RealtimeVoiceChannel)
└── twilio_media_stream.call        (whole call duration)
    ├── openai_realtime.connect          (model connect + session config)
    ├── openai_realtime.response_latency (per turn: caller stops talking -> first audio back)
    ├── openai_realtime.barge_in         (per interruption: truncated?, played_ms)
    └── openai_realtime.tool_call        (per tool call: tool_name)

ConversationRelay Voice Call                 (VoiceChannel)
└── conversation_relay.call             (whole call duration)
    ├── conversation_relay.response_latency  (per turn: final prompt -> first response token sent)
    ├── model_invocation                      (per turn: LLM call; time_to_first_token_ms) *
    └── conversation_relay.barge_in           (per interruption)
```

`conversation_relay.response_latency` is the direct comparison point for
`openai_realtime.response_latency` — both measure "caller finishes talking"
to "assistant's first audio/token back," just for the two different voice
architectures.

Two spans differ because of *where the model lives*:

- **`model_invocation` (`*`) is emitted by the example, not by TAC.** In the
  cascaded path the application invokes the LLM inside `on_message_ready`, so
  TAC never sees the model call and can't time it — `voice_cascaded.py` wraps
  it itself using `voice_channel.trace_context(conv_id)` to nest under the
  call trace. It's the cascaded stand-in for `openai_realtime.connect` and
  breaks out the model's share of `response_latency` (which also includes
  sending the reply back to Twilio). The realtime channel needs no equivalent
  app span because TAC owns and times that model connection directly.
- **ConversationRelay has no `tool_call` span** for the same reason: tool
  calls run inside your callback (the LLM framework's own agent loop), not
  inside TAC.

## Services

| Service | Port | Purpose |
|---------|------|---------|
| Langfuse | 3001 | Trace UI |
| Postgres (`langfuse-db`) | - | Langfuse's relational store |
| ClickHouse | - | Langfuse's trace/observation store |
| MinIO | 9092/9093 | S3-compatible blob storage Langfuse uses internally |
| Redis | - | Langfuse's queue/cache |

## Cleanup

```bash
# Stop services (keeps data)
docker compose down

# Stop and delete all data
docker compose down -v
```
