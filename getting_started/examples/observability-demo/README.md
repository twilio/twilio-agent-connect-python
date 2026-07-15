# RealtimeVoiceChannel Observability Demo (Langfuse)

Local Langfuse stack for viewing the OpenTelemetry spans `RealtimeVoiceChannel`
emits for each call: connect latency, per-turn response latency, tool calls,
and barge-ins. See `src/tac/core/tracing.py` for the exporter setup.

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

### 3. Run the realtime voice example

```bash
cd getting_started/examples
# .env needs Twilio + OpenAI + Langfuse credentials, and
# TWILIO_VOICE_PUBLIC_DOMAIN set to your ngrok domain.
uv run python features/voice_realtime.py
```

Expose it with ngrok (`ngrok http --url=<your-domain> 8000`) and point your
Twilio number's voice webhook at `https://<your-domain>/twiml-realtime`.

### 4. Make a call and check Langfuse

Call your Twilio number, talk to the assistant, hang up. In Langfuse
(http://localhost:3001), open **Traces** and find the **Realtime Voice Call**
trace for that call:

```
Realtime Voice Call
└── twilio_media_stream.call        (whole call duration)
    ├── openai_realtime.connect          (model connect + session config)
    ├── openai_realtime.response_latency (per turn: caller stops talking -> first audio back)
    ├── openai_realtime.barge_in         (per interruption: truncated?, played_ms)
    └── openai_realtime.tool_call        (per tool call: tool_name)
```

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
