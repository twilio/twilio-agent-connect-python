# TAC OpenTelemetry Observability Demo

This demo showcases TAC's OpenTelemetry integration:
- **Metrics** → Prometheus → Grafana (P95 latencies, message counts)
- **Traces** → Langfuse (nested span visualization for debugging)

## Architecture

```
TAC Application
    ↓ (OTLP Protocol)
OpenTelemetry Collector
    ↓
    ├─→ Metrics → Prometheus → Grafana
    └─→ Traces → Langfuse
```

## Quick Start

### 1. Start Services

```bash
cd getting_started/examples/observability-demo
docker compose up -d
```

Wait 30-60 seconds for services to be ready.

### 2. Configure Langfuse

1. Open http://localhost:3001
2. Create an account and project
3. Go to **Settings → API Keys**
4. Copy the keys and add to your `.env` file:

```bash
# In getting_started/examples/.env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

### 3. Run the Example

```bash
# Set up environment (if not already done)
cd getting_started/examples
# Make sure .env has all required credentials (TAC + OpenAI + Langfuse)

# Run the example
uv run python observability-demo/openai_with_observability.py

# Expose with ngrok
ngrok http 8000
```

### 4. Configure Twilio Webhooks

Point Twilio to your ngrok URL:
- **SMS**: `https://your-ngrok.ngrok.io/channels/sms`
- **Voice**: `https://your-ngrok.ngrok.io/twiml`

### 5. Send Messages

Send SMS or make a voice call. You'll see:

**Metrics in Grafana** (http://localhost:3000):
- P95 Callback Latency by Channel
- Message Volume (received/sent)
- API Request Duration

**Traces in Langfuse** (http://localhost:3001):
- Click **Traces** → Select any `message.processing` trace
- See nested operations:
  ```
  message.processing (1.2s)
  ├── conversation.start (0.01s)
  ├── memory.retrieve (0.3s)
  ├── conversation.ready (0.8s)  ← User callback
  ├── message.send (0.09s)
  └── conversation.end (0.01s)
  ```

## Services

| Service | Port | Purpose |
|---------|------|---------|
| Grafana | 3000 | Metrics dashboard |
| Langfuse | 3001 | Trace visualization |
| Prometheus | 9090 | Metrics storage |
| OTel Collector | 4318 | OTLP receiver |

## Files

- `openai_with_observability.py` - Example application (SMS + Voice + OpenAI)
- `docker-compose.yml` - Infrastructure stack
- `otel-collector-config.yml` - Collector configuration
- `prometheus.yml` - Prometheus config
- `grafana-datasource.yml` - Grafana datasource
- `grafana-dashboard.yml` - Dashboard provisioning
- `grafana-tac-dashboard.json` - TAC metrics dashboard

## What Gets Measured

### Metrics (Grafana)
- **Message Counts**: received, sent, errors
- **Latency**: P50/P95/P99 for conversation lifecycle stages
- **API Requests**: duration and errors by client type

Labels: `channel` (sms/voice/whatsapp/rcs/chat), `client_type` (conversation/memory/knowledge)

### Traces (Langfuse)
- Complete request lifecycle with nested spans
- Timing for each operation
- Conversation ID and channel metadata
- Parent-child relationships

## Integration Pattern

Add observability to your TAC application:

```python
from tac import TAC, TACConfig
from tac.telemetry import TACTelemetry
import os

# 1. Configure OTLP endpoint
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"

# 2. Setup telemetry BEFORE creating TAC
telemetry = TACTelemetry()
telemetry.setup_meter(enable_otlp_exporter=True)   # Metrics
telemetry.setup_tracer(enable_otlp_exporter=True)  # Traces

# 3. Create TAC - metrics and traces are automatic
tac = TAC(config=TACConfig.from_env())
```

## Cleanup

```bash
# Stop services
docker compose down

# Remove all data
docker compose down -v
```
