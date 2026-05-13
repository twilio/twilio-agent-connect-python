# Proposal: OpenTelemetry Integration for TAC

## 1. Overview

This proposal suggests adding **OpenTelemetry observability** to TAC with two components:

### 1. Metrics (Monitoring)
Aggregate numerical data for monitoring system health - message counts, latency, error rates.

**Backend:** Prometheus → Grafana

### 2. Traces (Debugging)
Individual request lifecycle showing complete execution flow with nested operations.

**Backend:** Jaeger, Langfuse, or any OTLP-compatible system

---

## 2. Background & Motivation

### The Problem

Users deploying TAC in production need visibility into their system's performance and health. Without built-in observability, users cannot:

- **Monitor performance** - Identify latency bottlenecks or degradation
- **Track errors** - Understand failure rates and error patterns
- **Measure throughput** - Know how many messages are being processed
- **Debug issues** - Investigate production problems with data
- **Set up alerts** - Get notified when SLAs are violated

### Why OpenTelemetry?

**OpenTelemetry (OTel)** is the industry standard for observability instrumentation:

✅ **Universal compatibility** - Works with Grafana, Datadog, New Relic, Prometheus, Langfuse, etc.  
✅ **Vendor-neutral** - Not locked into any specific platform  
✅ **Widely adopted** - Used by major projects (Strands SDK, cloud providers, etc.)  
✅ **Unified protocol** - OTLP (OpenTelemetry Protocol) for metrics and traces  
✅ **Production-ready** - Mature, stable, and well-documented

By instrumenting TAC with OpenTelemetry, users can plug TAC into their existing observability infrastructure without vendor lock-in.

---

## 3. Metrics

Metrics provide aggregate numerical data about system behavior over time - counts, rates, and latencies. They answer questions like "how many?", "how fast?", and "how often?"

### Category 1: Message Processing

Track message throughput, success rate, and errors per channel.

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `tac.message.received` | Counter | Total messages received from Twilio | `channel` (sms, voice, whatsapp, rcs, chat) |
| `tac.message.sent` | Counter | Total messages successfully sent | `channel` |
| `tac.message.success` | Counter | Messages processed successfully end-to-end | `channel` |
| `tac.message.error` | Counter | Message processing errors | `channel`, `error_type` |

**Use case:** Monitor message throughput, calculate success rate per channel, track error patterns.

**Example queries:**
- Success rate: `rate(tac.message.success) / rate(tac.message.received)`
- Error rate by channel: `rate(tac.message.error{channel="sms"})`
- Total throughput: `sum(rate(tac.message.received)) by (channel)`

---

### Category 2: Lifecycle Latency

Measure latency at each stage of the conversation lifecycle.

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `tac.conversation.start` | Histogram | Time to start a new conversation (on_conversation_start) | `channel` |
| `tac.conversation.ready` | Histogram | Time spent in user callback (on_message_ready) | `channel` |
| `tac.conversation.end` | Histogram | Time to clean up ended conversation (on_conversation_end) | `channel`, `reason` (completed, timeout, error) |

**Use case:** Monitor performance of the three main lifecycle stages: starting conversations, processing messages, and ending conversations.

**Example queries:**
- P95 callback latency: `histogram_quantile(0.95, rate(tac_conversation_ready_seconds_bucket))`
- P99 conversation start: `histogram_quantile(0.99, rate(tac_conversation_start_seconds_bucket))`
- P95 conversation end: `histogram_quantile(0.95, rate(tac_conversation_end_seconds_bucket))`

---

### Category 3: API Requests

Monitor Twilio API performance and reliability.

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `tac.api.request` | Counter | Total API requests to Twilio services | `client_type` (conversation, memory, knowledge), `method` (GET, POST, PUT, DELETE) |
| `tac.api.success` | Counter | Successful API requests (2xx status) | `client_type`, `method` |
| `tac.api.error` | Counter | Failed API requests | `client_type`, `method`, `error_type`, `status_code` |
| `tac.api.request_duration` | Histogram | API request latency | `client_type`, `method` |

**Use case:** Detect Twilio service degradation, track API quota usage, calculate success rates per service.

**Example queries:**
- Orchestrator success rate: `rate(tac.api.success{client_type="conversation"}) / rate(tac.api.request{client_type="conversation"})`
- Memory API P95 latency: `histogram_quantile(0.95, rate(tac.api.request_duration_bucket{client_type="memory"}))`
- Knowledge API errors: `sum(rate(tac.api.error{client_type="knowledge"})) by (error_type)`

---

## 4. Traces

Traces would provide visibility into individual request execution, showing the complete lifecycle of a message with nested operations (spans).

### Example Trace Structure

```
message.processing (1.2s)
├── conversation.start (0.01s)
├── memory.retrieve (0.3s)
├── conversation.ready (0.8s)  ← User callback
├── message.send (0.09s)
└── conversation.end (0.01s)
```

### Trace Attributes

Each span includes:
- `conversation_id` - Unique conversation identifier
- `channel` - sms, voice, whatsapp, rcs, chat
- `service.name` - "tac"
- `service.version` - TAC version
- Custom attributes (model name, token usage, etc.)

### Use Cases

- **Debug slow requests** - "Why did this message take 5 seconds?"
- **Find bottlenecks** - "Is the delay in memory retrieval or LLM calls?"
- **Visualize flow** - See parent-child relationships in nested operations
- **Correlate with metrics** - Jump from high P95 latency in Grafana to specific slow traces

### Supported Backends

- **Jaeger** - Open-source distributed tracing platform
- **Langfuse** - LLM-focused observability with OpenTelemetry support
- **Datadog APM** - Enterprise monitoring with OTLP support
- **Any OTLP-compatible system**

---

## 5. Proposed API

### Dependencies

Core SDK is included by default (small footprint, zero overhead when not configured). Exporters are optional to minimize bundle size and let users choose their backend.

**Core (included by default):**
- `opentelemetry-api>=1.20.0`
- `opentelemetry-sdk>=1.20.0`

**Optional (install with `pip install tac[telemetry]`):**
- `opentelemetry-exporter-otlp-proto-http>=1.20.0` - OTLP exporter for production use

### Basic Setup (Metrics Only)

```python
from tac import TAC, TACConfig
from tac.telemetry import TACTelemetry

# Enable telemetry with console exporter (development)
TACTelemetry().setup_meter(enable_console_exporter=True)

# TAC automatically records metrics
tac = TAC(config=TACConfig.from_env())
```

### Production Setup (Metrics + Traces)

```python
import os
from tac import TAC, TACConfig
from tac.telemetry import TACTelemetry

# Configure OTLP endpoint
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://otel-collector:4318"

# Enable metrics + traces
telemetry = TACTelemetry()
telemetry.setup_meter(enable_otlp_exporter=True)
telemetry.setup_tracer(enable_otlp_exporter=True)  # Proposed

tac = TAC(config=TACConfig.from_env())
```

### Proposed Data Flow

```
TAC Application
    ↓ (OTLP Protocol)
OpenTelemetry Collector
    ↓
    ├─→ Metrics → Prometheus → Grafana
    └─→ Traces → Jaeger/Langfuse
```

---

## 6. Dashboard & Visualization

### Metrics (Grafana)

- **P95/P99 Latency** - Callback execution time by channel
- **Message Volume** - Messages received/sent per minute
- **Error Rates** - Errors by channel and type
- **API Performance** - Twilio API latency and success rates

### Traces (Jaeger/Langfuse)

- **Waterfall View** - Visual timeline of request execution
- **Nested Spans** - Parent-child relationships showing operation hierarchy
- **Detailed Attributes** - conversation_id, channel, timing, custom metadata
- **Debug Tools** - Filter by slow requests, errors, or specific conversations

---

## 7. References

- [OpenTelemetry Python Docs](https://opentelemetry.io/docs/languages/python/)
- [OpenTelemetry Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/)
- [Strands SDK Telemetry Implementation](https://github.com/anthropics/anthropic-strands-python)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
- [Grafana Dashboard Examples](https://grafana.com/grafana/dashboards/)

