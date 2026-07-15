"""OpenTelemetry tracing setup for Twilio Agent Connect.

Exports spans to Langfuse (via its OTLP/HTTP endpoint) when
``LANGFUSE_PUBLIC_KEY``/``LANGFUSE_SECRET_KEY`` are set, so spans show up as
traces in the Langfuse UI. Otherwise falls back to printing spans to stdout,
so latency is still visible with no external account required.
"""

import base64
import os

from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

_configured = False

DEFAULT_LANGFUSE_HOST = "https://cloud.langfuse.com"


def _build_span_processor() -> SpanProcessor:
    """Export to Langfuse if credentials are configured, else stdout."""
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if public_key and secret_key:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        host = os.environ.get("LANGFUSE_HOST", DEFAULT_LANGFUSE_HOST)
        auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        otlp_exporter = OTLPSpanExporter(
            endpoint=f"{host}/api/public/otel/v1/traces",
            headers={
                "Authorization": f"Basic {auth}",
                "x-langfuse-ingestion-version": "4",
            },
        )
        return BatchSpanProcessor(otlp_exporter)
    return BatchSpanProcessor(ConsoleSpanExporter())


def setup_tracing() -> None:
    """Configure a process-wide TracerProvider for the realtime voice path.

    Idempotent, so channels can call it unconditionally on init.
    """
    global _configured
    if _configured:
        return
    provider = TracerProvider()
    provider.add_span_processor(_build_span_processor())
    trace.set_tracer_provider(provider)
    _configured = True


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer instance for a specific module."""
    return trace.get_tracer(name)
