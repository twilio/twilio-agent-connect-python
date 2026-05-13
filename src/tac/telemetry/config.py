"""OpenTelemetry Configuration for TAC"""

import logging
import os

import opentelemetry.metrics as metrics_api
import opentelemetry.sdk.metrics as metrics_sdk
import opentelemetry.trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


def get_otel_resource() -> Resource:
    """Create OpenTelemetry resource with TAC service info"""
    return Resource.create(
        {
            "service.name": "tac",
            "service.version": "1.0.0",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
        }
    )


class TACTelemetry:
    """TAC OpenTelemetry Setup

    Usage:
        # Metrics only (development)
        TACTelemetry().setup_meter(enable_console_exporter=True)

        # Metrics + Traces (production)
        telemetry = TACTelemetry()
        telemetry.setup_meter(enable_otlp_exporter=True)
        telemetry.setup_tracer(enable_otlp_exporter=True)

    Environment Variables:
        OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL (default: http://localhost:4318)
        OTEL_EXPORTER_OTLP_HEADERS: Headers for OTLP requests
        TAC_TELEMETRY_INCLUDE_MESSAGE_CONTENT: Include input/output in traces (default: false)
            WARNING: Setting this to 'true' will send message content (user input and LLM output)
            to your observability backend. This may include PII and should only be enabled
            if you have proper data handling agreements and comply with privacy regulations.
    """

    def __init__(self):
        self.resource = get_otel_resource()
        self.meter_provider: metrics_sdk.MeterProvider | None = None
        self.tracer_provider: TracerProvider | None = None

        # Check if message content should be included in traces (default: False for privacy)
        self.include_message_content = os.getenv(
            "TAC_TELEMETRY_INCLUDE_MESSAGE_CONTENT", "false"
        ).lower() in ("true", "1", "yes")

    def setup_meter(
        self,
        enable_console_exporter: bool = False,
        enable_otlp_exporter: bool = False,
    ) -> "TACTelemetry":
        """Setup OpenTelemetry MeterProvider

        Args:
            enable_console_exporter: Output to console (for debugging)
            enable_otlp_exporter: Export to OTLP endpoint (for production)

        Returns:
            self (for method chaining)
        """
        logger.info("Setting up TAC telemetry")

        metric_readers = []

        # Console exporter (for debugging)
        if enable_console_exporter:
            from opentelemetry.sdk.metrics.export import (
                ConsoleMetricExporter,
                PeriodicExportingMetricReader,
            )

            console_reader = PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=10000,
            )
            metric_readers.append(console_reader)
            logger.info("Console metrics exporter enabled")

        # OTLP exporter (for Prometheus/Grafana)
        if enable_otlp_exporter:
            try:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                    OTLPMetricExporter,
                )
                from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

                otlp_reader = PeriodicExportingMetricReader(
                    OTLPMetricExporter(),
                    export_interval_millis=5000,  # Export every 5 seconds for demo
                )
                metric_readers.append(otlp_reader)
                logger.info("OTLP metrics exporter enabled")
            except Exception as e:
                logger.error(f"Failed to setup OTLP exporter: {e}")

        # Create and set global meter provider
        self.meter_provider = metrics_sdk.MeterProvider(
            resource=self.resource,
            metric_readers=metric_readers,
        )
        metrics_api.set_meter_provider(self.meter_provider)

        logger.info("TAC telemetry configured successfully")
        return self

    def setup_tracer(
        self,
        enable_console_exporter: bool = False,
        enable_otlp_exporter: bool = False,
        sampling_ratio: float = 1.0,
    ) -> "TACTelemetry":
        """Setup OpenTelemetry TracerProvider

        Args:
            enable_console_exporter: Output to console (for debugging)
            enable_otlp_exporter: Export to OTLP endpoint (for Jaeger/Langfuse)
            sampling_ratio: Trace sampling ratio (0.0 to 1.0, default 1.0 = 100%)

        Returns:
            self (for method chaining)
        """
        logger.info("Setting up TAC tracing")

        span_processors = []

        # Console exporter (for debugging)
        if enable_console_exporter:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

            console_processor = SimpleSpanProcessor(ConsoleSpanExporter())
            span_processors.append(console_processor)
            logger.info("Console trace exporter enabled")

        # OTLP exporter (for Jaeger/Langfuse)
        if enable_otlp_exporter:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                # Check if Langfuse endpoint
                endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
                is_langfuse = "langfuse" in endpoint.lower()

                if is_langfuse:
                    logger.info(f"Detected Langfuse endpoint: {endpoint}")
                    # Langfuse requires auth headers
                    headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
                    if headers:
                        logger.info("Using OTLP headers for authentication")

                otlp_processor = BatchSpanProcessor(OTLPSpanExporter())
                span_processors.append(otlp_processor)
                logger.info("OTLP trace exporter enabled")
            except Exception as e:
                logger.error(f"Failed to setup OTLP trace exporter: {e}")

        # Create sampler based on sampling ratio
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

        if sampling_ratio < 1.0:
            # Use parent-based sampling to respect upstream sampling decisions
            sampler = ParentBased(root=TraceIdRatioBased(sampling_ratio))
            logger.info(f"Trace sampling enabled: {sampling_ratio * 100:.1f}%")
        else:
            sampler = ParentBased(root=TraceIdRatioBased(1.0))

        # Create and set global tracer provider
        self.tracer_provider = TracerProvider(resource=self.resource, sampler=sampler)
        for processor in span_processors:
            self.tracer_provider.add_span_processor(processor)
        trace_api.set_tracer_provider(self.tracer_provider)

        logger.info("TAC tracing configured successfully")
        return self

    def shutdown(self, timeout_millis: int = 30000) -> None:
        """Shutdown telemetry and flush all pending data

        This ensures all metrics and traces are exported before the application exits.
        Should be called on application shutdown or when stopping the server.

        Args:
            timeout_millis: Maximum time to wait for shutdown (default: 30 seconds)
        """
        logger.info("Shutting down TAC telemetry...")

        # Shutdown meter provider
        if self.meter_provider:
            try:
                self.meter_provider.shutdown(timeout_millis=timeout_millis)
                logger.info("Meter provider shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down meter provider: {e}")

        # Shutdown tracer provider (forces span export)
        if self.tracer_provider:
            try:
                self.tracer_provider.shutdown()
                logger.info("Tracer provider shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down tracer provider: {e}")

        logger.info("TAC telemetry shutdown complete")
