"""TAC Tracer - Singleton for creating spans"""

import logging
import os
import threading
from typing import Optional

from opentelemetry import trace
from opentelemetry.trace import Tracer

logger = logging.getLogger(__name__)


class TracerClient:
    """Singleton tracer client for TAC

    Automatically creates spans for tracing TAC operations.
    If tracing is not configured, operations are no-op.

    Usage:
        from tac.telemetry.tracer import TracerClient

        tracer = TracerClient()
        with tracer.start_span("conversation.start", attributes={"channel": "sms"}):
            # ... your code ...
    """

    _instance: Optional["TracerClient"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "TracerClient":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        logger.info("Initializing TAC TracerClient")
        tracer_provider = trace.get_tracer_provider()
        self.tracer: Tracer = tracer_provider.get_tracer("tac.telemetry")

        # Check if message content should be included in traces (default: False for privacy)
        self.include_message_content = os.getenv(
            "TAC_TELEMETRY_INCLUDE_MESSAGE_CONTENT", "false"
        ).lower() in ("true", "1", "yes")
        if self.include_message_content:
            logger.warning(
                "TAC_TELEMETRY_INCLUDE_MESSAGE_CONTENT is enabled - message content "
                "(input/output) will be sent to your observability backend. "
                "Ensure this complies with your privacy policy."
            )

        self._initialized = True

    def start_span(self, name: str, attributes: dict[str, str] | None = None):
        """Start a new span

        Args:
            name: Span name (e.g., "conversation.start")
            attributes: Optional span attributes

        Returns:
            Span context manager
        """
        return self.tracer.start_as_current_span(name, attributes=attributes or {})
