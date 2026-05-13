"""TAC Metrics Client - Singleton for recording metrics"""

import logging
import threading
from typing import Optional

import opentelemetry.metrics as metrics_api
from opentelemetry.metrics import Counter, Histogram, Meter, UpDownCounter

from . import metrics_constants as constants

logger = logging.getLogger(__name__)


class MetricsClient:
    """Singleton metrics client for TAC

    Automatically records metrics to configured exporters.
    If telemetry is not configured, operations are no-op.

    Usage:
        metrics = MetricsClient()
        metrics.message_received_count.add(1, attributes={"channel": "sms"})
    """

    _instance: Optional["MetricsClient"] = None
    _lock = threading.Lock()

    # Message processing metrics
    message_received_count: Counter
    message_sent_count: Counter
    message_error_count: Counter

    # Lifecycle latency metrics
    conversation_start_duration: Histogram
    conversation_ready_duration: Histogram
    conversation_end_duration: Histogram

    # API request metrics
    api_request_count: Counter
    api_request_duration: Histogram
    api_error_count: Counter

    # Conversation state metrics
    conversation_active_count: UpDownCounter

    def __new__(cls) -> "MetricsClient":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        logger.info("Initializing TAC MetricsClient")
        meter_provider: metrics_api.MeterProvider = metrics_api.get_meter_provider()
        self.meter: Meter = meter_provider.get_meter("tac.telemetry", version="1.0.0")
        self._create_instruments()
        self._initialized = True

    def _create_instruments(self) -> None:
        """Create all metric instruments"""

        # Message processing metrics
        self.message_received_count = self.meter.create_counter(
            name=constants.TAC_MESSAGE_RECEIVED_COUNT,
            unit="1",
            description="Total messages received by TAC",
        )

        self.message_sent_count = self.meter.create_counter(
            name=constants.TAC_MESSAGE_SENT_COUNT,
            unit="1",
            description="Total messages sent by TAC",
        )

        self.message_error_count = self.meter.create_counter(
            name=constants.TAC_MESSAGE_ERROR_COUNT,
            unit="1",
            description="Total message processing errors",
        )

        # Lifecycle latency metrics
        self.conversation_start_duration = self.meter.create_histogram(
            name=constants.TAC_CONVERSATION_START_DURATION,
            unit="s",
            description="Time to start a new conversation (on_conversation_start)",
        )

        self.conversation_ready_duration = self.meter.create_histogram(
            name=constants.TAC_CONVERSATION_READY_DURATION,
            unit="s",
            description="Time spent in user callback (on_message_ready)",
        )

        self.conversation_end_duration = self.meter.create_histogram(
            name=constants.TAC_CONVERSATION_END_DURATION,
            unit="s",
            description="Time to clean up ended conversation (on_conversation_end)",
        )

        # API request metrics
        self.api_request_count = self.meter.create_counter(
            name=constants.TAC_API_REQUEST_COUNT,
            unit="1",
            description="Total API requests",
        )

        self.api_request_duration = self.meter.create_histogram(
            name=constants.TAC_API_REQUEST_DURATION,
            unit="s",
            description="API request duration",
        )

        self.api_error_count = self.meter.create_counter(
            name=constants.TAC_API_ERROR_COUNT,
            unit="1",
            description="Total API errors",
        )

        # Conversation state metrics
        self.conversation_active_count = self.meter.create_up_down_counter(
            name=constants.TAC_CONVERSATION_ACTIVE_COUNT,
            unit="1",
            description="Number of active conversations",
        )

        logger.info("TAC metrics instruments created")
