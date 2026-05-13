"""TAC Metrics Constants

Note: The service name "tac" is automatically prepended by Prometheus when scraping
from OpenTelemetry Collector, so these metric names don't include the "tac_" prefix.
Final metric names in Prometheus will be: tac_message_received_total, etc.
"""

# Message Processing Metrics
TAC_MESSAGE_RECEIVED_COUNT = "message_received_total"
TAC_MESSAGE_SENT_COUNT = "message_sent_total"
TAC_MESSAGE_ERROR_COUNT = "message_error_total"

# Lifecycle Latency Metrics
TAC_CONVERSATION_START_DURATION = "conversation_start_seconds"
TAC_CONVERSATION_READY_DURATION = "conversation_ready_seconds"
TAC_CONVERSATION_END_DURATION = "conversation_end_seconds"

# API Request Metrics
TAC_API_REQUEST_COUNT = "api_request_total"
TAC_API_REQUEST_DURATION = "api_request_duration_seconds"
TAC_API_ERROR_COUNT = "api_error_total"

# Conversation State Metrics
TAC_CONVERSATION_ACTIVE_COUNT = "conversation_active"
