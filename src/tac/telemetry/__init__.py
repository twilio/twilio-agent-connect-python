"""TAC Telemetry Module

Provides OpenTelemetry-based observability for TAC applications.
"""

from .config import TACTelemetry
from .metrics import MetricsClient
from .tracer import TracerClient

__all__ = [
    "TACTelemetry",
    "MetricsClient",
    "TracerClient",
]
