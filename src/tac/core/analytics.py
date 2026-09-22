"""Product usage telemetry.

Reports coarse usage events so the SDK's channels and voice providers can be
prioritized against how they're actually used. Never reports message content,
transcripts, or end-user identifiers. Opt out by setting
``TAC_ANALYTICS_DISABLED=true``.
"""

import atexit
import os
import threading
from importlib.metadata import version
from typing import Any

from segment.analytics import Client

from tac.core.logging import ContextLogger, get_logger

# Segment stamps only its own library name on an event, so events need an
# explicit marker naming the package that produced them.
_SDK_PACKAGE = "twilio-agent-connect-python"

# Batch aggressively enough that a short-lived process still reports, without
# a request per event.
_UPLOAD_SIZE = 20
_UPLOAD_INTERVAL_SECONDS = 10.0

# The library's own default is 10, which makes the worst-case flush long enough
# to matter at shutdown.
_MAX_RETRIES = 3

_FLUSH_TIMEOUT_SECONDS = 5.0

_client: Client | None = None
_logger: ContextLogger | None = None
_disabled: bool | None = None
_lock = threading.Lock()


def _get_logger() -> ContextLogger:
    global _logger
    if _logger is None:
        _logger = get_logger("analytics")
    return _logger


def _is_disabled() -> bool:
    global _disabled
    if _disabled is None:
        _disabled = os.environ.get("TAC_ANALYTICS_DISABLED") == "true"
    return _disabled


def _on_error(*args: Any) -> None:
    # Debug, not warning: telemetry is best-effort and its failures are not the
    # consumer's problem, so they shouldn't surface in an application's logs.
    # Signature is *args so this doesn't depend on the library's callback arity.
    _get_logger().debug("Segment analytics error", error=str(args[0]) if args else None)


def _get_client() -> Client | None:
    global _client
    if _is_disabled():
        return None
    with _lock:
        if _client is None:
            _client = Client(
                "oH5gLNxB4NEg60y81mBxHWZn4RAoXQTN",
                upload_size=_UPLOAD_SIZE,
                upload_interval=_UPLOAD_INTERVAL_SECONDS,
                max_retries=_MAX_RETRIES,
                on_error=_on_error,
            )
            # Registered after the client is built so atexit's LIFO ordering
            # runs this bounded flush before the library's own hook, which
            # joins its consumer thread without flushing the queue first.
            atexit.register(shutdown_analytics)
        return _client


def track_event(event: str, account_sid: str, **properties: Any) -> None:
    """Record a telemetry event. Never raises.

    A property whose value is ``None`` is dropped rather than sent. Every
    declared property is optional, so omitting one is always accepted, whereas
    a null against a typed property is a violation that discards the whole
    event. Pass an unknown value as ``None`` and it will simply be left out.

    Args:
        event: Event name.
        account_sid: Twilio account SID, also used as the anonymous id.
        **properties: Event properties. ``None`` values are omitted; ``False``
            and ``0`` are sent, being meaningful values in their own right.
    """
    try:
        client = _get_client()
        if client is None:
            return
        client.track(
            anonymous_id=account_sid,
            event=event,
            # `sdk_version`/`sdk_package` last so a caller-supplied value
            # cannot displace the SDK's own identity. `account_sid` is a named
            # parameter, so it can never arrive in `properties` at all.
            properties={
                "account_sid": account_sid,
                **{k: v for k, v in properties.items() if v is not None},
                "sdk_version": version("twilio-agent-connect"),
                "sdk_package": _SDK_PACKAGE,
            },
        )
        _get_logger().debug("Analytics event tracked", event=event)
    except Exception as e:
        _get_logger().debug("Analytics event failed", event=event, error=str(e))


def shutdown_analytics() -> None:
    """Flush pending events and close the client.

    Bounded and non-blocking: the library's ``shutdown()`` blocks with no
    timeout of its own, so it runs on a daemon thread this waits on only
    briefly. Losing a final batch is preferable to stalling a host
    application's exit.
    """
    global _client
    with _lock:
        client, _client = _client, None
    if client is None:
        return
    try:
        thread = threading.Thread(target=client.shutdown, daemon=True)
        thread.start()
        thread.join(timeout=_FLUSH_TIMEOUT_SECONDS)
    except Exception as e:
        # Includes the RuntimeError raised when a thread can no longer be
        # started because the interpreter is already shutting down.
        _get_logger().debug("Analytics shutdown failed", error=str(e))


def _reset_analytics() -> None:
    """Reset internal state — for testing only."""
    global _client, _logger, _disabled
    with _lock:
        _client = None
    _logger = None
    _disabled = None
