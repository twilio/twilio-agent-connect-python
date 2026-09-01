"""The voice teardown invariant, as one assertion.

However a call ends — normal hangup, abrupt disconnect, an exception during
setup, cancellation — the process must hold nothing for it afterwards. Every
voice test that opens a socket asserts this, so a new teardown path can't
quietly start leaking.
"""

from __future__ import annotations

from typing import Any

from tac.channels.voice.channel import VoiceChannel


def assert_no_residual_state(channel: VoiceChannel, conv_id: str) -> None:
    """Assert no local state survives for ``conv_id`` on ``channel``.

    Covers the channel's session, the provider's WebSocket registry and
    session manager, and (for Media Streams providers) the call state and
    stashed per-call session config.
    """
    assert conv_id not in channel._conversations, f"session for {conv_id} was not released"

    provider: Any = channel._provider

    ws_manager = getattr(provider, "_websocket_manager", None)
    if ws_manager is not None:
        assert not ws_manager.has_websocket(conv_id), f"websocket for {conv_id} was not removed"

    session_manager = getattr(provider, "session_manager", None)
    if session_manager is not None:
        assert not session_manager.has_session(conv_id), (
            f"session state for {conv_id} was not removed"
        )

    calls = getattr(provider, "_calls", None)
    if calls is not None:
        assert conv_id not in calls, f"call state for {conv_id} was not removed"

    session_configs = getattr(provider, "_call_session_configs", None)
    if session_configs is not None:
        assert conv_id not in session_configs, f"session config for {conv_id} was not removed"
