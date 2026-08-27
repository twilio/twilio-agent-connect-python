"""TwiML generation for ``OpenAIRealtimeProvider``.

See https://www.twilio.com/docs/voice/twiml/stream for the ``<Stream>`` verb.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from twilio.twiml.voice_response import Connect, VoiceResponse

from tac.core.config import TACConfig
from tac.models.voice import VoiceTwiMLOptionsMediaStreams

if TYPE_CHECKING:
    from tac.channels.voice.media_streams.openai_realtime.config import (
        OpenAIRealtimeProviderConfig,
    )


def generate_twiml(
    websocket_url: str | None = None,
    options: VoiceTwiMLOptionsMediaStreams | dict[str, Any] | None = None,
) -> str:
    """Generate TwiML that connects the call to a bidirectional Media Stream.

    The WebSocket URL may be passed positionally or as ``options.websocket_url``
    (positional wins when both are given), so a caller can pass everything in
    one object: ``generate_twiml(options=VoiceTwiMLOptionsMediaStreams(websocket_url=...))``.

    Args:
        websocket_url: Public ``wss://`` URL of the WebSocket endpoint Twilio
            should stream call audio to (the ``<Stream url=...>`` attribute).
            Optional if ``options.websocket_url`` is set.
        options: Optional ``VoiceTwiMLOptionsMediaStreams`` (or dict).

    Returns:
        TwiML XML string ready to return to Twilio.

    Raises:
        ValueError: If no WebSocket URL is provided via either source.
    """
    if options is None:
        options = VoiceTwiMLOptionsMediaStreams()
    elif isinstance(options, dict):
        options = VoiceTwiMLOptionsMediaStreams(**options)

    resolved_websocket_url = websocket_url if websocket_url is not None else options.websocket_url
    if not resolved_websocket_url:
        raise ValueError(
            "generate_twiml requires a WebSocket URL — pass it positionally or "
            "set options.websocket_url."
        )

    response = VoiceResponse()

    connect_kwargs: dict[str, str] = {}
    if options.action_url:
        connect_kwargs["action"] = options.action_url
    if options.action_method:
        connect_kwargs["method"] = options.action_method
    connect = Connect(**connect_kwargs)

    stream_kwargs: dict[str, Any] = {"url": resolved_websocket_url}
    if options.name:
        stream_kwargs["name"] = options.name
    if options.status_callback:
        stream_kwargs["status_callback"] = options.status_callback
    if options.status_callback_method:
        stream_kwargs["status_callback_method"] = options.status_callback_method
    stream = connect.stream(**stream_kwargs)

    if options.custom_parameters:
        for name, value in options.custom_parameters.items():
            if value is not None:
                stream.parameter(name=name, value=str(value))

    response.append(connect)
    return str(response)


class TwiMLBuilderMediaStreams:
    """Builds the TwiML for a Media Streams call, owning the layering and
    WebSocket URL resolution so ``OpenAIRealtimeProvider`` doesn't have to.
    """

    def __init__(self, tac_config: TACConfig, channel_config: OpenAIRealtimeProviderConfig) -> None:
        self.tac_config = tac_config
        self.channel_config = channel_config

    def build(
        self,
        caller: str,
        *,
        host: VoiceTwiMLOptionsMediaStreams | None = None,
        per_call: VoiceTwiMLOptionsMediaStreams | None = None,
        websocket_url: str | None = None,
    ) -> str:
        """Build the TwiML XML for one call.

        TwiML fields are merged per-field, highest precedence first:
          1. ``per_call`` — the ``on_inbound_call_twiml`` customizer's output
             for inbound, or ``InitiateVoiceConversationOptions.twiml_options``
             for outbound.
          2. ``OpenAIRealtimeProviderConfig.default_twiml_options`` — channel-wide defaults
          3. ``host`` — per-call transport facts supplied by the host (e.g. a
             per-call ``websocket_url`` with an affinity token)
          4. TAC defaults: the WebSocket URL derived from
             ``TACConfig.voice_public_domain`` + ``voice_websocket_path``

        Args:
            caller: Name of the calling method, used in the "no WebSocket URL"
                error so it points at the API the developer actually called.
            host: Per-call overrides from the host owning the route.
            per_call: Per-call overrides — the ``on_inbound_call_twiml``
                customizer's output for inbound, or
                ``InitiateVoiceConversationOptions.twiml_options`` for outbound.
            websocket_url: Dedicated per-call WebSocket override that wins over
                any ``websocket_url`` coming through the option layers. Used by
                outbound, which takes it as its own argument.

        Raises:
            ValueError: If no layer and no ``TACConfig``-derived default
                supplies a WebSocket URL.
        """
        merged = self._build_twiml_options(host, per_call)

        if websocket_url is not None:
            resolved_websocket_url = websocket_url
        elif merged.websocket_url is not None:
            resolved_websocket_url = merged.websocket_url
        elif self.tac_config.voice_public_domain:
            resolved_websocket_url = (
                f"wss://{self.tac_config.voice_public_domain}{self.tac_config.voice_websocket_path}"
            )
        else:
            raise ValueError(
                f"{caller} needs a WebSocket URL. Set TWILIO_VOICE_PUBLIC_DOMAIN "
                "(or TACConfig.voice_public_domain)."
            )

        return generate_twiml(resolved_websocket_url, merged)

    def _build_twiml_options(
        self,
        host: VoiceTwiMLOptionsMediaStreams | None,
        per_call: VoiceTwiMLOptionsMediaStreams | None,
    ) -> VoiceTwiMLOptionsMediaStreams:
        """Layer TwiML options, lowest precedence first: ``host`` →
        ``default_twiml_options`` → ``per_call`` (the ``on_inbound_call_twiml``
        customizer's output).
        """
        merged = VoiceTwiMLOptionsMediaStreams()
        if host is not None:
            self._overlay_fields(merged, host)
        if self.channel_config.default_twiml_options is not None:
            self._overlay_fields(merged, self.channel_config.default_twiml_options)
        if per_call is not None:
            self._overlay_fields(merged, per_call)
        return merged

    @staticmethod
    def _overlay_fields(
        target: VoiceTwiMLOptionsMediaStreams, source: VoiceTwiMLOptionsMediaStreams
    ) -> None:
        """Apply fields explicitly set on ``source`` onto ``target``.

        ``custom_parameters`` replaces wholesale when set at a higher-priority
        layer — there's no per-key merging.
        """
        for field in source.model_fields_set:
            setattr(target, field, getattr(source, field))
