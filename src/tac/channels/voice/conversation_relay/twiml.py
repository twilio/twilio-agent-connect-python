"""TwiML generation for voice channel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from twilio.twiml.voice_response import VoiceResponse

from tac.channels.voice.twiml import TwiMLBuilderBase
from tac.models.voice import VoiceTwiMLOptionsConversationRelay
from tac.tools.handoff import studio_voice_handoff_url

if TYPE_CHECKING:
    from tac.channels.voice.conversation_relay.config import ConversationRelayProviderConfig

# Fields on VoiceTwiMLOptionsConversationRelay that map to <ConversationRelay> attributes
# and are emitted via the snake_case → camelCase conversion done by twilio's SDK.
# Must stay in sync with VoiceTwiMLOptionsConversationRelay's field declarations —
# see _verify_attrs_in_sync.
_OPTIONAL_RELAY_ATTRS = (
    "welcome_greeting",
    "welcome_greeting_interruptible",
    "conversation_configuration",
    # language / TTS / STT
    "language",
    "tts_language",
    "transcription_language",
    "voice",
    "tts_provider",
    "transcription_provider",
    "speech_model",
    "elevenlabs_text_normalization",
    # turn detection / interruption
    "eot_threshold",
    "partial_prompts",
    "deepgram_smart_format",
    "speech_timeout",
    "interruptible",
    "interrupt_sensitivity",
    "report_input_during_agent_speech",
    "ignore_backchannel",
    "preemptible",
    "dtmf_detection",
    # hints / events / debug / intelligence
    "hints",
    "events",
    "debug",
    "intelligence_service",
)

# Fields on VoiceTwiMLOptionsConversationRelay that this module handles specially (not via the
# generic _OPTIONAL_RELAY_ATTRS loop) — the websocket_url (emitted as the
# ``<ConversationRelay url=...>`` attribute, resolved from the positional
# ``websocket_url`` arg or ``options.websocket_url``), the action_url, the
# <Language> children list, the <Parameter> children dict, and the extra
# escape hatch.
_HANDLED_OUTSIDE_LOOP = {
    "websocket_url",
    "action_url",
    "languages",
    "custom_parameters",
    "extra",
}


def _verify_attrs_in_sync() -> None:
    """Fail fast at import time if VoiceTwiMLOptionsConversationRelay grows a field that isn't
    accounted for here — either it's a new ConversationRelay attribute that
    needs to go in _OPTIONAL_RELAY_ATTRS, or it's special-cased and should
    be added to _HANDLED_OUTSIDE_LOOP.
    """
    declared = set(VoiceTwiMLOptionsConversationRelay.model_fields)
    accounted = set(_OPTIONAL_RELAY_ATTRS) | _HANDLED_OUTSIDE_LOOP
    missing = declared - accounted
    extra = accounted - declared
    if missing:
        raise RuntimeError(
            f"VoiceTwiMLOptionsConversationRelay field(s) {sorted(missing)} not handled by "
            "twiml.py — add to _OPTIONAL_RELAY_ATTRS or _HANDLED_OUTSIDE_LOOP."
        )
    if extra:
        raise RuntimeError(
            f"twiml.py references VoiceTwiMLOptionsConversationRelay field(s) {sorted(extra)} that "
            "no longer exist on the model."
        )


_verify_attrs_in_sync()


def generate_twiml(
    websocket_url: str | None = None,
    options: VoiceTwiMLOptionsConversationRelay | dict[str, Any] | None = None,
) -> str:
    """
    Generate TwiML XML for ConversationRelay.

    This is a low-level function. Most users should call
    ``VoiceChannel.handle_incoming_call`` instead — it layers in TAC defaults,
    static ``twiml_options`` from ``ConversationRelayProviderConfig``, and any per-call
    customizer output.

    The WebSocket URL may be passed positionally or as ``options.websocket_url``
    (positional wins when both are given), so a caller can pass everything in one
    object: ``generate_twiml(options=VoiceTwiMLOptionsConversationRelay(websocket_url=...))``.

    Args:
        websocket_url: Public WebSocket URL for ConversationRelay
            (e.g. ``'wss://example.ngrok.app/ws'``). Optional if
            ``options.websocket_url`` is set.
        options: Optional ``VoiceTwiMLOptionsConversationRelay`` (or dict). See that model
            for supported fields. Newly-added ConversationRelay attributes
            not yet typed on the model can be passed via ``extra``.

    Returns:
        TwiML XML string ready to return to Twilio.

    Raises:
        ValueError: If no WebSocket URL is provided via either source.

    Example:
        >>> twiml = generate_twiml(
        ...     "wss://example.com/voice",
        ...     VoiceTwiMLOptionsConversationRelay(
        ...         welcome_greeting="Hello!",
        ...         conversation_configuration="conv_configuration_xxxx",
        ...     ),
        ... )
    """
    if options is None:
        options = VoiceTwiMLOptionsConversationRelay()
    elif isinstance(options, dict):
        options = VoiceTwiMLOptionsConversationRelay(**options)

    # Positional arg wins when both are set; fall back to options.websocket_url.
    # (VoiceTwiMLOptionsConversationRelay rejects an empty options.websocket_url, but
    # the positional arg bypasses that pydantic validation entirely, so it still
    # needs its own whitespace check below.)
    resolved_websocket_url = websocket_url if websocket_url is not None else options.websocket_url
    if not resolved_websocket_url or not resolved_websocket_url.strip():
        raise ValueError(
            "generate_twiml requires a WebSocket URL — pass it positionally or "
            "set options.websocket_url."
        )

    response = VoiceResponse()

    # Create Connect verb with optional action
    connect_kwargs: dict[str, str] = {}
    if options.action_url:
        connect_kwargs["action"] = options.action_url
    connect = response.connect(**connect_kwargs)

    # Build ConversationRelay kwargs. The twilio SDK converts snake_case to
    # camelCase automatically, and serializes bool/str as TwiML attribute values.
    relay_kwargs: dict[str, Any] = {"url": resolved_websocket_url}
    for attr in _OPTIONAL_RELAY_ATTRS:
        value = getattr(options, attr)
        if value is None:
            continue
        # Twilio accepts True/False on `interruptible` for backward-compat
        # but the documented enum is none|dtmf|speech|any. Normalize so we
        # emit canonical values regardless of Twilio SDK's bool serialization.
        if attr == "interruptible" and isinstance(value, bool):
            value = "any" if value else "none"
        relay_kwargs[attr] = value

    # VoiceTwiMLOptionsConversationRelay's validator already rejects extra keys that shadow
    # typed fields, so we can pass everything through here as-is.
    if options.extra:
        relay_kwargs.update(options.extra)

    relay = connect.conversation_relay(**relay_kwargs)

    # Emit <Language> children, if any
    if options.languages:
        for lang in options.languages:
            lang_kwargs: dict[str, Any] = {"code": lang.code}
            for attr in ("voice", "tts_provider", "transcription_provider", "speech_model"):
                value = getattr(lang, attr)
                if value is not None:
                    lang_kwargs[attr] = value
            relay.language(**lang_kwargs)

    # Add custom parameters as <Parameter> children
    if options.custom_parameters:
        params_dict: dict[str, Any] = (
            options.custom_parameters.model_dump(by_alias=True, exclude_none=True)
            if isinstance(options.custom_parameters, BaseModel)
            else options.custom_parameters
        )
        for name, value in params_dict.items():
            if value is not None:
                relay.parameter(name=name, value=str(value))

    return str(response)


DEFAULT_WELCOME_GREETING = "Hello! How can I assist you today?"


class TwiMLBuilderConversationRelay(TwiMLBuilderBase):
    """Builds the TwiML for a ConversationRelay call, owning every layering
    and resolution decision so ``VoiceChannel`` doesn't have to.

    Takes ``TACConfig`` and ``ConversationRelayProviderConfig`` wholesale (not individual
    derived values) so a later change to either — a new field, a new default
    — is a change to this class alone, not a change to what ``VoiceChannel``
    has to compute and hand over.
    """

    channel_config: ConversationRelayProviderConfig

    def build(
        self,
        caller: str,
        *,
        host: VoiceTwiMLOptionsConversationRelay | None = None,
        per_call: VoiceTwiMLOptionsConversationRelay | None = None,
        websocket_url: str | None = None,
    ) -> str:
        """Build the TwiML XML for one call.

        Args:
            caller: Name of the calling method, used in the "no WebSocket URL"
                error so it points at the API the developer actually called.
            host: Per-call overrides from the host owning the route (e.g. a
                per-call ``websocket_url`` with an affinity token). Lowest of
                the three option layers.
            per_call: Per-call overrides — the ``on_inbound_call_twiml``
                customizer's output for inbound, or
                ``InitiateVoiceConversationOptions.twiml_options`` for outbound.
                Highest layer.
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
        elif (default_url := self._default_websocket_url()) is not None:
            resolved_websocket_url = default_url
        else:
            raise self._missing_websocket_url_error(caller)

        return generate_twiml(resolved_websocket_url, merged)

    def _build_twiml_options(
        self,
        host: VoiceTwiMLOptionsConversationRelay | None,
        per_call: VoiceTwiMLOptionsConversationRelay | None,
    ) -> VoiceTwiMLOptionsConversationRelay:
        """Layer TwiML options, lowest precedence first: TAC defaults →
        ``host`` (calling host's per-call values) → ``default_twiml_options`` →
        ``per_call`` (application customizer output for inbound, or
        ``InitiateVoiceConversationOptions.twiml_options`` for outbound).
        """
        merged = VoiceTwiMLOptionsConversationRelay(
            welcome_greeting=DEFAULT_WELCOME_GREETING,
            conversation_configuration=self.tac_config.conversation_configuration_id,
            action_url=self._resolve_action_url(host, per_call),
        )
        # action_url is resolved above via _resolve_action_url, so skip it here.
        if host is not None:
            self._overlay_fields(merged, host, skip={"action_url"})
        if self.channel_config.default_twiml_options is not None:
            self._overlay_fields(
                merged, self.channel_config.default_twiml_options, skip={"action_url"}
            )
        if per_call is not None:
            self._overlay_fields(merged, per_call, skip={"action_url"})
        return merged

    def _resolve_action_url(
        self,
        host: VoiceTwiMLOptionsConversationRelay | None,
        customized: VoiceTwiMLOptionsConversationRelay | None,
    ) -> str | None:
        """Resolve the TwiML ``<Connect action=...>`` URL.

        Precedence (highest to lowest):
          1. application customizer
          2. channel ``default_twiml_options``
          3. ``host`` (calling host's per-call options)
          4. Studio handoff (when ``studio_handoff_flow_sid`` is configured)
          5. Channel default — derived from ``TACConfig.voice_public_domain``
             + ``TACConfig.voice_action_path``.

        User-expressed intent (Studio handoff is configured explicitly on
        ``TACConfig``) beats the SDK's generated cleanup default. If a user
        sets both Studio handoff and runs in relay-only mode, Studio wins
        for that call — the session-cleanup URL is skipped, same as if they
        had set any other action_url via customizer or static options.

        Explicit ``action_url=None`` on a layer suppresses
        ``<Connect action=...>`` entirely — all lower layers are skipped.
        Use this to disable the cleanup callback for a specific call (e.g.
        from a customizer) or channel-wide. ``action_url`` left unset (not
        in ``model_fields_set``) falls through to the next layer.
        """
        if customized is not None and "action_url" in customized.model_fields_set:
            return customized.action_url
        if (
            self.channel_config.default_twiml_options is not None
            and "action_url" in self.channel_config.default_twiml_options.model_fields_set
        ):
            return self.channel_config.default_twiml_options.action_url
        if host is not None and "action_url" in host.model_fields_set:
            return host.action_url
        if self.tac_config.studio_handoff_flow_sid:
            return studio_voice_handoff_url(
                self.tac_config.account_sid,
                self.tac_config.studio_handoff_flow_sid,
            )
        # Channel default. None if voice_public_domain isn't set; that's fine
        # because every layer above this one is already exhausted.
        if self.tac_config.voice_public_domain:
            return (
                f"https://{self.tac_config.voice_public_domain}{self.tac_config.voice_action_path}"
            )
        return None
