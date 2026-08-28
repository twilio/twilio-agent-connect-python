"""Voice channel configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from tac._deprecation import resolve_deprecated_alias
from tac.channels.voice.conversation_relay.provider import ConversationRelayProvider
from tac.channels.voice.provider import VoiceProvider, VoiceProviderConfig
from tac.core.config import TACConfig

if TYPE_CHECKING:
    from tac.channels.voice.channel import VoiceChannel
from tac.models.outbound import CallOptions
from tac.models.voice import VoiceTwiMLOptionsConversationRelay
from tac.session import SessionManager, ThreadSafeSessionManager


class ConversationRelayProviderConfig(VoiceProviderConfig):
    """
    Configuration for Voice channel.

    TwiML configuration layers (highest precedence first):

      Inbound calls (``handle_incoming_call``):
        1. Output of the customizer registered via
           ``VoiceChannel.on_inbound_call_twiml(...)`` [optional]
        2. ``default_twiml_options``                     [optional]
        3. ``handle_incoming_call(host_twiml_options=...)`` [optional]
        4. TAC defaults

      Outbound calls (``initiate_outbound_conversation``):
        1. ``InitiateVoiceConversationOptions.twiml_options`` [optional]
        2. ``default_twiml_options``                          [optional]
        3. TAC defaults

      Calls-API parameters (``initiate_outbound_conversation``):
        1. ``InitiateVoiceConversationOptions.call_options`` [optional]
        2. ``default_call_options``                          [optional]
        3. Callback URLs derived from ``TACConfig.voice_public_domain`` +
           ``voice_call_event_path``, for handlers that are registered

    All layers merge per-field via Pydantic's ``model_fields_set`` — only
    fields a layer explicitly sets override lower layers. Lists (``languages``)
    and nested models (``custom_parameters``) replace wholesale when set.

    Attributes:
        session_manager: SessionManager for tracking and canceling in-flight tasks.
            Defaults to ThreadSafeSessionManager for automatic task cancellation on
            interrupts and new prompts. Set to None only for debugging/testing.
        memory_mode: Memory retrieval mode. Default is "never".
            - "always": Retrieve memory for every message with the query string
            - "once": Retrieve memory once at conversation start with empty query and cache it.
                     Cache is invalidated when conversation becomes INACTIVE.
            - "never": Skip memory retrieval
        default_twiml_options: Static ``VoiceTwiMLOptionsConversationRelay`` applied to every call
            (inbound and outbound). Controls the TwiML inside
            ``<ConversationRelay>`` — voice, language, transcription provider,
            welcome_greeting, ``<Language>`` children, etc. Use this when the
            same ConversationRelay configuration is correct for every call.
        default_call_options: Static ``CallOptions`` applied to every outbound
            call — the ``calls.create`` parameters, including the call-event
            callback URLs. This is the layer to use for a custom server or
            non-default routes.

    Per-call inbound customization is registered via
    ``VoiceChannel.on_inbound_call_twiml(...)`` (not on this config).
    """

    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    session_manager: SessionManager | None = Field(
        default_factory=ThreadSafeSessionManager,
        description=(
            "SessionManager for task cancellation. Defaults to ThreadSafeSessionManager. "
            "Set to None only for debugging/testing."
        ),
    )
    default_twiml_options: VoiceTwiMLOptionsConversationRelay | None = Field(
        default=None,
        description="Static VoiceTwiMLOptionsConversationRelay for the TwiML inside "
        "<ConversationRelay>, "
        "applied to every call (inbound and outbound). Per-call inbound "
        "customization is registered via VoiceChannel.on_inbound_call_twiml(...). "
        "Note: ``custom_parameters`` and ``languages`` replace wholesale when a "
        "higher-priority layer sets them — see "
        "tac.channels.voice.conversation_relay.twiml.TwiMLBuilderConversationRelay._overlay_fields.",
    )
    default_call_options: CallOptions | None = Field(
        default=None,
        description="Static CallOptions applied to every outbound call — AMD, "
        "recording, timeout, and the call-event callback URLs. Set the URLs here "
        "when TAC isn't serving the routes (custom server) or they're at "
        "non-default paths; they override the URLs TAC would derive from "
        "voice_public_domain + voice_call_event_path.",
    )

    def create_provider(self, channel: VoiceChannel, tac_config: TACConfig) -> VoiceProvider:
        return ConversationRelayProvider(channel, tac_config, self)


def __getattr__(name: str) -> Any:
    """Lazily resolve ``VoiceChannelConfig``, the deprecated pre-provider-split
    name for ``ConversationRelayProviderConfig``.

    A module-level ``__getattr__`` (PEP 562) — rather than a subclass with its own
    ``__init__`` — keeps ``VoiceChannelConfig`` a true alias:
    ``VoiceChannelConfig is ConversationRelayProviderConfig``, so
    ``isinstance``/``==`` behave exactly as they did before the rename, and the
    warning fires once per access rather than once per instantiation. Every
    other re-export of this name (``tac.channels``, ``tac.channels.voice``,
    ``tac.channels.voice.conversation_relay``) resolves it the same way,
    directly — see ``tac._deprecation.resolve_deprecated_alias``.

    TODO(3.0): remove.
    """
    if name == "VoiceChannelConfig":
        return resolve_deprecated_alias("VoiceChannelConfig", ConversationRelayProviderConfig)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
