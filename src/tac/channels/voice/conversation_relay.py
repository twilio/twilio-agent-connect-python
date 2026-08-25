"""``ConversationRelayProvider``: the default ``VoiceProvider`` — Twilio
ConversationRelay's managed setup/prompt/interrupt loop over one WebSocket,
with Twilio doing ASR/TTS.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from tac.channels.voice.config import ConversationRelayProviderConfig
from tac.channels.voice.provider import VoiceProvider
from tac.channels.voice.twiml import generate_twiml
from tac.channels.websocket_manager import WebSocketManager
from tac.channels.websocket_protocol import WebSocketDisconnectError, WebSocketProtocol
from tac.core.config import CallEventKind
from tac.models.memory import MemoryMode
from tac.models.outbound import (
    CallOptions,
    InitiateVoiceConversationOptions,
    InitiateVoiceConversationResult,
)
from tac.models.session import AuthorInfo
from tac.models.voice import (
    ConversationRelayCallbackPayload,
    InterruptMessage,
    PromptMessage,
    SetupMessage,
    TwiMLOptions,
    TwiMLRequest,
)
from tac.session import SessionState
from tac.tools.handoff import studio_voice_handoff_url
from tac.utils.redaction import mask_phone, redact_twiml_parameters

if TYPE_CHECKING:
    from tac.channels.voice.channel import VoiceChannel

DEFAULT_WELCOME_GREETING = "Hello! How can I assist you today?"

_POLL_ATTEMPTS = 10
_POLL_BASE_DELAY = 0.25
# Caps the exponential backoff so a higher _POLL_ATTEMPTS doesn't blow up the
# total wait time — total worst case is comfortably bounded (~11s of sleep)
# instead of growing exponentially with attempt count.
_POLL_MAX_DELAY = 1.5


class ConversationRelayProvider(VoiceProvider):
    """Twilio ConversationRelay: Twilio handles ASR/TTS and exchanges JSON
    ``setup``/``prompt``/``interrupt`` messages over one WebSocket.

    This is the default provider ``VoiceChannel`` builds when none is passed
    explicitly.
    """

    def __init__(
        self,
        config: ConversationRelayProviderConfig | dict[str, Any] | None = None,
    ) -> None:
        if isinstance(config, dict):
            config = ConversationRelayProviderConfig(**config)
        elif config is None:
            config = ConversationRelayProviderConfig()
        self.config = config
        self._websocket_manager = WebSocketManager()
        self._on_inbound_call_twiml: Callable[[TwiMLRequest], Awaitable[TwiMLOptions]] | None = None

    @property
    def channel_name(self) -> str:
        return "VOICE"

    @property
    def memory_mode(self) -> MemoryMode:
        return self.config.memory_mode

    def build_twiml(self, websocket_url: str, options: TwiMLOptions) -> str:
        return generate_twiml(websocket_url, options)

    def on_inbound_call_twiml(
        self, callback: Callable[[TwiMLRequest], Awaitable[TwiMLOptions]]
    ) -> None:
        """Register a callback that produces per-call overrides for the
        TwiML inside ``<ConversationRelay>`` on inbound calls.

        The callback receives a framework-neutral ``TwiMLRequest`` (parsed
        from the Twilio webhook form) and returns a ``TwiMLOptions``. Fields
        the callback explicitly sets override ``default_twiml_options`` and
        TAC defaults; unset fields fall through.

        Outbound calls don't use this — pass per-call TwiML via
        ``InitiateVoiceConversationOptions.twiml_options`` directly.
        """
        self._on_inbound_call_twiml = callback

    async def handle_incoming_call(
        self,
        channel: VoiceChannel,
        twiml_request: TwiMLRequest | None,
        host_twiml_options: TwiMLOptions | None,
    ) -> str:
        """
        Generate TwiML response for incoming voice calls.

        ConversationRelay automatically handles conversation creation and participant
        management via the ``conversation_configuration`` parameter.

        The WebSocket URL and default session-cleanup action URL are derived
        from ``TACConfig.voice_public_domain`` + ``TACConfig.voice_websocket_path``
        / ``voice_action_path``.

        TwiML fields are merged per-field, highest precedence first:
          1. Output of the customizer registered via
             ``on_inbound_call_twiml(...)`` if configured and ``twiml_request``
             is given. (Application-owned.)
          2. ``self.config.default_twiml_options`` — per-channel defaults.
          3. ``host_twiml_options`` — per-call transport facts supplied by the
             host (the code owning the route), e.g. a per-call ``websocket_url``
             with an affinity token.
          4. TAC defaults: a fixed default ``welcome_greeting``,
             ``conversation_configuration`` from ``TACConfig``,
             ``action_url`` resolved via Studio handoff (when
             ``studio_handoff_flow_sid`` is configured), else derived from
             ``TACConfig.voice_public_domain`` + ``voice_action_path``, and the
             ``websocket_url`` derived from ``TACConfig.voice_public_domain`` +
             ``voice_websocket_path``.

        Fields not set at a layer fall through to lower layers. Lists
        (``languages``) and nested models (``custom_parameters``) replace
        wholesale when set at a higher-priority layer. ``websocket_url`` falls
        back to the ``TACConfig``-derived URL if unset at every layer.

        The two arguments are complementary, not alternatives — a custom host
        typically passes both on the same call: ``twiml_request`` carries the
        inbound call's data (so the application's customizer can run), and
        ``host_twiml_options`` carries the host's own per-call overrides.
        """
        customized: TwiMLOptions | None = None
        if self._on_inbound_call_twiml is not None and twiml_request is not None:
            customized = await self._on_inbound_call_twiml(twiml_request)

        merged = self._build_twiml_options(channel, host_twiml_options, customized)
        # merged.websocket_url is either a validated non-empty URL (set by some
        # layer) or None; fall back to the TACConfig-derived URL only when None.
        websocket_url = (
            merged.websocket_url
            if merged.websocket_url is not None
            else channel._resolve_websocket_url("handle_incoming_call")
        )
        return self.build_twiml(websocket_url, merged)

    def _build_twiml_options(
        self,
        channel: VoiceChannel,
        host: TwiMLOptions | None,
        per_call: TwiMLOptions | None,
    ) -> TwiMLOptions:
        """Layer TwiML options, lowest precedence first: TAC defaults →
        ``host`` (calling host's per-call values) → ``self.config.default_twiml_options``
        → ``per_call`` (application customizer output for inbound, or
        ``InitiateVoiceConversationOptions.twiml_options`` for outbound).
        """
        merged = TwiMLOptions(
            welcome_greeting=DEFAULT_WELCOME_GREETING,
            conversation_configuration=channel.tac.config.conversation_configuration_id,
            action_url=self._resolve_action_url(channel, host, per_call),
        )
        if host is not None:
            self._overlay_fields(merged, host)
        if self.config.default_twiml_options is not None:
            self._overlay_fields(merged, self.config.default_twiml_options)
        if per_call is not None:
            self._overlay_fields(merged, per_call)
        return merged

    @staticmethod
    def _overlay_fields(target: TwiMLOptions, source: TwiMLOptions) -> None:
        """Apply fields explicitly set on ``source`` onto ``target``.

        Nested models (``custom_parameters``), lists (``languages``), and
        dicts (``extra``) replace wholesale — there's no per-key merging.
        If you add a field that should merge (e.g. a dict of headers),
        special-case it here instead of getting the default overwrite behavior.

        ``action_url`` is skipped here on purpose — it's resolved once via
        ``_resolve_action_url`` looking at every layer at once, and that
        resolved value is written into ``target`` before this overlay runs.
        Letting it through here would let a higher-priority layer that didn't
        set action_url silently clobber a lower layer that did.
        """
        for field in source.model_fields_set:
            if field == "action_url":
                continue
            setattr(target, field, getattr(source, field))

    def _resolve_action_url(
        self,
        channel: VoiceChannel,
        host: TwiMLOptions | None,
        customized: TwiMLOptions | None,
    ) -> str | None:
        """Resolve the TwiML ``<Connect action=...>`` URL.

        Precedence (highest to lowest):
          1. application customizer
          2. ``self.config.default_twiml_options``
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
            self.config.default_twiml_options is not None
            and "action_url" in self.config.default_twiml_options.model_fields_set
        ):
            return self.config.default_twiml_options.action_url
        if host is not None and "action_url" in host.model_fields_set:
            return host.action_url
        if channel.tac.config.studio_handoff_flow_sid:
            return studio_voice_handoff_url(
                channel.tac.config.account_sid,
                channel.tac.config.studio_handoff_flow_sid,
            )
        return self._resolve_default_action_url(channel)

    @staticmethod
    def _resolve_default_action_url(channel: VoiceChannel) -> str | None:
        """Resolve the default ``<Connect action=...>`` cleanup URL.

        Returns None if ``voice_public_domain`` isn't set; that's fine because
        action_url has higher-priority layers (customizer, twiml_options,
        Studio handoff) above this fallback.
        """
        if channel.tac.config.voice_public_domain:
            return (
                f"https://{channel.tac.config.voice_public_domain}"
                f"{channel.tac.config.voice_action_path}"
            )
        return None

    async def handle_conversation_relay_callback(
        self,
        channel: VoiceChannel,
        payload_dict: dict[str, str],
    ) -> None:
        """Handle ConversationRelay callback webhook from Twilio.

        In relay-only mode, this is a secondary mechanism for cleaning up
        conversation state when a call ends (the primary mechanism is websocket
        disconnect). In orchestrated mode, conversation lifecycle is managed by
        CO webhooks, so this is a no-op.
        """
        try:
            payload = ConversationRelayCallbackPayload(**payload_dict)
        except ValidationError:
            channel.logger.warning(
                "Invalid ConversationRelay callback payload, ignoring",
                payload_keys=list(payload_dict.keys()),
            )
            return

        if payload.account_sid != channel.tac.config.account_sid:
            channel.logger.warning(
                "ConversationRelay callback account_sid mismatch, ignoring",
                expected=channel.tac.config.account_sid,
                received=payload.account_sid,
            )
            return

        channel.logger.debug(
            "ConversationRelay callback received",
            call_sid=payload.call_sid,
            call_status=payload.call_status,
        )

        if payload.call_status == "completed" and not channel.tac.is_orchestrator_enabled():
            if payload.call_sid in channel._conversations:
                await channel._end_conversation(payload.call_sid)

    def _merge_call_options(self, per_call: CallOptions | None) -> CallOptions | None:
        """Overlay ``per_call`` onto ``self.config.default_call_options``.

        Per-field via ``model_fields_set``, same as ``_overlay_fields`` does for
        TwiMLOptions. The merged result is re-validated so a combination only
        reachable by layering — per-call clearing ``machine_detection`` while the
        default set ``async_amd`` — still fails instead of reaching Twilio.
        """
        default = self.config.default_call_options
        if default is None or per_call is None:
            return per_call or default

        merged = default.model_dump(by_alias=True, exclude_none=True)
        # model_fields_set covers extras too, since CallOptions allows them.
        for field in per_call.model_fields_set:
            merged[field] = getattr(per_call, field)
        return CallOptions(**merged)

    def _build_call_kwargs(
        self, channel: VoiceChannel, call_options: CallOptions | None
    ) -> dict[str, Any]:
        """Build the extra kwargs for ``client.calls.create``.

        Layers, highest precedence first: this call's ``call_options``,
        ``self.config.default_call_options``, then callback URLs derived
        from ``voice_public_domain`` + ``voice_call_event_path``.

        A URL is derived only when its handler is registered. That's a deliberate
        deviation from ``websocket_url`` / ``action_url``, which derive
        unconditionally: those are load-bearing, so a wrong one fails loudly on
        the first call, whereas an unwanted call-event URL fails as silent 11200
        alerts for a feature nobody asked for. Set the URLs in
        ``default_call_options`` when TAC isn't serving the routes.
        """
        merged = self._merge_call_options(call_options)
        call_kwargs = merged.to_call_kwargs() if merged else {}

        wiring: list[tuple[CallEventKind, str, Callable[..., Any] | None]] = [
            ("status", "status_callback", channel._on_call_status),
            ("amd", "async_amd_status_callback", channel._on_amd),
            ("recording", "recording_status_callback", channel._on_recording),
        ]
        for kind, param, handler in wiring:
            if handler is None:
                continue
            url = channel.tac.config.call_event_url(kind)
            if url is not None:
                call_kwargs.setdefault(param, url)

        return call_kwargs

    async def initiate_outbound_conversation(
        self,
        channel: VoiceChannel,
        options: InitiateVoiceConversationOptions,
    ) -> InitiateVoiceConversationResult:
        """Initiate an outbound voice conversation.

        Places an outbound call with inline TwiML that connects to ConversationRelay.
        The conversationConfiguration attribute tells CO to create and manage the
        conversation during passive hydration. The session is initialized lazily
        on the first prompt when the conversation is discovered by callSid.

        TwiML fields are merged per-field, highest precedence first:
          1. ``options.twiml_options`` — per-call overrides
          2. ``self.config.default_twiml_options`` — channel-wide defaults
          3. TAC defaults: welcome greeting, ``conversation_configuration``
             from ``TACConfig``, and ``action_url`` from Studio handoff (if
             configured), else derived from ``TACConfig.voice_public_domain``
             + ``voice_action_path``.

        Fields not set at a layer fall through to lower layers. Lists
        (``languages``) and nested models (``custom_parameters``) replace
        wholesale when set at a higher-priority layer.

        The WebSocket URL is derived from ``TACConfig.voice_public_domain`` +
        ``TACConfig.voice_websocket_path``, unless overridden per-call via
        ``options.websocket_url``.
        """
        from_number = channel.tac.config.phone_number

        channel.logger.info(
            "Initiating outbound voice conversation",
            to=mask_phone(options.to),
            from_number=mask_phone(from_number),
        )

        # Outbound has no inbound customizer and no server layer; the per-call
        # override is options.twiml_options.
        merged = self._build_twiml_options(channel, None, options.twiml_options)

        # ``options.websocket_url`` is the dedicated per-call outbound override
        # and wins over any websocket_url that came through the layered
        # ``twiml_options`` merge; both fall back to the TACConfig-derived URL.
        if options.websocket_url is not None:
            websocket_url = options.websocket_url
        elif merged.websocket_url is not None:
            websocket_url = merged.websocket_url
        else:
            websocket_url = channel._resolve_websocket_url("initiate_outbound_conversation")

        call_kwargs = self._build_call_kwargs(channel, options.call_options)

        try:
            twiml_xml = self.build_twiml(websocket_url, merged)

            # The inline TwiML handed to Twilio, useful for debugging the
            # <Connect action> handoff target. custom_parameters values are
            # masked — they're arbitrary developer data (profile IDs, caller
            # names), unlike the WS/action URLs and conversation config.
            channel.logger.debug(
                "Outbound call TwiML",
                twiml=redact_twiml_parameters(twiml_xml),
                to=mask_phone(options.to),
            )

            client = channel._get_twilio_client()
            call = await asyncio.to_thread(
                client.calls.create,
                to=options.to,
                from_=from_number,
                twiml=twiml_xml,
                **call_kwargs,
            )

            channel.logger.info(
                "Outbound voice call placed",
                call_sid=call.sid,
                to=mask_phone(options.to),
            )

            return InitiateVoiceConversationResult(call_sid=call.sid)

        except Exception as e:
            channel.logger.error(
                "Failed to initiate outbound call",
                to=mask_phone(options.to),
                error=str(e),
                exc_info=True,
            )
            raise

    def get_websocket(self, conversation_id: str) -> WebSocketProtocol | None:
        return self._websocket_manager.get_websocket(conversation_id)

    @staticmethod
    def _caller_address(setup_msg: SetupMessage) -> str | None:
        """Return the phone number of the remote caller/callee from the setup message."""
        if setup_msg.direction and setup_msg.direction.upper() == "OUTBOUND":
            return setup_msg.to_number
        return setup_msg.from_number

    async def _initialize_conversation(
        self,
        channel: VoiceChannel,
        call_sid: str,
        setup_msg: SetupMessage,
        websocket: WebSocketProtocol,
    ) -> tuple[str, SessionState | None]:
        """Poll CO for the conversation created by ConversationRelay, resolve
        the customer participant, and initialize the local session."""
        conversation_orchestrator_client = channel.tac.conversation_orchestrator_client
        if conversation_orchestrator_client is None:
            raise RuntimeError("_initialize_conversation called without Conversation Orchestrator")

        conversations: list[Any] = []
        for attempt in range(_POLL_ATTEMPTS):
            conversations = await conversation_orchestrator_client.list_conversations(
                channel_id=call_sid,
                status=["ACTIVE"],
            )
            if len(conversations) == 1:
                break
            if attempt < _POLL_ATTEMPTS - 1:
                channel.logger.debug(
                    "Conversation not ready yet, polling again",
                    call_sid=call_sid,
                    attempt=attempt + 1,
                    found=len(conversations),
                )
                await asyncio.sleep(min(_POLL_BASE_DELAY * (2**attempt), _POLL_MAX_DELAY))

        if len(conversations) != 1:
            raise RuntimeError(
                f"Expected exactly 1 conversation for "
                f"call_sid {call_sid}, but found "
                f"{len(conversations)} after "
                f"{_POLL_ATTEMPTS} attempts."
            )

        conversation = conversations[0]
        conv_id = conversation.id

        participants = await conversation_orchestrator_client.list_participants(conv_id)

        customer_participant = next(
            (p for p in participants if p.type == "CUSTOMER"),
            None,
        )
        customer_address = (
            next(
                (a.address for a in customer_participant.addresses if a.channel == "VOICE"),
                None,
            )
            if customer_participant and customer_participant.addresses
            else None
        )
        profile_lookup_address = customer_address or self._caller_address(setup_msg)
        profile_id = customer_participant.profile_id if customer_participant else None

        # Resolve the agent participant so ai_agent_info is populated on the
        # session, matching the messaging channels. The agent is the participant
        # that owns TAC's address (the configured phone number) on the VOICE
        # channel and has an agent type. A HUMAN_AGENT added by a
        # redirected/escalated call is NOT TAC and is not adopted here.
        agent_participant = channel._find_agent_participant(
            participants, "VOICE", channel.tac.config.phone_number
        )
        agent_address = (
            next(
                (a.address for a in agent_participant.addresses if a.channel == "VOICE"),
                None,
            )
            if agent_participant and agent_participant.addresses
            else None
        )

        self._websocket_manager.add_websocket(conv_id, websocket)
        session = channel._start_conversation(conv_id, profile_id)
        # In orchestrator mode conv_id is the Orchestrator conversation id, so
        # record the CallSid so out-of-band call webhooks can reach this session
        # (resolved via get_conversation_session_by_call_sid).
        session.call_sid = call_sid

        session_state = None
        if self.config.session_manager is not None:
            session_state = self.config.session_manager.get_or_create_session(conv_id)

        if profile_lookup_address:
            session.author_info = AuthorInfo(address=profile_lookup_address)

        if agent_participant:
            # Fall back to the configured phone number we matched on — the
            # participant owns it by definition, so it's a meaningful address
            # even in the unlikely case it carries no explicit VOICE address.
            session.ai_agent_info = AuthorInfo(
                address=agent_address or channel.tac.config.phone_number,
                participant_id=agent_participant.id,
            )

        return conv_id, session_state

    async def handle_websocket(self, channel: VoiceChannel, websocket: WebSocketProtocol) -> None:
        """
        Handle voice streaming WebSocket connection lifecycle.

        This method manages the entire websocket connection:
        - Accepts the connection
        - Processes incoming messages
        - Tracks and cancels in-flight tasks (if session_manager provided)
        - Cleans up on disconnect
        """
        await websocket.accept()
        channel.logger.debug("WebSocket connection established")

        conv_id: str | None = None
        session_state = None
        # This call's background CO conversation lookup task (kicked off
        # below on "setup"), consumed and reset to None on the first
        # "prompt". Still non-None in `finally` means the call ended before
        # any prompt arrived.
        init_task: asyncio.Task[tuple[str, SessionState | None]] | None = None

        try:
            # First message should be 'setup'
            data = await websocket.receive_json()
            if data.get("type") == "setup":
                setup_msg = SetupMessage(**data)
                call_sid = setup_msg.call_sid

                # Kick off the CO conversation lookup now, in the background,
                # instead of waiting for the first prompt. ConversationRelay
                # creates the CO conversation as soon as the call connects, not
                # when the caller speaks, so this overlaps CO's
                # list_conversations/list_participants polling with the wait
                # for the caller's first utterance (transcription) rather than
                # paying that latency serially once the first prompt lands.
                if call_sid and channel.tac.is_orchestrator_enabled():
                    init_task = asyncio.create_task(
                        self._initialize_conversation(channel, call_sid, setup_msg, websocket)
                    )

                # Process all subsequent messages
                while True:
                    data = await websocket.receive_json()
                    msg_type = data.get("type")

                    if msg_type == "prompt":
                        if not conv_id and call_sid:
                            if init_task is not None:
                                # Clear before awaiting so a failure here
                                # doesn't leave `finally` re-awaiting (and
                                # re-logging) this same task. If still
                                # polling, this just waits it out. (None here
                                # only means relay-only mode — see "setup".)
                                task_to_await = init_task
                                init_task = None
                                conv_id, session_state = await task_to_await
                            else:
                                conv_id = call_sid
                                self._websocket_manager.add_websocket(conv_id, websocket)
                                session = channel._start_conversation(conv_id, profile_id=None)
                                # Relay-only: conv_id == call_sid.
                                session.call_sid = call_sid

                                caller = self._caller_address(setup_msg)
                                if caller:
                                    channel._conversations[conv_id].author_info = AuthorInfo(
                                        address=caller,
                                    )

                                if self.config.session_manager is not None:
                                    session_state = (
                                        self.config.session_manager.get_or_create_session(conv_id)
                                    )

                        if conv_id:
                            await self._handle_prompt_async(channel, conv_id, data, session_state)
                        else:
                            channel.logger.warning(
                                "Received prompt before conversation initialized"
                            )
                    elif msg_type == "interrupt":
                        if conv_id:
                            await self._handle_interrupt_async(
                                channel, conv_id, data, session_state
                            )
                        else:
                            channel.logger.warning(
                                "Received interrupt before conversation initialized"
                            )
                    else:
                        channel.logger.debug(f"Skip message type received: {msg_type}")
            else:
                channel.logger.warning("First message was not 'setup'. Closing connection.")
                await websocket.close()
                return
        except WebSocketDisconnectError:
            channel.logger.info("WebSocket connection closed", conversation_id=conv_id)
        except Exception as e:
            channel.logger.error(f"WebSocket error: {str(e)}")
        finally:
            cancelled_error: asyncio.CancelledError | None = None
            we_cancelled_it = False
            if init_task is not None:
                # Call ended before any prompt arrived, so the background
                # lookup was never awaited.
                if not init_task.done():
                    init_task.cancel()
                    we_cancelled_it = True
                result: tuple[str, SessionState | None] | None
                try:
                    result = await init_task
                except asyncio.CancelledError as e:
                    # Defer re-raise decision until after cleanup below;
                    # we_cancelled_it (not init_task.cancelled(), which can't
                    # tell the two apart) decides whether to.
                    cancelled_error = e
                    result = None
                except Exception as e:
                    # No prompt ever arrived to surface this failure via the
                    # outer except Exception above, so log it here instead.
                    channel.logger.error(
                        f"Background CO conversation lookup failed: {e}",
                        call_sid=call_sid,
                    )
                    result = None
                if result is not None and conv_id is None:
                    # The lookup already registered the session/websocket
                    # before anyone claimed conv_id — adopt it so it's
                    # cleaned up instead of leaked.
                    conv_id = result[0]
            if conv_id:
                channel.logger.debug("Cleanup - removing WebSocket", conversation_id=conv_id)
                await self._cleanup_connection(channel, conv_id)
            if cancelled_error is not None and not we_cancelled_it:
                # A real external cancellation, not the one we caused above —
                # propagate it now that cleanup ran.
                raise cancelled_error

    async def _handle_prompt_async(
        self,
        channel: VoiceChannel,
        conv_id: str,
        data: dict[str, Any],
        session_state: SessionState | None,
    ) -> None:
        """Handle prompt message asynchronously with task tracking."""
        try:
            should_process = data.get("final", True)
            if should_process:
                prompt_msg = PromptMessage(**data)
                conv_id = prompt_msg.conversation_id or conv_id

                # Cancel previous stream task if session manager is enabled
                if session_state:
                    await session_state.cancel_stream_task()

                    # Create new task using unified flow (memory retrieval + callback)
                    session_state.stream_task = asyncio.create_task(
                        self._handle_prompt(channel, conv_id, prompt_msg)
                    )
                    # Yield to event loop to let task start
                    await asyncio.sleep(0)
                else:
                    await self._handle_prompt(channel, conv_id, prompt_msg)
        except Exception as e:
            channel.logger.error(f"Failed to handle prompt: {str(e)}")

    async def _handle_interrupt_async(
        self,
        channel: VoiceChannel,
        conv_id: str,
        data: dict[str, Any],
        session_state: SessionState | None,
    ) -> None:
        """Handle interrupt message asynchronously with task cancellation."""
        try:
            interrupt_msg = InterruptMessage(**data)
            conv_id = interrupt_msg.conversation_id or conv_id

            # Cancel in-flight stream task if session manager is enabled
            if session_state:
                await session_state.cancel_stream_task()

                # Send acknowledgment to Twilio after cancelling
                websocket = self._websocket_manager.get_websocket(conv_id)
                if websocket:
                    try:
                        await websocket.send_text(
                            json.dumps({"type": "text", "token": "", "last": True})
                        )
                    except (WebSocketDisconnectError, RuntimeError):
                        channel.logger.debug(
                            f"WebSocket closed before sending interrupt acknowledgment "
                            f"for {conv_id}."
                        )

            # Call the interrupt handler
            self._handle_interrupt(channel, conv_id, interrupt_msg)
        except Exception as e:
            channel.logger.error(f"Failed to handle interrupt: {str(e)}")

    async def send_response(
        self,
        channel: VoiceChannel,
        conversation_id: str,
        response: str | AsyncGenerator[str | dict[str, Any], None],
        role: str | None = None,
    ) -> None:
        """
        Send voice response through the websocket connection for this conversation.

        Supports both simple string responses and streaming async generators.
        """
        # Validate response type before processing
        if not isinstance(response, (str, AsyncGenerator)):
            raise TypeError("Voice channel requires string or async generator for response")

        # Get WebSocket from manager
        websocket = self._websocket_manager.get_websocket(conversation_id)
        if not websocket:
            channel.logger.error("No websocket connection", conversation_id=conversation_id)
            return

        full_response = ""

        try:
            # Check if response is an async generator (streaming)
            if isinstance(response, AsyncGenerator):
                # Streaming response
                json_template = {"type": "text", "token": "", "last": False}
                closed = False
                response_gen: AsyncGenerator[str | dict[str, Any], None] = response

                try:
                    async for chunk in response_gen:
                        # Handle different chunk types (plain text or dict with metadata)
                        if isinstance(chunk, dict):
                            if "output" in chunk:
                                token = chunk["output"]
                            else:
                                token = str(chunk)
                        else:
                            token = chunk

                        full_response += token
                        json_template["token"] = token

                        try:
                            await websocket.send_text(json.dumps(json_template))
                        except (WebSocketDisconnectError, RuntimeError):
                            channel.logger.info(
                                "WebSocket closed during streaming",
                                conversation_id=conversation_id,
                            )
                            closed = True
                            break

                    # Send final message marker
                    if not closed:
                        try:
                            await websocket.send_text(
                                json.dumps({"type": "text", "token": "", "last": True})
                            )
                        except (WebSocketDisconnectError, RuntimeError):
                            channel.logger.info(
                                "WebSocket closed before sending final marker",
                                conversation_id=conversation_id,
                            )
                except asyncio.CancelledError:
                    # Let Python's async generator cleanup handle closing the generator
                    raise
            else:
                await websocket.send_text(
                    json.dumps({"type": "text", "token": response, "last": True})
                )

            # If a handoff is pending, send the WS "end" message now that the
            # LLM's final response has been delivered to the caller.
            if conversation_id in channel._conversations:
                session = channel._conversations[conversation_id]
                if session.pending_handoff_data is not None:
                    try:
                        await websocket.send_text(
                            session.pending_handoff_data.model_dump_json(by_alias=True)
                        )
                        session.pending_handoff_data = None
                    except (WebSocketDisconnectError, RuntimeError):
                        channel.logger.warning(
                            "WebSocket closed before sending handoff end message; "
                            "caller will not be transferred",
                            conversation_id=conversation_id,
                        )

        except asyncio.CancelledError:
            # Re-raise to propagate cancellation up the call stack.
            # Partial responses from interrupted streams are NOT saved to
            # Conversation Orchestrator. Incomplete responses shouldn't be
            # part of conversation history.
            raise
        except (WebSocketDisconnectError, RuntimeError):
            channel.logger.info(
                "WebSocket closed before sending response", conversation_id=conversation_id
            )
        except Exception as e:
            channel.logger.error(
                f"Error sending response: {e}", conversation_id=conversation_id, exc_info=True
            )

    async def _handle_prompt(
        self, channel: VoiceChannel, conv_id: str, message: PromptMessage
    ) -> None:
        """Handle incoming voice prompt (user speech)."""
        if conv_id not in channel._conversations:
            channel.logger.error(
                f"Received prompt for unknown conversation {conv_id}. "
                "Conversation should be initialized on first prompt.",
                conversation_id=conv_id,
            )
            return

        message_body = message.voice_prompt or ""
        session = channel._conversations[conv_id]

        # Retrieve memory if memory_mode is enabled and Twilio Memory is configured
        memory_response = await channel._retrieve_memory_if_enabled(session, message_body, conv_id)

        # Trigger message ready callback
        try:
            response = await channel.tac.trigger_message_ready(
                message_body, session, memory_response
            )
            # Auto-send if callback returned a string (None = manual send_response flow)
            if response is not None:
                await self.send_response(channel, conv_id, response, role="assistant")
        except Exception as e:
            channel.logger.error(
                "Error in message ready callback",
                conversation_id=conv_id,
                error=str(e),
                exc_info=True,
            )

    def _handle_interrupt(
        self, channel: VoiceChannel, conv_id: str, message: InterruptMessage
    ) -> None:
        """Handle interrupt message when user interrupts the agent.

        Note: Task cancellation is handled by the async wrapper
        (_handle_interrupt_async) when called from the WebSocket message
        handler. This method only triggers the TAC interrupt callback.
        """
        # Trigger interrupt callback if conversation exists
        if conv_id in channel._conversations:
            session = channel._conversations[conv_id]
            channel.tac.trigger_interrupt(session, message)
        else:
            channel.logger.warning(
                f"Received interrupt for unknown conversation {conv_id}, skipping callback"
            )

    async def _cleanup_connection(self, channel: VoiceChannel, conv_id: str) -> None:
        """Clean up WebSocket and session resources when connection closes.

        In orchestrated mode, the conversation remains tracked in
        channel._conversations until the CONVERSATION_UPDATED/CLOSED webhook
        arrives from Conversation Orchestrator. In relay-only mode there is
        no such webhook, so we also end the conversation here.
        """
        # Remove WebSocket from manager
        if self._websocket_manager.has_websocket(conv_id):
            self._websocket_manager.remove_websocket(conv_id)

        # Cancel running stream task and cleanup session if session manager is enabled
        if self.config.session_manager is not None and self.config.session_manager.has_session(
            conv_id
        ):
            session_state = self.config.session_manager.get_or_create_session(conv_id)
            # Cancel any running task (user hung up, no point continuing)
            await session_state.cancel_stream_task()
            self.config.session_manager.remove_session(conv_id)

        if not channel.tac.is_orchestrator_enabled() and conv_id in channel._conversations:
            await channel._end_conversation(conv_id)

        channel.logger.debug(
            "Cleaned up WebSocket and session resources",
            conversation_id=conv_id,
        )


__all__ = ["ConversationRelayProvider"]
