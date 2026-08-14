from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from twilio.rest import Client

from pydantic import ValidationError

from tac.channels.base import BaseChannel
from tac.channels.websocket_manager import WebSocketManager
from tac.channels.websocket_protocol import WebSocketDisconnectError, WebSocketProtocol
from tac.core.config import CallEventKind
from tac.core.tac import TAC
from tac.models.outbound import (
    CallOptions,
    InitiateVoiceConversationOptions,
    InitiateVoiceConversationResult,
)
from tac.models.session import AuthorInfo, ConversationSession
from tac.models.voice import (
    AmdEvent,
    CallStatusEvent,
    ConversationRelayCallbackPayload,
    InterruptMessage,
    PromptMessage,
    RecordingEvent,
    SetupMessage,
    TwiMLOptions,
    TwiMLRequest,
)
from tac.session import SessionState
from tac.tools.handoff import studio_voice_handoff_url
from tac.utils.redaction import mask_phone, redact_twiml_parameters

from . import twiml
from .config import (
    AmdHandler,
    CallStatusHandler,
    InboundCallTwiMLHandler,
    RecordingHandler,
    VoiceChannelConfig,
)

_POLL_ATTEMPTS = 10
_POLL_BASE_DELAY = 0.25
# Caps the exponential backoff so a higher _POLL_ATTEMPTS doesn't blow up the
# total wait time — total worst case is comfortably bounded (~11s of sleep)
# instead of growing exponentially with attempt count.
_POLL_MAX_DELAY = 1.5

DEFAULT_WELCOME_GREETING = "Hello! How can I assist you today?"


class VoiceChannel(BaseChannel):
    """
    Voice Channel for handling voice-based conversations via WebSocket.

    Key features:
    - TwiML generation for incoming calls (see twiml module)
    - WebSocket connection management for real-time voice streaming
    - Conversation lifecycle management (inherited from BaseChannel)
    - Outbound call initiation

    This channel is framework-agnostic and accepts any WebSocket implementation
    satisfying WebSocketProtocol. For a batteries-included FastAPI server, use
    tac.server.TACFastAPIServer.
    """

    def __init__(
        self,
        tac: TAC,
        config: VoiceChannelConfig | dict[str, Any] | None = None,
    ):
        """
        Initialize Voice channel for websocket protocol handling.

        Args:
            tac: TAC instance for memory/context operations
            config: Voice channel configuration (VoiceChannelConfig or dict).
                If None, uses default configuration.

        Examples:
            >>> channel = VoiceChannel(tac, config={"memory_mode": "always"})
            >>> channel = VoiceChannel(tac, config=VoiceChannelConfig(session_manager=sm))
            >>> channel = VoiceChannel(tac)  # Use defaults
        """
        # Convert dict to config model or use defaults
        if isinstance(config, dict):
            config = VoiceChannelConfig(**config)
        elif config is None:
            config = VoiceChannelConfig()

        super().__init__(tac, memory_mode=config.memory_mode)
        self.config = config
        self.session_manager = config.session_manager
        self._on_inbound_call_twiml: InboundCallTwiMLHandler | None = None
        self._on_call_status: CallStatusHandler | None = None
        self._on_amd: AmdHandler | None = None
        self._on_recording: RecordingHandler | None = None
        self._websocket_manager = WebSocketManager()
        self._twilio_client: Client | None = None

    def on_inbound_call_twiml(self, callback: InboundCallTwiMLHandler) -> None:
        """Register a callback that produces per-call overrides for the
        TwiML inside ``<ConversationRelay>`` on inbound calls.

        The callback receives a framework-neutral ``TwiMLRequest`` (parsed
        from the Twilio webhook form) and returns a ``TwiMLOptions``. Fields
        the callback explicitly sets override ``default_twiml_options`` and
        TAC defaults; unset fields fall through.

        Example:
            ```python
            async def by_country(req: TwiMLRequest) -> TwiMLOptions:
                if req.caller_country == "MX":
                    return TwiMLOptions(language="es-MX", welcome_greeting="¡Hola!")
                return TwiMLOptions()


            voice_channel.on_inbound_call_twiml(by_country)
            ```

        Outbound calls don't use this — pass per-call TwiML via
        ``InitiateVoiceConversationOptions.twiml_options`` directly.
        """
        self._on_inbound_call_twiml = callback

    def on_call_status(self, callback: CallStatusHandler) -> None:
        """Register a handler for Twilio ``status_callback`` webhooks.

        This is the Calls-API status callback (call disposition), not the
        ConversationRelay session callback — see
        :meth:`handle_conversation_relay_callback`.

        Registering does two things: it stores the handler, and it makes later
        outbound calls pass ``status_callback`` to ``calls.create``. With no
        handler registered TAC omits that parameter, so Twilio has nowhere to
        post and the event never arrives.

        Twilio reports only the terminal event by default, which covers every
        disposition; set ``CallOptions.status_callback_event`` for
        ringing/answered.

        Example:
            ```python
            async def on_call_status(event: CallStatusEvent) -> None:
                if event.is_unreached:
                    ...  # queue a retry


            voice_channel.on_call_status(on_call_status)
            ```
        """
        self._on_call_status = callback

    def on_amd(self, callback: AmdHandler) -> None:
        """Register a handler for Twilio ``async_amd_status_callback`` webhooks.

        Registering makes later outbound calls pass
        ``async_amd_status_callback`` to ``calls.create``; without a handler TAC
        omits it and Twilio has nowhere to post the result. It does not enable
        detection — that's per-call, via ``CallOptions.machine_detection`` and
        ``async_amd``, both of which are required for this to fire (at most once
        per call).

        Example:
            ```python
            async def on_amd(event: AmdEvent) -> None:
                if event.is_machine:
                    await voice_channel.end_call(event.call_sid)  # voicemail → hang up


            voice_channel.on_amd(on_amd)
            ```
        """
        self._on_amd = callback

    def on_recording(self, callback: RecordingHandler) -> None:
        """Register a handler for Twilio ``recording_status_callback`` webhooks.

        Registering makes later outbound calls pass
        ``recording_status_callback`` to ``calls.create``; without a handler TAC
        omits it and Twilio has nowhere to post. It does not start recording —
        that's ``CallOptions.record``, which is required for this to fire.

        Example:
            ```python
            async def on_recording(event: RecordingEvent) -> None:
                if event.recording_status == "completed":
                    ...  # store event.recording_url


            voice_channel.on_recording(on_recording)
            ```
        """
        self._on_recording = callback

    def _resolve_websocket_url(self, action: str) -> str:
        """Resolve the public WebSocket URL from
        ``TACConfig.voice_public_domain`` + ``TACConfig.voice_websocket_path``.
        Raises if ``voice_public_domain`` isn't set.
        """
        if self.tac.config.voice_public_domain:
            return (
                f"wss://{self.tac.config.voice_public_domain}{self.tac.config.voice_websocket_path}"
            )
        raise ValueError(
            f"{action} needs a WebSocket URL. Set TWILIO_VOICE_PUBLIC_DOMAIN "
            "(or TACConfig.voice_public_domain)."
        )

    def _resolve_default_action_url(self) -> str | None:
        """Resolve the default ``<Connect action=...>`` cleanup URL.

        Returns None if ``voice_public_domain`` isn't set; that's fine because
        action_url has higher-priority layers (customizer, twiml_options,
        Studio handoff) above this fallback.
        """
        if self.tac.config.voice_public_domain:
            return (
                f"https://{self.tac.config.voice_public_domain}{self.tac.config.voice_action_path}"
            )
        return None

    @staticmethod
    def _caller_address(setup_msg: SetupMessage) -> str | None:
        """Return the phone number of the remote caller/callee from the setup message."""
        if setup_msg.direction and setup_msg.direction.upper() == "OUTBOUND":
            return setup_msg.to_number
        return setup_msg.from_number

    def _get_twilio_client(self) -> Client:
        if self._twilio_client is None:
            from twilio.rest import Client

            self._twilio_client = Client(
                self.tac.config.api_key,
                self.tac.config.api_secret,
                self.tac.config.account_sid,
            )
        return self._twilio_client

    async def handle_incoming_call(
        self,
        twiml_request: TwiMLRequest | None = None,
        *,
        host_twiml_options: TwiMLOptions | None = None,
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
             ``VoiceChannel.on_inbound_call_twiml(...)`` if configured
             and ``twiml_request`` is given. (Application-owned.)
          2. ``VoiceChannelConfig.default_twiml_options`` — per-channel defaults.
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

        Args:
            twiml_request: The incoming Twilio voice webhook, parsed into a
                framework-neutral form (From, To, CallSid, CallerCountry, …).
                Supplied by Twilio; forwarded to the ``on_inbound_call_twiml``
                customizer so the application can produce per-call overrides.
            host_twiml_options: Per-call TwiML overrides supplied by the *host*
                (the code owning the route — e.g. a custom server), for
                transport facts the SDK can't derive, such as a per-call
                ``websocket_url`` with an affinity token. Layered below
                ``default_twiml_options`` and the application customizer, so a
                developer's explicit settings still win.

        Returns:
            TwiML XML string for call connection.
        """
        customized: TwiMLOptions | None = None
        if self._on_inbound_call_twiml is not None and twiml_request is not None:
            customized = await self._on_inbound_call_twiml(twiml_request)

        merged = self._build_twiml_options(host_twiml_options, customized)
        # merged.websocket_url is either a validated non-empty URL (set by some
        # layer) or None; fall back to the TACConfig-derived URL only when None.
        websocket_url = (
            merged.websocket_url
            if merged.websocket_url is not None
            else self._resolve_websocket_url("handle_incoming_call")
        )
        return twiml.generate_twiml(websocket_url, merged)

    def _build_twiml_options(
        self,
        host: TwiMLOptions | None,
        per_call: TwiMLOptions | None,
    ) -> TwiMLOptions:
        """Layer TwiML options, lowest precedence first: TAC defaults →
        ``host`` (calling host's per-call values) → ``default_twiml_options`` →
        ``per_call`` (application customizer output for inbound, or
        ``InitiateVoiceConversationOptions.twiml_options`` for outbound).
        """
        merged = TwiMLOptions(
            welcome_greeting=DEFAULT_WELCOME_GREETING,
            conversation_configuration=self.tac.config.conversation_configuration_id,
            action_url=self._resolve_action_url(host, per_call),
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
        host: TwiMLOptions | None,
        customized: TwiMLOptions | None,
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
            self.config.default_twiml_options is not None
            and "action_url" in self.config.default_twiml_options.model_fields_set
        ):
            return self.config.default_twiml_options.action_url
        if host is not None and "action_url" in host.model_fields_set:
            return host.action_url
        if self.tac.config.studio_handoff_flow_sid:
            return studio_voice_handoff_url(
                self.tac.config.account_sid,
                self.tac.config.studio_handoff_flow_sid,
            )
        return self._resolve_default_action_url()

    async def handle_conversation_relay_callback(
        self,
        payload_dict: dict[str, str],
    ) -> None:
        """Handle ConversationRelay callback webhook from Twilio.

        In relay-only mode, this is a secondary mechanism for cleaning up
        conversation state when a call ends (the primary mechanism is websocket
        disconnect). In orchestrated mode, conversation lifecycle is managed by
        CO webhooks, so this is a no-op.

        Args:
            payload_dict: Raw form data dict from the webhook request.
        """
        try:
            payload = ConversationRelayCallbackPayload(**payload_dict)
        except ValidationError:
            self.logger.warning(
                "Invalid ConversationRelay callback payload, ignoring",
                payload_keys=list(payload_dict.keys()),
            )
            return

        if payload.account_sid != self.tac.config.account_sid:
            self.logger.warning(
                "ConversationRelay callback account_sid mismatch, ignoring",
                expected=self.tac.config.account_sid,
                received=payload.account_sid,
            )
            return

        self.logger.debug(
            "ConversationRelay callback received",
            call_sid=payload.call_sid,
            call_status=payload.call_status,
        )

        if payload.call_status == "completed" and not self.tac.is_orchestrator_enabled():
            if payload.call_sid in self._conversations:
                await self._end_conversation(payload.call_sid)

    def _call_event_account_ok(self, payload_dict: dict[str, str]) -> bool:
        """Whether a call-webhook payload belongs to the configured account.

        Twilio signature validation already gates the route; this is defense in
        depth. A payload with no ``AccountSid`` is allowed through.

        Subaccounts: events carry the SID the call was placed on, so configure
        TAC with that account or its events get dropped here.
        """
        account_sid = payload_dict.get("AccountSid")
        if account_sid and account_sid != self.tac.config.account_sid:
            self.logger.warning(
                "Call event account_sid mismatch, ignoring",
                expected=self.tac.config.account_sid,
                received=account_sid,
            )
            return False
        return True

    async def handle_call_status_event(self, payload_dict: dict[str, str]) -> None:
        """Handle a Twilio ``status_callback`` webhook.

        The developer routes the request here (``TACFastAPIServer`` does this
        automatically for its ``/status`` call-event route). Parsed into a
        :class:`CallStatusEvent` and dispatched to the :meth:`on_call_status`
        handler. No-op if no handler is registered.

        Args:
            payload_dict: Raw form data dict from the webhook request.
        """
        if self._on_call_status is None or not self._call_event_account_ok(payload_dict):
            return
        event = CallStatusEvent.from_form(payload_dict)
        self.logger.debug(
            "Call status event received",
            call_sid=event.call_sid,
            call_status=event.call_status,
        )
        await self._on_call_status(event)

    async def handle_amd_event(self, payload_dict: dict[str, str]) -> None:
        """Handle a Twilio ``async_amd_status_callback`` webhook.

        The developer routes the request here (``TACFastAPIServer`` does this
        automatically for its ``/amd`` call-event route). Parsed into an
        :class:`AmdEvent` and dispatched to the :meth:`on_amd` handler. No-op if
        no handler is registered.

        Args:
            payload_dict: Raw form data dict from the webhook request.
        """
        if self._on_amd is None or not self._call_event_account_ok(payload_dict):
            return
        event = AmdEvent.from_form(payload_dict)
        self.logger.debug(
            "Call AMD event received",
            call_sid=event.call_sid,
            answered_by=event.answered_by,
        )
        await self._on_amd(event)

    async def handle_recording_event(self, payload_dict: dict[str, str]) -> None:
        """Handle a Twilio ``recording_status_callback`` webhook.

        The developer routes the request here (``TACFastAPIServer`` does this
        automatically for its ``/recording`` call-event route). Parsed into a
        :class:`RecordingEvent` and dispatched to the :meth:`on_recording`
        handler. No-op if no handler is registered.

        Args:
            payload_dict: Raw form data dict from the webhook request.
        """
        if self._on_recording is None or not self._call_event_account_ok(payload_dict):
            return
        event = RecordingEvent.from_form(payload_dict)
        self.logger.debug(
            "Call recording event received",
            call_sid=event.call_sid,
            recording_status=event.recording_status,
        )
        await self._on_recording(event)

    async def end_call(self, call_sid: str) -> bool:
        """Hang up a call and clean up its ConversationRelay session.

        Works on ``call_sid`` alone, whether or not a session exists yet.
        No-ops the session cleanup if none is tracked.

        Does not raise — hanging up an already-ended call is routine (the callee
        hangs up while AMD is still resolving), and handlers shouldn't have to
        guard against it.

        Args:
            call_sid: Twilio Call SID (from a call event, the outbound result, or
                ``ConversationSession.call_sid``).

        Returns:
            True if Twilio accepted the hangup, False if it failed (logged).
            Session cleanup runs either way.
        """
        client = self._get_twilio_client()
        hung_up = True
        try:
            await asyncio.to_thread(client.calls(call_sid).update, status="completed")
        except Exception as e:
            hung_up = False
            self.logger.error(
                "Failed to hang up call",
                call_sid=call_sid,
                error=str(e),
                exc_info=True,
            )

        session = self.get_conversation_session_by_call_sid(call_sid)
        if session is not None:
            await self._end_conversation(session.conversation_id)
        return hung_up

    def get_conversation_session_by_call_sid(self, call_sid: str) -> ConversationSession | None:
        """Look up the active voice session for a Twilio Call SID.

        Out-of-band code holding a CallSid — a dashboard route, an operator
        action, a call-event handler — can't reach the session-facing methods,
        which are keyed by conversation id: the Orchestrator conversation id in
        orchestrator mode, the CallSid only in ConversationRelay-only mode.

        Relay-only mode creates the session on the caller's first prompt.
        Orchestrator mode creates it earlier — as soon as the background CO
        lookup started at WebSocket setup finishes — so it may already exist
        before the caller has said anything, including before ``on_amd``
        fires. Either way, treat this as racy and use :meth:`end_call` to hang
        up, which works whether or not a session exists yet.

        At the other end, orchestrator mode keeps the session until Conversation
        Orchestrator's CLOSED webhook, so it outlives the call and
        ``on_call_status`` / ``on_recording`` do resolve. Relay-only mode tears
        down on the ConversationRelay callback instead, which races them.

        Named for ``ConversationSession``; ``session_manager`` deals in
        ``SessionState``, a different type.

        Example:
            ```python
            async def nudge(call_sid: str) -> None:
                session = voice_channel.get_conversation_session_by_call_sid(call_sid)
                if session is not None:
                    await voice_channel.send_response(session.conversation_id, "Still there?")
            ```

        Args:
            call_sid: Twilio Call SID, e.g. from
                ``InitiateVoiceConversationResult.call_sid`` or a call event.

        Returns:
            The session, or ``None`` — not created yet (relay-only mode, or
            orchestrator mode where the background CO lookup hasn't finished),
            the call ended, or it landed on another instance (see the
            horizontal-scaling note in CLAUDE.md).
        """
        for session in self._conversations.values():
            if session.call_sid == call_sid:
                return session
        return None

    async def _initialize_conversation(
        self,
        call_sid: str,
        setup_msg: SetupMessage,
        websocket: WebSocketProtocol,
    ) -> tuple[str, SessionState | None]:
        """Poll CO for the conversation created by ConversationRelay, resolve
        the customer participant, and initialize the local session."""
        conversation_orchestrator_client = self.tac.conversation_orchestrator_client
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
                self.logger.debug(
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
        agent_participant = self._find_agent_participant(
            participants, "VOICE", self.tac.config.phone_number
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
        session = self._start_conversation(conv_id, profile_id)
        # In orchestrator mode conv_id is the Orchestrator conversation id, so
        # record the CallSid so out-of-band call webhooks can reach this session
        # (resolved via get_conversation_session_by_call_sid).
        session.call_sid = call_sid

        session_state = None
        if self.session_manager is not None:
            session_state = self.session_manager.get_or_create_session(conv_id)

        if profile_lookup_address:
            session.author_info = AuthorInfo(address=profile_lookup_address)

        if agent_participant:
            # Fall back to the configured phone number we matched on — the
            # participant owns it by definition, so it's a meaningful address
            # even in the unlikely case it carries no explicit VOICE address.
            session.ai_agent_info = AuthorInfo(
                address=agent_address or self.tac.config.phone_number,
                participant_id=agent_participant.id,
            )

        return conv_id, session_state

    async def handle_websocket(self, websocket: WebSocketProtocol) -> None:
        """
        Handle voice streaming WebSocket connection lifecycle.

        This method manages the entire websocket connection:
        - Accepts the connection
        - Processes incoming messages
        - Tracks and cancels in-flight tasks (if session_manager provided)
        - Cleans up on disconnect

        Args:
            websocket: Any WebSocket implementation satisfying WebSocketProtocol
        """
        await websocket.accept()
        self.logger.debug("WebSocket connection established")

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
                if call_sid and self.tac.is_orchestrator_enabled():
                    init_task = asyncio.create_task(
                        self._initialize_conversation(call_sid, setup_msg, websocket)
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
                                session = self._start_conversation(conv_id, profile_id=None)
                                # Relay-only: conv_id == call_sid.
                                session.call_sid = call_sid

                                caller = self._caller_address(setup_msg)
                                if caller:
                                    self._conversations[conv_id].author_info = AuthorInfo(
                                        address=caller,
                                    )

                                if self.session_manager is not None:
                                    session_state = self.session_manager.get_or_create_session(
                                        conv_id
                                    )

                        if conv_id:
                            await self._handle_prompt_async(conv_id, data, session_state)
                        else:
                            self.logger.warning("Received prompt before conversation initialized")
                    elif msg_type == "interrupt":
                        if conv_id:
                            await self._handle_interrupt_async(conv_id, data, session_state)
                        else:
                            self.logger.warning(
                                "Received interrupt before conversation initialized"
                            )
                    else:
                        self.logger.debug(f"Skip message type received: {msg_type}")
            else:
                self.logger.warning("First message was not 'setup'. Closing connection.")
                await websocket.close()
                return
        except WebSocketDisconnectError:
            self.logger.info("WebSocket connection closed", conversation_id=conv_id)
        except Exception as e:
            self.logger.error(f"WebSocket error: {str(e)}")
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
                    self.logger.error(
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
                self.logger.debug("Cleanup - removing WebSocket", conversation_id=conv_id)
                await self._cleanup_connection(conv_id)
            if cancelled_error is not None and not we_cancelled_it:
                # A real external cancellation, not the one we caused above —
                # propagate it now that cleanup ran.
                raise cancelled_error

    def _merge_call_options(self, per_call: CallOptions | None) -> CallOptions | None:
        """Overlay ``per_call`` onto ``VoiceChannelConfig.default_call_options``.

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

    def _build_call_kwargs(self, call_options: CallOptions | None) -> dict[str, Any]:
        """Build the extra kwargs for ``client.calls.create``.

        Layers, highest precedence first: this call's ``call_options``,
        ``VoiceChannelConfig.default_call_options``, then callback URLs derived
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
            ("status", "status_callback", self._on_call_status),
            ("amd", "async_amd_status_callback", self._on_amd),
            ("recording", "recording_status_callback", self._on_recording),
        ]
        for kind, param, handler in wiring:
            if handler is None:
                continue
            url = self.tac.config.call_event_url(kind)
            if url is not None:
                call_kwargs.setdefault(param, url)

        return call_kwargs

    async def initiate_outbound_conversation(
        self,
        options: InitiateVoiceConversationOptions,
    ) -> InitiateVoiceConversationResult:
        """Initiate an outbound voice conversation.

        Places an outbound call with inline TwiML that connects to ConversationRelay.
        The conversationConfiguration attribute tells CO to create and manage the
        conversation during passive hydration. The session is initialized lazily
        on the first prompt when the conversation is discovered by callSid.

        TwiML fields are merged per-field, highest precedence first:
          1. ``options.twiml_options`` — per-call overrides
          2. ``VoiceChannelConfig.default_twiml_options`` — channel-wide defaults
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
        from_number = self.tac.config.phone_number

        self.logger.info(
            "Initiating outbound voice conversation",
            to=mask_phone(options.to),
            from_number=mask_phone(from_number),
        )

        # Outbound has no inbound customizer and no server layer; the per-call
        # override is options.twiml_options.
        merged = self._build_twiml_options(None, options.twiml_options)

        # ``options.websocket_url`` is the dedicated per-call outbound override
        # and wins over any websocket_url that came through the layered
        # ``twiml_options`` merge; both fall back to the TACConfig-derived URL.
        if options.websocket_url is not None:
            websocket_url = options.websocket_url
        elif merged.websocket_url is not None:
            websocket_url = merged.websocket_url
        else:
            websocket_url = self._resolve_websocket_url("initiate_outbound_conversation")

        call_kwargs = self._build_call_kwargs(options.call_options)

        try:
            twiml_xml = twiml.generate_twiml(websocket_url, merged)

            # The inline TwiML handed to Twilio, useful for debugging the
            # <Connect action> handoff target. custom_parameters values are
            # masked — they're arbitrary developer data (profile IDs, caller
            # names), unlike the WS/action URLs and conversation config.
            self.logger.debug(
                "Outbound call TwiML",
                twiml=redact_twiml_parameters(twiml_xml),
                to=mask_phone(options.to),
            )

            client = self._get_twilio_client()
            call = await asyncio.to_thread(
                client.calls.create,
                to=options.to,
                from_=from_number,
                twiml=twiml_xml,
                **call_kwargs,
            )

            self.logger.info(
                "Outbound voice call placed",
                call_sid=call.sid,
                to=mask_phone(options.to),
            )

            return InitiateVoiceConversationResult(call_sid=call.sid)

        except Exception as e:
            self.logger.error(
                "Failed to initiate outbound call",
                to=mask_phone(options.to),
                error=str(e),
                exc_info=True,
            )
            raise

    async def _handle_prompt_async(
        self,
        conv_id: str,
        data: dict[str, Any],
        session_state: SessionState | None,
    ) -> None:
        """
        Handle prompt message asynchronously with task tracking.

        Args:
            conv_id: Conversation ID
            data: Raw message data
            session_state: Session state object (if session_manager provided)
        """
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
                        self._handle_prompt(conv_id, prompt_msg)
                    )
                    # Yield to event loop to let task start
                    await asyncio.sleep(0)
                else:
                    await self._handle_prompt(conv_id, prompt_msg)
        except Exception as e:
            self.logger.error(f"Failed to handle prompt: {str(e)}")

    async def _handle_interrupt_async(
        self,
        conv_id: str,
        data: dict[str, Any],
        session_state: SessionState | None,
    ) -> None:
        """
        Handle interrupt message asynchronously with task cancellation.

        Args:
            conv_id: Conversation ID
            data: Raw message data
            session_state: Session state object (if session_manager provided)
        """
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
                        self.logger.debug(
                            f"WebSocket closed before sending interrupt acknowledgment "
                            f"for {conv_id}."
                        )

            # Call the interrupt handler
            self._handle_interrupt(conv_id, interrupt_msg)
        except Exception as e:
            self.logger.error(f"Failed to handle interrupt: {str(e)}")

    async def process_webhook(
        self, webhook_data: dict[str, Any], idempotency_token: str | None = None
    ) -> None:
        """Process conversation webhooks for cleanup and cache invalidation.

        Voice channel processes CONVERSATION_UPDATED events:
        - CLOSED status: Clean up local session state
        - INACTIVE status: Invalidate cached memory (memory will be updated by
          Conversation Orchestrator)

        Note: Conversation tracking uses instance-local memory. In multi-instance
        deployments, webhooks may route to a different instance, preventing cleanup.
        See CLAUDE.md for horizontal scaling considerations.

        Args:
            webhook_data: Raw webhook event data from Twilio
            idempotency_token: Optional Twilio idempotency token from request headers
        """
        if not self._is_event_for_this_channel(webhook_data):
            return

        if idempotency_token:
            if self._is_duplicate_webhook(idempotency_token):
                return

        event_type = webhook_data.get("eventType")
        event_data = webhook_data.get("data")

        if not isinstance(event_data, dict):
            self.logger.warning(
                "Webhook missing or malformed data field, skipping",
                event_type=event_type,
            )
            return

        if event_type == "CONVERSATION_UPDATED":
            conv_id = event_data.get("id")
            status = event_data.get("status")

            if not conv_id:
                return

            session = self._conversations.get(conv_id)
            if not session or session.channel != self.get_channel_name():
                return

            if status == "CLOSED":
                await self._end_conversation(conv_id)
            elif status == "INACTIVE" and self.memory_mode == "once":
                # Invalidate cached memory when conversation becomes inactive
                # Memory is updated by Conversation Orchestrator on INACTIVE transition
                async with session.cache_lock:
                    if session.cached_memory is not None:
                        session.cached_memory = None
                        self.logger.debug(
                            "Invalidated cached memory on INACTIVE status",
                            conversation_id=conv_id,
                        )

    async def send_response(
        self,
        conversation_id: str,
        response: str | AsyncGenerator[str | dict[str, Any], None],
        role: str | None = None,
    ) -> None:
        """
        Send voice response through the websocket connection for this conversation.

        Supports both simple string responses and streaming async generators.

        Args:
            conversation_id: Conversation ID
            response: Response text (string) or async generator for streaming
            role: Optional message role (not used in this implementation, but kept
                  for API consistency with BaseChannel interface)
        """
        # Validate response type before processing
        if not isinstance(response, (str, AsyncGenerator)):
            raise TypeError("Voice channel requires string or async generator for response")

        # Get WebSocket from manager
        websocket = self._websocket_manager.get_websocket(conversation_id)
        if not websocket:
            self.logger.error("No websocket connection", conversation_id=conversation_id)
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
                            self.logger.info(
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
                            self.logger.info(
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
            if conversation_id in self._conversations:
                session = self._conversations[conversation_id]
                if session.pending_handoff_data is not None:
                    try:
                        await websocket.send_text(
                            session.pending_handoff_data.model_dump_json(by_alias=True)
                        )
                        session.pending_handoff_data = None
                    except (WebSocketDisconnectError, RuntimeError):
                        self.logger.warning(
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
            self.logger.info(
                "WebSocket closed before sending response", conversation_id=conversation_id
            )
        except Exception as e:
            self.logger.error(
                f"Error sending response: {e}", conversation_id=conversation_id, exc_info=True
            )

    def get_channel_name(self) -> str:
        return "VOICE"

    def get_websocket(self, conversation_id: str) -> WebSocketProtocol | None:
        """
        Get the WebSocket connection for a specific conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            WebSocket connection if exists, None otherwise
        """
        return self._websocket_manager.get_websocket(conversation_id)

    async def _handle_prompt(self, conv_id: str, message: PromptMessage) -> None:
        """
        Handle incoming voice prompt (user speech).

        Args:
            conv_id: Conversation ID
            message: Parsed PromptMessage containing user's transcribed speech
        """
        if conv_id not in self._conversations:
            self.logger.error(
                f"Received prompt for unknown conversation {conv_id}. "
                "Conversation should be initialized on first prompt.",
                conversation_id=conv_id,
            )
            return

        message_body = message.voice_prompt or ""
        session = self._conversations[conv_id]

        # Retrieve memory if memory_mode is enabled and Twilio Memory is configured
        memory_response = await self._retrieve_memory_if_enabled(session, message_body, conv_id)

        # Trigger message ready callback
        try:
            response = await self.tac.trigger_message_ready(message_body, session, memory_response)
            # Auto-send if callback returned a string (None = manual send_response flow)
            if response is not None:
                await self.send_response(conv_id, response, role="assistant")
        except Exception as e:
            self.logger.error(
                "Error in message ready callback",
                conversation_id=conv_id,
                error=str(e),
                exc_info=True,
            )

    def _handle_interrupt(self, conv_id: str, message: InterruptMessage) -> None:
        """
        Handle interrupt message when user interrupts the agent.

        Note: Task cancellation is handled by the async wrapper (_handle_interrupt_async)
        when called from the WebSocket message handler. This method only triggers the
        TAC interrupt callback.

        Args:
            conv_id: Conversation ID
            message: Parsed InterruptMessage with interruption details
        """
        # Trigger interrupt callback if conversation exists
        if conv_id in self._conversations:
            session = self._conversations[conv_id]
            self.tac.trigger_interrupt(session, message)
        else:
            self.logger.warning(
                f"Received interrupt for unknown conversation {conv_id}, skipping callback"
            )

    async def _cleanup_connection(self, conv_id: str) -> None:
        """
        Clean up WebSocket and session resources when connection closes.

        In orchestrated mode, the conversation remains tracked in
        self._conversations until the CONVERSATION_UPDATED/CLOSED webhook
        arrives from Conversation Orchestrator. In relay-only mode there is no such webhook,
        so we also end the conversation here.

        Args:
            conv_id: Conversation ID
        """
        # Remove WebSocket from manager
        if self._websocket_manager.has_websocket(conv_id):
            self._websocket_manager.remove_websocket(conv_id)

        # Cancel running stream task and cleanup session if session manager is enabled
        if self.session_manager is not None and self.session_manager.has_session(conv_id):
            session_state = self.session_manager.get_or_create_session(conv_id)
            # Cancel any running task (user hung up, no point continuing)
            await session_state.cancel_stream_task()
            self.session_manager.remove_session(conv_id)

        if not self.tac.is_orchestrator_enabled() and conv_id in self._conversations:
            await self._end_conversation(conv_id)

        self.logger.debug(
            "Cleaned up WebSocket and session resources",
            conversation_id=conv_id,
        )
