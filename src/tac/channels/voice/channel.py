from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from twilio.rest import Client

from tac.channels.base import BaseChannel
from tac.channels.websocket_protocol import WebSocketProtocol
from tac.core.tac import TAC
from tac.models.outbound import (
    InitiateVoiceConversationOptions,
    InitiateVoiceConversationResult,
)
from tac.models.session import AuthorInfo, ConversationSession
from tac.models.voice import (
    AmdEvent,
    CallStatusEvent,
    RecordingEvent,
    TwiMLRequest,
    VoiceTwiMLOptions,
)

from .conversation_relay import ConversationRelayProviderConfig
from .provider import VoiceProviderConfig

InboundCallTwiMLHandler = Callable[[TwiMLRequest], Awaitable[VoiceTwiMLOptions]]
CallStatusHandler = Callable[[CallStatusEvent], Awaitable[None]]
AmdHandler = Callable[[AmdEvent], Awaitable[None]]
RecordingHandler = Callable[[RecordingEvent], Awaitable[None]]
CallEndedHandler = Callable[[ConversationSession], Awaitable[None]]

#: Channel name Conversation Orchestrator uses for voice participants. Not
#: ``get_channel_name()``, which is provider-specific.
CO_VOICE_CHANNEL = "VOICE"

#: Default seconds :meth:`VoiceChannel.aclose` waits for calls to end.
DEFAULT_DRAIN_GRACE_PERIOD = 30.0


class VoiceChannel(BaseChannel):
    """
    Voice Channel for handling voice-based conversations via WebSocket.

    Owns the Twilio Calls API lifecycle and conversation bookkeeping
    (inherited from BaseChannel). The real-time media transport itself —
    TwiML generation, WebSocket protocol handling, outbound call
    initiation — is delegated to a pluggable ``VoiceProvider``
    (``ConversationRelayProvider`` is the only implementation today).

    This channel is framework-agnostic and accepts any WebSocket implementation
    satisfying WebSocketProtocol. For a batteries-included FastAPI server, use
    tac.server.TACFastAPIServer.
    """

    def __init__(
        self,
        tac: TAC,
        config: VoiceProviderConfig | dict[str, Any] | None = None,
    ):
        """
        Initialize Voice channel for websocket protocol handling.

        Args:
            tac: TAC instance for memory/context operations
            config: Voice channel configuration — a ``VoiceProviderConfig``
                instance for any provider, or a dict. The dict form is
                shorthand for ``ConversationRelayProviderConfig`` (the default
                ConversationRelay provider) specifically, not a generic
                constructor — it's hydrated as
                ``ConversationRelayProviderConfig(**config)`` and fails if it
                has fields that config doesn't have. To configure a different
                provider, construct that provider's config and pass it
                directly instead of a dict. If None, uses
                ``ConversationRelayProviderConfig()``.

        Examples:
            >>> channel = VoiceChannel(tac, config={"memory_mode": "always"})
            >>> channel = VoiceChannel(
            ...     tac, config=ConversationRelayProviderConfig(session_manager=sm)
            ... )
            >>> channel = VoiceChannel(tac)  # Use defaults
        """
        # dict is shorthand for ConversationRelayProviderConfig specifically —
        # see the config Args note above. Not a generic dict-to-provider-config
        # constructor.
        if isinstance(config, dict):
            config = ConversationRelayProviderConfig(**config)
        elif config is None:
            config = ConversationRelayProviderConfig()

        super().__init__(tac, memory_mode=config.memory_mode)
        # Live sessions by conversation id. Instance-local by design: a call
        # is pinned to the process holding its WebSocket.
        self._conversations: dict[str, ConversationSession] = {}
        self._provider = config.create_provider(self, tac.config)
        self._on_inbound_call_twiml: InboundCallTwiMLHandler | None = None
        self._on_call_status: CallStatusHandler | None = None
        self._on_amd: AmdHandler | None = None
        self._on_recording: RecordingHandler | None = None
        self._on_call_ended: CallEndedHandler | None = None
        self._twilio_client: Client | None = None
        self._accepting_calls = True  # cleared by aclose()

    def on_inbound_call_twiml(self, callback: InboundCallTwiMLHandler) -> None:
        """Register a callback that produces per-call overrides for the
        active provider's inbound-call TwiML.

        The callback receives a framework-neutral ``TwiMLRequest`` (parsed
        from the Twilio webhook form) and returns a ``VoiceTwiMLOptions`` —
        the concrete subclass the active provider expects (e.g.
        ``VoiceTwiMLOptionsConversationRelay`` for the default
        ConversationRelay provider; see that provider's
        ``handle_incoming_call`` for its merge/precedence rules).

        Outbound calls don't use this — pass per-call TwiML via
        ``InitiateVoiceConversationOptions.twiml_options`` directly.
        """
        self._on_inbound_call_twiml = callback

    def on_call_status(self, callback: CallStatusHandler) -> None:
        """Register a handler for Twilio ``status_callback`` webhooks.

        This is the Calls-API status callback (call disposition), not the
        active provider's own out-of-band lifecycle webhook — see
        :meth:`handle_twilio_provider_callback`.

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

    def on_call_ended(self, callback: CallEndedHandler) -> None:
        """Register a handler for WebSocket teardown.

        Fires once per call, always, on the instance that held the call. It
        receives the live session just before it is discarded, so it is the
        only place late-call in-memory state — the transcript, `call_sid`,
        anything you put on `metadata` — is still reachable.

        Not the same as `on_conversation_ended` (fires when the *conversation*
        closes, on any instance, from a session that may be rebuilt) or
        `on_call_status` (a Twilio Calls-API webhook, no session). Handler
        exceptions are logged and swallowed.

        Example:
            ```python
            async def on_call_ended(session: ConversationSession) -> None:
                await archive(session.call_sid, session.metadata.get("transcript", []))


            voice_channel.on_call_ended(on_call_ended)
            ```
        """
        self._on_call_ended = callback

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
        host_twiml_options: VoiceTwiMLOptions | None = None,
    ) -> str:
        """Generate TwiML response for incoming voice calls. Delegates to the
        active provider — see ``ConversationRelayProvider.handle_incoming_call``
        for the full merge/precedence rules (only meaningful for that provider;
        a non-TwiML provider ignores ``host_twiml_options``).

        ``host_twiml_options`` is typed against the ``VoiceTwiMLOptions`` base —
        the active provider defines the concrete shape it expects (e.g.
        ``VoiceTwiMLOptionsConversationRelay``) and validates it at runtime.
        """
        return await self._provider.handle_incoming_call(
            twiml_request, host_twiml_options=host_twiml_options
        )

    async def handle_twilio_provider_callback(
        self,
        payload_dict: dict[str, str],
    ) -> None:
        """Handle the active provider's own out-of-band lifecycle webhook, if
        it has one — e.g. ConversationRelay's ``<Connect action=...>`` callback.

        In relay-only mode, this is a secondary mechanism for cleaning up
        conversation state when a call ends (the primary mechanism is websocket
        disconnect). In orchestrated mode, conversation lifecycle is managed by
        CO webhooks, so this is a no-op.

        Not every provider has an equivalent webhook — see
        ``VoiceProvider.handle_twilio_provider_callback``.

        Args:
            payload_dict: Raw form data dict from the webhook request.
        """
        await self._provider.handle_twilio_provider_callback(payload_dict)

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
        """Hang up a call and clean up its session.

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
            await self._release_session(session.conversation_id)
        return hung_up

    def _start_conversation(
        self,
        conv_id: str,
        profile_id: str | None = None,
    ) -> ConversationSession:
        """Track a new session for a call, or return the existing one.

        Profile data is fetched lazily during retrieve_memory() when needed.
        """
        if conv_id in self._conversations:
            self.logger.debug(
                "Conversation already exists, skipping initialization",
                conversation_id=conv_id,
                channel=self.get_channel_name(),
            )
            return self._conversations[conv_id]

        self._conversations[conv_id] = ConversationSession(
            conversation_id=conv_id,
            profile_id=profile_id,
            channel=self.get_channel_name(),
        )

        self.logger.info(
            f"CONVERSATION | Started {self.get_channel_name()} conversation",
            conversation_id=conv_id,
            profile_id=profile_id,
        )
        return self._conversations[conv_id]

    async def _release_session(self, conv_id: str) -> ConversationSession | None:
        """Free a finished call's session and fire the end-of-call hooks.

        Called from every teardown path and idempotent, so a second call is a
        no-op returning ``None``. Always fires ``on_call_ended``; also fires
        ``on_conversation_ended`` unless a Conversation Orchestrator CLOSED
        webhook will do that later (see
        ``VoiceProvider._conversation_closed_by_orchestrator``).
        """
        session = self._conversations.pop(conv_id, None)
        if session is None:
            return None

        if self._on_call_ended is not None:
            try:
                await self._on_call_ended(session)
            except Exception as e:
                self.logger.error(
                    "Error in call ended callback",
                    conversation_id=conv_id,
                    error=str(e),
                    exc_info=True,
                )

        if not self._provider._conversation_closed_by_orchestrator:
            await self._trigger_conversation_ended(session)

        self.logger.debug(
            "Released voice session",
            conversation_id=conv_id,
            channel=self.get_channel_name(),
        )
        return session

    async def _handle_conversation_closed(self, conv_id: str, call_sid: str | None = None) -> None:
        """Fire ``on_conversation_ended`` for a CLOSED webhook.

        The session is normally already released (the socket closed when the
        caller hung up), so the usual path is a rebuild from Conversation
        Orchestrator — which is also what lets the hook fire on whichever
        instance received the webhook.
        """
        session = self._conversations.pop(conv_id, None)
        if session is None:
            if not self.tac._has_conversation_ended_callback():
                return
            session = await self._rebuild_session(conv_id, call_sid)
            if session is None:
                return
        await self._trigger_conversation_ended(session)

    async def _rebuild_session(
        self, conv_id: str, call_sid: str | None = None
    ) -> ConversationSession | None:
        """Reconstruct a session from Conversation Orchestrator.

        Carries identity only — conversation id, call_sid, profile, both
        participants. Live in-memory state (transcript, metadata) is gone by
        now; use ``on_call_ended`` for that. Returns ``None`` if no
        participant is on the voice channel, which is how another channel's
        CLOSED is filtered out.
        """
        client = self.tac.conversation_orchestrator_client
        if client is None:
            return None

        try:
            participants = await client.list_participants(conv_id)
        except Exception as e:
            self.logger.error(
                "Failed to list participants while rebuilding a closed voice conversation",
                conversation_id=conv_id,
                error=str(e),
            )
            return None

        if not any(a.channel == CO_VOICE_CHANNEL for p in participants for a in p.addresses):
            return None

        customer = next((p for p in participants if p.type == "CUSTOMER"), None)
        agent = self._find_agent_participant(
            participants, CO_VOICE_CHANNEL, self.tac.config.phone_number
        )

        session = ConversationSession(
            conversation_id=conv_id,
            call_sid=call_sid,
            channel=self.get_channel_name(),
            profile_id=customer.profile_id if customer else None,
        )

        if customer is not None:
            customer_address = next(
                (a.address for a in customer.addresses if a.channel == CO_VOICE_CHANNEL), None
            )
            if customer_address:
                session.author_info = AuthorInfo(
                    address=customer_address, participant_id=customer.id
                )
        if agent is not None:
            session.ai_agent_info = AuthorInfo(
                address=next(
                    (a.address for a in agent.addresses if a.channel == CO_VOICE_CHANNEL),
                    self.tac.config.phone_number,
                ),
                participant_id=agent.id,
            )
        return session

    async def aclose(self, *, grace_period: float = DEFAULT_DRAIN_GRACE_PERIOD) -> None:
        """Drain live calls at shutdown: refuse new ones, wait up to
        ``grace_period`` seconds for the rest to end, then force-release them.

        Without this a scale-in drops live calls with no callback at all.
        Fail your readiness probe before calling it, so the load balancer
        stops routing here first. Idempotent.

        Args:
            grace_period: Seconds to wait for calls to end on their own.
        """
        self._accepting_calls = False

        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(grace_period, 0.0)
        while self._conversations and loop.time() < deadline:
            await asyncio.sleep(0.1)

        remaining = list(self._conversations)
        if remaining:
            self.logger.warning(
                "Draining voice sessions still active at shutdown",
                count=len(remaining),
            )
        for conv_id in remaining:
            await self._release_session(conv_id)

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

        At the other end the session is released as soon as the call's
        WebSocket closes, in every mode — so out-of-band events that arrive
        after the hangup (``on_call_status``, ``on_recording``) generally find
        nothing here. Read late-call state from the session ``on_call_ended``
        hands you instead.

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

    async def handle_websocket(self, websocket: WebSocketProtocol) -> None:
        """
        Handle voice streaming WebSocket connection lifecycle. Delegates to
        the active provider.

        Refuses the connection outright once :meth:`aclose` has been called —
        a draining instance must not adopt a call it is about to drop.

        Args:
            websocket: Any WebSocket implementation satisfying WebSocketProtocol
        """
        if not self._accepting_calls:
            self.logger.warning("Refusing voice WebSocket: channel is draining for shutdown")
            await websocket.close()
            return
        await self._provider.handle_websocket(websocket)

    async def initiate_outbound_conversation(
        self,
        options: InitiateVoiceConversationOptions,
    ) -> InitiateVoiceConversationResult:
        """Initiate an outbound voice conversation.

        Only ``ConversationRelayProvider`` supports outbound calls today —
        raises ``NotImplementedError`` for any other provider.
        """
        return await self._provider.initiate_outbound_conversation(options)

    async def process_webhook(
        self, webhook_data: dict[str, Any], idempotency_token: str | None = None
    ) -> None:
        """Process conversation webhooks for cleanup and cache invalidation.

        Voice channel processes CONVERSATION_UPDATED events:

        - **CLOSED**: fire ``on_conversation_ended``. The call's session is
          normally already released (the WebSocket closed when the caller hung
          up), so this rebuilds it from Conversation Orchestrator — which is
          also what makes the hook fire on whichever instance received the
          webhook rather than only on the one that held the call.
        - **INACTIVE**: invalidate cached memory, if this instance holds the
          session. A call is pinned to one instance for its lifetime, so an
          INACTIVE landing elsewhere has no cache to clear and is ignored.

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

        if event_type != "CONVERSATION_UPDATED":
            return

        conv_id = event_data.get("id")
        status = event_data.get("status")
        if not conv_id:
            return

        if event_data.get("configurationId") != self.tac.config.conversation_configuration_id:
            return

        if status == "CLOSED":
            if not self._provider._conversation_closed_by_orchestrator:
                # This provider has no Conversation Orchestrator conversation
                # behind its calls; on_conversation_ended already fired at
                # teardown and firing again here would double it.
                return
            await self._handle_conversation_closed(conv_id, event_data.get("channelId"))
        elif status == "INACTIVE" and self.memory_mode == "once":
            session = self._conversations.get(conv_id)
            if session is None:
                return
            # Memory is updated by Conversation Orchestrator on the INACTIVE
            # transition, so the cached copy is stale from here on.
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
        Send a response back through this channel's active provider.

        Args:
            conversation_id: Conversation ID
            response: Response text (string) or async generator for streaming
            role: Optional message role (not used by ConversationRelayProvider, but
                  kept for API consistency with BaseChannel interface)
        """
        await self._provider.send_response(conversation_id, response, role)

    def get_channel_name(self) -> str:
        return self._provider.channel_name

    def get_websocket(self, conversation_id: str) -> WebSocketProtocol | None:
        """
        Get the WebSocket connection for a specific conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            WebSocket connection if exists, None otherwise
        """
        return self._provider.get_websocket(conversation_id)
