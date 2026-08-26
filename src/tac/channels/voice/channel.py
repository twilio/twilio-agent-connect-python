from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
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
from tac.models.session import ConversationSession
from tac.models.voice import (
    AmdEvent,
    CallStatusEvent,
    RecordingEvent,
    TwiMLOptions,
    TwiMLRequest,
)

from .config import (
    AmdHandler,
    CallStatusHandler,
    InboundCallTwiMLHandler,
    RecordingHandler,
    VoiceChannelConfig,
)
from .provider import VoiceProviderConfig


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
            config: Voice channel configuration (a ``VoiceProviderConfig`` —
                ``VoiceChannelConfig`` today — or dict). If None, uses default
                configuration.

        Examples:
            >>> channel = VoiceChannel(tac, config={"memory_mode": "always"})
            >>> channel = VoiceChannel(tac, config=VoiceChannelConfig(session_manager=sm))
            >>> channel = VoiceChannel(tac)  # Use defaults
        """
        # Convert dict to config model or use defaults.
        if isinstance(config, dict):
            config = VoiceChannelConfig(**config)
        elif config is None:
            config = VoiceChannelConfig()

        super().__init__(tac, memory_mode=config.memory_mode)
        self._provider = config.create_provider(self, tac.config)
        self._on_inbound_call_twiml: InboundCallTwiMLHandler | None = None
        self._on_call_status: CallStatusHandler | None = None
        self._on_amd: AmdHandler | None = None
        self._on_recording: RecordingHandler | None = None
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
        """Generate TwiML response for incoming voice calls. Delegates to the
        active provider — see ``ConversationRelayProvider.handle_incoming_call``
        for the full merge/precedence rules (only meaningful for that provider;
        a non-TwiML provider ignores ``host_twiml_options``).
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

    async def handle_websocket(self, websocket: WebSocketProtocol) -> None:
        """
        Handle voice streaming WebSocket connection lifecycle. Delegates to
        the active provider.

        Args:
            websocket: Any WebSocket implementation satisfying WebSocketProtocol
        """
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
