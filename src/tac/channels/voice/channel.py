from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from twilio.rest import Client

from tac.channels.base import BaseChannel
from tac.channels.websocket_protocol import WebSocketProtocol
from tac.core.analytics import analytics_enabled, track_event
from tac.core.tac import TAC
from tac.models.conversation import ConversationResponse
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
from tac.utils.timestamps import elapsed_ms, parse_iso8601

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

#: After the grace period, how long aclose() waits for cancelled calls to run
#: their teardown before releasing whatever sessions remain.
_FORCED_STOP_TIMEOUT_SECONDS = 5.0


def _created_at_key(conversation: ConversationResponse) -> datetime:
    """Parse ``ConversationResponse.created_at`` for chronological sorting.

    Naive timestamps are treated as UTC. A missing or unparseable value
    sorts first (``datetime.min``), so a conversation with good data always
    wins over one without.
    """
    return parse_iso8601(conversation.created_at) or datetime.min.replace(tzinfo=timezone.utc)


class VoiceChannel(BaseChannel):
    """
    Voice Channel for handling voice-based conversations via WebSocket.

    Owns the Twilio Calls API lifecycle and conversation bookkeeping
    (inherited from BaseChannel). The real-time media transport itself —
    TwiML generation, WebSocket protocol handling, outbound call
    initiation — is delegated to a pluggable ``VoiceProvider``.

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
        # Call work aclose() must wait for besides live sessions: admitted
        # WebSocket handlers (one may still be in setup, with no session yet)
        # and teardowns still running their end-of-call callbacks.
        self._handler_tasks: set[asyncio.Task[Any]] = set()
        self._releases_in_flight = 0

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

        # With a CO conversation behind the call, CO's PARTICIPANT_ADDED
        # reports the start instead — see _track_conversation_started.
        if not self._provider._conversation_closed_by_orchestrator:
            track_event(
                "Conversation Started",
                self.tac.config.account_sid,
                channel=self._telemetry_channel,
                conversation_id=conv_id,
                has_profile_id=profile_id is not None,
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
        # The session is gone from the store but its callbacks haven't run:
        # count it so aclose() doesn't mistake an empty store for a finished drain.
        self._releases_in_flight += 1
        try:
            return await self._run_release(conv_id, session)
        finally:
            self._releases_in_flight -= 1

    async def _run_release(self, conv_id: str, session: ConversationSession) -> ConversationSession:
        """The end-of-call work for a session `_release_session` just popped."""
        # Measured before the callbacks below, which are application-owned:
        # they are awaited and may do network I/O or mutate the session, and
        # neither their latency nor their edits belong in the reported duration.
        duration_ms = int((datetime.now() - session.started_at).total_seconds() * 1000)

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

        # Only when no CO conversation sits behind the call (relay-only, Media
        # Streams): otherwise CO's CLOSED reports the end, with CO's duration.
        if not self._provider._conversation_closed_by_orchestrator:
            track_event(
                "Conversation Ended",
                self.tac.config.account_sid,
                channel=self._telemetry_channel,
                conversation_id=conv_id,
                duration_ms=duration_ms,
            )

        self.logger.debug(
            "Released voice session",
            conversation_id=conv_id,
            channel=self.get_channel_name(),
        )
        return session

    async def _handle_conversation_closed(
        self, conv_id: str, duration_ms: int | None = None
    ) -> None:
        """Fire ``on_conversation_ended`` for a CLOSED webhook.

        The session is normally already released (the socket closed when the
        caller hung up), so the usual path is a rebuild from Conversation
        Orchestrator — which is also what lets the hook fire on whichever
        instance received the webhook.

        If this instance still holds the session, the call is live: CO closed
        the conversation mid-call (a closed timeout during a long hold, or the
        app closing it). The hook fires now, from a snapshot, and the session
        stays for the call's own teardown to release — which is what keeps
        ``on_call_ended`` firing and the rest of the call working. The call
        then carries on under the closed conversation's id; CO starts a new
        conversation for its later traffic, whose own CLOSED arrives here
        later and takes the rebuild path.

        The hook is at-least-once: a duplicate CLOSED delivery that isn't
        deduped by its idempotency token fires it again, same as
        `on_message_ready`.

        Also reports "Conversation Ended" (``duration_ms`` is CO's
        ``updatedAt − createdAt``). On the live path the session in hand already
        proves the conversation is on voice, so no lookup is needed; otherwise
        it is reported only if the rebuilt conversation is on voice.
        """
        live = self._conversations.get(conv_id)
        if live is not None:
            self.logger.warning(
                "Conversation closed while its call is still live; the call continues "
                "under the closed conversation's id",
                conversation_id=conv_id,
                call_sid=live.call_sid,
            )
            self._track_conversation_ended(conv_id, duration_ms)
            await self._trigger_conversation_ended(self._detached_snapshot(live))
            return

        notify = self.tac._has_conversation_ended_callback()
        report = analytics_enabled()
        if not (notify or report):
            return
        session = await self._rebuild_session(conv_id, shared=True)
        if session is None:
            return
        if report:
            self._track_conversation_ended(conv_id, duration_ms)
        if notify:
            await self._trigger_conversation_ended(session)

    def _detached_snapshot(self, live: ConversationSession) -> ConversationSession:
        """A copy of a live call's session that shares no mutable state with it.

        `on_conversation_ended` may keep or edit what it's given while the call
        carries on, so the callback-visible data — `metadata` and the nested
        models — is deep-copied. The snapshot gets its own `cache_lock` rather
        than the live call's, so nothing the callback does can block the call.
        A `metadata` value that can't be copied (an app-owned handle, say) is
        shared rather than failing the hook.
        """
        conv_id = live.conversation_id

        def detached(value: Any, what: str) -> Any:
            try:
                return copy.deepcopy(value)
            except Exception as e:
                self.logger.debug(
                    "Sharing a value that can't be copied into the CLOSED snapshot",
                    conversation_id=conv_id,
                    value=what,
                    error=str(e),
                )
                return value

        return live.model_copy(
            update={
                "metadata": {k: detached(v, f"metadata[{k!r}]") for k, v in live.metadata.items()},
                "profile": detached(live.profile, "profile"),
                "author_info": detached(live.author_info, "author_info"),
                "ai_agent_info": detached(live.ai_agent_info, "ai_agent_info"),
                "pending_handoff_data": detached(live.pending_handoff_data, "pending_handoff_data"),
                "cached_memory": detached(live.cached_memory, "cached_memory"),
                "cache_lock": asyncio.Lock(),
            }
        )

    async def _rebuild_session(
        self, conv_id: str, call_sid: str | None = None, *, shared: bool = False
    ) -> ConversationSession | None:
        """Reconstruct a session from Conversation Orchestrator.

        Carries identity only — conversation id, call_sid, profile, both
        participants. Live in-memory state (transcript, metadata) is gone by
        now; use ``on_call_ended`` for that. ``call_sid`` defaults to a VOICE
        participant address's ``channelId`` — the customer's if present, else
        the agent's, else any other participant's. Returns ``None`` if no
        participant is on the voice channel, which is how another channel's
        CLOSED is filtered out.

        ``shared=True`` (the CLOSED path) shares the participant lookup with
        the other channels handling the same webhook; other callers need
        fresh participants.
        """
        client = self.tac.conversation_orchestrator_client
        if client is None:
            return None

        try:
            if shared:
                participants = await self.tac._list_participants_shared(conv_id)
            else:
                participants = await client.list_participants(conv_id)
        except Exception as e:
            self.logger.error(
                "Failed to list participants while rebuilding a voice session",
                conversation_id=conv_id,
                error=str(e),
            )
            return None

        if not any(a.channel == CO_VOICE_CHANNEL for p in participants for a in p.addresses):
            return None

        # A conversation grouped across channels may list another channel's
        # customer first; the caller is the CUSTOMER on VOICE.
        customer = next(
            (
                p
                for p in participants
                if p.type == "CUSTOMER" and any(a.channel == CO_VOICE_CHANNEL for a in p.addresses)
            ),
            None,
        )
        # TAC may own several numbers; the agent is whichever one this call used.
        agent = next(
            (
                found
                for number in self.tac.config.phone_numbers
                if (found := self._find_agent_participant(participants, CO_VOICE_CHANNEL, number))
                is not None
            ),
            None,
        )

        if call_sid is None:
            # CO's conversation webhooks carry no channel ids; the call a
            # voice participant is on is recorded on its VOICE address.
            # Prefer the customer's, then the agent's, then anyone else's.
            ordered = [p for p in (customer, agent) if p is not None] + [
                p for p in participants if p is not customer and p is not agent
            ]
            call_sid = next(
                (
                    a.channel_id
                    for p in ordered
                    for a in p.addresses
                    if a.channel == CO_VOICE_CHANNEL and a.channel_id
                ),
                None,
            )
            if call_sid is None:
                self.logger.debug("Rebuilt voice session has no call_sid", conversation_id=conv_id)

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

    def _has_call_work(self) -> bool:
        """Whether any call is still live, being set up, or being torn down."""
        return bool(self._conversations or self._handler_tasks or self._releases_in_flight)

    async def aclose(self, *, grace_period: float = DEFAULT_DRAIN_GRACE_PERIOD) -> None:
        """Drain live calls at shutdown: refuse new ones, wait up to
        ``grace_period`` seconds for the rest to end, then force-release them.

        It waits for every admitted call to finish — including one still in
        setup that has no session yet — and for end-of-call callbacks
        (`on_call_ended`, `on_conversation_ended`) still running. After the
        grace period, calls still running are cancelled (their teardown still
        runs and fires the end-of-call hooks). Any call still held after that
        is force-closed through its provider: the transport is closed and the
        provider's full teardown runs — sockets, stream tasks, model
        connection and provider state — before the session is released.

        Without this a scale-in drops live calls with no callback at all.
        Fail your readiness probe before calling it, so the load balancer
        stops routing here first. Idempotent.

        Args:
            grace_period: Seconds to wait for calls to end on their own.
        """
        self._accepting_calls = False

        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(grace_period, 0.0)
        while self._has_call_work() and loop.time() < deadline:
            await asyncio.sleep(0.1)

        current = asyncio.current_task()
        stragglers = [t for t in self._handler_tasks if t is not current and not t.done()]
        if stragglers:
            self.logger.warning(
                "Cancelling voice calls still running at shutdown",
                count=len(stragglers),
            )
            for task in stragglers:
                task.cancel()
            # Let their teardown (which fires the end-of-call hooks) run.
            await asyncio.wait(stragglers, timeout=_FORCED_STOP_TIMEOUT_SECONDS)

        remaining = list(self._conversations)
        if remaining:
            self.logger.warning(
                "Draining voice sessions still active at shutdown",
                count=len(remaining),
            )
        for conv_id in remaining:
            # The provider closes the transport and runs its full teardown
            # (sockets, stream tasks, provider state), which also releases the
            # channel session and fires the end-of-call hooks.
            await self._provider._force_close_call(conv_id)

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
            the call ended, or it landed on another instance. For the last
            case, :meth:`resolve_conversation_session_by_call_sid` falls back
            to Conversation Orchestrator.
        """
        for session in self._conversations.values():
            if session.call_sid == call_sid:
                return session
        return None

    async def resolve_conversation_session_by_call_sid(
        self, call_sid: str
    ) -> ConversationSession | None:
        """Look up a call's session on any instance, falling back to
        Conversation Orchestrator when this instance doesn't hold the call.

        Out-of-band call events (``on_call_status``, ``on_amd``,
        ``on_recording``) carry only a CallSid and, without
        ``TACConfig.instance_public_domain``, can reach any replica. This
        returns the live session when the call is held here — same as
        :meth:`get_conversation_session_by_call_sid` — and otherwise, for a
        ConversationRelay call in orchestrator mode, a session rebuilt from
        Conversation Orchestrator.

        A rebuilt session carries identity only (conversation id,
        ``call_sid``, profile, both participants), is not tracked, and can't
        be used with :meth:`send_response`: the call's WebSocket is on
        another instance. If CO holds several conversations for the call
        (it starts a new one when a conversation closes mid-call), the
        active one wins, else the newest — this rule applies only to the
        rebuild path. On the instance holding the call, the live session is
        returned as-is, so after a mid-call close it still carries the
        closed conversation's id while other instances return the new one;
        correlate by ``call_sid`` if you need a stable key.

        To tell live from rebuilt, check
        ``voice_channel.get_websocket(session.conversation_id) is not None``
        — true only for a session live on this instance. A rebuilt
        session's ``started_at`` is the rebuild time, not the call's, and
        its ``metadata`` is always empty.

        Cost: on a local miss this makes up to two Conversation Orchestrator
        requests (list conversations, then participants). Prefer
        :meth:`get_conversation_session_by_call_sid` on hot paths, or when
        ``TACConfig.instance_public_domain`` already routes call events to
        the instance holding the call.

        Example:
            ```python
            async def on_status(event: CallStatusEvent) -> None:
                session = await voice_channel.resolve_conversation_session_by_call_sid(
                    event.call_sid
                )
                if session is not None:
                    audit_log(session.conversation_id, session.profile_id, event.call_status)
            ```

        Args:
            call_sid: Twilio Call SID, e.g. from a call event.

        Returns:
            The live or rebuilt session, or ``None`` — relay-only or Media
            Streams mode (no CO conversation behind the call), CO knows no
            conversation for it, or the lookup failed (logged).
        """
        session = self.get_conversation_session_by_call_sid(call_sid)
        if session is not None:
            return session

        client = self.tac.conversation_orchestrator_client
        if client is None or not self._provider._conversation_closed_by_orchestrator:
            return None

        try:
            conversations = await client.list_conversations(channel_id=call_sid)
        except Exception as e:
            self.logger.error(
                "Failed to look up the conversation for a call",
                call_sid=call_sid,
                error=str(e),
                exc_info=True,
            )
            return None
        if not conversations:
            return None

        active = [c for c in conversations if c.status == "ACTIVE"]
        chosen = max(active or conversations, key=_created_at_key)
        return await self._rebuild_session(chosen.id, call_sid)

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
        # Socket-open half of the Websocket Connected/Disconnected pair.
        # Tracked here rather than in a provider so every transport is
        # covered; no conversation identifier exists yet at this point.
        # After the drain check: a refused socket never connected.
        track_event(
            "Websocket Connected",
            self.tac.config.account_sid,
            channel=self._telemetry_channel,
            provider=self._provider.provider_id,
            orchestrator_enabled=self.tac.is_orchestrator_enabled(),
        )
        # Tracked from admission to completion, so aclose() waits for a call
        # still in setup — before it has a session to see.
        task = asyncio.current_task()
        if task is not None:
            self._handler_tasks.add(task)
        try:
            await self._provider.handle_websocket(websocket)
        finally:
            if task is not None:
                self._handler_tasks.discard(task)

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

        Voice channel processes these events:

        - **PARTICIPANT_ADDED**: report "Conversation Started" when the customer
          joins on voice (orchestrated ConversationRelay only).
        - **CLOSED** (CONVERSATION_UPDATED): fire ``on_conversation_ended``.
          The call's session is normally already released (the WebSocket
          closed when the caller hung up), so this rebuilds it from
          Conversation Orchestrator — which is also what makes the hook fire
          on whichever instance received the webhook rather than only on the
          one that held the call. If this instance still holds the call's
          session (the call is live), the hook fires now instead, with a
          snapshot of that live session, and the session stays until the
          call's own teardown — which still fires ``on_call_ended``. A single
          call can then produce a second ``on_conversation_ended``, under a
          different conversation id, when Conversation Orchestrator later
          closes the conversation it started for the call's remaining
          traffic. It also reports "Conversation Ended", with CO's duration.
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

        if event_type == "PARTICIPANT_ADDED":
            # Only with a CO conversation behind calls; otherwise the session
            # reports the start and counting this too would double it.
            if self._provider._conversation_closed_by_orchestrator:
                self._track_conversation_started(event_data)
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
            await self._handle_conversation_closed(
                conv_id, elapsed_ms(event_data.get("createdAt"), event_data.get("updatedAt"))
            )
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
        # Response Sent is reported by the provider, not here: a reply from the
        # message-ready callback is auto-sent directly through the provider and
        # would otherwise go unreported.
        await self._provider.send_response(conversation_id, response, role)

    def get_channel_name(self) -> str:
        return self._provider.channel_name

    @property
    def _co_channel(self) -> str:
        # get_channel_name() is the provider's transport label; CO always
        # records voice participants as VOICE.
        return CO_VOICE_CHANNEL

    def _is_own_co_address(self, address: str) -> bool:
        return address in self.tac.config.phone_numbers

    @property
    def _telemetry_channel(self) -> str:
        # Not derived from `get_channel_name()` like the base class does: that
        # returns the provider's transport (e.g.
        # ``"VOICE_MEDIA_STREAM_OPENAI_GPT_LIVE"``), whereas telemetry reports
        # the transport separately as `provider`.
        return "voice"

    def get_websocket(self, conversation_id: str) -> WebSocketProtocol | None:
        """
        Get the WebSocket connection for a specific conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            WebSocket connection if exists, None otherwise
        """
        return self._provider.get_websocket(conversation_id)
