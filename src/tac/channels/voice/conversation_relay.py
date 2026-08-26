"""``ConversationRelayProvider``: the default ``VoiceProvider``.

Twilio ConversationRelay's managed setup/prompt/interrupt loop over one
WebSocket, with Twilio doing ASR/TTS.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from tac.channels.voice.provider import VoiceProvider
from tac.channels.websocket_manager import WebSocketManager
from tac.channels.websocket_protocol import WebSocketDisconnectError, WebSocketProtocol
from tac.core.config import CallEventKind, TACConfig
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
    TwiMLRequest,
    VoiceTwiMLOptions,
    VoiceTwiMLOptionsConversationRelay,
)
from tac.session import SessionState
from tac.utils.redaction import mask_phone, redact_twiml_parameters

from . import twiml

if TYPE_CHECKING:
    from tac.channels.voice.channel import VoiceChannel
    from tac.channels.voice.config import ConversationRelayProviderConfig

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
        channel: VoiceChannel,
        tac_config: TACConfig,
        config: ConversationRelayProviderConfig,
    ) -> None:
        super().__init__(channel)
        self.config = config
        self.session_manager = config.session_manager
        self._websocket_manager = WebSocketManager()
        self._twiml = twiml.TwiMLBuilderConversationRelay(tac_config, config)

    @property
    def channel_name(self) -> str:
        return "VOICE"

    @staticmethod
    def _caller_address(setup_msg: SetupMessage) -> str | None:
        """Return the phone number of the remote caller/callee from the setup message."""
        if setup_msg.direction and setup_msg.direction.upper() == "OUTBOUND":
            return setup_msg.to_number
        return setup_msg.from_number

    async def handle_incoming_call(
        self,
        twiml_request: TwiMLRequest | None = None,
        *,
        host_twiml_options: VoiceTwiMLOptions | None = None,
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
        if host_twiml_options is not None and not isinstance(
            host_twiml_options, VoiceTwiMLOptionsConversationRelay
        ):
            raise TypeError(
                "ConversationRelayProvider.handle_incoming_call requires host_twiml_options "
                f"to be a VoiceTwiMLOptionsConversationRelay, got "
                f"{type(host_twiml_options).__name__}"
            )

        customized: VoiceTwiMLOptionsConversationRelay | None = None
        if self.channel._on_inbound_call_twiml is not None and twiml_request is not None:
            customized = await self.channel._on_inbound_call_twiml(twiml_request)

        return self._twiml.build(
            "handle_incoming_call",
            host=host_twiml_options,
            per_call=customized,
        )

    async def handle_twilio_provider_callback(
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

        if payload.account_sid != self.channel.tac.config.account_sid:
            self.logger.warning(
                "ConversationRelay callback account_sid mismatch, ignoring",
                expected=self.channel.tac.config.account_sid,
                received=payload.account_sid,
            )
            return

        self.logger.debug(
            "ConversationRelay callback received",
            call_sid=payload.call_sid,
            call_status=payload.call_status,
        )

        if payload.call_status == "completed" and not self.channel.tac.is_orchestrator_enabled():
            if payload.call_sid in self.channel._conversations:
                await self.channel._end_conversation(payload.call_sid)

    async def _initialize_conversation(
        self,
        call_sid: str,
        setup_msg: SetupMessage,
        websocket: WebSocketProtocol,
    ) -> tuple[str, SessionState | None]:
        """Poll CO for the conversation created by ConversationRelay, resolve
        the customer participant, and initialize the local session."""
        conversation_orchestrator_client = self.channel.tac.conversation_orchestrator_client
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
        agent_participant = self.channel._find_agent_participant(
            participants, "VOICE", self.channel.tac.config.phone_number
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
        session = self.channel._start_conversation(conv_id, profile_id)
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
                address=agent_address or self.channel.tac.config.phone_number,
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
                if call_sid and self.channel.tac.is_orchestrator_enabled():
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
                                session = self.channel._start_conversation(conv_id, profile_id=None)
                                # Relay-only: conv_id == call_sid.
                                session.call_sid = call_sid

                                caller = self._caller_address(setup_msg)
                                if caller:
                                    self.channel._conversations[conv_id].author_info = AuthorInfo(
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
            ("status", "status_callback", self.channel._on_call_status),
            ("amd", "async_amd_status_callback", self.channel._on_amd),
            ("recording", "recording_status_callback", self.channel._on_recording),
        ]
        for kind, param, handler in wiring:
            if handler is None:
                continue
            url = self.channel.tac.config.call_event_url(kind)
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
        from_number = self.channel.tac.config.phone_number

        self.logger.info(
            "Initiating outbound voice conversation",
            to=mask_phone(options.to),
            from_number=mask_phone(from_number),
        )

        # Outbound has no inbound customizer and no server layer; the per-call
        # override is options.twiml_options. ``options.websocket_url`` is the
        # dedicated per-call override and wins over any websocket_url that came
        # through the layered merge; both fall back to the TACConfig-derived URL.
        twiml_xml = self._twiml.build(
            "initiate_outbound_conversation",
            per_call=options.twiml_options,
            websocket_url=options.websocket_url,
        )

        call_kwargs = self._build_call_kwargs(options.call_options)

        try:
            # The inline TwiML handed to Twilio, useful for debugging the
            # <Connect action> handoff target. custom_parameters values are
            # masked — they're arbitrary developer data (profile IDs, caller
            # names), unlike the WS/action URLs and conversation config.
            self.logger.debug(
                "Outbound call TwiML",
                twiml=redact_twiml_parameters(twiml_xml),
                to=mask_phone(options.to),
            )

            client = self.channel._get_twilio_client()
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
            if conversation_id in self.channel._conversations:
                session = self.channel._conversations[conversation_id]
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
        if conv_id not in self.channel._conversations:
            self.logger.error(
                f"Received prompt for unknown conversation {conv_id}. "
                "Conversation should be initialized on first prompt.",
                conversation_id=conv_id,
            )
            return

        message_body = message.voice_prompt or ""
        session = self.channel._conversations[conv_id]

        # Retrieve memory if memory_mode is enabled and Twilio Memory is configured
        memory_response = await self.channel._retrieve_memory_if_enabled(
            session, message_body, conv_id
        )

        # Trigger message ready callback
        try:
            response = await self.channel.tac.trigger_message_ready(
                message_body, session, memory_response
            )
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
        if conv_id in self.channel._conversations:
            session = self.channel._conversations[conv_id]
            self.channel.tac.trigger_interrupt(session, message)
        else:
            self.logger.warning(
                f"Received interrupt for unknown conversation {conv_id}, skipping callback"
            )

    async def _cleanup_connection(self, conv_id: str) -> None:
        """
        Clean up WebSocket and session resources when connection closes.

        In orchestrated mode, the conversation remains tracked in
        self.channel._conversations until the CONVERSATION_UPDATED/CLOSED webhook
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

        if (
            not self.channel.tac.is_orchestrator_enabled()
            and conv_id in self.channel._conversations
        ):
            await self.channel._end_conversation(conv_id)

        self.logger.debug(
            "Cleaned up WebSocket and session resources",
            conversation_id=conv_id,
        )
