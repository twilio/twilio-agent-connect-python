from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from twilio.rest import Client

from pydantic import ValidationError

from tac import tracing
from tac.channels.base import BaseChannel
from tac.channels.websocket_manager import WebSocketManager
from tac.channels.websocket_protocol import WebSocketDisconnectError, WebSocketProtocol
from tac.core.tac import TAC
from tac.models.outbound import InitiateVoiceConversationOptions, InitiateVoiceConversationResult
from tac.models.session import AuthorInfo
from tac.models.voice import (
    ConversationRelayCallbackPayload,
    InterruptMessage,
    PromptMessage,
    SetupMessage,
    TwiMLOptions,
)
from tac.session import SessionState
from tac.utils.redaction import mask_phone

from . import twiml
from .config import VoiceChannelConfig

_POLL_ATTEMPTS = 5
_POLL_BASE_DELAY = 0.25
_CO_INIT_WARN_THRESHOLD_S = 0.8


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
        self.session_manager = config.session_manager
        self.transcription_provider = config.transcription_provider
        self.speech_model = config.speech_model
        self._websocket_manager = WebSocketManager()
        self._twilio_client: Client | None = None
        self._stt_chunk_counts: dict[str, int] = {}
        self._stt_processed_counts: dict[str, int] = {}

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
        options: TwiMLOptions | dict[str, Any],
    ) -> str:
        """
        Generate TwiML response for incoming voice calls.

        ConversationRelay automatically handles conversation creation and participant
        management via the conversation_configuration parameter.

        Args:
            options: TwiML generation options (TwiMLOptions or dict) containing:
                - websocket_url (required): WebSocket URL for ConversationRelay
                - custom_parameters (optional): Additional custom parameters
                - welcome_greeting (optional): Initial greeting message
                - action_url (optional): URL for call completion webhook

        Returns:
            TwiML XML string for call connection

        Example:
            >>> twiml = await voice_channel.handle_incoming_call(
            ...     options={
            ...         "websocket_url": "wss://example.com/ws",
            ...         "custom_parameters": {"session_id": "sess_123"},
            ...         "welcome_greeting": "Hello!",
            ...         "action_url": "https://example.com/callback",
            ...     },
            ... )
        """
        # Handle dict input (convert to TwiMLOptions)
        if isinstance(options, dict):
            options = TwiMLOptions(**options)

        # Set default welcome greeting if not provided
        if options.welcome_greeting is None:
            options.welcome_greeting = "Hello! How can I assist you today?"

        # ConversationRelay automatically creates conversation and participants
        return twiml.generate_twiml(
            TwiMLOptions(
                websocket_url=options.websocket_url,
                custom_parameters=options.custom_parameters,
                welcome_greeting=options.welcome_greeting,
                action_url=options.action_url,
                conversation_configuration=self.tac.config.conversation_configuration_id,
                transcription_provider=self.transcription_provider,
                speech_model=self.speech_model,
            )
        )

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
        co_init_start = time.monotonic()
        with tracing.co_init_span(call_sid):
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
                    await asyncio.sleep(_POLL_BASE_DELAY * (2**attempt))
        co_init_elapsed = time.monotonic() - co_init_start
        if co_init_elapsed >= _CO_INIT_WARN_THRESHOLD_S:
            self.logger.warning(
                "co_init exceeded threshold — CO may be cold",
                call_sid=call_sid,
                co_init_elapsed_ms=round(co_init_elapsed * 1000),
                threshold_ms=round(_CO_INIT_WARN_THRESHOLD_S * 1000),
            )

        if len(conversations) != 1:
            raise RuntimeError(
                f"Expected exactly 1 conversation for "
                f"call_sid {call_sid}, but found "
                f"{len(conversations)} after "
                f"{_POLL_ATTEMPTS} attempts."
            )

        conversation = conversations[0]
        conv_id = conversation.id

        with tracing.profile_lookup_span(call_sid):
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

        self._websocket_manager.add_websocket(conv_id, websocket)
        session = self._start_conversation(conv_id, profile_id)
        session.metadata["call_sid"] = call_sid

        session_state = None
        if self.session_manager is not None:
            session_state = self.session_manager.get_or_create_session(conv_id)

        if profile_lookup_address:
            session.author_info = AuthorInfo(address=profile_lookup_address)

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
        call_sid: str | None = None
        session_state = None
        init_task: asyncio.Task[tuple[str, SessionState | None]] | None = None

        try:
            # First message should be 'setup'
            data = await websocket.receive_json()
            if data.get("type") == "setup":
                setup_msg = SetupMessage(**data)
                call_sid = setup_msg.call_sid

                self.logger.info("Call started", call_sid=call_sid)
                if call_sid:
                    tracing.start_call(call_sid)

                # Fire init immediately so CO polling runs during user speech
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
                            if self.tac.is_orchestrator_enabled():
                                with tracing.first_prompt_wait_span(call_sid):
                                    conv_id, session_state = await init_task  # type: ignore[misc]
                                init_task = None
                            else:
                                conv_id = call_sid
                                self._websocket_manager.add_websocket(conv_id, websocket)
                                self._start_conversation(conv_id, profile_id=None)

                                caller = self._caller_address(setup_msg)
                                if caller:
                                    self._conversations[conv_id].author_info = AuthorInfo(
                                        address=caller,
                                    )

                                if self.session_manager is not None:
                                    session_state = self.session_manager.get_or_create_session(
                                        conv_id
                                    )

                            # Store call_sid so turn handlers can tag traces/logs
                            if conv_id and conv_id in self._conversations:
                                self._conversations[conv_id].metadata["call_sid"] = call_sid

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
            if init_task is not None:
                if not init_task.done():
                    init_task.cancel()
                try:
                    await init_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.warning(
                        "Early co_init task failed",
                        call_sid=call_sid,
                        error=str(e),
                    )
                if not conv_id and not init_task.cancelled():
                    try:
                        resolved_conv_id, _ = init_task.result()
                        self._websocket_manager.remove_websocket(resolved_conv_id)
                    except Exception:
                        pass

            if conv_id:
                self.logger.debug("Cleanup - removing WebSocket", conversation_id=conv_id)
                await self._cleanup_connection(conv_id)
            elif call_sid:
                tracing.end_call(call_sid)

    async def initiate_outbound_conversation(
        self,
        options: InitiateVoiceConversationOptions,
    ) -> InitiateVoiceConversationResult:
        """Initiate an outbound voice conversation.

        Places an outbound call with inline TwiML that connects to ConversationRelay.
        The conversationConfiguration attribute tells CO to create and manage the
        conversation during passive hydration. The session is initialized lazily
        on the first prompt when the conversation is discovered by callSid.

        ``options.websocket_url`` must be the publicly accessible WebSocket
        endpoint (e.g., ``wss://your-domain.ngrok.app/ws``). Unlike inbound calls
        where TACServer sets this automatically, outbound calls require it
        explicitly since there is no incoming HTTP request to derive the host from.
        """
        from_number = self.tac.config.phone_number

        self.logger.info(
            "Initiating outbound voice conversation",
            to=mask_phone(options.to),
            from_number=mask_phone(from_number),
        )

        try:
            twiml_xml = twiml.generate_twiml(
                TwiMLOptions(
                    websocket_url=options.websocket_url,
                    welcome_greeting=options.welcome_greeting,
                    action_url=options.action_url,
                    conversation_configuration=self.tac.config.conversation_configuration_id,
                    custom_parameters=options.custom_parameters,
                    transcription_provider=self.transcription_provider,
                    speech_model=self.speech_model,
                )
            )

            client = self._get_twilio_client()
            call = await asyncio.to_thread(
                client.calls.create, to=options.to, from_=from_number, twiml=twiml_xml
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
            is_final = data.get("final", "MISSING")
            utterance = data.get("voicePrompt", data.get("voice_prompt", ""))

            # STT chunking diagnostics
            self._stt_chunk_counts[conv_id] = self._stt_chunk_counts.get(conv_id, 0) + 1
            chunks_so_far = self._stt_chunk_counts[conv_id]
            processed_so_far = self._stt_processed_counts.get(conv_id, 0)
            model_label = self.speech_model or "default"
            self.logger.debug(
                "STT chunk received",
                model=model_label,
                final=is_final,
                chunks=chunks_so_far,
                processed=processed_so_far,
                text=utterance,
            )

            if should_process:
                self._stt_processed_counts[conv_id] = self._stt_processed_counts.get(conv_id, 0) + 1
                total_chunks = self._stt_chunk_counts.get(conv_id, 0)
                total_processed = self._stt_processed_counts[conv_id]
                self.logger.debug(
                    "STT turn firing",
                    turn=total_processed,
                    processed=total_processed,
                    total_chunks=total_chunks,
                    text=utterance,
                )

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

                first_token_recorded = False
                _stream_session = self._conversations.get(conversation_id)
                _stream_call_sid = (
                    _stream_session.metadata.get("call_sid", conversation_id)
                    if _stream_session
                    else conversation_id
                )
                try:
                    with tracing.llm_response_stream_span(_stream_call_sid):
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
                                if not first_token_recorded:
                                    tracing.record_first_token(_stream_call_sid)
                                    first_token_recorded = True
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
        return "voice"

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
        call_sid: str = session.metadata.get("call_sid", conv_id)

        with tracing.turn_span(call_sid, message_body):
            # Retrieve memory if memory_mode is enabled and Twilio Memory is configured
            with tracing.memory_span(call_sid):
                memory_response = await self._retrieve_memory_if_enabled(
                    session, message_body, conv_id
                )

            # Trigger message ready callback (LLM call + response send happen inside)
            try:
                with tracing.llm_span(call_sid):
                    response = await self.tac.trigger_message_ready(
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

        In orchestrated mode, immediately closes the conversation via the CO API
        so Memora's extraction pipeline starts without waiting for the inactive timeout.
        In relay-only mode there is no CO webhook, so we end the conversation locally.

        Args:
            conv_id: Conversation ID
        """
        # Remove WebSocket from manager
        if self._websocket_manager.has_websocket(conv_id):
            self._websocket_manager.remove_websocket(conv_id)

        total_chunks = self._stt_chunk_counts.pop(conv_id, 0)
        total_processed = self._stt_processed_counts.pop(conv_id, 0)
        if total_chunks:
            skipped = total_chunks - total_processed
            model_label = self.speech_model or "default"
            self.logger.debug(
                "STT call summary",
                model=model_label,
                total_chunks=total_chunks,
                processed=total_processed,
                skipped=skipped,
                skipped_pct=round(skipped / total_chunks * 100),
            )

        # Cancel running stream task and cleanup session if session manager is enabled
        if self.session_manager is not None and self.session_manager.has_session(conv_id):
            session_state = self.session_manager.get_or_create_session(conv_id)
            # Cancel any running task (user hung up, no point continuing)
            await session_state.cancel_stream_task()
            self.session_manager.remove_session(conv_id)

        if self.tac.is_orchestrator_enabled():
            # Close conversation immediately so Memora extraction starts right away
            # instead of waiting for the inactive timeout.
            co_client = self.tac.conversation_orchestrator_client
            if co_client is not None and conv_id in self._conversations:
                try:
                    await co_client.update_conversation(conv_id, status="CLOSED")
                    self.logger.debug("Closed conversation on call end", conversation_id=conv_id)
                except Exception as e:
                    self.logger.warning(
                        "Failed to close conversation on call end",
                        conversation_id=conv_id,
                        error=str(e),
                    )
        elif conv_id in self._conversations:
            await self._end_conversation(conv_id)

        session = self._conversations.get(conv_id)
        call_sid = session.metadata.get("call_sid", conv_id) if session else conv_id
        tracing.end_call(call_sid)

        self.logger.debug(
            "Cleaned up WebSocket and session resources",
            conversation_id=conv_id,
        )
