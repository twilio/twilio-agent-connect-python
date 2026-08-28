"""``OpenAIRealtimeProvider``: bridges Twilio Media Streams to OpenAI's
Realtime API.

Twilio streams call audio to our own WebSocket via ``<Connect><Stream>``, and
this provider relays it to/from a second WebSocket it opens to OpenAI's
Realtime API. Session lifecycle is independent of Conversation Orchestrator —
this provider always starts the local session with ``profile_id=None``, the
same as ConversationRelay's relay-only mode.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import uuid
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any

import websockets

from tac.channels.voice.media_streams.openai_realtime.models import _CallState
from tac.channels.voice.media_streams.openai_realtime.twiml import (
    TwiMLBuilderMediaStreams,
    VoiceTwiMLOptionsMediaStreams,
)
from tac.channels.voice.provider import VoiceProvider
from tac.channels.websocket_protocol import WebSocketDisconnectError, WebSocketProtocol
from tac.core.config import CallEventKind, TACConfig
from tac.models.outbound import (
    CallOptions,
    InitiateVoiceConversationOptions,
    InitiateVoiceConversationOptionsOpenAIRealtime,
    InitiateVoiceConversationResult,
)
from tac.models.session import ConversationSession
from tac.models.stream import StreamStartMessage
from tac.models.voice import TwiMLRequest, VoiceTwiMLOptions
from tac.tools import TACTool
from tac.utils.redaction import mask_phone, redact_twiml_parameters

if TYPE_CHECKING:
    from tac.channels.voice.channel import VoiceChannel
    from tac.channels.voice.media_streams.openai_realtime.config import (
        OpenAIRealtimeProviderConfig,
    )


#: Twilio Media Streams always sends/expects 8kHz G.711 u-law audio — per
#: https://www.twilio.com/docs/voice/media-streams/websocket-messages, this
#: isn't a provider choice or a default, it's the only format Twilio's
#: bidirectional Stream supports. No ``rate`` field — OpenAI's Realtime
#: ``session.audio.*.format`` schema rejects it as unknown (g711 is
#: inherently fixed-rate, so it isn't settable).
TWILIO_MEDIA_STREAM_AUDIO_FORMAT: dict[str, Any] = {"type": "audio/pcmu"}

#: G.711 u-law at 8kHz is 1 byte/sample, 8000 samples/sec — a fixed,
#: non-configurable rate, so audio byte count converts to milliseconds by
#: this constant alone, regardless of session_config.
_PCMU_BYTES_PER_MS = 8

#: Reserved <Stream> custom_parameters key used to correlate an outbound
#: call's session_config override to its WebSocket start event. calls.create()
#: returning call.sid doesn't happen-before Twilio connecting the stream, so
#: call.sid can't be the correlation key — this token, embedded in the TwiML
#: before the call is placed, can.
_SESSION_CONFIG_TOKEN_PARAM = "_tac_session_config_token"


class OpenAIRealtimeProvider(VoiceProvider):
    """``VoiceProvider`` bridging Twilio Media Streams to OpenAI Realtime.

    Example:
        ```python
        channel = VoiceChannel(tac, config=OpenAIRealtimeProviderConfig(default_session_config=...))
        ```
    """

    def __init__(
        self,
        channel: VoiceChannel,
        tac_config: TACConfig,
        config: OpenAIRealtimeProviderConfig,
    ) -> None:
        super().__init__(channel)

        self.config = config
        self.tac_config = tac_config
        self._tools_by_name: dict[str, TACTool] = {tool.name: tool for tool in config.tools}
        self._calls: dict[str, _CallState] = {}
        self._twiml = TwiMLBuilderMediaStreams(tac_config, config)
        # Keyed by call_sid; set in handle_incoming_call, popped in _connect_model.
        self._call_session_configs: dict[str, dict[str, Any]] = {}

    @property
    def channel_name(self) -> str:
        return "VOICE_MEDIA_STREAM_OPENAI_REALTIME"

    def get_transcript(self, conversation_id: str) -> list[dict[str, str]]:
        """Return the transcript captured so far for an in-progress call.

        Lives on ``ConversationSession.metadata["transcript"]``, so once the
        call ends (and the session is popped from ``channel._conversations``)
        it's no longer reachable here — read it from the session an
        ``on_conversation_ended`` handler receives instead.
        """
        session = self.channel._conversations.get(conversation_id)
        return list(session.metadata.get("transcript", [])) if session else []

    def get_websocket(self, conversation_id: str) -> WebSocketProtocol | None:
        call = self._calls.get(conversation_id)
        return call.twilio_ws if call else None

    async def handle_incoming_call(
        self,
        twiml_request: TwiMLRequest | None = None,
        *,
        host_twiml_options: VoiceTwiMLOptions | None = None,
    ) -> str:
        """Build the ``<Connect><Stream>`` TwiML for an inbound call.

        TwiML fields are merged per-field, highest precedence first:
          1. Output of the customizer registered via
             ``VoiceChannel.on_inbound_call_twiml(...)`` if configured
             and ``twiml_request`` is given.
          2. ``OpenAIRealtimeProviderConfig.default_twiml_options`` — per-channel defaults.
          3. ``host_twiml_options`` — per-call transport facts supplied by the host.
          4. TAC defaults: the WebSocket URL derived from
             ``TACConfig.voice_public_domain`` + ``voice_websocket_path``.

        Also runs ``on_inbound_call_session_config`` (if set) and stashes its
        result for ``_connect_model`` to pick up once the call connects.
        """
        if host_twiml_options is not None and not isinstance(
            host_twiml_options, VoiceTwiMLOptionsMediaStreams
        ):
            raise TypeError(
                "OpenAIRealtimeProvider.handle_incoming_call requires host_twiml_options "
                f"to be a VoiceTwiMLOptionsMediaStreams, got {type(host_twiml_options).__name__}"
            )

        customized: VoiceTwiMLOptionsMediaStreams | None = None
        if self.channel._on_inbound_call_twiml is not None and twiml_request is not None:
            result = await self.channel._on_inbound_call_twiml(twiml_request)
            if not isinstance(result, VoiceTwiMLOptionsMediaStreams):
                raise TypeError(
                    "OpenAIRealtimeProvider.handle_incoming_call requires the "
                    "on_inbound_call_twiml customizer to return a "
                    f"VoiceTwiMLOptionsMediaStreams, got {type(result).__name__}"
                )
            customized = result

        if (
            self.config.on_inbound_call_session_config is not None
            and twiml_request is not None
            and twiml_request.call_sid is not None
        ):
            session_config = await self.config.on_inbound_call_session_config(twiml_request)
            if session_config is not None:
                self._call_session_configs[twiml_request.call_sid] = session_config

        return self._twiml.build(
            "handle_incoming_call", host=host_twiml_options, per_call=customized
        )

    def _build_call_kwargs(self, call_options: CallOptions | None) -> dict[str, Any]:
        """Build the extra kwargs for ``client.calls.create``.

        Layers, highest precedence first: this call's ``call_options``, then
        callback URLs derived from ``voice_public_domain`` + ``voice_call_event_path``,
        one per registered handler.
        """
        call_kwargs = call_options.to_call_kwargs() if call_options else {}

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

        Places an outbound call with inline TwiML that connects to a Media
        Stream. Unlike inbound, there's no local session yet at this point —
        it's created when Twilio's WebSocket ``start`` event arrives, the
        same as ``_register_call`` does for inbound.

        TwiML fields are merged per-field — see ``TwiMLBuilderMediaStreams.build``.
        The WebSocket URL is derived from ``TACConfig.voice_public_domain`` +
        ``TACConfig.voice_websocket_path``, unless overridden per-call via
        ``options.websocket_url``.

        Pass ``InitiateVoiceConversationOptionsOpenAIRealtime`` with
        ``session_config`` set to override the default for this call.
        """
        twiml_options = options.twiml_options
        if twiml_options is not None and not isinstance(
            twiml_options, VoiceTwiMLOptionsMediaStreams
        ):
            raise TypeError(
                "OpenAIRealtimeProvider.initiate_outbound_conversation requires "
                "options.twiml_options to be a VoiceTwiMLOptionsMediaStreams, got "
                f"{type(twiml_options).__name__}"
            )

        # A token embedded in the TwiML below correlates this override to its
        # WebSocket start event — not call.sid, since Twilio connecting the
        # stream doesn't happen-after calls.create() returning call.sid.
        # Building the token/TwiML here doesn't touch _call_session_configs
        # yet; that's deferred until right before the call is placed (below),
        # so a failure in _twiml.build() has nothing to leak.
        session_config = (
            options.session_config
            if isinstance(options, InitiateVoiceConversationOptionsOpenAIRealtime)
            else None
        )
        session_config_token: str | None = None
        if session_config is not None:
            session_config_token = uuid.uuid4().hex
            existing_params = (twiml_options.custom_parameters or {}) if twiml_options else {}
            twiml_options = (twiml_options or VoiceTwiMLOptionsMediaStreams()).model_copy(
                update={
                    "custom_parameters": {
                        **existing_params,
                        _SESSION_CONFIG_TOKEN_PARAM: session_config_token,
                    }
                }
            )

        from_number = self.channel.tac.config.phone_number

        self.logger.info(
            "Initiating outbound voice conversation",
            to=mask_phone(options.to),
            from_number=mask_phone(from_number),
        )

        twiml_xml = self._twiml.build(
            "initiate_outbound_conversation",
            per_call=twiml_options,
            websocket_url=options.websocket_url,
        )

        call_kwargs = self._build_call_kwargs(options.call_options)

        if session_config_token is not None and session_config is not None:
            self._call_session_configs[session_config_token] = session_config

        try:
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
            if session_config_token is not None:
                self._call_session_configs.pop(session_config_token, None)
            self.logger.error(
                "Failed to initiate outbound call",
                to=mask_phone(options.to),
                error=str(e),
                exc_info=True,
            )
            raise

    async def handle_websocket(self, websocket: WebSocketProtocol) -> None:
        """Drive one Twilio Media Stream connection from accept to disconnect.

        Races the Twilio read against the OpenAI model-event reader so that
        if the model side disconnects first, we stop pumping caller audio
        into a dead socket and tear the call down immediately instead of
        leaving the caller connected to silence.
        """
        await websocket.accept()

        conv_id: str | None = None
        model_reader: asyncio.Task[None] | None = None

        try:
            while True:
                recv_task: asyncio.Task[dict[str, Any]] = asyncio.create_task(
                    websocket.receive_json()
                )
                waiters: list[asyncio.Task[Any]] = [recv_task]
                if model_reader is not None:
                    waiters.append(model_reader)

                done, _pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)

                if model_reader is not None and model_reader in done:
                    recv_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await recv_task
                    self.logger.info("Model connection ended", conversation_id=conv_id)
                    break

                data = recv_task.result()
                event = data.get("event")

                if event == "start":
                    conv_id = self._register_call(data.get("start") or {}, websocket)
                    await self._connect_model(conv_id)
                    model_reader = asyncio.create_task(self._handle_model_events(conv_id))

                elif event == "media":
                    media = data.get("media") or {}
                    if conv_id is not None:
                        if media.get("payload"):
                            await self._model_send(
                                conv_id,
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": media["payload"],
                                },
                            )

                elif event == "stop":
                    self.logger.info("Media stream stopped", conversation_id=conv_id)
                    break

        except WebSocketDisconnectError:
            self.logger.info("Media stream WebSocket closed", conversation_id=conv_id)
        except Exception as e:
            self.logger.error(f"Media stream WebSocket error: {e}", exc_info=True)
        finally:
            if model_reader is not None and not model_reader.done():
                model_reader.cancel()
            if model_reader is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await model_reader
            if conv_id is not None:
                await self._cleanup_call(conv_id)

    def _register_call(self, start: dict[str, Any], websocket: WebSocketProtocol) -> str:
        """Handle Twilio's ``start`` event, returning the conversation id."""
        message = StreamStartMessage(**start)
        conv_id = message.conversation_id

        token = message.custom_parameters.get(_SESSION_CONFIG_TOKEN_PARAM)
        if token is not None:
            session_config = self._call_session_configs.pop(token, None)
            if session_config is not None:
                self._call_session_configs[conv_id] = session_config

        self._calls[conv_id] = _CallState(twilio_ws=websocket)

        session = self.channel._start_conversation(conv_id, profile_id=None)
        session.call_sid = message.call_sid
        session.metadata.update({"stream_sid": message.stream_sid, "transcript": []})

        self.logger.debug(
            "Media stream started", conversation_id=conv_id, media_format=message.media_format
        )
        return conv_id

    async def _connect_model(self, conv_id: str) -> None:
        """Open the OpenAI Realtime WebSocket and send the session config.

        Uses this call's ``_call_session_configs`` entry if one was stashed
        (by ``handle_incoming_call`` or ``initiate_outbound_conversation``),
        else falls back to ``default_session_config``.
        """
        session_config = self._call_session_configs.pop(conv_id, None)
        if session_config is None:
            session_config = self.config.default_session_config
        if session_config is None:
            raise ValueError(
                f"No session_config available for call {conv_id} — this call supplied none "
                "and default_session_config isn't set either."
            )
        if not session_config.get("model"):
            raise ValueError(
                f"session_config for call {conv_id} must include a 'model' field — it's "
                "used as the ?model= query param when opening the OpenAI Realtime WebSocket."
            )
        for direction in ("input", "output"):
            fmt = ((session_config.get("audio") or {}).get(direction) or {}).get("format")
            if fmt != TWILIO_MEDIA_STREAM_AUDIO_FORMAT:
                raise ValueError(
                    f"session_config for call {conv_id} has audio.{direction}.format={fmt!r} — "
                    f"Twilio Media Streams always sends/expects "
                    f"{TWILIO_MEDIA_STREAM_AUDIO_FORMAT!r}, this isn't configurable. Set "
                    f"audio.{direction}.format to TWILIO_MEDIA_STREAM_AUDIO_FORMAT."
                )

        model_ws = await websockets.connect(
            f"wss://api.openai.com/v1/realtime?model={session_config['model']}",
            additional_headers={"Authorization": f"Bearer {self.config.openai_api_key}"},
        )
        call = self._calls.get(conv_id)
        if call is not None:
            call.model_ws = model_ws
        self.logger.info("Connected to OpenAI Realtime", conversation_id=conv_id)

        await self._model_send(conv_id, {"type": "session.update", "session": session_config})

        # VAD waits for the caller to speak first; request a response to greet.
        response = self.config.welcome_greeting_response
        if response is not None:
            await self._model_send(conv_id, {"type": "response.create", "response": response})

    async def _handle_model_events(self, conv_id: str) -> None:
        """Read OpenAI Realtime events until the socket closes, dispatching each one.

        A failure handling one event (e.g. a malformed delta) is logged and
        skipped rather than ending the loop — only the socket actually
        closing (raised by ``async for`` itself, not from inside it) does.
        """
        call = self._calls.get(conv_id)
        if call is None or call.model_ws is None:
            return
        try:
            async for raw in call.model_ws:
                try:
                    event = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                    session = self.channel._conversations.get(conv_id)
                    if session is None:
                        continue
                    await self._dispatch_model_event(conv_id, session, event)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.error(
                        f"Error handling model event: {e}",
                        conversation_id=conv_id,
                        exc_info=True,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.debug(f"Model read loop ended: {e}", conversation_id=conv_id)

    async def _dispatch_model_event(
        self, conv_id: str, session: ConversationSession, event: dict[str, Any]
    ) -> None:
        event_type = event.get("type")
        call = self._calls.get(conv_id)
        if call is None:
            return

        if event_type == "error":
            error = event.get("error") or {}
            if error.get("code") == "response_cancel_not_active":
                # Benign race: our response.cancel (sent while
                # barge_in.response_active was true) lost to the model's own
                # response.done arriving first. Nothing to cancel anymore,
                # which is exactly what we wanted.
                self.logger.debug(
                    "response.cancel raced response.done", conversation_id=conv_id, error=error
                )
            else:
                self.logger.error(
                    "OpenAI Realtime error event", conversation_id=conv_id, error=error
                )

        elif event_type == "input_audio_buffer.speech_started":
            self.logger.debug("Caller speech detected (VAD)", conversation_id=conv_id)
            await self._handle_barge_in(conv_id, session, call)

        elif event_type == "response.created":
            call.barge_in.response_active = True

        elif event_type == "conversation.item.input_audio_transcription.completed":
            if event.get("transcript"):
                session.metadata.setdefault("transcript", []).append(
                    {"role": "user", "text": event["transcript"]}
                )

        elif event_type == "response.output_item.done":
            # Fires per-item, ahead of response.done — lower tool-call latency.
            # "completed" excludes calls cut short by an interruption, whose
            # arguments JSON may be a truncated fragment.
            item = event.get("item", {})
            if item.get("type") == "function_call" and item.get("status") == "completed":
                await self._handle_function_call(conv_id, item)

        elif event_type == "response.done":
            # last_assistant_item stays set — the model generates faster than
            # realtime, so Twilio may still be playing this when it arrives.
            # response_active clears though: nothing left to cancel.
            call.barge_in.response_active = False
            for item in event.get("response", {}).get("output", []):
                if item.get("role") != "assistant":
                    continue
                for content in item.get("content", []):
                    if content.get("transcript"):
                        session.metadata.setdefault("transcript", []).append(
                            {"role": "assistant", "text": content["transcript"]}
                        )

        elif event_type == "response.output_audio.delta" and event.get("delta"):
            barge_in = call.barge_in
            item_id = event.get("item_id")
            if item_id and item_id == barge_in.muted_item_id:
                # Stale audio for an item already truncated by barge-in.
                return

            if item_id and item_id != barge_in.last_assistant_item:
                barge_in.last_assistant_item = item_id
                barge_in.current_item_audio_ms = 0

            barge_in.current_item_audio_ms += (
                len(base64.b64decode(event["delta"])) // _PCMU_BYTES_PER_MS
            )

            await self._twilio_send(
                conv_id,
                {
                    "event": "media",
                    "streamSid": session.metadata.get("stream_sid"),
                    "media": {"payload": event["delta"]},
                },
            )

    async def _handle_barge_in(
        self, conv_id: str, session: ConversationSession, call: _CallState
    ) -> None:
        """Caller started talking. Cancel any response still generating,
        truncate the model's memory of the last reply at the point actually
        heard, then clear Twilio's buffered audio so playback stops
        immediately. If no assistant audio has been sent since the last
        barge-in, there's nothing queued at Twilio to clear, so this is a
        no-op."""
        barge_in = call.barge_in

        last_assistant_item = barge_in.last_assistant_item
        if last_assistant_item is None:
            self.logger.debug("Barge-in: no assistant item to interrupt", conversation_id=conv_id)
            return

        self.logger.debug("Barge-in: truncating assistant reply", conversation_id=conv_id)
        if barge_in.response_active:
            # Stop the model from generating more of a reply nobody will
            # hear — otherwise it keeps burning tokens on discarded audio.
            # Only send this while a response is actually in flight —
            # response.cancel with nothing to cancel is itself an error event.
            await self._model_send(conv_id, {"type": "response.cancel"})
            barge_in.response_active = False
        await self._model_send(
            conv_id,
            {
                "type": "conversation.item.truncate",
                "item_id": last_assistant_item,
                "content_index": 0,
                # current_item_audio_ms is the exact duration of audio sent
                # for this item, so it never exceeds the item's real content.
                "audio_end_ms": barge_in.current_item_audio_ms,
            },
        )
        await self._twilio_send(
            conv_id, {"event": "clear", "streamSid": session.metadata.get("stream_sid")}
        )

        barge_in.muted_item_id = last_assistant_item
        barge_in.last_assistant_item = None
        barge_in.current_item_audio_ms = 0

    async def _handle_function_call(self, conv_id: str, item: dict[str, Any]) -> None:
        """Run a model-requested tool call and hand the result back.

        Always sends a function_call_output — even a tool that ran
        successfully can return a non-JSON-serializable object (a datetime,
        a Pydantic model, ...), and the model would otherwise be left
        waiting on a call_id it never gets a result for.
        """
        call_id = item.get("call_id")
        name = item.get("name")
        output = await self._run_tool_call(conv_id, name, item.get("arguments"))
        try:
            output_json = json.dumps(output)
        except TypeError as e:
            self.logger.error(
                f"Tool '{name}' returned a non-JSON-serializable result: {e}",
                conversation_id=conv_id,
            )
            output_json = json.dumps(
                {"error": f"Tool '{name}' returned a non-serializable result."}
            )

        await self._model_send(
            conv_id,
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_json,
                },
            },
        )
        await self._model_send(conv_id, {"type": "response.create"})

    async def _run_tool_call(
        self, conv_id: str, name: str | None, arguments_json: str | None
    ) -> object:
        """Look up a model-requested tool by name, run it, and return its output.

        Errors are returned as part of the output rather than raised, so a bad
        tool call doesn't kill the call.
        """
        tool = self._tools_by_name.get(name) if name else None

        self.logger.debug(f"Tool call: {name}({arguments_json})", conversation_id=conv_id)

        if tool is None:
            return {"error": f"Unknown tool '{name}'"}

        try:
            arguments = json.loads(arguments_json or "{}")
            output = await tool(**arguments)
            self.logger.debug(f"Tool result: {name} -> {output}", conversation_id=conv_id)
            return output
        except Exception as e:
            self.logger.error(f"Tool '{name}' failed: {e}", conversation_id=conv_id, exc_info=True)
            return {"error": f"Tool '{name}' failed to execute."}

    async def _model_send(self, conv_id: str, obj: dict[str, Any]) -> None:
        call = self._calls.get(conv_id)
        if call is None or call.model_ws is None:
            return
        try:
            await call.model_ws.send(json.dumps(obj))
        except Exception as e:
            self.logger.debug(f"Failed to send to model: {e}", conversation_id=conv_id)

    async def _twilio_send(self, conv_id: str, obj: dict[str, Any]) -> None:
        call = self._calls.get(conv_id)
        if call is None or call.twilio_ws is None:
            return
        try:
            await call.twilio_ws.send_text(json.dumps(obj))
        except (WebSocketDisconnectError, RuntimeError) as e:
            self.logger.debug(f"Failed to send to Twilio: {e}", conversation_id=conv_id)

    async def _cleanup_call(self, conv_id: str) -> None:
        call = self._calls.pop(conv_id, None)
        if call is not None and call.model_ws is not None:
            try:
                await call.model_ws.close()
            except Exception as e:
                self.logger.debug(f"Error closing model socket: {e}", conversation_id=conv_id)
        await self.channel._end_conversation(conv_id)

    async def send_response(
        self,
        conversation_id: str,
        response: str | AsyncGenerator[str | dict[str, Any], None],
        role: str | None = None,
    ) -> None:
        """Not supported: the model streams audio to Twilio directly, so there
        is no text response for this provider to send."""
        raise NotImplementedError(
            f"{type(self).__name__} produces audio via the model; it has no text send_response."
        )
