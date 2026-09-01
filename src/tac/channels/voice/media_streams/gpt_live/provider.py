"""``GPTLiveProvider``: bridges Twilio Media Streams to OpenAI's GPT-Live alpha.

GPT-Live is an unreleased OpenAI alpha API.

Unlike ``OpenAIRealtimeProvider``, GPT-Live is full-duplex — there's no
client-driven barge-in truncate, the model handles interruption itself — and
tool calls go through Responses delegation instead of direct function-calling
events.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from typing import TYPE_CHECKING, Any

import websockets

from tac.channels.voice.media_streams.gpt_live.models import _CallState
from tac.channels.voice.media_streams.shared.openai_provider import (
    OPENAI_USER_AGENT,
    MediaStreamsOpenAIProvider,
)
from tac.channels.websocket_protocol import WebSocketDisconnectError, WebSocketProtocol
from tac.models.outbound import (
    InitiateVoiceConversationOptions,
    InitiateVoiceConversationOptionsGPTLive,
    InitiateVoiceConversationResult,
)
from tac.models.session import ConversationSession
from tac.models.stream import StreamStartMessage
from tac.models.voice import VoiceTwiMLOptionsMediaStreams
from tac.utils.redaction import mask_phone, redact_twiml_parameters

if TYPE_CHECKING:
    from tac.channels.voice.media_streams.gpt_live.config import GPTLiveProviderConfig

# TODO: remove once GPT-Live is released (GA drops the alpha header requirement).
#: Required on every GPT-Live alpha request — omitting it is rejected.
_OPENAI_ALPHA_HEADER_NAME = "OpenAI-Alpha"
_OPENAI_ALPHA_HEADER_VALUE = "quicksilver=v2"

#: Reserved <Stream> custom_parameters key used to correlate an outbound
#: call's session_config override to its WebSocket start event. calls.create()
#: returning call.sid doesn't happen-before Twilio connecting the stream, so
#: call.sid can't be the correlation key — this token, embedded in the TwiML
#: before the call is placed, can.
_SESSION_CONFIG_TOKEN_PARAM = "_tac_session_config_token"

#: GPT-Live speaks Twilio's exact wire format natively — unlike the Realtime
#: API's separate session.audio.input/output.format, this is a single shared
#: session.audio.format, selected once at WebSocket startup and immutable
#: after. No transcoding needed on either leg between Twilio and GPT-Live.
TWILIO_MEDIA_STREAM_AUDIO_FORMAT: dict[str, Any] = {"type": "audio/pcmu", "rate": 8000}


class GPTLiveProvider(MediaStreamsOpenAIProvider[_CallState]):
    """``VoiceProvider`` bridging Twilio Media Streams to OpenAI's GPT-Live alpha.

    Example:
        ```python
        channel = VoiceChannel(tac, config=GPTLiveProviderConfig(default_session_config=...))
        ```
    """

    config: GPTLiveProviderConfig

    @property
    def channel_name(self) -> str:
        return "VOICE_MEDIA_STREAM_OPENAI_GPT_LIVE"

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

        Pass ``InitiateVoiceConversationOptionsGPTLive`` with ``session_config``
        set to override the default for this call.
        """
        twiml_options = options.twiml_options
        if twiml_options is not None and not isinstance(
            twiml_options, VoiceTwiMLOptionsMediaStreams
        ):
            raise TypeError(
                "GPTLiveProvider.initiate_outbound_conversation requires "
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
            if isinstance(options, InitiateVoiceConversationOptionsGPTLive)
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

        Races the Twilio read against the GPT-Live model-event reader so that
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
                    if conv_id is not None and media.get("payload"):
                        await self._model_send(
                            conv_id,
                            {"type": "input_audio.append", "audio": media["payload"]},
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
        """Open the GPT-Live WebSocket and send the session config.

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
        audio_format = (session_config.get("audio") or {}).get("format")
        if audio_format != TWILIO_MEDIA_STREAM_AUDIO_FORMAT:
            raise ValueError(
                f"session_config for call {conv_id} has audio.format={audio_format!r} — "
                f"Twilio Media Streams always sends/expects {TWILIO_MEDIA_STREAM_AUDIO_FORMAT!r}, "
                "this isn't configurable. Set audio.format to TWILIO_MEDIA_STREAM_AUDIO_FORMAT."
            )
        if "model" in session_config:
            # The model is selected via the WebSocket URL's ?model= query
            # param; GPT-Live rejects repeating it in the initial session.update.
            raise ValueError(
                f"session_config for call {conv_id} must not include 'model' — GPT-Live "
                "selects it from the ?model= query param (GPTLiveProviderConfig.model), "
                "not from session_config."
            )

        model_ws = await websockets.connect(
            f"wss://api.openai.com/v1/live?model={self.config.model}",
            additional_headers={
                "Authorization": f"Bearer {self.config.openai_api_key}",
                "User-Agent": OPENAI_USER_AGENT,
                _OPENAI_ALPHA_HEADER_NAME: _OPENAI_ALPHA_HEADER_VALUE,
            },
        )
        call = self._calls.get(conv_id)
        if call is not None:
            call.model_ws = model_ws
        self.logger.info("Connected to GPT-Live", conversation_id=conv_id)

        await self._model_send(conv_id, {"type": "session.update", "session": session_config})
        # welcome_instruction is sent once session.started arrives — see _dispatch_model_event.

    async def _dispatch_model_event(
        self, conv_id: str, session: ConversationSession, event: dict[str, Any]
    ) -> None:
        event_type = event.get("type")

        if event_type == "error":
            self.logger.error(
                "GPT-Live error event", conversation_id=conv_id, error=event.get("error")
            )

        elif event_type == "session.started":
            # session.context.append before this event is undocumented behavior.
            instruction = self.config.welcome_instruction
            if instruction is not None:
                # Sent verbatim — caller must word it as an instruction, not a
                # bare greeting, or the model won't speak first.
                await self._model_send(
                    conv_id,
                    {
                        "type": "session.context.append",
                        "channel": "speakable",
                        "content": [{"type": "input_text", "text": instruction}],
                    },
                )

        elif event_type == "turn.done":
            # One assembled entry per turn — unlike Realtime's fragment-level
            # *_transcript.added events.
            turn = event.get("turn") or {}
            text = turn.get("transcript")
            role = turn.get("role")
            if text and role:
                session.metadata.setdefault("transcript", []).append({"role": role, "text": text})

        elif event_type == "output_audio.delta" and event.get("audio"):
            # No item_id/barge-in bookkeeping needed — GPT-Live is
            # full-duplex and handles interruption server-side.
            await self._twilio_send(
                conv_id,
                {
                    "event": "media",
                    "streamSid": session.metadata.get("stream_sid"),
                    "media": {"payload": event["audio"]},
                },
            )

        elif event_type == "response.output_item.done":
            # "completed" excludes calls cut short mid-generation.
            item = event.get("item") or {}
            if item.get("type") == "function_call" and item.get("status") == "completed":
                await self._handle_function_call(conv_id, item)

    async def _handle_function_call(self, conv_id: str, item: dict[str, Any]) -> None:
        """Run a Responses-delegated tool call and hand the result back.

        Unlike Realtime, no follow-up ``response.create`` is sent — GPT-Live
        continues on its own once the delegation output is acknowledged.
        Always sends a function_call_output when call_id is present — even a
        tool that ran successfully can return a non-JSON-serializable object
        (a datetime, a Pydantic model, ...), and the model would otherwise be
        left waiting on a call_id it never gets a result for. Without a
        call_id there's nothing to reply to, so the item is dropped instead.
        """
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            self.logger.error(
                "Received malformed function_call item without call_id",
                conversation_id=conv_id,
                item=item,
            )
            return

        name = item.get("name")
        if not isinstance(name, str) or not name:
            self.logger.error(
                "Received malformed function_call item without tool name",
                conversation_id=conv_id,
                call_id=call_id,
                item=item,
            )
            output_json = json.dumps({"error": "Malformed function call: missing tool name."})
        else:
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
                "type": "delegation.function_call_output.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_json,
                },
            },
        )

    async def _cleanup_call(self, conv_id: str) -> None:
        call = self._calls.pop(conv_id, None)
        if call is not None and call.model_ws is not None:
            # Known limitation: doesn't wait for session.closed before
            # closing the socket out from under it.
            with contextlib.suppress(Exception):
                await call.model_ws.send(json.dumps({"type": "session.close"}))
            try:
                await call.model_ws.close()
            except Exception as e:
                self.logger.debug(f"Error closing model socket: {e}", conversation_id=conv_id)
        await self.channel._end_conversation(conv_id)
