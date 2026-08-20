"""OpenAI Realtime (Media Streams) channel: bridges Twilio audio to OpenAI.

This channel connects a phone caller to a speech-to-speech model. It sits
between two WebSockets and relays raw audio in both directions — the model
does its own speech recognition, response generation, speech synthesis, and
turn detection, so this channel never touches text:

    caller  <->  Twilio Media Stream  <->  [this channel]  <->  OpenAI Realtime

Unlike ``OpenAIRealtimeSipChannel`` (which hands the call to OpenAI at the SIP
level, never touching audio), this channel *is* the audio path — Twilio
forwards call audio to our own WebSocket, and we relay it to/from OpenAI's
Realtime WebSocket ourselves. That means everything (transcript, tool
calling, barge-in) happens inline in the same connection, with no separate
sideband channel to open.

How it works:

1. Twilio opens a Media Stream WebSocket to us and sends JSON events:
   ``start`` (call metadata), ``media`` (base64 G.711 u-law audio frames), and
   ``stop``.
2. On ``start`` we open a second WebSocket to the OpenAI Realtime API and
   send whatever session config ``TAC.on_message_ready`` builds for this call
   (as a JSON-encoded string), then spawn a background task to read from it.
3. Each caller ``media`` frame is forwarded to the model as
   ``input_audio_buffer.append``; each model ``response.output_audio.delta``
   is forwarded back to Twilio as a ``media`` frame.
4. When the caller speaks over the assistant, the model emits
   ``input_audio_buffer.speech_started``; we truncate the in-flight reply and
   clear Twilio's playback buffer (barge-in).

State for one call is split two ways:
- Both sockets (Twilio-facing and the outbound OpenAI connection) live in
  ``RealtimeWebSocketManager``, keyed by conversation id — pure connection
  routing, no call-specific data.
- Everything else (barge-in bookkeeping, transcript) lives on the
  ``ConversationSession`` this channel already tracks via
  ``BaseChannel._conversations`` — looked up by conversation id like any
  other channel, not stashed on a channel-specific session class.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

from tac import TAC
from tac.channels.base import BaseChannel
from tac.channels.openai_realtime_media_streams.config import (
    OpenAIRealtimeMediaStreamsChannelConfig,
)
from tac.channels.openai_realtime_media_streams.messages import StreamStartMessage
from tac.channels.openai_realtime_media_streams.twiml import generate_stream_twiml
from tac.channels.openai_realtime_media_streams.websocket_manager import (
    RealtimeWebSocketManager,
)
from tac.channels.websocket_protocol import WebSocketDisconnectError, WebSocketProtocol
from tac.tools import TACTool

try:
    import websockets
except ImportError as e:
    raise ImportError(
        "OpenAIRealtimeMediaStreamsChannel requires the 'websockets' package to connect "
        "to the OpenAI Realtime WebSocket. Install with: "
        "pip install tac[openai-realtime-media-streams]"
    ) from e

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"


class OpenAIRealtimeMediaStreamsChannel(BaseChannel):
    """Voice channel bridging Twilio Media Streams to OpenAI Realtime.

    Framework-agnostic: accepts any ``WebSocketProtocol``. See
    ``OpenAIRealtimeMediaStreamsServer`` for a batteries-included FastAPI host.

    Unlike ``OpenAIRealtimeSipChannel``, there's no accept/reject webhook —
    Twilio hands us the call's audio directly once ``<Connect><Stream>`` TwiML
    connects it, and the channel's job is entirely the audio bridge. What the
    session looks like is supplied by ``TAC.on_message_ready`` (reused here —
    see ``_connect_model``), not this channel's config.
    """

    def __init__(
        self,
        tac: TAC,
        config: OpenAIRealtimeMediaStreamsChannelConfig | dict[str, Any] | None = None,
    ) -> None:
        if isinstance(config, dict):
            config = OpenAIRealtimeMediaStreamsChannelConfig(**config)
        elif config is None:
            config = OpenAIRealtimeMediaStreamsChannelConfig()

        super().__init__(tac)

        if not config.openai_api_key:
            raise ValueError(
                "openai_api_key is required. Set the OPENAI_API_KEY environment "
                "variable or provide openai_api_key in OpenAIRealtimeMediaStreamsChannelConfig."
            )

        self.config = config
        self._tools_by_name: dict[str, TACTool] = {tool.name: tool for tool in config.tools}
        self._sockets = RealtimeWebSocketManager()

    def get_channel_name(self) -> str:
        return "OPENAI_REALTIME_MEDIA_STREAMS"

    def build_stream_twiml(self, websocket_url: str) -> str:
        """Return the ``<Connect><Stream>`` TwiML that points Twilio at this channel."""
        return generate_stream_twiml(websocket_url)

    def get_transcript(self, conversation_id: str) -> list[dict[str, str]]:
        """Return the transcript captured so far for an in-progress call.

        Once the call ends, the transcript stays on
        ``ConversationSession.metadata["transcript"]`` for ``on_conversation_ended``.
        """
        session = self._conversations.get(conversation_id)
        if session is None:
            return []
        return list(session.metadata.get("transcript", []))

    async def handle_websocket(self, websocket: WebSocketProtocol) -> None:
        """Drive one Twilio Media Stream connection from accept to disconnect."""
        await websocket.accept()

        conv_id: str | None = None
        model_reader: asyncio.Task[None] | None = None

        try:
            while True:
                data = await websocket.receive_json()
                event = data.get("event")

                if event == "start":
                    conv_id = self._register_call(data.get("start") or {})
                    self._sockets.add_twilio_socket(conv_id, websocket)
                    await self._connect_model(conv_id)
                    model_reader = asyncio.create_task(self._handle_model_events(conv_id))

                elif event == "media":
                    media = data.get("media") or {}
                    if conv_id is not None:
                        session = self._conversations.get(conv_id)
                        if session is not None:
                            session.metadata["latest_media_timestamp"] = int(
                                media.get("timestamp", 0)
                            )
                        if media.get("payload"):
                            await self._model_send(
                                conv_id,
                                {"type": "input_audio_buffer.append", "audio": media["payload"]},
                            )

                elif event == "stop":
                    self.logger.info("Media stream stopped", conversation_id=conv_id)
                    break

        except WebSocketDisconnectError:
            self.logger.info("Media stream WebSocket closed", conversation_id=conv_id)
        except Exception as e:
            self.logger.error(f"Media stream WebSocket error: {e}", exc_info=True)
        finally:
            if model_reader is not None:
                model_reader.cancel()
            if conv_id is not None:
                await self._sockets.pop_sockets(conv_id)
                await self._end_conversation(conv_id)

    def _register_call(self, start: dict[str, Any]) -> str:
        """Handle Twilio's ``start`` event: register the conversation and seed
        its metadata. Returns the conversation id.
        """
        message = StreamStartMessage(**start)
        conv_id = message.conversation_id

        session = self._start_conversation(conv_id, profile_id=None)
        session.metadata.update(
            {
                "stream_sid": message.stream_sid,
                "transcript": [],
                "latest_media_timestamp": 0,
                "response_start_timestamp": None,
                "last_assistant_item": None,
                "muted_item_id": None,
                "response_active": False,
            }
        )

        self.logger.info(
            "Media stream started",
            conversation_id=conv_id,
            media_format=message.media_format,
        )
        return conv_id

    async def _connect_model(self, conv_id: str) -> None:
        """Open the OpenAI Realtime WebSocket and configure the session."""
        session = self._conversations[conv_id]

        model_ws = await websockets.connect(
            f"{OPENAI_REALTIME_URL}?model={self.config.model}",
            additional_headers={"Authorization": f"Bearer {self.config.openai_api_key}"},
        )
        self._sockets.add_model_socket(conv_id, model_ws)
        self.logger.info(
            "Connected to OpenAI Realtime", conversation_id=conv_id, model=self.config.model
        )

        # What the session looks like (model, voice, turn detection,
        # instructions, tools) is entirely the application's call. Reuses
        # TAC.on_message_ready (user_message="", memory_response=None — neither
        # applies to a Realtime session) rather than a separate callback; the
        # handler must return a JSON-encoded session dict, since
        # trigger_message_ready's contract is str | None, not dict.
        session_config_json = await self.tac.trigger_message_ready("", session, None)
        if not session_config_json:
            raise RuntimeError(
                "on_message_ready returned nothing — OpenAIRealtimeMediaStreamsChannel "
                "needs it to return a JSON-encoded session config dict."
            )
        session_config = json.loads(session_config_json)
        await self._model_send(conv_id, {"type": "session.update", "session": session_config})

        # With VAD the model waits for the caller to speak. To greet first,
        # explicitly request a response; the per-response instructions steer
        # only this turn.
        if self.config.welcome_greeting:
            await self._model_send(
                conv_id,
                {
                    "type": "response.create",
                    "response": {
                        "instructions": (
                            f"Greet the caller in English by saying: {self.config.welcome_greeting}"
                        )
                    },
                },
            )

    async def _handle_model_events(self, conv_id: str) -> None:
        """Read OpenAI events until the model socket closes, forwarding assistant
        audio to Twilio and handling barge-in."""
        model_ws = self._sockets.get_model_socket(conv_id)
        if model_ws is None:
            return
        try:
            async for raw in model_ws:
                event = json.loads(raw)
                event_type = event.get("type")
                session = self._conversations.get(conv_id)
                if session is None:
                    continue

                if event_type == "error":
                    self.logger.error(
                        "OpenAI Realtime error event",
                        conversation_id=conv_id,
                        error=event.get("error"),
                    )

                elif event_type == "input_audio_buffer.speech_started":
                    self.logger.info("Caller speech detected (VAD)", conversation_id=conv_id)
                    await self._handle_barge_in(conv_id)

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    if event.get("transcript"):
                        session.metadata.setdefault("transcript", []).append(
                            {"role": "user", "text": event["transcript"]}
                        )

                elif event_type == "response.done":
                    # Deliberately NOT resetting last_assistant_item/response_start_timestamp
                    # here: response.done means the model finished *generating* this reply,
                    # but Twilio can still be several seconds into *playing back* audio we
                    # already forwarded it. Resetting now would make a barge-in landing in
                    # that gap think there's nothing to truncate/clear, so the stale audio
                    # already sitting in Twilio's playback buffer would just keep playing.
                    # The next response's first delta below re-anchors this state instead.
                    # response_active does flip off, though: a barge-in landing in this gap
                    # should only clear Twilio's leftover buffer, not truncate — the model
                    # already finished this item, so there's nothing left to cut short.
                    session.metadata["response_active"] = False

                    for item in event.get("response", {}).get("output", []):
                        if item.get("type") == "function_call":
                            await self._handle_function_call(conv_id, item)
                            continue
                        if item.get("role") != "assistant":
                            continue
                        for content in item.get("content", []):
                            if content.get("transcript"):
                                session.metadata.setdefault("transcript", []).append(
                                    {"role": "assistant", "text": content["transcript"]}
                                )

                elif event_type == "response.output_audio.delta" and event.get("delta"):
                    item_id = event.get("item_id")
                    if item_id and item_id == session.metadata.get("muted_item_id"):
                        # Stale audio for a response we already truncated on barge-in —
                        # the model hadn't fully stopped generating it yet. Drop it so
                        # the assistant doesn't keep audibly talking after the interrupt.
                        continue

                    if item_id and item_id != session.metadata.get("last_assistant_item"):
                        # First delta of a new item: (re)anchor barge-in timing here,
                        # keyed off the item actually changing rather than trusting
                        # response_start_timestamp is None (see response.done above).
                        session.metadata["last_assistant_item"] = item_id
                        session.metadata["response_start_timestamp"] = session.metadata.get(
                            "latest_media_timestamp", 0
                        )
                        session.metadata["response_active"] = True

                    await self._twilio_send(
                        conv_id,
                        {
                            "event": "media",
                            "streamSid": session.metadata.get("stream_sid"),
                            "media": {"payload": event["delta"]},
                        },
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.debug(f"Model read loop ended: {e}", conversation_id=conv_id)

    async def _handle_barge_in(self, conv_id: str) -> None:
        """Caller started talking. If a reply is still actively streaming, tell
        the model to truncate it at the point actually heard. Either way, clear
        Twilio's buffered audio so playback stops immediately."""
        session = self._conversations.get(conv_id)
        if session is None:
            return

        last_assistant_item = session.metadata.get("last_assistant_item")
        if last_assistant_item is None:
            self.logger.debug("Barge-in: no assistant item to interrupt", conversation_id=conv_id)
            return

        response_start_timestamp = session.metadata.get("response_start_timestamp")
        if session.metadata.get("response_active") and response_start_timestamp is not None:
            played_ms = max(
                session.metadata.get("latest_media_timestamp", 0) - response_start_timestamp, 0
            )
            self.logger.info("Barge-in: truncating assistant reply", conversation_id=conv_id)
            await self._model_send(
                conv_id,
                {
                    "type": "conversation.item.truncate",
                    "item_id": last_assistant_item,
                    "content_index": 0,
                    "audio_end_ms": played_ms,
                },
            )
        else:
            # The model already finished generating this item — nothing left to
            # truncate server-side. Just clear whatever Twilio hasn't played yet.
            self.logger.info(
                "Barge-in: reply already finished, clearing leftover playback",
                conversation_id=conv_id,
            )
        await self._twilio_send(
            conv_id, {"event": "clear", "streamSid": session.metadata.get("stream_sid")}
        )

        session.metadata["muted_item_id"] = last_assistant_item
        session.metadata["last_assistant_item"] = None
        session.metadata["response_start_timestamp"] = None
        session.metadata["response_active"] = False

    async def _handle_function_call(self, conv_id: str, item: dict[str, Any]) -> None:
        """Run a model-requested tool call and hand the result back.

        ``item`` is a ``function_call`` entry from ``response.done``'s output
        (has ``name``, ``call_id``, and a JSON-encoded ``arguments`` string).
        Looks the tool up by name, runs it, and reports the outcome back via
        ``function_call_output`` + ``response.create`` so the model continues
        the turn with the result (errors are reported to the model rather than
        raised, so a bad tool call doesn't kill the call).
        """
        name = item.get("name")
        call_id = item.get("call_id")
        tool = self._tools_by_name.get(name) if name else None

        self.logger.info(f"Tool call: {name}({item.get('arguments')})", conversation_id=conv_id)

        output: object
        if tool is None:
            output = {"error": f"Unknown tool '{name}'"}
        else:
            try:
                arguments = json.loads(item.get("arguments") or "{}")
                output = await tool(**arguments)
                self.logger.info(f"Tool result: {name} -> {output}", conversation_id=conv_id)
            except Exception as e:
                self.logger.error(
                    f"Tool '{name}' failed: {e}", conversation_id=conv_id, exc_info=True
                )
                output = {"error": str(e)}

        await self._model_send(
            conv_id,
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(output),
                },
            },
        )
        await self._model_send(conv_id, {"type": "response.create"})

    # -- Low-level sends (each swallows a dead-socket write, logging at debug) --

    async def _model_send(self, conv_id: str, obj: dict[str, Any]) -> None:
        model_ws = self._sockets.get_model_socket(conv_id)
        if model_ws is None:
            return
        try:
            await model_ws.send(json.dumps(obj))
        except Exception as e:
            self.logger.debug(f"Failed to send to model: {e}", conversation_id=conv_id)

    async def _twilio_send(self, conv_id: str, obj: dict[str, Any]) -> None:
        websocket = self._sockets.get_twilio_socket(conv_id)
        if websocket is None:
            return
        try:
            await websocket.send_text(json.dumps(obj))
        except (WebSocketDisconnectError, RuntimeError) as e:
            self.logger.debug(f"Failed to send to Twilio: {e}", conversation_id=conv_id)

    # -- BaseChannel abstract methods that don't apply to the audio path ----

    async def process_webhook(
        self, webhook_data: dict[str, Any], idempotency_token: str | None = None
    ) -> None:
        """No-op: lifecycle is driven entirely by the Media Stream WebSocket."""
        return None

    async def send_response(
        self,
        conversation_id: str,
        response: str | AsyncGenerator[str | dict[str, Any], None],
        role: str | None = None,
    ) -> None:
        """Not supported: the model streams audio to Twilio directly, so there is
        no text response for the channel to send."""
        raise NotImplementedError(
            "OpenAIRealtimeMediaStreamsChannel produces audio via the model; it has "
            "no text send_response. Use VoiceChannel (ConversationRelay) or "
            "OpenAIRealtimeSipChannel for text-based/webhook-based responses."
        )
