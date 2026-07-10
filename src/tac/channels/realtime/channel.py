"""Realtime voice channel: bridges Twilio Media Streams to the OpenAI Realtime API.

This channel connects a phone caller to a speech-to-speech model. It sits between
two WebSockets and relays raw audio in both directions — the model does its own
speech recognition, response generation, speech synthesis, and turn detection, so
this channel never touches text:

    caller  ⇄  Twilio Media Stream  ⇄  [this channel]  ⇄  OpenAI Realtime

How it works:

1. Twilio opens a Media Stream WebSocket to us and sends JSON events:
   ``start`` (call metadata), ``media`` (base64 G.711 u-law audio frames), and
   ``stop``. Other events (``connected``, ``mark``, …) are ignored.
2. On ``start`` we open a second WebSocket to the OpenAI Realtime API, configure
   the session (u-law audio, server-side turn detection), and spawn a background
   task to read from it.
3. Each caller ``media`` frame is forwarded to the model as
   ``input_audio_buffer.append``; each model ``response.output_audio.delta`` is
   forwarded back to Twilio as a ``media`` frame.
4. When the caller speaks over the assistant, the model emits
   ``input_audio_buffer.speech_started``; we truncate the in-flight reply and
   clear Twilio's playback buffer (barge-in).
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from tac.channels.base import BaseChannel
from tac.channels.realtime.config import TWILIO_AUDIO_FORMAT, RealtimeVoiceChannelConfig
from tac.channels.realtime.twiml import generate_stream_twiml
from tac.channels.websocket_protocol import WebSocketDisconnectError, WebSocketProtocol
from tac.core.tac import TAC

try:
    from websockets.asyncio.client import connect as _ws_connect
except ImportError:  # pragma: no cover - exercised only without the extra
    _ws_connect = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from websockets.asyncio.client import ClientConnection

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"


class _RealtimeCallSession:
    """Mutable state for one bridged call (one Twilio WebSocket connection).

    One instance per connection, so concurrent calls never share state.
    """

    def __init__(self, twilio_ws: WebSocketProtocol) -> None:
        self.twilio_ws = twilio_ws
        self.model_ws: ClientConnection | None = None
        self.conv_id: str | None = None
        self.stream_sid: str | None = None

        # Barge-in bookkeeping. played_ms = latest_media_timestamp - response_start_timestamp.
        self.latest_media_timestamp: int = 0
        self.response_start_timestamp: int | None = None
        self.last_assistant_item: str | None = None


class RealtimeVoiceChannel(BaseChannel):
    """Voice channel bridging Twilio Media Streams to OpenAI Realtime.

    Framework-agnostic: accepts any :class:`WebSocketProtocol`. See
    ``RealtimeVoiceServer`` for a batteries-included FastAPI host.
    """

    def __init__(
        self,
        tac: TAC,
        config: RealtimeVoiceChannelConfig | dict[str, Any],
    ) -> None:
        if isinstance(config, dict):
            config = RealtimeVoiceChannelConfig(**config)
        super().__init__(tac)
        self.config = config

    def get_channel_name(self) -> str:
        return "voice"

    def build_stream_twiml(self, websocket_url: str) -> str:
        """Return the ``<Connect><Stream>`` TwiML that points Twilio at this channel."""
        return generate_stream_twiml(websocket_url)

    async def handle_websocket(self, websocket: WebSocketProtocol) -> None:
        """Drive one Twilio Media Stream connection from accept to disconnect."""
        await websocket.accept()

        session = _RealtimeCallSession(websocket)
        model_reader: asyncio.Task[None] | None = None

        try:
            while True:
                data = await websocket.receive_json()
                event = data.get("event")

                if event == "start":
                    model_reader = await self._on_stream_start(session, data.get("start") or {})

                elif event == "media":
                    media = data.get("media") or {}
                    session.latest_media_timestamp = int(media.get("timestamp", 0))
                    if media.get("payload") and session.model_ws is not None:
                        await self._model_send(
                            session,
                            {"type": "input_audio_buffer.append", "audio": media["payload"]},
                        )

                elif event == "stop":
                    self.logger.info(
                        "Realtime voice stream stopped", conversation_id=session.conv_id
                    )
                    break

        except WebSocketDisconnectError:
            self.logger.info("Realtime voice WebSocket closed", conversation_id=session.conv_id)
        except Exception as e:
            self.logger.error(f"Realtime voice WebSocket error: {e}", exc_info=True)
        finally:
            if model_reader is not None:
                model_reader.cancel()
            if session.model_ws is not None:
                try:
                    await session.model_ws.close()
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass
            if session.conv_id and session.conv_id in self._conversations:
                await self._end_conversation(session.conv_id)

    async def _on_stream_start(
        self, session: _RealtimeCallSession, start: dict[str, Any]
    ) -> asyncio.Task[None]:
        """Handle Twilio's ``start`` event: register the conversation, connect the
        model, and return the background task that relays model audio to Twilio."""
        session.stream_sid = start.get("streamSid")
        session.conv_id = start.get("callSid") or session.stream_sid or "unknown-call"
        self._start_conversation(session.conv_id, profile_id=None)

        self.logger.info(
            "Realtime voice stream started",
            conversation_id=session.conv_id,
            media_format=start.get("mediaFormat"),
        )

        await self._connect_model(session)
        return asyncio.create_task(self._pump_model_to_twilio(session))

    async def _connect_model(self, session: _RealtimeCallSession) -> None:
        """Open the OpenAI Realtime WebSocket and configure the session."""
        if _ws_connect is None:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "RealtimeVoiceChannel needs the 'websockets' package. "
                "Install it with: pip install 'twilio-agent-connect[realtime]'"
            )

        # No "OpenAI-Beta" header: it selects the old beta request shape, which
        # gpt-realtime rejects with close code 4000 (beta_api_shape_disabled).
        session.model_ws = await _ws_connect(
            f"{OPENAI_REALTIME_URL}?model={self.config.model}",
            additional_headers={"Authorization": f"Bearer {self.config.openai_api_key}"},
        )

        # TWILIO_AUDIO_FORMAT is u-law, matching Twilio's stream, so no transcoding.
        await self._model_send(
            session,
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": self.config.model,
                    "output_modalities": ["audio"],
                    "instructions": self.config.instructions,
                    "audio": {
                        "input": {
                            "format": TWILIO_AUDIO_FORMAT,
                            "turn_detection": {"type": "server_vad"},
                            "transcription": {"model": "whisper-1"},
                        },
                        "output": {
                            "format": TWILIO_AUDIO_FORMAT,
                            "voice": self.config.voice,
                        },
                    },
                },
            },
        )

        # With server VAD the model waits for the caller to speak. To greet first,
        # explicitly request a response; the per-response instructions steer only
        # this turn.
        if self.config.welcome_greeting:
            await self._model_send(
                session,
                {
                    "type": "response.create",
                    "response": {
                        "instructions": (
                            f"Greet the caller in English by saying: {self.config.welcome_greeting}"
                        )
                    },
                },
            )

    async def _pump_model_to_twilio(self, session: _RealtimeCallSession) -> None:
        """Read OpenAI events until the model socket closes, forwarding assistant
        audio to Twilio and handling barge-in."""
        if session.model_ws is None:
            return
        try:
            async for raw in session.model_ws:
                event = json.loads(raw)
                event_type = event.get("type")

                if event_type == "error":
                    self.logger.error(
                        "OpenAI Realtime error event",
                        conversation_id=session.conv_id,
                        error=event.get("error"),
                    )

                elif event_type == "input_audio_buffer.speech_started":
                    await self._handle_barge_in(session)

                elif event_type == "response.done":
                    # Reply finished on its own; clear barge-in tracking so the
                    # next turn measures playback from its own start.
                    session.last_assistant_item = None
                    session.response_start_timestamp = None

                elif event_type == "response.output_audio.delta" and event.get("delta"):
                    # Track the item and its start time so barge-in can truncate it.
                    if session.response_start_timestamp is None:
                        session.response_start_timestamp = session.latest_media_timestamp
                    if event.get("item_id"):
                        session.last_assistant_item = event["item_id"]

                    await self._twilio_send(
                        session,
                        {
                            "event": "media",
                            "streamSid": session.stream_sid,
                            "media": {"payload": event["delta"]},
                        },
                    )
                    await self._twilio_send(
                        session, {"event": "mark", "streamSid": session.stream_sid}
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.debug(f"Model read loop ended: {e}", conversation_id=session.conv_id)

    async def _handle_barge_in(self, session: _RealtimeCallSession) -> None:
        """Caller started talking mid-reply. Tell the model to truncate the
        in-flight item at the point actually heard, and clear Twilio's buffered
        audio so playback stops immediately."""
        if session.last_assistant_item is None or session.response_start_timestamp is None:
            return

        played_ms = session.latest_media_timestamp - session.response_start_timestamp
        await self._model_send(
            session,
            {
                "type": "conversation.item.truncate",
                "item_id": session.last_assistant_item,
                "content_index": 0,
                "audio_end_ms": max(played_ms, 0),
            },
        )
        await self._twilio_send(session, {"event": "clear", "streamSid": session.stream_sid})

        session.last_assistant_item = None
        session.response_start_timestamp = None

    # -- Low-level sends (each swallows a dead-socket write, logging at debug) --

    async def _model_send(self, session: _RealtimeCallSession, obj: dict[str, Any]) -> None:
        if session.model_ws is None:
            return
        try:
            await session.model_ws.send(json.dumps(obj))
        except Exception as e:
            self.logger.debug(f"Failed to send to model: {e}", conversation_id=session.conv_id)

    async def _twilio_send(self, session: _RealtimeCallSession, obj: dict[str, Any]) -> None:
        try:
            await session.twilio_ws.send_text(json.dumps(obj))
        except (WebSocketDisconnectError, RuntimeError) as e:
            self.logger.debug(f"Failed to send to Twilio: {e}", conversation_id=session.conv_id)

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
            "RealtimeVoiceChannel produces audio via the model; it has no text send_response. "
            "Use VoiceChannel (ConversationRelay) for text-based responses."
        )
