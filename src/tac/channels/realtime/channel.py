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
from tac.tools.base import TACTool

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
        # True while last_assistant_item is still actively streaming (between its
        # first delta and response.done). Only truncate while this is true — once
        # the model has finished generating, it already knows the reply in full,
        # so there's nothing left to truncate; a barge-in in that gap should only
        # clear Twilio's leftover playback buffer, not ask the model to cut a
        # completed item short (which OpenAI rejects: "Audio content of Xms is
        # already shorter than Yms").
        self.response_active: bool = False

        # item_id truncated by the most recent barge-in. The model may already have
        # more of that (now-cancelled) response queued up server-side, so late
        # response.output_audio.delta events for this item still arrive after we've
        # told Twilio to stop playing it — drop them instead of re-forwarding.
        self.muted_item_id: str | None = None

        # Turn-by-turn text transcript, built from the model's transcription
        # events. Each entry is {"role": "user" | "assistant", "text": str}.
        self.transcript: list[dict[str, str]] = []


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
        self._tools_by_name: dict[str, TACTool] = {tool.name: tool for tool in config.tools}

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
                # Stash the collected transcript on the session so it's available
                # to the on_conversation_ended callback. This keeps history in
                # memory for now; persisting it (Conversation Orchestrator, a
                # database, etc.) is left to that callback.
                self._conversations[session.conv_id].metadata["transcript"] = session.transcript
                self.logger.info(
                    "Realtime voice transcript",
                    conversation_id=session.conv_id,
                    transcript=session.transcript,
                )
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
        self.logger.info(
            "Connected to OpenAI Realtime", conversation_id=session.conv_id, model=self.config.model
        )

        session_config: dict[str, Any] = {
            "type": "realtime",
            "model": self.config.model,
            "output_modalities": ["audio"],
            "instructions": self.config.instructions,
            "audio": {
                "input": {
                    # TWILIO_AUDIO_FORMAT is u-law, matching Twilio's stream, so no transcoding.
                    "format": TWILIO_AUDIO_FORMAT,
                    "turn_detection": {"type": "server_vad"},
                    "transcription": {"model": "whisper-1"},
                },
                "output": {
                    "format": TWILIO_AUDIO_FORMAT,
                    "voice": self.config.voice,
                },
            },
        }
        if self.config.tools:
            session_config["tools"] = [tool.to_realtime_format() for tool in self.config.tools]
            session_config["tool_choice"] = "auto"

        await self._model_send(session, {"type": "session.update", "session": session_config})

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
                    self.logger.info(
                        "Caller speech detected (VAD)", conversation_id=session.conv_id
                    )
                    await self._handle_barge_in(session)

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    if event.get("transcript"):
                        session.transcript.append({"role": "user", "text": event["transcript"]})

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
                    session.response_active = False

                    # response.done carries the full per-turn text already, so
                    # there's no need to accumulate response.output_audio_transcript.delta.
                    for item in event.get("response", {}).get("output", []):
                        if item.get("type") == "function_call":
                            await self._handle_function_call(session, item)
                            continue
                        if item.get("role") != "assistant":
                            continue
                        for content in item.get("content", []):
                            if content.get("transcript"):
                                session.transcript.append(
                                    {"role": "assistant", "text": content["transcript"]}
                                )

                elif event_type == "response.output_audio.delta" and event.get("delta"):
                    item_id = event.get("item_id")
                    if item_id and item_id == session.muted_item_id:
                        # Stale audio for a response we already truncated on barge-in —
                        # the model hadn't fully stopped generating it yet. Drop it so
                        # the assistant doesn't keep audibly talking after the interrupt.
                        continue

                    if item_id and item_id != session.last_assistant_item:
                        # First delta of a new item: (re)anchor barge-in timing here,
                        # keyed off the item actually changing rather than trusting
                        # response_start_timestamp is None (see response.done above).
                        session.last_assistant_item = item_id
                        session.response_start_timestamp = session.latest_media_timestamp
                        session.response_active = True

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
        """Caller started talking. If a reply is still actively streaming, tell
        the model to truncate it at the point actually heard. Either way, clear
        Twilio's buffered audio so playback stops immediately."""
        if session.last_assistant_item is None:
            self.logger.debug(
                "Barge-in: no assistant item to interrupt", conversation_id=session.conv_id
            )
            return

        if session.response_active and session.response_start_timestamp is not None:
            played_ms = session.latest_media_timestamp - session.response_start_timestamp
            self.logger.info(
                "Barge-in: truncating assistant reply", conversation_id=session.conv_id
            )
            await self._model_send(
                session,
                {
                    "type": "conversation.item.truncate",
                    "item_id": session.last_assistant_item,
                    "content_index": 0,
                    "audio_end_ms": max(played_ms, 0),
                },
            )
        else:
            # The model already finished generating this item — nothing left to
            # truncate server-side. Just clear whatever Twilio hasn't played yet.
            self.logger.info(
                "Barge-in: reply already finished, clearing leftover playback",
                conversation_id=session.conv_id,
            )
        await self._twilio_send(session, {"event": "clear", "streamSid": session.stream_sid})

        session.muted_item_id = session.last_assistant_item
        session.last_assistant_item = None
        session.response_start_timestamp = None
        session.response_active = False

    async def _handle_function_call(
        self, session: _RealtimeCallSession, item: dict[str, Any]
    ) -> None:
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
        self.logger.info(
            f"Tool call: {name}({item.get('arguments')})", conversation_id=session.conv_id
        )

        output: object
        if tool is None:
            output = {"error": f"Unknown tool '{name}'"}
        else:
            try:
                arguments = json.loads(item.get("arguments") or "{}")
                output = await tool(**arguments)
                self.logger.info(
                    f"Tool result: {name} -> {output}", conversation_id=session.conv_id
                )
            except Exception as e:
                self.logger.error(
                    f"Tool '{name}' failed: {e}",
                    conversation_id=session.conv_id,
                    exc_info=True,
                )
                output = {"error": str(e)}

        await self._model_send(
            session,
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(output),
                },
            },
        )
        await self._model_send(session, {"type": "response.create"})

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
