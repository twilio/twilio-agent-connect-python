"""``MediaStreamsOpenAIProvider``: pieces shared verbatim by every ``VoiceProvider``
bridging Twilio Media Streams to an OpenAI real-time voice API.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from tac import __version__
from tac.channels.voice.media_streams.shared.config import MediaStreamsOpenAIProviderConfig
from tac.channels.voice.media_streams.shared.models import MediaStreamsOpenAICallState
from tac.channels.voice.media_streams.twiml import TwiMLBuilderMediaStreams
from tac.channels.voice.provider import VoiceProvider
from tac.channels.websocket_protocol import WebSocketDisconnectError, WebSocketProtocol
from tac.core.config import TACConfig
from tac.models.outbound import CallOptions
from tac.models.voice import TwiMLRequest, VoiceTwiMLOptions, VoiceTwiMLOptionsMediaStreams
from tac.tools import TACTool

if TYPE_CHECKING:
    from tac.channels.voice.channel import VoiceChannel


#: Identifies this SDK to OpenAI on every WebSocket connection, per OpenAI's
#: requested User-Agent pattern: [Company/Library name]/[Language] [Version].
OPENAI_USER_AGENT = f"twilio-agent-connect-python/Python {__version__}"

TCallState = TypeVar("TCallState", bound=MediaStreamsOpenAICallState)


class MediaStreamsOpenAIProvider(VoiceProvider, Generic[TCallState]):
    """Shared scaffolding for a Media Streams provider bridging to an OpenAI
    real-time voice API. Holds only what's identical across every such
    provider; each subclass still owns its own ``initiate_outbound_conversation``,
    ``handle_websocket``, ``_register_call``, ``_connect_model``,
    ``_dispatch_model_event``, ``_handle_function_call``, and ``_cleanup_call``.

    Generic over ``TCallState`` (bound to ``MediaStreamsOpenAICallState``) so
    ``_calls`` keeps each subclass's own call-state shape instead of widening
    to the shared base everywhere it's read.
    """

    def __init__(
        self,
        channel: VoiceChannel,
        tac_config: TACConfig,
        config: MediaStreamsOpenAIProviderConfig,
    ) -> None:
        super().__init__(channel)

        self.config = config
        self.tac_config = tac_config
        self._tools_by_name: dict[str, TACTool] = {tool.name: tool for tool in config.tools}
        self._calls: dict[str, TCallState] = {}
        self._twiml = TwiMLBuilderMediaStreams(tac_config, config)
        # Keyed by call_sid (inbound) or a token rekeyed to call_sid on
        # connect (outbound); set in handle_incoming_call or
        # initiate_outbound_conversation, popped in _connect_model.
        self._call_session_configs: dict[str, dict[str, Any]] = {}

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
          2. ``MediaStreamsProviderConfig.default_twiml_options`` — per-channel defaults.
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
                "MediaStreamsOpenAIProvider.handle_incoming_call requires host_twiml_options "
                f"to be a VoiceTwiMLOptionsMediaStreams, got {type(host_twiml_options).__name__}"
            )

        customized: VoiceTwiMLOptionsMediaStreams | None = None
        if self.channel._on_inbound_call_twiml is not None and twiml_request is not None:
            result = await self.channel._on_inbound_call_twiml(twiml_request)
            if not isinstance(result, VoiceTwiMLOptionsMediaStreams):
                raise TypeError(
                    "MediaStreamsOpenAIProvider.handle_incoming_call requires the "
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
        return self._apply_call_event_callbacks(call_kwargs)

    async def _handle_model_events(self, conv_id: str) -> None:
        """Read model events until the socket closes, dispatching each one.

        A failure handling one event is logged and skipped rather than
        ending the loop — only the socket actually closing (raised by
        ``async for`` itself, not from inside it) does.
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
        self, conv_id: str, session: Any, event: dict[str, Any]
    ) -> None:
        """Interpret one event received from the model. Protocol-specific — implemented
        by each subclass."""
        raise NotImplementedError

    async def _run_tool_call(
        self, conv_id: str, name: str | None, arguments_json: str | None
    ) -> object:
        """Look up a model-requested tool by name, run it, and return its output.

        Errors are returned as part of the output rather than raised, so a bad
        tool call doesn't kill the call.
        """
        self.logger.debug(f"Tool call: {name}({arguments_json})", conversation_id=conv_id)

        if not name:
            return {"error": "Tool name is required"}

        tool = self._tools_by_name.get(name)
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
