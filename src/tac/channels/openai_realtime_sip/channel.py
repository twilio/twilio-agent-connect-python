"""OpenAI Realtime (SIP) channel implementation for TAC.

Bridges phone calls to OpenAI's Realtime API over SIP: a Twilio Elastic SIP
Trunk forwards the call at the SIP level directly to OpenAI, so audio never
passes through TAC. TAC's role is limited to the ``realtime.call.incoming``
webhook (deciding whether/how to accept the call) and, optionally, a
JSON-only control WebSocket for transcript capture and tool-calling.

See https://developers.openai.com/api/docs/guides/realtime-sip.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import websockets

from tac import TAC
from tac.channels.base import BaseChannel
from tac.channels.openai_realtime_sip.client import (
    OpenAIRealtimeSipClient,
    verify_and_parse_incoming_call,
)
from tac.channels.openai_realtime_sip.config import (
    CallIncomingHandler,
    OpenAIRealtimeSipChannelConfig,
)
from tac.channels.openai_realtime_sip.models import OpenAIRealtimeSipCallIncoming
from tac.tools import TACTool

TranscriptEntry = dict[str, str]


class OpenAIRealtimeSipChannel(BaseChannel):
    """Voice channel that hands calls off to OpenAI's Realtime API via SIP.

    Unlike ``VoiceChannel`` (ConversationRelay), this channel never sees
    audio or per-turn text — the model's spoken conversation happens
    entirely between Twilio and OpenAI. TAC's job is deciding whether/how
    to accept each incoming call.

    Register the acceptance decision with ``on_call_incoming``::

        channel = OpenAIRealtimeSipChannel(tac)


        @channel.on_call_incoming
        async def handle_incoming(
            event: OpenAIRealtimeSipCallIncoming,
        ) -> OpenAIRealtimeSipSessionConfig:
            return OpenAIRealtimeSipSessionConfig(model="gpt-realtime-2.1", voice="alloy")

    Accepting a call is enough for basic voice conversation — nothing else
    is required. Getting a text transcript or enabling tool calling both
    additionally require a "sideband" control WebSocket, which this channel
    opens automatically once a call is accepted (only if a tool is
    registered via ``register_tool`` or ``OpenAIRealtimeSipSessionConfig``
    sets ``input_transcription_model`` — otherwise no control connection is
    made). That connection carries JSON events only, never audio.
    """

    def __init__(
        self,
        tac: TAC,
        config: OpenAIRealtimeSipChannelConfig | dict[str, Any] | None = None,
    ):
        if isinstance(config, dict):
            config = OpenAIRealtimeSipChannelConfig(**config)
        elif config is None:
            config = OpenAIRealtimeSipChannelConfig()

        super().__init__(tac, memory_mode=config.memory_mode)

        if not config.openai_api_key:
            raise ValueError(
                "openai_api_key is required. Set the OPENAI_API_KEY environment "
                "variable or provide openai_api_key in OpenAIRealtimeSipChannelConfig."
            )

        self.config = config
        self.client = OpenAIRealtimeSipClient(api_key=config.openai_api_key)
        self._call_incoming_handler: CallIncomingHandler | None = None
        self._tools: dict[str, TACTool] = {}
        self._transcripts: dict[str, list[TranscriptEntry]] = {}
        self._control_tasks: dict[str, asyncio.Task[None]] = {}

    def get_channel_name(self) -> str:
        return "OPENAI_REALTIME_SIP"

    def on_call_incoming(self, handler: CallIncomingHandler) -> CallIncomingHandler:
        """Register the callback invoked for each ``realtime.call.incoming`` webhook.

        The handler receives the parsed incoming-call event and returns the
        ``OpenAIRealtimeSipSessionConfig`` to accept the call with. Use like a
        decorator::

            @channel.on_call_incoming
            async def handle_incoming(
                event: OpenAIRealtimeSipCallIncoming,
            ) -> OpenAIRealtimeSipSessionConfig: ...
        """
        self._call_incoming_handler = handler
        return handler

    def register_tool(self, tool: TACTool) -> TACTool:
        """Register a tool to make available on every accepted call.

        Auto-fills ``OpenAIRealtimeSipSessionConfig.tools`` for calls whose
        ``on_call_incoming`` handler didn't set ``tools`` explicitly. When the
        model calls it, the control-channel event loop executes it and sends
        the result back on the same connection.
        """
        self._tools[tool.name] = tool
        return tool

    def get_transcript(self, call_id: str) -> list[TranscriptEntry]:
        """Return the transcript captured so far for a call (``[{"role", "text"}, ...]``).

        Only populated if a control WebSocket was opened for the call (see
        class docstring) and only reflects an in-progress call — once the
        call ends, the transcript moves to
        ``ConversationSession.metadata["transcript"]`` for ``on_conversation_ended``.
        """
        return list(self._transcripts.get(call_id, []))

    def verify_webhook(
        self, payload: str | bytes, headers: dict[str, str]
    ) -> OpenAIRealtimeSipCallIncoming | None:
        """Verify an OpenAI webhook's signature and parse it, if it's a call-incoming event.

        Raises:
            openai.InvalidWebhookSignatureError: if the signature doesn't verify.
            ValueError: if no webhook secret is configured.
        """
        if not self.config.openai_webhook_secret:
            raise ValueError(
                "openai_webhook_secret is required to verify webhooks. Set the "
                "OPENAI_WEBHOOK_SECRET environment variable or provide "
                "openai_webhook_secret in OpenAIRealtimeSipChannelConfig."
            )
        return verify_and_parse_incoming_call(payload, headers, self.config.openai_webhook_secret)

    async def process_webhook(
        self, webhook_data: dict[str, Any], idempotency_token: str | None = None
    ) -> None:
        """Handle an already-verified ``realtime.call.incoming`` event.

        ``webhook_data`` here is the *parsed* ``OpenAIRealtimeSipCallIncoming`` model,
        already produced by ``verify_webhook`` at the HTTP layer — unlike other TAC
        channels, verification can't happen inside this method because OpenAI's
        webhook body must be read raw (bytes) for signature verification, before
        any JSON parsing.
        """
        event = OpenAIRealtimeSipCallIncoming.model_validate(webhook_data)

        if self._call_incoming_handler is None:
            self.logger.warning(
                "No on_call_incoming handler registered; rejecting call",
                call_id=event.call_id,
            )
            await self.client.reject_call(event.call_id)
            return

        self._start_conversation(event.call_id)
        session_config = await self._call_incoming_handler(event)
        if session_config.tools is None and self._tools:
            session_config = session_config.model_copy(
                update={"tools": [t.to_realtime_format() for t in self._tools.values()]}
            )

        await self.client.accept_call(event.call_id, session_config)
        self.logger.info(
            "CONVERSATION | Accepted OpenAI Realtime call",
            call_id=event.call_id,
            model=session_config.model,
        )

        if self._tools or session_config.input_transcription_model:
            task = asyncio.create_task(self._run_control_channel(event.call_id))
            self._control_tasks[event.call_id] = task

    async def _run_control_channel(self, call_id: str) -> None:
        """Hold the sideband control WebSocket open for the life of the call.

        Captures transcript events and dispatches function calls until the
        connection closes (the call ended), then runs conversation cleanup.
        """
        try:
            async with self.client.control_connection(call_id) as ws:
                async for raw_message in ws:
                    event = json.loads(raw_message)
                    await self._handle_control_event(call_id, ws, event)
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("Control WebSocket closed — call ended", call_id=call_id)
        except Exception as e:
            self.logger.warning(
                "Control WebSocket ended unexpectedly", call_id=call_id, error=str(e), exc_info=True
            )
        finally:
            self._control_tasks.pop(call_id, None)
            session = self._conversations.get(call_id)
            if session is not None:
                session.metadata["transcript"] = self._transcripts.pop(call_id, [])
            await self._end_conversation(call_id)

    async def _handle_control_event(self, call_id: str, ws: Any, event: dict[str, Any]) -> None:
        event_type = event.get("type")

        if event_type == "response.output_audio_transcript.done":
            self._transcripts.setdefault(call_id, []).append(
                {"role": "assistant", "text": event.get("transcript", "")}
            )
        elif event_type == "conversation.item.input_audio_transcription.completed":
            self._transcripts.setdefault(call_id, []).append(
                {"role": "caller", "text": event.get("transcript", "")}
            )
        elif event_type == "response.done":
            output = event.get("response", {}).get("output", [])
            for item in output:
                if item.get("type") == "function_call":
                    await self._handle_function_call(ws, item)

    async def _handle_function_call(self, ws: Any, item: dict[str, Any]) -> None:
        name = item.get("name", "")
        fc_call_id = item.get("call_id", "")
        tool = self._tools.get(name)

        if tool is None:
            output = json.dumps({"error": f"Unknown tool: {name}"})
        else:
            try:
                kwargs = json.loads(item.get("arguments") or "{}")
                result = await tool(**kwargs)
                output = result if isinstance(result, str) else json.dumps(result)
            except Exception as e:
                self.logger.error("Tool execution failed", tool=name, error=str(e), exc_info=True)
                output = json.dumps({"error": str(e)})

        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": fc_call_id,
                        "output": output,
                    },
                }
            )
        )
        await ws.send(json.dumps({"type": "response.create"}))

    async def end_call(self, call_id: str) -> None:
        """Hang up an in-progress call and clean up its local session.

        If a control WebSocket is open for this call, ``_run_control_channel``
        notices the connection close and runs conversation cleanup itself —
        this only forces the hangup and doesn't duplicate that cleanup.
        """
        await self.client.hangup_call(call_id)
        if call_id not in self._control_tasks:
            await self._end_conversation(call_id)

    async def send_response(
        self,
        conversation_id: str,
        response: str | AsyncGenerator[str | dict[str, Any], None],
        role: str | None = None,
    ) -> None:
        """Not applicable to this channel.

        There is no per-turn text-push model here — the spoken conversation is
        driven entirely by the Realtime session between Twilio and OpenAI once
        the call is accepted. Steer an in-progress call via the control
        WebSocket (``OpenAIRealtimeSipClient.control_connection``) instead of
        this method.
        """
        raise NotImplementedError(
            "OpenAIRealtimeSipChannel has no per-turn text-push model; use "
            "OpenAIRealtimeSipClient.control_connection(call_id) to steer an "
            "in-progress call instead."
        )
