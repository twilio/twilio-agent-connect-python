"""Pydantic models for the OpenAI Realtime (SIP) channel.

See:
- https://developers.openai.com/api/docs/guides/realtime-sip
- https://developers.openai.com/api/reference/resources/webhooks
"""

from typing import Any

from pydantic import BaseModel, Field


class OpenAIRealtimeSipCallIncoming(BaseModel):
    """Parsed ``realtime.call.incoming`` webhook event from OpenAI.

    OpenAI POSTs this when a SIP call arrives at the project's configured
    SIP endpoint (routed there by a Twilio Elastic SIP Trunk's origination
    URI). ``call_id`` is required to accept/reject/hangup/refer the call and
    to open the per-call control WebSocket.
    """

    model_config = {"extra": "ignore"}

    call_id: str = Field(
        description="Identifier for this call. Used in accept/reject/hangup/refer "
        "requests and as the `call_id` query parameter for the control WebSocket."
    )
    sip_headers: list[dict[str, str]] = Field(
        default_factory=list,
        description="Raw SIP headers from the incoming INVITE (From, To, Call-ID, ...).",
    )

    def sip_header(self, name: str) -> str | None:
        """Look up a SIP header by name (case-insensitive)."""
        for header in self.sip_headers:
            if header.get("name", "").lower() == name.lower():
                return header.get("value")
        return None

    @classmethod
    def from_webhook_event(cls, event: dict[str, Any]) -> "OpenAIRealtimeSipCallIncoming":
        """Build from the unwrapped webhook event body (``event["data"]``)."""
        data = event.get("data", {})
        return cls(call_id=data["call_id"], sip_headers=data.get("sip_headers", []))


class OpenAIRealtimeSipSessionConfig(BaseModel):
    """Session configuration returned by the user's ``on_call_incoming`` callback.

    Serialized directly into the body of the ``POST /v1/realtime/calls/{call_id}/accept``
    request — same shape as the OpenAI Realtime "create client secret" / session config.
    """

    model_config = {"extra": "forbid"}

    model: str = Field(description="Realtime model to use, e.g. 'gpt-realtime-2.1'.")
    instructions: str | None = Field(
        default=None, description="System instructions for the voice agent."
    )
    voice: str | None = Field(
        default=None, description="Voice to use for the model's spoken output, e.g. 'marin'."
    )
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description="Realtime-format tool definitions "
        "({'type': 'function', 'name', 'description', 'parameters'}). Auto-filled "
        "from tools registered via OpenAIRealtimeSipChannel.register_tool(...) "
        "if left unset.",
    )
    input_transcription_model: str | None = Field(
        default=None,
        description="Transcription model for the caller's speech (e.g. "
        "'gpt-live-transcribe'). Unset means no text transcript of what the "
        "caller said is produced — only the model's own spoken output gets "
        "transcribed automatically.",
    )

    def to_accept_payload(self) -> dict[str, Any]:
        """Build the JSON body for the ``/accept`` request, omitting unset fields.

        GA session config nests output-audio settings under ``audio.output``
        rather than a flat top-level ``voice`` field — the ``/accept`` endpoint
        accepts a flat ``voice`` with a 200 but silently fails to configure it,
        which then causes the SIP session to fail with a 400 shortly after.
        """
        payload: dict[str, Any] = {"type": "realtime", "model": self.model}
        if self.instructions is not None:
            payload["instructions"] = self.instructions
        audio: dict[str, Any] = {}
        if self.voice is not None:
            audio["output"] = {"voice": self.voice}
        if self.input_transcription_model is not None:
            audio["input"] = {"transcription": {"model": self.input_transcription_model}}
        if audio:
            payload["audio"] = audio
        if self.tools is not None:
            payload["tools"] = self.tools
        return payload
