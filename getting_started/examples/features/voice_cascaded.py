"""
Voice Cascaded Example — ConversationRelay + OpenAI Agents SDK.

This is the *cascaded* counterpart to voice_realtime.py: Twilio's
ConversationRelay does speech-to-text and text-to-speech, and TAC hands the
transcribed turn to an LLM (here, the OpenAI Agents SDK) for a text reply,
which ConversationRelay then speaks back to the caller.

    caller ⇄ Twilio ASR/TTS ⇄ ConversationRelay ⇄ [this channel] ⇄ LLM

Built to be a fair, side-by-side Langfuse comparison against voice_realtime.py
(the speech-to-speech alternative): same tier of model, greeting, tone, and
tool, with no TAC memory involved on either side. Run both, call each number,
and compare the "response_latency" spans in Langfuse:
  - conversation_relay.response_latency (this example) — final transcript
    received -> first LLM token sent back for ConversationRelay to speak.
  - openai_realtime.response_latency (voice_realtime.py) — caller stops
    talking -> model's first audio byte back.

This example also emits a "model_invocation" span (with a
time_to_first_token_ms attribute) around the LLM call in the callback below.
In the cascaded path the application invokes the model, not TAC, so TAC can't
time it for you — this shows how to nest your own model span under the call
trace via voice_channel.trace_context(). It's the cascaded analog of the
realtime channel's openai_realtime.connect span, and it breaks out the model's
share of response_latency (which also includes sending audio back to Twilio).

Runs in relay-only mode (no Conversation Orchestrator) so the comparison
isolates the ASR+LLM+TTS cascade from CO/memory overhead — voice_realtime.py
never touches CO or memory either.

Env vars required:
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_API_KEY, TWILIO_API_SECRET
- TWILIO_PHONE_NUMBER
- TWILIO_VOICE_PUBLIC_DOMAIN (ngrok or similar, e.g. 'example.ngrok.app')
- OPENAI_API_KEY

Env vars that should NOT be set (this example uses relay-only mode):
- TWILIO_CONVERSATION_CONFIGURATION_ID
"""

import time
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

from agents import Agent, Runner, function_tool, set_tracing_disabled
from dotenv import load_dotenv

from tac import TAC, TACConfig
from tac.channels.voice import TwiMLOptions, VoiceChannel, VoiceChannelConfig
from tac.core.tracing import get_tracer
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.server import TACFastAPIServer

load_dotenv()
# The OpenAI Agents SDK ships its own tracer; disable it so the only spans in
# Langfuse are TAC's call/response_latency/barge_in plus the model_invocation
# span below — keeping this trace directly comparable to voice_realtime.py's.
set_tracing_disabled(True)

# Same tracer setup TAC's channels use, so spans this example emits from the
# message-ready callback export to Langfuse alongside the channel's spans.
tracer = get_tracer(__name__)

# No conversation_configuration_id — relay-only mode, same as voice_realtime.py
# never touching Conversation Orchestrator. Keeps the comparison to just the
# voice architecture (cascaded vs. speech-to-speech), not CO/memory overhead.
tac = TAC(config=TACConfig.from_env())
assert not tac.is_orchestrator_enabled(), (
    "This example expects relay-only mode — unset TWILIO_CONVERSATION_CONFIGURATION_ID."
)

# Same greeting text as voice_realtime.py's DEFAULT_WELCOME_GREETING, so the
# caller hears an identical opening line on both numbers.
WELCOME_GREETING = "Hello! How can I help you today?"

# Paraphrases voice_realtime.py's realtime instructions for a text-completion
# model rather than a speech-to-speech one — same tone/pacing/format rules.
SYSTEM_INSTRUCTIONS = (
    "You are a warm, friendly voice assistant speaking with a caller over the phone — "
    "like a helpful colleague, not a script. Keep responses short, a sentence or two per "
    "turn, and vary your wording so you don't sound robotic. Always speak English. Do "
    "not use markdown, asterisks, bullet lists, or emojis; your words will be spoken aloud."
)

# memory_mode="never" (the default) is explicit here since the whole point of
# this example is an apples-to-apples comparison against voice_realtime.py,
# which has no memory concept at all.
voice_channel = VoiceChannel(
    tac,
    config=VoiceChannelConfig(
        memory_mode="never",
        default_twiml_options=TwiMLOptions(welcome_greeting=WELCOME_GREETING),
    ),
)


@function_tool()
def get_current_time() -> str:
    """Get the current date and time. Use this if the caller asks what time or day it is."""
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


# gpt-5.4-mini is the usual default for cost, but voice_realtime.py runs
# OpenAI's flagship speech model (gpt-realtime) — the flagship text model
# here keeps the comparison fair rather than pitting flagship against mini.
MODEL = "gpt-5.4"

agent = Agent(
    name="Voice Assistant",
    instructions=SYSTEM_INSTRUCTIONS,
    model=MODEL,
    tools=[get_current_time],
)

conversation_history: dict[str, list[Any]] = {}


async def handle_message_ready(
    user_message: str,
    context: ConversationSession,
    memory_response: TACMemoryResponse | None,
) -> None:
    """Stream voice responses through the OpenAI Agents SDK.

    Returns None and manually calls send_response() with an async generator
    so tokens are sent to the caller (and ConversationRelay's TTS) as they
    arrive from the LLM.
    """
    conv_id = context.conversation_id

    history = conversation_history.get(conv_id, [])
    agent_input = history + [{"role": "user", "content": user_message}]

    async def stream_tokens() -> AsyncGenerator[str, None]:
        # Time the LLM call itself. In the cascaded path TAC never invokes the
        # model — the application does, right here — so unlike the realtime
        # channel (which owns and times its model WebSocket via
        # openai_realtime.connect), instrumenting the model call is the app's
        # job. voice_channel.trace_context nests this under the same
        # conversation_relay.call trace, so it sits beside the channel's
        # response_latency span rather than forming a disconnected root trace.
        span = tracer.start_span(
            "model_invocation",
            context=voice_channel.trace_context(conv_id),
            attributes={
                "conversation_id": conv_id,
                "gen_ai.request.model": MODEL,
                # Langfuse renders a span carrying a model attribute as a
                # "generation" observation.
                "langfuse.observation.type": "generation",
            },
        )
        started = time.monotonic()
        first_token_seen = False
        try:
            result = Runner.run_streamed(agent, agent_input)
            async for event in result.stream_events():
                if event.type == "raw_response_event" and hasattr(event.data, "delta"):
                    if not first_token_seen:
                        first_token_seen = True
                        # Model time-to-first-token — the LLM's share of the
                        # channel's response_latency (which also includes the
                        # send back to Twilio).
                        span.set_attribute(
                            "time_to_first_token_ms",
                            round((time.monotonic() - started) * 1000),
                        )
                    yield event.data.delta
            conversation_history[conv_id] = result.to_input_list()
        finally:
            span.end()

    await voice_channel.send_response(conv_id, stream_tokens())


tac.on_message_ready(handle_message_ready)


async def handle_conversation_ended(context: ConversationSession) -> None:
    conversation_history.pop(context.conversation_id, None)


tac.on_conversation_ended(handle_conversation_ended)


if __name__ == "__main__":
    server = TACFastAPIServer(tac=tac, voice_channel=voice_channel)
    server.start()
