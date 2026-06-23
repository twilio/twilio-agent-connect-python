"""
Conversation Intelligence Quickstart

AI-powered customer simulation with real-time Script Adherence monitoring and
post-call Summary generation using Twilio Conversation Intelligence v3 + TAC voice channel.

What you get:
- GPT-4o AI customer that coaches the agent through a support script
- Script Adherence operator fires after each agent utterance → dashboard shows checkpoints live
- Summary operator fires when the call ends → stored in Conversation Memory
- Dashboard at http://localhost:3340 shows live transcript + CI events

Prerequisites:
1. Run the setup wizard: uv run getting_started/examples/features/cintel_quickstart/setup_server.py
2. Start ngrok on port 3340: ngrok http 3340
3. Add env vars from wizard to getting_started/examples/.env
4. Point your Twilio number's Voice webhook to: https://<ngrok>/tac/twiml

Usage:
    uv run getting_started/examples/features/cintel_quickstart/app.py
"""

import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from tac import TAC, TACConfig
from tac.channels.voice import VoiceChannel, VoiceChannelConfig
from tac.core.logging import get_logger
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.server import TACFastAPIServer
from tac.server.config import TACServerConfig

load_dotenv()

logger = get_logger(__name__)

# ── AI Customer Prompt ────────────────────────────────────────────────────────
SCRIPT_COACH_CUSTOMER_PROMPT = """You are a SCRIPT COACH playing the role of a frustrated but reasonable customer who has called Owl Internet tech support. Your dual purpose:

1. AS A CUSTOMER: You have a specific problem - your internet keeps disconnecting every few hours. You're frustrated but not abusive. You want your problem solved. An agent from Owl Internet is calling you — you just picked up and said hello, now you're waiting for the agent to introduce themselves.

2. AS A COACH: You're secretly helping a new support agent practice their script. Without breaking character, you naturally guide them through the required checkpoints:

REQUIRED SCRIPT CHECKPOINTS:
1. GREETING - Wait for them to say "Hi, this is [name] calling from Owl Internet"
2. IDENTITY VERIFICATION - Wait for them to ask for your account number. Provide: "Ben Smith, account 12345678"
3. RESOLUTION STEPS - Let them troubleshoot. Mention you've already tried restarting the router.
4. BRAND-APPROVED CLOSING - Wait for them to thank you for choosing Owl Internet and ask if there's anything else.

COACHING BEHAVIOR:
- If they skip a step, find a natural way to prompt them. Example: If they don't greet you properly, say "Oh, I didn't catch your name?"
- If they're doing well, be a cooperative customer who follows their lead
- If they go off-script, express mild confusion: "Wait, don't you need my account number first?"

RESOLUTION PHASE:
- If the agent says they will trigger a remote reset or restart your modem remotely, IMMEDIATELY respond positively: "Oh wow! Yes, I can see my router lights are blinking now... and... oh it's coming back online! The internet is working again! Thank you so much!"

VOICE CHARACTERISTICS:
- Speak naturally, with occasional "um" and "uh"
- Show frustration about the problem, not at the agent
- Sound relieved when they're helpful"""

# ── TAC + channel setup ───────────────────────────────────────────────────────
ci_config_id = os.environ.get("CONVERSATION_INTELLIGENCE_CONFIGURATION_ID")
if not ci_config_id:
    print(
        "ERROR: CONVERSATION_INTELLIGENCE_CONFIGURATION_ID is not set.\n"
        "Run the setup wizard first:\n"
        "  uv run getting_started/examples/features/cintel_quickstart/setup_server.py\n"
        "Then add CONVERSATION_INTELLIGENCE_CONFIGURATION_ID to your .env file.",
        file=sys.stderr,
    )
    sys.exit(1)

tac = TAC(config=TACConfig.from_env())
voice_channel = VoiceChannel(tac, config=VoiceChannelConfig(memory_mode="always"))

from cintel_webhook import parse_cintel_webhook  # noqa: E402
from openai import (  # noqa: E402
    APIConnectionError,
    APIError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)
from sse_manager import sse_manager  # noqa: E402

openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
conversation_history: dict[str, list] = {}


async def handle_message_ready(
    user_message: str,
    context: ConversationSession,
    memory_response: TACMemoryResponse | None,
) -> None:
    conv_id = context.conversation_id
    logger.info(f"[{conv_id}] Agent said: {user_message[:50]}...")

    if conv_id not in conversation_history:
        conversation_history[conv_id] = []

    conversation_history[conv_id].append({"role": "user", "content": user_message})

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": SCRIPT_COACH_CUSTOMER_PROMPT},
                *conversation_history[conv_id],
            ],
        )

        llm_response = response.choices[0].message.content or ""
        conversation_history[conv_id].append({"role": "assistant", "content": llm_response})

        logger.info(f"[{conv_id}] AI customer: {llm_response[:50]}...")

        await voice_channel.send_response(conv_id, llm_response)

        sse_manager.broadcast(
            "transcript-update",
            {"speaker": "agent", "text": user_message, "interim": False},
        )
        sse_manager.broadcast(
            "transcript-update",
            {"speaker": "customer", "text": llm_response, "interim": False},
        )

    except AuthenticationError:
        logger.error(f"[{conv_id}] OpenAI authentication failed - check API key")
        await voice_channel.send_response(
            conv_id, "I'm having trouble connecting. Please try again later."
        )

    except RateLimitError:
        logger.warning(f"[{conv_id}] OpenAI rate limit reached, using fallback")
        await voice_channel.send_response(
            conv_id, "Sorry, could you repeat that? I'm having trouble hearing you."
        )

    except (APIConnectionError, APIError) as e:
        logger.error(f"[{conv_id}] OpenAI API error: {e}")
        await voice_channel.send_response(
            conv_id, "I'm experiencing technical difficulties. Let me try that again."
        )

    except Exception as e:
        logger.error(f"[{conv_id}] Unexpected error in message handler: {e}")


async def handle_conversation_ended(context: ConversationSession) -> None:
    conv_id = context.conversation_id
    conversation_history.pop(conv_id, None)
    sse_manager.broadcast("call-ended", {})


tac.on_message_ready(handle_message_ready)
tac.on_conversation_ended(handle_conversation_ended)


# ── FastAPI app ───────────────────────────────────────────────────────────────
_static_dir = Path(__file__).parent / "static"


def _make_app() -> Any:
    from fastapi import FastAPI

    application = FastAPI(title="TAC CINTEL Quickstart")
    application.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @application.get("/")
    async def root() -> HTMLResponse:
        return HTMLResponse(content=(_static_dir / "index.html").read_text())

    @application.post("/ci-webhook")
    async def cintel_webhook(request: Request) -> dict[str, str]:
        payload = await request.json()
        logger.info("Received CINTEL webhook: %s", json.dumps(payload, indent=2))

        results = parse_cintel_webhook(payload)

        if results["script_adherence"]:
            for checkpoint in results["script_adherence"]:
                sse_manager.broadcast("checkpoint-update", checkpoint)

        if results["summary"]:
            logger.info(f"Summary received: {results['summary']['summary_text'][:50]}...")
            sse_manager.broadcast("summary-update", results["summary"])

        return {"status": "ok"}

    @application.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        return StreamingResponse(
            sse_manager.event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @application.get("/api/config")
    async def get_config() -> JSONResponse:
        return JSONResponse({"phone_number": os.environ.get("TWILIO_PHONE_NUMBER", "")})

    @application.get("/api/script")
    async def get_script() -> JSONResponse:
        """Fetch Script Adherence operator script param and return per-category hints."""
        api_key = os.environ.get("TWILIO_API_KEY", "")
        api_secret = os.environ.get("TWILIO_API_SECRET", "")
        credentials = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()
        headers = {"Authorization": f"Basic {credentials}"}

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"https://intelligence.twilio.com/v3/ControlPlane/Configurations/{ci_config_id}",
                    headers=headers,
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning(f"Could not fetch Intelligence Config: {e}")
            return JSONResponse({})

        script_text = ""
        for rule in data.get("rules", []):
            for op in rule.get("operators", []):
                script = op.get("parameters", {}).get("script", "")
                if script:
                    script_text = script
                    break
            if script_text:
                break

        if not script_text:
            logger.warning("Script Adherence: no script parameter found in Intelligence Config")
            return JSONResponse({})

        hints = _parse_script_hints(script_text)
        logger.info("Script hints loaded: %s", hints)
        return JSONResponse(hints)

    return application


def _parse_script_hints(script: str) -> dict[str, str]:
    """Extract the first Example: value per category from the script."""
    hints: dict[str, str] = {}
    current_category = ""
    for line in script.splitlines():
        cat_match = re.match(r"^Category:\s*(\S+)", line.strip())
        if cat_match:
            current_category = cat_match.group(1)
            continue
        example_match = re.search(r'Example:\s*"([^"]+)"?', line)
        if example_match and current_category and current_category not in hints:
            hints[current_category] = example_match.group(1)
    return hints


# ── Server ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    server_config = TACServerConfig.from_env()
    server_config.cintel_webhook_path = "/ci-webhook"
    if not os.environ.get("TWILIO_SERVER_PORT"):
        server_config.port = 3340

    server = TACFastAPIServer(
        tac=tac,
        voice_channel=voice_channel,
        config=server_config,
    )

    # Mount dashboard routes onto the TAC app
    app = _make_app()
    server.app.mount("/", app)

    logger.info(
        f"Conversation Intelligence Quickstart running.\n"
        f"  Dashboard : http://localhost:{server_config.port}\n"
        f"  TwiML URL : https://<ngrok>/tac/twiml  (configure on your Twilio number)\n"
        f"  CI Webhook: https://<ngrok>/ci-webhook  (configure in your Intelligence Configuration)\n"
        f"  CI Config : {ci_config_id}"
    )
    server.start()
