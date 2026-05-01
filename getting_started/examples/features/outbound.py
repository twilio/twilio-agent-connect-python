"""
Feature: Outbound Conversations (SMS, RCS, and Voice)

Demonstrates agent-initiated (outbound) conversations using TAC. Sends an SMS,
RCS message, or places a voice call, then handles the full conversation loop with
the OpenAI Agents SDK.

Usage:
    python outbound.py --to +16505551234 --channel sms --message "Hello!"
    python outbound.py --to +16505551234 --channel rcs --message "Hello!"
    python outbound.py --to +16505551234 --channel voice
    python outbound.py --to +16505551234 --channel voice --welcome-greeting "Hi there!"

Requires ``OPENAI_API_KEY`` in addition to the usual TAC env vars.
For voice calls, ``TWILIO_VOICE_PUBLIC_DOMAIN`` must also be set (e.g. via ngrok).
For RCS, ``TWILIO_RCS_SENDER_ID`` must be set in environment variables.
"""

import argparse
import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from agents import Agent, Runner, set_tracing_disabled
from dotenv import load_dotenv
from fastapi import FastAPI

from tac import TAC, TACConfig
from tac.channels.rcs import RCSChannel, RCSChannelConfig
from tac.channels.sms import SMSChannel, SMSChannelConfig
from tac.channels.voice import VoiceChannel, VoiceChannelConfig
from tac.models.outbound import (
    InitiateMessagingConversationOptions,
    InitiateVoiceConversationOptions,
)
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.server import TACFastAPIServer
from tac.server.config import TACServerConfig

load_dotenv()
set_tracing_disabled(True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Outbound conversations with TAC")
    parser.add_argument("--to", required=True, help="Destination phone number or RCS address")
    parser.add_argument(
        "--channel",
        required=True,
        choices=["sms", "rcs", "voice"],
        help="Channel (sms, rcs, or voice)",
    )
    parser.add_argument("--message", help="Initial message (required for SMS and RCS)")
    parser.add_argument("--welcome-greeting", help="Optional voice welcome greeting")
    return parser.parse_args()


SYSTEM_INSTRUCTIONS = (
    "You are a friendly, helpful AI assistant. You initiated this outbound "
    "conversation by reaching out to the customer. When the customer first "
    "speaks (e.g., 'hello?'), introduce yourself and explain why you are "
    "calling -- for example: 'Hi! This is an AI assistant calling on behalf "
    "of Acme Corp about your recent order.' Be conversational and helpful. "
    "You do not have the ability to transfer calls or connect to human agents. "
    "Only offer capabilities you actually have."
)

tac = TAC(config=TACConfig.from_env())
voice_channel = VoiceChannel(tac, config=VoiceChannelConfig(auto_retrieve_memory=True))
sms_channel = SMSChannel(tac, config=SMSChannelConfig(auto_retrieve_memory=True))

# RCS channel requires rcs_sender_id configured in TAC config
rcs_channel = RCSChannel(
    tac,
    config=RCSChannelConfig(
        auto_retrieve_memory=True,
    ),
)

conversation_history: dict[str, list[Any]] = {}


async def handle_message_ready(
    user_message: str,
    context: ConversationSession,
    memory_response: TACMemoryResponse | None,
) -> str:
    instructions = SYSTEM_INSTRUCTIONS
    if memory_response:
        memory_sections = memory_response.build_memory_prompts()
        if memory_sections:
            instructions += "\n\n" + "\n\n".join(memory_sections)

    agent = Agent(name="Outbound Agent", instructions=instructions)

    history = conversation_history.get(context.conversation_id, [])
    agent_input = history + [{"role": "user", "content": user_message}]

    result = await Runner.run(agent, agent_input)

    conversation_history[context.conversation_id] = result.to_input_list()
    return result.final_output_as(str)


tac.on_message_ready(handle_message_ready)


async def initiate_outbound(args: argparse.Namespace) -> None:
    print(f"\n  Outbound {args.channel.upper()} conversation\n")

    try:
        if args.channel == "sms":
            sms_result = await sms_channel.initiate_outbound_conversation(
                InitiateMessagingConversationOptions(to=args.to, message=args.message)
            )
            print(f"SMS sent to {args.to} (conversation: {sms_result.conversation_id})")
            print(f"[{sms_result.conversation_id}] Agent: {args.message}")
            print("\nWaiting for replies... (Ctrl+C to exit)\n")

        elif args.channel == "rcs":
            # Validate RCS configuration
            if not tac.config.rcs_sender_id:
                print("Error: RCS requires TWILIO_RCS_SENDER_ID environment variable to be set.")
                sys.exit(1)

            rcs_result = await rcs_channel.initiate_outbound_conversation(
                InitiateMessagingConversationOptions(to=args.to, message=args.message)
            )
            print(f"RCS message sent to {args.to} (conversation: {rcs_result.conversation_id})")
            print(f"[{rcs_result.conversation_id}] Agent: {args.message}")
            print("\nWaiting for replies... (Ctrl+C to exit)\n")

        elif args.channel == "voice":
            server_config = TACServerConfig.from_env()
            public_domain = server_config.public_domain
            if not public_domain:
                print("TWILIO_VOICE_PUBLIC_DOMAIN is required for voice calls.")
                sys.exit(1)

            voice_result = await voice_channel.initiate_outbound_conversation(
                InitiateVoiceConversationOptions(
                    to=args.to,
                    websocket_url=f"wss://{public_domain}/ws",
                    welcome_greeting=args.welcome_greeting,
                    action_url=f"https://{public_domain}/conversation-relay-callback",
                )
            )
            print(f"Call placed to {args.to} (SID: {voice_result.call_sid})")
            print("\nConversation in progress... (Ctrl+C to exit)\n")

    except Exception as e:
        print(f"\nFailed to initiate outbound conversation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()

    if args.channel in ("sms", "rcs") and not args.message:
        print(f"--message is required for {args.channel.upper()} channel.")
        sys.exit(1)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        asyncio.create_task(initiate_outbound(args))
        yield

    app = FastAPI(title="TAC Outbound Example", lifespan=lifespan)

    server = TACFastAPIServer(
        tac=tac,
        voice_channel=voice_channel,
        messaging_channels=[sms_channel, rcs_channel],
        app=app,
    )

    server.start()
