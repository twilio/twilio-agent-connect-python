<div align="center">
  <div>
    <img src="https://raw.githubusercontent.com/twilio/twilio-agent-connect-python/main/logo.svg" alt="TAC Logo" width="120" height="120">
  </div>

  <h1>
    Twilio Agent Connect
  </h1>

  <h2>
    A powerful SDK for building intelligent, context-aware AI agents with Twilio's communication technologies.
  </h2>

  <div align="center">
    <a href="https://github.com/twilio/twilio-agent-connect-python"><img alt="Python SDK" src="https://img.shields.io/badge/Python-3.10+-3776AB.svg"/></a>
    <a href="https://pypi.org/project/twilio-agent-connect/"><img alt="PyPI" src="https://img.shields.io/pypi/v/twilio-agent-connect.svg"/></a>
    <a href="https://github.com/twilio/twilio-agent-connect-python/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/twilio/twilio-agent-connect-python/actions/workflows/ci.yml/badge.svg"/></a>
    <a href="https://github.com/twilio/twilio-agent-connect-python/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg"/></a>
    <a href="https://www.twilio.com/docs/conversations/agent-connect/quickstart"><img alt="Getting Started" src="https://img.shields.io/badge/Getting%20Started-Quickstart-F22F46.svg"/></a>
  </div>
  
  <p>
    <a href="https://www.twilio.com/docs/conversations/agent-connect">Documentation</a>
    ◆ <a href="https://github.com/twilio/twilio-agent-connect-python">Python SDK</a>
    ◆ <a href="https://github.com/twilio/twilio-agent-connect-typescript">TypeScript SDK</a>
    ◆ <a href="https://github.com/twilio/twilio-agent-connect-python/tree/main/getting_started/examples">Examples</a>
  </p>
</div>

Seamlessly integrate with Twilio Conversation Memory and Conversation Orchestrator to build LLM-powered agents with persistent memory and conversation context.

> [!TIP]
> **Building AI agents on AWS or Microsoft?** Connect them to Twilio's voice, messaging, and conversation context with these dedicated packages:
> - **[TAC for AWS](https://github.com/twilio/twilio-agent-connect-aws)** — Strands, Bedrock Agents, Bedrock AgentCore
> - **[TAC for Microsoft](https://github.com/twilio/twilio-agent-connect-microsoft)** — Microsoft Agent Framework, Azure AI Foundry (incl. Voice Live), Azure OpenAI

---

## Key Features

- **Multi-Channel Support**: Built-in handling for Voice (ConversationRelay), SMS, RCS, WhatsApp, and Chat
- **Outbound Conversations**: Agent-initiated conversations across all supported channels
- **ConversationRelay-Only Mode**: Get started quickly with TAC's voice plumbing (TwiML, WebSocket, callbacks) before adding Conversation Orchestrator or Conversation Memory
- **Memory Management**: Automatic integration with Twilio Conversation Memory for persistent user context
- **Conversation Lifecycle**: Automatic tracking of conversation sessions and state
- **Human Handoff**: Built-in tool to route conversations to human agents via Twilio Studio Flows (including Flex)

## Installation

```bash
pip install twilio-agent-connect
```

For server support (includes FastAPI and uvicorn for TACFastAPIServer):

```bash
pip install "twilio-agent-connect[server]"
```

TAC requires **Python 3.10 or newer**.

## Quick Examples

**Option 1: Use the Setup Wizard**

Use the [Twilio Setup Wizard](https://github.com/twilio/twilio-agent-connect-python/tree/main/getting_started/twilio_setup/) to automatically create a Memory Store and Conversation Configuration and generate your `.env` file:

```bash
git clone https://github.com/twilio/twilio-agent-connect-python.git
cd twilio-agent-connect-python
make setup  # Open http://localhost:8080
```

**Option 2: Manual Setup**

You can also create a Memory Store and Conversation Configuration manually through the [Twilio Console](https://1console.twilio.com). For a full walkthrough — credentials, Console navigation, and webhook configuration — see the [TAC Quickstart](https://www.twilio.com/docs/conversations/agent-connect/quickstart).

---

After completing setup, here's a minimal example to get started:

### Multi-Channel with OpenAI SDK

Use the OpenAI adapter to automatically inject conversation memory and user context into your OpenAI API calls across Voice, SMS, RCS, WhatsApp, and Chat channels.

First, install the required packages:

```bash
pip install openai python-dotenv
```

> **Note**: `python-dotenv` is optional — TAC works with environment variables from any source (`.env` files, Docker, Kubernetes, CI/CD, shell exports, etc.).

Then create your application:

```python
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tac import TAC, TACConfig
from tac.adapters.openai import with_tac_memory
from tac.channels.sms import SMSChannel
from tac.channels.voice import VoiceChannel
from tac.server import TACFastAPIServer

load_dotenv()

tac = TAC(config=TACConfig.from_env())
voice_channel = VoiceChannel(tac)
sms_channel = SMSChannel(tac)
openai_client = AsyncOpenAI()

conversation_history = {}
SYSTEM_INSTRUCTIONS = (
    "You are a customer service agent speaking with a user over voice or SMS. "
    "Keep responses short and conversational — a sentence or two. "
    "Do not use markdown, asterisks, bullets, or emojis; your words will be "
    "spoken aloud or sent as plain text."
)

async def handle_message_ready(user_message, context, memory_response):
    conv_id = context.conversation_id

    if conv_id not in conversation_history:
        conversation_history[conv_id] = []
    conversation_history[conv_id].append({"role": "user", "content": user_message})

    client = with_tac_memory(openai_client, memory_response, context)

    response = await client.responses.create(
        model="gpt-5.4-mini",
        instructions=SYSTEM_INSTRUCTIONS,
        input=conversation_history[conv_id]
    )

    llm_response = response.output_text
    conversation_history[conv_id].append({"role": "assistant", "content": llm_response})

    return llm_response

tac.on_message_ready(handle_message_ready)
TACFastAPIServer(tac=tac, voice_channel=voice_channel, messaging_channels=[sms_channel]).start()
```

> **Note**: See the [getting started guide](https://github.com/twilio/twilio-agent-connect-python/blob/main/getting_started/README.md) for complete setup instructions and `.env` configuration details.

**That's it!** The server automatically:
- Creates FastAPI app with `/twiml`, `/ws`, and `/webhook` endpoints
- Handles Voice, SMS, RCS, WhatsApp, and Chat conversations
- Routes responses to the appropriate channel
- Injects conversation memory and user profile into OpenAI calls

For configuration details and environment variables, see the [getting started guide](https://github.com/twilio/twilio-agent-connect-python/blob/main/getting_started/README.md).

## How It Works

TAC simplifies building AI agents by handling the integration between Twilio's communication channels and your LLM:

### Message Flow

1. **Webhook/Connection Received**: Twilio sends webhook (SMS) or WebSocket connection (Voice) to your server
2. **Channel Processing**: Channel validates and processes the incoming event
3. **Memory Retrieval**: TAC optionally retrieves user memories and profile from Memory
4. **Callback Invoked**: Your `on_message_ready` callback receives user message, context, and optional memory response
5. **Response Handling**: Your callback returns a response string that TAC routes to the appropriate channel

For detailed architecture and advanced usage, see [CLAUDE.md](https://github.com/twilio/twilio-agent-connect-python/blob/main/CLAUDE.md).

## Learn More

**Examples & Guides:**
- **[Getting Started Guide](https://github.com/twilio/twilio-agent-connect-python/tree/main/getting_started/)** - Setup wizard, examples, and comprehensive documentation
- **[Partner SDK Examples](https://github.com/twilio/twilio-agent-connect-python/tree/main/getting_started/examples/partners/)** - Integration examples for OpenAI (Chat Completions, Responses API, Agents SDK), LangChain, AWS Bedrock Agent, AWS Bedrock AgentCore, and AWS Strands
- **[ConversationRelay-Only Mode](https://github.com/twilio/twilio-agent-connect-python/blob/main/getting_started/examples/features/relay_only.py)** - Get started with voice using just ConversationRelay
- More examples coming soon

**AWS and Microsoft connectors:**
- **[TAC for AWS](https://github.com/twilio/twilio-agent-connect-aws)** — `StrandsConnector`, `BedrockConnector`, `BedrockAgentCoreConnector` for AWS Strands, Bedrock Agents, and Bedrock AgentCore
- **[TAC for Microsoft](https://github.com/twilio/twilio-agent-connect-microsoft)** — `AgentFrameworkConnector` and `VoiceLiveConnector` for Microsoft Agent Framework, Azure AI Foundry (including Voice Live), and Azure OpenAI

**Documentation:**
- **[CLAUDE.md](https://github.com/twilio/twilio-agent-connect-python/blob/main/CLAUDE.md)** - Architecture, development guide, and API reference
- **[Getting Started Guide](https://github.com/twilio/twilio-agent-connect-python/blob/main/getting_started/README.md)** - Setup instructions, environment variables, and troubleshooting

---

# TAC Development / Contribution

TAC uses [`uv`](https://docs.astral.sh/uv/) for package management. Ensure you have it installed:

```bash
uv --version
```

### Setup Development Environment

```bash
# Install all dependencies (including dev tools)
make sync
```

### Running Tests and Checks

```bash
# Format code
make format

# Run linting
make lint

# Run type checking
make type-check

# Run tests
make test

# Run all checks at once
make check
```
