<div align="center">
  <div>
    <img src="logo.svg" alt="TAC Logo" width="120" height="120">
  </div>

  <h1>
    Twilio Agent Connect
  </h1>

  <h2>
    A powerful SDK for building intelligent, context-aware AI agents with Twilio's communication technologies.
  </h2>

  <div align="center">
    <a href="https://github.com/twilio/twilio-agent-connect-python"><img alt="Python SDK" src="https://img.shields.io/badge/Python-3.10+-3776AB.svg"/></a>
    <a href="https://github.com/twilio/twilio-agent-connect-typescript"><img alt="TypeScript SDK" src="https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg"/></a>
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-green.svg"/></a>
    <a href="https://www.twilio.com/docs/platform/tac/quickstart"><img alt="Getting Started" src="https://img.shields.io/badge/Getting%20Started-Quickstart-F22F46.svg"/></a>
  </div>
  
  <p>
    <a href="https://www.twilio.com/docs/platform/tac/overview">Documentation</a>
    ◆ <a href="https://github.com/twilio/twilio-agent-connect-python">Python SDK</a>
    ◆ <a href="https://github.com/twilio/twilio-agent-connect-typescript">TypeScript SDK</a>
    ◆ <a href="getting_started/examples">Examples</a>
  </p>
</div>

Seamlessly integrate with Twilio's Memory Store and Conversation Orchestrator to build LLM-powered agents with persistent memory and conversation context.

---

## Key Features

- **Multi-Channel Support**: Built-in webhook handling for SMS, RCS, WhatsApp, and Chat conversations
- **Voice Channel Support**: WebSocket protocol handling for Twilio Voice with ConversationRelay
- **Outbound Conversations**: Agent-initiated conversations via SMS, RCS, WhatsApp, and Voice channels
- **ConversationRelay-Only Mode**: Get started quickly with TAC's voice plumbing (TwiML, WebSocket, callbacks) before adding Conversation Orchestrator
- **Memory Management**: Automatic integration with Twilio Memory for persistent user context
- **Conversation Lifecycle**: Automatic tracking of conversation sessions and state
- **Type-Safe**: Full type hints and Pydantic models throughout
- **Callback-Based**: Simple `on_message_ready` callback for LLM integration with optional memory retrieval
- **Production Ready**: Comprehensive test coverage and error handling

## Get Started

To get started, set up your Python environment (Python 3.10 or newer required), and then install TAC SDK package.

### uv (Recommended)

We recommend using [uv](https://docs.astral.sh/uv/) for the best development experience:

```bash
uv init
uv add git+https://github.com/twilio/twilio-agent-connect-python.git

# Install with server support (includes FastAPI and uvicorn for TACFastAPIServer)
uv add git+https://github.com/twilio/twilio-agent-connect-python.git --extra server
```

### pip/venv (Alternative)

If you prefer using pip and venv:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install git+https://github.com/twilio/twilio-agent-connect-python.git

# Install with server support
pip install "git+https://github.com/twilio/twilio-agent-connect-python.git[server]"
```

## Quick Examples

**Option 1: Use the Setup Wizard**

Use the [Twilio Setup Wizard](getting_started/twilio_setup/) to automatically create a Memory Store and Conversation Configuration and generate your `.env` file:

```bash
git clone https://github.com/twilio/twilio-agent-connect-python.git
cd twilio-agent-connect-python
make setup  # Open http://localhost:8080
```

**Option 2: Manual Setup**

You can also create a Memory Store and Conversation Configuration manually through the [Twilio Console](https://1console.twilio.com). For a full walkthrough — credentials, Console navigation, and webhook configuration — see the [TAC Quickstart](https://www.twilio.com/docs/platform/tac/quickstart).

---

After completing setup, here's a minimal example to get started:

### Multi-Channel with OpenAI SDK

Use the OpenAI adapter to automatically inject conversation memory and user context into your OpenAI API calls across Voice, SMS, RCS, WhatsApp, and Chat channels.

First, install the required packages:

```bash
uv add openai python-dotenv
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

> **Note**: See the [getting started guide](getting_started/README.md) for complete setup instructions and `.env` configuration details.

**That's it!** The server automatically:
- Creates FastAPI app with `/twiml`, `/ws`, and `/webhook` endpoints
- Handles Voice, SMS, RCS, WhatsApp, and Chat conversations
- Routes responses to the appropriate channel
- Injects conversation memory and user profile into OpenAI calls

For configuration details and environment variables, see the [getting started guide](getting_started/README.md).

## How It Works

TAC simplifies building AI agents by handling the integration between Twilio's communication channels and your LLM:

### Message Flow

1. **Webhook/Connection Received**: Twilio sends webhook (SMS) or WebSocket connection (Voice) to your server
2. **Channel Processing**: Channel validates and processes the incoming event
3. **Memory Retrieval**: TAC optionally retrieves user memories and profile from Memory
4. **Callback Invoked**: Your `on_message_ready` callback receives user message, context, and optional memory response
5. **Response Handling**: Your callback returns a response string that TAC routes to the appropriate channel

For detailed architecture and advanced usage, see [CLAUDE.md](CLAUDE.md).

## Learn More

**Examples & Guides:**
- **[Getting Started Guide](getting_started/)** - Setup wizard, examples, and comprehensive documentation
- **[Partner SDK Examples](getting_started/examples/partners/)** - OpenAI Chat Completions and Responses API examples
- **[ConversationRelay-Only Mode](getting_started/examples/features/relay_only.py)** - Get started with voice using just ConversationRelay
- More examples coming soon

**Documentation:**
- **[CLAUDE.md](CLAUDE.md)** - Architecture, development guide, and API reference
- **[Getting Started Guide](getting_started/README.md)** - Setup instructions, environment variables, and troubleshooting

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

# Or manually with uv
uv sync --all-extras --all-groups
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
