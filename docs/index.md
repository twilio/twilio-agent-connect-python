# Twilio Agent Connect (Python)

Twilio Agent Connect (TAC) is a Python SDK — middleware (not an agent runtime) that
enables LLM applications (OpenAI Agents SDK, Bedrock, LangChain, etc.) to use Twilio
primitives: Conversation Memory for memory, Conversation Orchestrator for conversations,
and ConversationRelay for voice.

## Installation

```bash
pip install twilio-agent-connect
# with the optional FastAPI server
pip install "twilio-agent-connect[server]"
```

## Quick start

```python
from tac import TAC, TACConfig

tac = TAC(TACConfig(
    api_key="SK...",
    api_secret="...",
    conversation_configuration_id="CF...",
))
```

See the [API Reference](api/core.md) for the full public API.
