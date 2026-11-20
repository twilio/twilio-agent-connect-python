# TAC Tools

Tool framework for LLM function calling that works with OpenAI and Anthropic APIs. Supports dependency injection to hide runtime values like API keys and client instances from LLM schemas.

## Core Concepts

### TACTool

The `TACTool` class represents a tool/function that can be used with LLMs:

- `name`: Function name
- `description`: What the tool does
- `params_json_schema`: JSON Schema for parameters (auto-generated from type hints)
- `implementation`: Property that returns an async callable with clean signature for LLM SDK introspection
- `configure_injection()`: Configure runtime dependencies to inject when the tool is called
- `to_openai_format()`: Convert to OpenAI function calling format
- `to_anthropic_format()`: Convert to Anthropic tool format
- `to_openai_agents_sdk_tool()`: Convert to an OpenAI Agents SDK `FunctionTool` (requires `pip install openai-agents`)

### Dependency Injection

Parameters marked with `Annotated[T, InjectedToolArg]` are:
- Hidden from the LLM schema
- Injected at runtime via `configure_injection()`
- Not visible to LLM SDK introspection

This allows tools to use API clients, credentials, and session context without exposing them to the LLM.

## Creating Tools

### Using the Decorator

The `@function_tool()` decorator automatically extracts function metadata:

```python
from tac.tools import function_tool

@function_tool()
def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> dict:
    """Calculate tip amount and total bill."""
    tip_amount = bill_amount * (tip_percentage / 100)
    return {"tip": tip_amount, "total": bill_amount + tip_amount}
```

Schema generation:
- Name: Uses function name (override with `name=` parameter)
- Description: Uses docstring (override with `description=` parameter)
- Parameters: Extracted from type hints
  - `str` → `"string"`
  - `int` → `"integer"`
  - `float` → `"number"`
  - `bool` → `"boolean"`
  - `list[T]` → `"array"` with items type T
  - `Optional[T]` → same as T but not required
  - `Literal["a", "b"]` → `"string"` with enum constraint
- Required vs Optional: Parameters with default values are optional, others are required

### With Dependency Injection

Use `Annotated[T, InjectedToolArg]` to inject runtime dependencies:

```python
from typing import Annotated
from tac.tools import function_tool, InjectedToolArg
from tac.context.memory import MemoryClient

async def retrieve_profile_memory(
    query: str,
    conversation_memory_client: Annotated[MemoryClient, InjectedToolArg],
    profile_id: Annotated[str, InjectedToolArg],
) -> dict:
    """Search and retrieve relevant memories for the current profile."""
    memory_response = await conversation_memory_client.retrieve_memory(
        profile_id=profile_id,
        query=query,
    )
    return memory_response.model_dump(by_alias=True, exclude_none=True)

# Wrap with decorator and configure injection
tool = function_tool()(retrieve_profile_memory)
tool.configure_injection(
    conversation_memory_client=my_conversation_memory_client,
    profile_id="prof_123"
)

# LLM only sees: retrieve_profile_memory(query: str)
# conversation_memory_client and profile_id are hidden from schema
```

### Manual Creation

```python
from tac.tools import create_tool

def my_function(param1: str, param2: int = 5) -> str:
    return f"Got {param1} and {param2}"

tool = create_tool(
    name="my_tool",
    description="Does something useful",
    params_json_schema={
        "type": "object",
        "properties": {
            "param1": {"type": "string"},
            "param2": {"type": "integer"}
        },
        "required": ["param1"]
    },
    implementation=my_function
)
```

## Built-in Tools

### Memory Tool

Retrieve memories from Twilio Conversation Memory:

```python
from tac import TAC, TACConfig
from tac.tools.memory import create_memory_tool
from tac.models.session import ConversationSession

# Initialize TAC
tac = TAC(config=TACConfig.from_env())

# Create session
session = ConversationSession(
    profile_id="prof_123",
    conversation_id="conv_456",
    channel="sms"
)

# Create tool with injected dependencies
memory_tool = create_memory_tool(tac.conversation_memory_client, session)

# LLM only sees: retrieve_profile_memory(query: str)
# Execute tool (async)
result = await memory_tool(query="user preferences about food")
```

### Knowledge Tool

Search a knowledge base via Twilio Conversation Memory:

```python
from tac.tools.knowledge import create_knowledge_tool

knowledge_tool = await create_knowledge_tool(
    knowledge_client=tac.knowledge_client,
    knowledge_base_id="know_knowledgebase_000000000000000000000000",
    name="search_product_faq",          # optional; defaults to search_<kb_display_name>
    description="Search product FAQs",  # optional; defaults to the KB's description
    top_k=3,
)

# LLM only sees: search_product_faq(query: str)
results = await knowledge_tool(query="What is the return policy?")
```

### Handoff Tool

Hand the conversation off to a human agent via a Twilio Studio Flow (e.g. one that routes to Flex). Works on both voice and SMS. Requires `TWILIO_STUDIO_HANDOFF_FLOW_SID` in your environment (or `tac.config.studio_handoff_flow_sid`).

```python
from tac.tools.handoff import create_studio_handoff_tool

handoff_tool = create_studio_handoff_tool(
    tac,
    session,
    attributes={"department": "support", "priority": "normal"},
)

# LLM only sees: handoff(reason: str)
# On digital channels, posts to the Studio Flow Executions API.
# On voice, stores the payload on the session so the voice channel
# can send the WS "end" message with handoffData after the final reply.
```

The tool also sets the conversation to `INACTIVE` and clears status callbacks so TAC stops receiving webhooks while the human handles the conversation. See `getting_started/examples/features/handoff.py` for a complete example.

## Implementation Property

The `implementation` property returns an async callable with clean signature:

```python
tool = function_tool()(my_function)

# Get callable with clean signature (non-injected params only)
callable_func = tool.implementation

# Inspect signature
import inspect
sig = inspect.signature(callable_func)
# Only non-injected parameters appear in signature

# Call it (always async, handles sync/async implementations)
result = await callable_func(param1="value")
```

This property is cached and automatically cleared when injection configuration changes.

## Schema Output Examples

Function:
```python
@function_tool()
def send_message(to: str, message: str, priority: int = 1) -> dict:
    """Send a message to a recipient."""
    pass
```

OpenAI format:
```json
{
  "type": "function",
  "function": {
    "name": "send_message",
    "description": "Send a message to a recipient.",
    "parameters": {
      "type": "object",
      "properties": {
        "to": {"type": "string"},
        "message": {"type": "string"},
        "priority": {"type": "integer"}
      },
      "required": ["to", "message"]
    }
  }
}
```

Anthropic format:
```json
{
  "name": "send_message",
  "description": "Send a message to a recipient.",
  "input_schema": {
    "type": "object",
    "properties": {
      "to": {"type": "string"},
      "message": {"type": "string"},
      "priority": {"type": "integer"}
    },
    "required": ["to", "message"]
  }
}
```
