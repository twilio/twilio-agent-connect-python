"""
Example: Framework-Agnostic Memory Injection

Shows how to use TAC memory with ANY agent framework (OpenAI, AWS Bedrock, Azure AI,
GCP Vertex AI, custom agents, etc.).

Key Concepts:
1. TAC provides memory via `memory_response` and `context` parameters
2. Use `MemoryPromptBuilder.build()` to format memory and profile data
3. Inject the formatted prompt into your agent's system prompt or context
4. Works with any LLM or agent framework

For SDK-specific examples with more detail, see the `partners/` directory.
"""

from typing import Any

from dotenv import load_dotenv

from tac import TAC, TACConfig
from tac.adapters.prompt_builder import MemoryPromptBuilder
from tac.channels.sms import SMSChannel
from tac.channels.voice import VoiceChannel
from tac.core.logging import get_logger
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.server import TACFastAPIServer

load_dotenv()

logger = get_logger(__name__)

# Initialize TAC and channels
tac = TAC(config=TACConfig.from_env())
voice_channel = VoiceChannel(tac)
sms_channel = SMSChannel(tac)

# Store conversation history per conversation
conversation_messages: dict[str, list[Any]] = {}


# No custom formatting needed! TAC provides MemoryPromptBuilder for this.
# You can use it as-is or customize it for your specific needs.


async def handle_message_ready(
    user_message: str,
    context: ConversationSession,
    memory_response: TACMemoryResponse | None,
) -> str:
    """
    Process incoming messages using manual memory injection.

    This pattern works with ANY agent framework:
    - OpenAI (see partners/ examples for adapter)
    - AWS Bedrock (see partners/aws_*.py examples)
    - Azure AI (upcoming example)
    - GCP Vertex AI (upcoming example)
    - Custom agents
    """
    conv_id = context.conversation_id

    try:
        if conv_id not in conversation_messages:
            conversation_messages[conv_id] = []

        # Build your system prompt with memory context using compose()
        # This is the official way to format TAC memory for any agent/framework
        base_prompt = (
            "You are a customer service agent speaking with a user over voice or SMS. "
            "Keep responses short and conversational — a sentence or two. "
            "Do not use markdown, asterisks, bullets, or emojis; your words will be "
            "spoken aloud or sent as plain text."
        )
        system_prompt = MemoryPromptBuilder.compose(
            base_prompt,
            memory_response=memory_response,
            context=context,
        )

        # Add user message to conversation history
        conversation_messages[conv_id].append({"role": "user", "content": user_message})

        # Now use system_prompt with your agent/LLM
        # Add it to your agent's messages, system parameter, or context
        # See SDK-specific examples in partners/ for detailed usage
        #
        # Example usage patterns:
        # - OpenAI: messages=[{"role": "system", "content": system_prompt}, ...]
        # - AWS Bedrock: system=[{"text": system_prompt}]
        # - Azure AI: messages=[{"role": "system", "content": system_prompt}, ...]
        # - Custom: Pass system_prompt to your agent however needed

        # For this example, simulate a response (showing system_prompt is used)
        llm_response = (
            f"[Simulated response with system prompt: {system_prompt[:50]}...]\n\n"
            f"Received: {user_message}"
        )

        conversation_messages[conv_id].append({"role": "assistant", "content": llm_response})

        return llm_response

    except Exception as e:
        logger.error("Error processing message", conversation_id=conv_id, error=str(e))
        return "Sorry, I encountered an error processing your message."


# Register the message handler
tac.on_message_ready(handle_message_ready)

if __name__ == "__main__":
    """
    Key Takeaways:

    1. TAC provides memory via `memory_response` and `context` parameters

    2. Use `MemoryPromptBuilder.compose(base_prompt, memory_response, context)` to
       automatically append memory to your system prompt - no if/else needed!

    3. Pass the composed prompt to your agent/LLM:
       - OpenAI: Add to system message
       - AWS Bedrock: Add to system prompt
       - Azure AI: Add to system message
       - GCP Vertex AI: Add to system instructions
       - Custom agents: Add wherever you need context

    This approach is framework-agnostic and works with ANY agent or LLM.

    For automatic injection with zero config, see the OpenAI adapter examples:
    - `examples/partners/openai_chat_completions.py` - Chat Completions API
    - `examples/partners/openai_responses_api.py` - Responses API
    """
    server = TACFastAPIServer(
        tac=tac, voice_channel=voice_channel, messaging_channels=[sms_channel]
    )
    server.start()
