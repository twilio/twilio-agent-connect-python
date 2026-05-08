"""
Example: Using LangChain with TAC

Demonstrates how to connect Twilio Agent Connect with LangChain
for voice and SMS channels.

Prerequisites:
    pip install langchain-core langchain-openai

Environment Variables:
    OPENAI_API_KEY - Your OpenAI API key
"""

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from tac import TAC, TACConfig
from tac.adapters import MemoryPromptBuilder
from tac.channels.sms import SMSChannel, SMSChannelConfig
from tac.channels.voice import VoiceChannel, VoiceChannelConfig
from tac.core.logging import get_logger
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.server import TACFastAPIServer

load_dotenv()

logger = get_logger(__name__)

# Initialize TAC with configuration from environment variables
tac = TAC(config=TACConfig.from_env())

# Create channel handlers for Voice and SMS
voice_channel = VoiceChannel(tac, config=VoiceChannelConfig(memory_mode="once"))
sms_channel = SMSChannel(tac, config=SMSChannelConfig(memory_mode="always"))

# Create LangChain LLM
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

# Create prompt template with conversation history support
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a customer service agent speaking with a user over voice or SMS. "
            "Keep responses short and conversational — a sentence or two. "
            "Do not use markdown, asterisks, bullets, or emojis; your words will be "
            "spoken aloud or sent as plain text.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{user_message}"),
    ]
)

# Create chain using LCEL (pipe operator)
chain = prompt_template | llm

# Store conversation history per conversation
conversation_history: dict[str, list[HumanMessage | AIMessage]] = {}


async def handle_message_ready(
    user_message: str,
    context: ConversationSession,
    memory_response: TACMemoryResponse | None,
) -> str:
    """
    Callback invoked when a message is ready to be processed.

    This example uses LangChain with LCEL to process the message
    and maintains conversation history per conversation.

    Args:
        user_message: The customer's message text
        context: Session data (conversation_id, channel, profile, etc.)
        memory_response: Optional retrieved memories (observations, summaries, communications)

    Returns:
        Response string to send to the channel
    """
    conv_id = context.conversation_id

    try:
        # Initialize conversation history for new conversations
        if conv_id not in conversation_history:
            conversation_history[conv_id] = []

        # Compose user message with memory context (user message first, then memory)
        prompt = MemoryPromptBuilder.compose(
            system_prompt=user_message,
            memory_response=memory_response,
            context=context,
        )

        # Invoke the chain with conversation history and composed prompt
        response = await chain.ainvoke(
            {
                "chat_history": conversation_history[conv_id],
                "user_message": prompt,
            }
        )

        # Extract text content from the response
        llm_response = response.content if hasattr(response, "content") else str(response)

        # Update conversation history
        conversation_history[conv_id].append(HumanMessage(content=user_message))
        conversation_history[conv_id].append(AIMessage(content=llm_response))

        return llm_response

    except Exception as e:
        logger.error("Error processing message", conversation_id=conv_id, error=str(e))
        return "Sorry, I encountered an error processing your message."


# Register the message handler callback
tac.on_message_ready(handle_message_ready)

if __name__ == "__main__":
    # TACFastAPIServer creates a FastAPI app with all required endpoints:
    # - /twiml: Voice call webhook (returns TwiML with ConversationRelay)
    # - /ws: WebSocket endpoint for Voice channel
    # - /webhook: Conversation webhook for all channels
    server = TACFastAPIServer(
        tac=tac, voice_channel=voice_channel, messaging_channels=[sms_channel]
    )
    server.start()
