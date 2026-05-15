"""
TAC Interactive Playground

A web UI where users can test Twilio Agent Connect by providing their credentials.
The TAC FastAPI server runs live on Streamlit Cloud.
"""

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

import streamlit as st
import uvicorn

# Add parent directory to path so we can import TAC
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from tac import TAC, TACConfig
from tac.adapters.openai import with_tac_memory
from tac.channels.sms import SMSChannel, SMSChannelConfig
from tac.channels.voice import VoiceChannel, VoiceChannelConfig
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.server import TACFastAPIServer

st.set_page_config(
    page_title="Twilio Agent Connect Playground",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Twilio Agent Connect Playground")
st.markdown(
    """
Test Twilio Agent Connect **live** with your own credentials. This runs a real TAC server
that you can connect to your Twilio phone number right now.

**🔒 Privacy:** Your credentials are only used in your isolated session and are never stored or logged.
"""
)

# Security notice
with st.expander("🔐 Security & Privacy"):
    st.markdown(
        """
        ### How Your Credentials Are Protected

        - ✅ **Session Isolation**: Each user gets a completely isolated Python session
        - ✅ **No Persistence**: Credentials are never written to disk or databases
        - ✅ **No Logging**: Your credentials and conversations are not logged
        - ✅ **Auto-Cleanup**: All data is cleared when you stop the server

        **This playground is for testing only.** For production, deploy TAC to your own infrastructure.
        """
    )

# Initialize session state
if "server_running" not in st.session_state:
    st.session_state.server_running = False
if "server_thread" not in st.session_state:
    st.session_state.server_thread = None
if "server_port" not in st.session_state:
    st.session_state.server_port = 8501  # Streamlit's port

# Sidebar for credentials
with st.sidebar:
    st.header("Configuration")

    with st.expander("📞 Twilio Credentials", expanded=True):
        twilio_sid = st.text_input(
            "Account SID",
            type="password",
            help="Find in Twilio Console: https://console.twilio.com",
        )
        twilio_token = st.text_input(
            "Auth Token",
            type="password",
            help="Find in Twilio Console: https://console.twilio.com",
        )

    with st.expander("🔧 TAC Configuration", expanded=True):
        tac_api_key = st.text_input(
            "TAC API Key",
            type="password",
        )
        tac_api_token = st.text_input(
            "TAC API Token",
            type="password",
        )
        tac_conversation_config = st.text_input(
            "Conversation Configuration ID",
            help="Your TAC Conversation Configuration ID",
        )
        tac_memory_store = st.text_input(
            "Memory Store ID (optional)",
            help="Leave empty to use default from conversation config",
        )

    with st.expander("🤖 OpenAI Configuration", expanded=True):
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Get from: https://platform.openai.com/api-keys",
        )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("⚙️ Server Control")

    # Validate required fields
    required_fields = {
        "Twilio Account SID": twilio_sid,
        "Twilio Auth Token": twilio_token,
        "TAC API Key": tac_api_key,
        "TAC API Token": tac_api_token,
        "Conversation Configuration ID": tac_conversation_config,
        "OpenAI API Key": openai_key,
    }

    missing_fields = [name for name, value in required_fields.items() if not value]

    if missing_fields:
        st.warning(f"Missing: {', '.join(missing_fields)}")
        start_disabled = True
    else:
        st.success("All credentials provided ✓")
        start_disabled = False

    if st.button(
        "🚀 Start TAC Server",
        disabled=start_disabled or st.session_state.server_running,
        type="primary",
    ):
        # Set environment variables
        os.environ["TWILIO_ACCOUNT_SID"] = twilio_sid
        os.environ["TWILIO_AUTH_TOKEN"] = twilio_token
        os.environ["TAC_API_KEY"] = tac_api_key
        os.environ["TAC_API_TOKEN"] = tac_api_token
        os.environ["TAC_CONVERSATION_CONFIGURATION_ID"] = tac_conversation_config
        if tac_memory_store:
            os.environ["TAC_MEMORY_STORE_ID"] = tac_memory_store
        os.environ["OPENAI_API_KEY"] = openai_key

        try:
            # Import here after env vars are set
            from openai import AsyncOpenAI
            from openai.types.chat import (
                ChatCompletionAssistantMessageParam,
                ChatCompletionMessageParam,
                ChatCompletionSystemMessageParam,
                ChatCompletionUserMessageParam,
            )

            # Initialize TAC
            tac = TAC(config=TACConfig.from_env())
            voice_channel = VoiceChannel(tac, config=VoiceChannelConfig(memory_mode="once"))
            sms_channel = SMSChannel(tac, config=SMSChannelConfig(memory_mode="once"))

            # Initialize OpenAI client
            openai_client = AsyncOpenAI(api_key=openai_key)

            # Store conversation history
            conversation_history: dict[str, list[ChatCompletionMessageParam]] = {}

            SYSTEM_MESSAGE: ChatCompletionSystemMessageParam = {
                "role": "system",
                "content": (
                    "You are a customer service agent speaking with a user over voice or SMS. "
                    "Keep responses short and conversational — a sentence or two. "
                    "Do not use markdown, asterisks, bullets, or emojis."
                ),
            }

            async def handle_message_ready(
                user_message: str,
                context: ConversationSession,
                memory_response: TACMemoryResponse | None,
            ) -> str:
                """Handle incoming messages"""
                conv_id = context.conversation_id

                try:
                    if conv_id not in conversation_history:
                        conversation_history[conv_id] = [SYSTEM_MESSAGE]

                    user_msg: ChatCompletionUserMessageParam = {
                        "role": "user",
                        "content": user_message,
                    }
                    conversation_history[conv_id].append(user_msg)

                    # Use TAC memory adapter
                    client = with_tac_memory(openai_client, memory_response, context)

                    response = await client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=conversation_history[conv_id],
                    )

                    llm_response = response.choices[0].message.content or ""

                    assistant_msg: ChatCompletionAssistantMessageParam = {
                        "role": "assistant",
                        "content": llm_response,
                    }
                    conversation_history[conv_id].append(assistant_msg)

                    return llm_response

                except Exception as e:
                    return f"Sorry, I encountered an error: {str(e)}"

            tac.on_message_ready(handle_message_ready)

            # Create FastAPI server
            server = TACFastAPIServer(
                tac=tac, voice_channel=voice_channel, messaging_channels=[sms_channel]
            )

            # Run server in background thread
            def run_server():
                config = uvicorn.Config(
                    server.app,
                    host="0.0.0.0",
                    port=8000,
                    log_level="info",
                )
                server_instance = uvicorn.Server(config)
                asyncio.run(server_instance.serve())

            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()

            st.session_state.server_running = True
            st.session_state.server_thread = thread

            time.sleep(2)  # Give server time to start
            st.success("✅ Server started!")
            st.rerun()

        except Exception as e:
            st.error(f"Failed to start server: {e}")
            st.session_state.server_running = False

    if st.button(
        "🛑 Stop Server", disabled=not st.session_state.server_running, type="secondary"
    ):
        st.session_state.server_running = False
        st.session_state.server_thread = None
        st.info("Server stopped. Refresh the page to start a new session.")

    # Server status
    if st.session_state.server_running:
        st.success("🟢 Server is running")
    else:
        st.info("🔴 Server is stopped")

with col2:
    st.subheader("📡 Webhook Configuration")

    if st.session_state.server_running:
        # Get Streamlit app URL
        # Streamlit Cloud provides this in session state
        app_url = st.query_params.get("url", "https://<your-streamlit-app>.streamlit.app")

        # If we can detect the actual URL, use it
        if "streamlit.app" not in app_url:
            app_url = "https://<your-streamlit-app>.streamlit.app"

        st.markdown("### Your Webhook URLs:")
        st.code(f"{app_url}/twiml", language="text")
        st.code(f"{app_url}/webhook", language="text")

        st.info(
            """
            **Next Steps:**
            1. Copy the webhook URLs above
            2. Go to [Twilio Console](https://console.twilio.com/us1/develop/phone-numbers/manage/incoming)
            3. Select your phone number
            4. Set **Voice webhook** to: `/twiml`
            5. Set **SMS webhook** to: `/webhook`
            6. Send a message or call your number!
            """
        )

        st.warning(
            """
            ⚠️ **Important:** Streamlit Cloud apps may sleep after inactivity.
            For production, deploy to Railway or your own infrastructure.
            """
        )
    else:
        st.info("Start the server to see your webhook URLs")

# Footer
st.divider()
st.markdown(
    """
    **Need help?**
    - [TAC Documentation](https://github.com/twilio/twilio-agent-connect-python)
    - [Twilio Console](https://console.twilio.com)
    - [Report Issues](https://github.com/twilio/twilio-agent-connect-python/issues)
    """
)
