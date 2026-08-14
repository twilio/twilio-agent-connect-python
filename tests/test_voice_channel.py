"""Tests for Voice Channel."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tac import TAC
from tac.channels.voice import VoiceChannel, generate_twiml
from tac.models.conversation import ConversationResponse
from tac.models.handoff import PendingHandoffData
from tac.models.memory import MemoryRetrievalResponse
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse
from tac.models.voice import (
    InterruptMessage,
    PromptMessage,
    SetupMessage,
    TwiMLOptions,
)


def get_test_config() -> dict:
    """Get a valid test configuration."""
    return {
        "account_sid": "ACtest123",
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_token",
        "conversation_configuration_id": "conv_configuration_test123",
        "phone_number": "+15551234567",
        "voice_public_domain": "example.com",
    }


class TestVoiceChannel:
    """Test Voice Channel functionality."""

    def test_initialization(self) -> None:
        """Test Voice channel initialization."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        assert channel.tac == tac
        assert channel._websocket_manager is not None
        assert len(channel._websocket_manager) == 0

    def test_get_channel_name(self) -> None:
        """Test get_channel_name returns 'voice'."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        assert channel.get_channel_name() == "VOICE"

    @pytest.mark.asyncio
    async def test_handle_prompt_message_without_memory_retrieval(self) -> None:
        """Test handling prompt message when memory_mode="never"."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Setup conversation first
        channel._start_conversation("CALL123", "profile_test_123")

        # Create prompt message
        prompt_msg = PromptMessage(
            type="prompt",
            conversationId="CALL123",
            voicePrompt="Hello, I need help",
        )

        # Call handler directly
        await channel._handle_prompt("CALL123", prompt_msg)

        # With memory_mode="never", memory is not fetched - test passes if no exception

    @pytest.mark.asyncio
    async def test_handle_prompt_message_with_memory_retrieval(self) -> None:
        """Test handling prompt message retrieves memory when memory_mode="always"."""
        # Create config with memory enabled
        config = get_test_config()
        from tac.core.config import TwilioMemoryConfig

        config["memory_config"] = TwilioMemoryConfig(trait_groups=["Contact"])
        tac = TAC(config)

        # Manually create memory_client for this test
        from tac.context.memory import MemoryClient

        tac.conversation_memory_client = MemoryClient(
            store_id="MGtest123",
            api_key=tac.config.api_key,
            api_secret=tac.config.api_secret,
        )

        # Mock the memory retrieval
        mock_memory_response = MemoryRetrievalResponse(
            observations=[],
            summaries=[],
            communications=[],
        )
        tac.conversation_memory_client.retrieve_memory = AsyncMock(
            return_value=mock_memory_response
        )

        # Create channel with memory_mode enabled (default is "never")
        channel = VoiceChannel(tac, config={"memory_mode": "always"})

        # Setup conversation with profile_id
        channel._start_conversation("CALL123", "profile_test_123")

        # Create prompt message
        prompt_msg = PromptMessage(
            type="prompt",
            conversationId="CALL123",
            voicePrompt="Hello, I need help",
        )

        # Call handler directly
        await channel._handle_prompt("CALL123", prompt_msg)

        # Verify memory retrieval was called
        tac.conversation_memory_client.retrieve_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_conversation_populates_ai_agent_info(self) -> None:
        """_initialize_conversation resolves the agent participant by its address
        (TAC's phone number) and accepts the AGENT type voice uses, setting
        ai_agent_info — matching the messaging channels' address-based match."""
        from tac.models.conversation import (
            ConversationResponse,
            ParticipantAddress,
            ParticipantResponse,
        )

        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        conversation = ConversationResponse(id="conv_abc", accountId="ACtest123", status="ACTIVE")
        customer = ParticipantResponse(
            id="part_customer",
            conversationId="conv_abc",
            accountId="ACtest123",
            name="Customer",
            type="CUSTOMER",
            profileId="profile_xyz",
            addresses=[ParticipantAddress(channel="VOICE", address="+15559998888")],
        )
        # Voice's agent participant is typed AGENT (not AI_AGENT) and sits at
        # TAC's configured phone number.
        agent = ParticipantResponse(
            id="part_agent",
            conversationId="conv_abc",
            accountId="ACtest123",
            name="Agent",
            type="AGENT",
            addresses=[ParticipantAddress(channel="VOICE", address="+15551234567")],
        )

        co_client = MagicMock()
        co_client.list_conversations = AsyncMock(return_value=[conversation])
        co_client.list_participants = AsyncMock(return_value=[customer, agent])
        tac.conversation_orchestrator_client = co_client

        setup_msg = SetupMessage(type="setup", callSid="CALL123", **{"from": "+15559998888"})

        conv_id, _ = await channel._initialize_conversation("CALL123", setup_msg, MagicMock())

        assert conv_id == "conv_abc"
        session = channel._conversations["conv_abc"]
        assert session.ai_agent_info is not None
        assert session.ai_agent_info.participant_id == "part_agent"
        assert session.ai_agent_info.address == "+15551234567"
        # Customer side still resolved as before.
        assert session.author_info is not None
        assert session.author_info.address == "+15559998888"

    @pytest.mark.asyncio
    async def test_initialize_conversation_skips_human_agent_for_ai_agent_info(self) -> None:
        """A redirected/escalated call's HUMAN_AGENT participant at TAC's address
        must not be treated as the AI agent — its type is not an agent type, so
        ai_agent_info stays None."""
        from tac.models.conversation import (
            ConversationResponse,
            ParticipantAddress,
            ParticipantResponse,
        )

        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        conversation = ConversationResponse(id="conv_def", accountId="ACtest123", status="ACTIVE")
        customer = ParticipantResponse(
            id="part_customer",
            conversationId="conv_def",
            accountId="ACtest123",
            name="Customer",
            type="CUSTOMER",
            addresses=[ParticipantAddress(channel="VOICE", address="+15559998888")],
        )
        # Human agent sits at TAC's address but is a real person — must not be
        # adopted as the AI agent.
        human_agent = ParticipantResponse(
            id="part_human",
            conversationId="conv_def",
            accountId="ACtest123",
            name="Human Agent",
            type="HUMAN_AGENT",
            addresses=[ParticipantAddress(channel="VOICE", address="+15551234567")],
        )

        co_client = MagicMock()
        co_client.list_conversations = AsyncMock(return_value=[conversation])
        co_client.list_participants = AsyncMock(return_value=[customer, human_agent])
        tac.conversation_orchestrator_client = co_client

        setup_msg = SetupMessage(type="setup", callSid="CALL456", **{"from": "+15559998888"})

        conv_id, _ = await channel._initialize_conversation("CALL456", setup_msg, MagicMock())

        session = channel._conversations[conv_id]
        assert session.ai_agent_info is None

    @pytest.mark.asyncio
    async def test_handle_interrupt_message(self) -> None:
        """Test handling interrupt message."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Setup conversation first
        channel._start_conversation("CALL123", None)

        # Create interrupt message
        interrupt_msg = InterruptMessage(
            type="interrupt",
            utteranceUntilInterrupt="Hello, I was saying...",
            durationUntilInterruptMs=1500,
        )

        # Call handler directly
        channel._handle_interrupt("CALL123", interrupt_msg)

        # Test passes if no exception is raised

    @pytest.mark.asyncio
    async def test_send_response(self) -> None:
        """Test sending voice response through websocket."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Start conversation directly
        channel._start_conversation("CALL123", "profile_test")

        # Mock websocket and register it with the manager
        mock_websocket = AsyncMock()
        channel._websocket_manager.add_websocket("CALL123", mock_websocket)

        # Send response without role
        await channel.send_response("CALL123", "Hello there")

        # Verify websocket.send_text was called once
        assert mock_websocket.send_text.call_count == 1

        # Send response with role
        await channel.send_response("CALL123", "How can I help?", role="assistant")

        # Verify websocket.send_text was called again
        assert mock_websocket.send_text.call_count == 2

    @pytest.mark.asyncio
    async def test_send_response_flushes_pending_handoff_after_final_response(self) -> None:
        """After the LLM's final response, a pending handoff end message is flushed
        and cleared on the session."""
        import json

        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        channel._start_conversation("CALL123", "profile_test")
        session = channel._conversations["CALL123"]
        session.pending_handoff_data = PendingHandoffData(
            handoff_data='{"conversationId":"CALL123"}',
        )

        mock_websocket = AsyncMock()
        channel._websocket_manager.add_websocket("CALL123", mock_websocket)

        await channel.send_response("CALL123", "Transferring you now.")

        assert mock_websocket.send_text.call_count == 2
        final_call = mock_websocket.send_text.call_args_list[-1][0][0]
        assert json.loads(final_call) == {
            "type": "end",
            "handoffData": '{"conversationId":"CALL123"}',
        }
        assert session.pending_handoff_data is None

    @pytest.mark.asyncio
    async def test_send_response_without_websocket(self) -> None:
        """Test sending response without active websocket logs error."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Start conversation directly
        channel._start_conversation("CALL123", "profile_test")

        # No websocket registered in manager
        # (don't add websocket to manager, so lookup returns None)

        # Should log error and return early (no exception raised)
        await channel.send_response("CALL123", "Hello there")

    @pytest.mark.asyncio
    async def test_end_conversation_cleanup(self) -> None:
        """Test ending conversation cleans up WebSocket but keeps conversation tracked."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Start conversation directly
        channel._start_conversation("CALL123", "profile_test")

        # Add a mock websocket to the manager
        mock_websocket = MagicMock()
        channel._websocket_manager.add_websocket("CALL123", mock_websocket)

        # Verify websocket is registered
        assert channel._websocket_manager.has_websocket("CALL123")
        assert "CALL123" in channel._conversations

        # Clean up connection (WebSocket only)
        await channel._cleanup_connection("CALL123")

        # Verify WebSocket cleanup but conversation still tracked
        assert not channel._websocket_manager.has_websocket("CALL123")
        assert "CALL123" in channel._conversations

    @pytest.mark.asyncio
    async def test_process_webhook_conversation_closed(self) -> None:
        """Test that process_webhook cleans up on CONVERSATION_UPDATED with CLOSED status."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Start a conversation
        channel._start_conversation("CONV123", "profile_123")
        assert "CONV123" in channel._conversations

        # Process CONVERSATION_UPDATED with CLOSED status
        webhook_data = {
            "eventType": "CONVERSATION_UPDATED",
            "data": {"id": "CONV123", "status": "CLOSED"},
        }
        await channel.process_webhook(webhook_data)

        # Should clean up the conversation
        assert "CONV123" not in channel._conversations

    @pytest.mark.asyncio
    async def test_process_webhook_conversation_inactive(self) -> None:
        """Test that INACTIVE invalidates cached memory when memory_mode='once'."""

        from tac.channels.voice.config import VoiceChannelConfig

        tac = TAC(get_test_config())
        # Enable "once" mode to trigger cache invalidation
        channel = VoiceChannel(tac, config=VoiceChannelConfig(memory_mode="once"))

        # Start a conversation
        session = channel._start_conversation("CONV123", "profile_123")
        assert "CONV123" in channel._conversations

        # Simulate cached memory with proper type
        empty_response = MemoryRetrievalResponse(observations=[], summaries=[], communications=[])
        session.cached_memory = TACMemoryResponse(empty_response)

        # Process CONVERSATION_UPDATED with INACTIVE status
        webhook_data = {
            "eventType": "CONVERSATION_UPDATED",
            "data": {"id": "CONV123", "status": "INACTIVE"},
        }
        await channel.process_webhook(webhook_data)

        # Should NOT clean up conversation (only CLOSED triggers cleanup)
        assert "CONV123" in channel._conversations
        # But should invalidate cached memory (because memory_mode="once")
        assert session.cached_memory is None

    @pytest.mark.asyncio
    async def test_process_webhook_not_tracked_locally(self) -> None:
        """Test that process_webhook ignores conversations not tracked locally."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Don't start conversation - not tracked locally

        # Process CONVERSATION_UPDATED for unknown conversation
        webhook_data = {
            "eventType": "CONVERSATION_UPDATED",
            "data": {"id": "CONV_UNKNOWN", "status": "CLOSED"},
        }
        await channel.process_webhook(webhook_data)

        # Should not raise, just ignore
        assert "CONV_UNKNOWN" not in channel._conversations

    @pytest.mark.asyncio
    async def test_process_webhook_filters_communication_created(self) -> None:
        """Test that process_webhook filters COMMUNICATION_CREATED by channel."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        channel._start_conversation("CONV123", "profile_123")

        # VOICE communication should be accepted (but voice doesn't process COMMUNICATION_CREATED)
        webhook_data = {
            "eventType": "COMMUNICATION_CREATED",
            "data": {
                "conversationId": "CONV123",
                "author": {"channel": "VOICE", "address": "+1234567890"},
                "content": {"text": "hello"},
            },
        }
        await channel.process_webhook(webhook_data)

        # SMS communication should be rejected
        webhook_data_sms = {
            "eventType": "COMMUNICATION_CREATED",
            "data": {
                "conversationId": "CONV123",
                "author": {"channel": "SMS", "address": "+1234567890"},
                "content": {"text": "hello"},
            },
        }
        await channel.process_webhook(webhook_data_sms)

        # Conversation should still be there (voice doesn't process COMMUNICATION_CREATED)
        assert "CONV123" in channel._conversations

    @pytest.mark.asyncio
    async def test_message_callback_integration(self) -> None:
        """Test message callback is invoked with conversation context."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Callback to capture context
        captured_context = None
        captured_memories = None
        captured_user_message = None

        async def message_callback(
            user_message: str,
            context: ConversationSession,
            memory_response: TACMemoryResponse | None,
        ) -> None:
            nonlocal captured_context, captured_memories, captured_user_message
            captured_context = context
            captured_memories = memory_response
            captured_user_message = user_message

        tac.on_message_ready(message_callback)

        # Setup conversation first
        channel._start_conversation("CALL123", "profile_test")

        # Create and handle prompt message
        prompt_msg = PromptMessage(
            type="prompt",
            conversationId="CALL123",
            voicePrompt="Test message",
        )
        await channel._handle_prompt("CALL123", prompt_msg)

        # Verify callback was invoked
        assert captured_context is not None
        assert captured_context.conversation_id == "CALL123"
        assert captured_context.profile_id == "profile_test"
        assert captured_context.channel == "VOICE"
        # Voice channel doesn't fetch memory, so it should be None
        assert captured_memories is None
        assert captured_user_message == "Test message"

    @pytest.mark.asyncio
    async def test_callback_auto_send_response(self) -> None:
        """Test that callback returning a string automatically sends response via websocket."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Callback that returns a string (should auto-send)
        async def message_callback(
            user_message: str,
            context: ConversationSession,
            memory_response: TACMemoryResponse | None,
        ) -> str:
            return "This is my automated response"

        tac.on_message_ready(message_callback)

        # Setup conversation
        channel._start_conversation("CALL_AUTO_SEND", "profile_auto_send")

        # Mock websocket and register it
        mock_websocket = AsyncMock()
        channel._websocket_manager.add_websocket("CALL_AUTO_SEND", mock_websocket)

        # Create and handle prompt message
        prompt_msg = PromptMessage(
            type="prompt",
            conversationId="CALL_AUTO_SEND",
            voicePrompt="Test message",
        )
        await channel._handle_prompt("CALL_AUTO_SEND", prompt_msg)

        # Verify websocket.send_text was called once with the auto-sent response
        assert mock_websocket.send_text.call_count == 1
        call_args = mock_websocket.send_text.call_args[0][0]
        assert "This is my automated response" in call_args

    @pytest.mark.asyncio
    async def test_callback_no_auto_send_on_none(self) -> None:
        """Test that callback returning None does not auto-send (manual send_response required)."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Callback that returns None (manual send_response flow)
        async def message_callback(
            user_message: str,
            context: ConversationSession,
            memory_response: TACMemoryResponse | None,
        ) -> None:
            # User will manually call channel.send_response() later
            pass

        tac.on_message_ready(message_callback)

        # Setup conversation
        channel._start_conversation("CALL_NO_AUTO", "profile_no_auto")

        # Mock websocket and register it
        mock_websocket = AsyncMock()
        channel._websocket_manager.add_websocket("CALL_NO_AUTO", mock_websocket)

        # Create and handle prompt message
        prompt_msg = PromptMessage(
            type="prompt",
            conversationId="CALL_NO_AUTO",
            voicePrompt="Test message",
        )
        await channel._handle_prompt("CALL_NO_AUTO", prompt_msg)

        # Verify websocket.send_text was NOT called (callback returned None)
        assert mock_websocket.send_text.call_count == 0

    @pytest.mark.asyncio
    async def test_handle_incoming_call(self) -> None:
        """Test handle_incoming_call generates valid TwiML with conversation_configuration."""
        from tac.channels.voice import VoiceChannelConfig

        tac = TAC(get_test_config())
        channel = VoiceChannel(
            tac,
            config=VoiceChannelConfig(
                default_twiml_options=TwiMLOptions(welcome_greeting="Welcome!"),
            ),
        )

        twiml = await channel.handle_incoming_call()

        assert '<?xml version="1.0" encoding="UTF-8"?>' in twiml
        assert "<Response>" in twiml
        assert '<Connect action="https://example.com/conversation-relay-callback">' in twiml
        assert "<ConversationRelay" in twiml
        assert 'url="wss://example.com/ws"' in twiml
        assert 'welcomeGreeting="Welcome!"' in twiml
        assert 'conversationConfiguration="conv_configuration_test123"' in twiml
        assert "</Connect>" in twiml
        assert "</Response>" in twiml

    @pytest.mark.asyncio
    async def test_handle_incoming_call_default_greeting(self) -> None:
        """Test handle_incoming_call uses default greeting."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        twiml = await channel.handle_incoming_call()

        assert 'welcomeGreeting="Hello! How can I assist you today?"' in twiml
        assert 'conversationConfiguration="conv_configuration_test123"' in twiml

    @pytest.mark.asyncio
    async def test_handle_incoming_call_websocket_url_via_customizer(self) -> None:
        """Test an on_inbound_call_twiml customizer can override websocket_url per call.

        websocket_url is a normal TwiMLOptions field, so it rides the same
        layered merge as every other attribute. This is the affinity-routed-host
        case (e.g. Azure Hosted Agents) appending a per-call token to the
        upgrade URL — done through the existing customizer, no new API surface.
        """
        from tac.channels.voice import VoiceChannelConfig
        from tac.models.voice import TwiMLRequest

        tac = TAC(get_test_config())
        channel = VoiceChannel(
            tac,
            config=VoiceChannelConfig(
                default_twiml_options=TwiMLOptions(welcome_greeting="Welcome!"),
            ),
        )

        override = "wss://example.com/ws?agent_session_id=CA123"

        async def customize(req: TwiMLRequest) -> TwiMLOptions:
            return TwiMLOptions(
                websocket_url=f"wss://example.com/ws?agent_session_id={req.call_sid}"
            )

        channel.on_inbound_call_twiml(customize)

        twiml = await channel.handle_incoming_call(
            twiml_request=TwiMLRequest.from_form({"CallSid": "CA123"})
        )

        # The customizer's URL is emitted verbatim...
        assert f'url="{override}"' in twiml
        # ...and the bare derived URL (without the query string) is NOT used.
        assert 'url="wss://example.com/ws"' not in twiml
        # The override only changes the URL; other layered fields still apply.
        assert 'welcomeGreeting="Welcome!"' in twiml
        assert 'conversationConfiguration="conv_configuration_test123"' in twiml

    @pytest.mark.asyncio
    async def test_handle_incoming_call_websocket_url_via_default_options(self) -> None:
        """Test default_twiml_options.websocket_url overrides the derived URL."""
        from tac.channels.voice import VoiceChannelConfig

        tac = TAC(get_test_config())
        override = "wss://static.example.com/socket"
        channel = VoiceChannel(
            tac,
            config=VoiceChannelConfig(
                default_twiml_options=TwiMLOptions(websocket_url=override),
            ),
        )

        twiml = await channel.handle_incoming_call()

        assert f'url="{override}"' in twiml
        assert 'url="wss://example.com/ws"' not in twiml

    @pytest.mark.asyncio
    async def test_handle_incoming_call_customizer_beats_default_options(self) -> None:
        """Test customizer websocket_url wins over default_twiml_options (precedence)."""
        from tac.channels.voice import VoiceChannelConfig
        from tac.models.voice import TwiMLRequest

        tac = TAC(get_test_config())
        channel = VoiceChannel(
            tac,
            config=VoiceChannelConfig(
                default_twiml_options=TwiMLOptions(websocket_url="wss://static.example.com/socket"),
            ),
        )

        async def customize(req: TwiMLRequest) -> TwiMLOptions:
            return TwiMLOptions(websocket_url="wss://per-call.example.com/ws")

        channel.on_inbound_call_twiml(customize)

        twiml = await channel.handle_incoming_call(
            twiml_request=TwiMLRequest.from_form({"CallSid": "CA999"})
        )

        assert 'url="wss://per-call.example.com/ws"' in twiml
        assert "static.example.com" not in twiml

    @pytest.mark.asyncio
    async def test_handle_incoming_call_default_websocket_url(self) -> None:
        """Test handle_incoming_call derives the URL when no layer sets it."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        twiml = await channel.handle_incoming_call()

        assert 'url="wss://example.com/ws"' in twiml

    @pytest.mark.asyncio
    async def test_prompt_with_empty_voice_prompt(self) -> None:
        """Test handling prompt message with empty voice_prompt."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Setup conversation first
        channel._start_conversation("CALL111", "profile_test")

        # Create prompt message with None voicePrompt
        prompt_msg = PromptMessage(
            type="prompt",
            conversationId="CALL111",
            voicePrompt=None,
        )

        # Call handler directly
        await channel._handle_prompt("CALL111", prompt_msg)

        # Voice channel doesn't fetch memory - test passes if no exception raised

    @pytest.mark.asyncio
    async def test_multiple_concurrent_conversations(self) -> None:
        """Test managing multiple concurrent conversations with separate websockets."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Start three concurrent conversations
        channel._start_conversation("CALL_001", "profile_001")
        channel._start_conversation("CALL_002", "profile_002")
        channel._start_conversation("CALL_003", "profile_003")

        # Create mock websockets for each conversation
        mock_ws_1 = AsyncMock()
        mock_ws_2 = AsyncMock()
        mock_ws_3 = AsyncMock()

        # Register websockets with the manager
        channel._websocket_manager.add_websocket("CALL_001", mock_ws_1)
        channel._websocket_manager.add_websocket("CALL_002", mock_ws_2)
        channel._websocket_manager.add_websocket("CALL_003", mock_ws_3)

        # Verify all conversations and websockets are tracked
        assert len(channel._conversations) == 3
        assert len(channel._websocket_manager) == 3
        assert channel._websocket_manager.has_websocket("CALL_001")
        assert channel._websocket_manager.has_websocket("CALL_002")
        assert channel._websocket_manager.has_websocket("CALL_003")

        # Send responses to each conversation independently
        await channel.send_response("CALL_001", "Response to call 1")
        await channel.send_response("CALL_002", "Response to call 2")
        await channel.send_response("CALL_003", "Response to call 3")

        # Verify each websocket received only its own message
        assert mock_ws_1.send_text.call_count == 1
        assert mock_ws_2.send_text.call_count == 1
        assert mock_ws_3.send_text.call_count == 1

        # Verify correct messages were sent to each websocket
        call_args_1 = mock_ws_1.send_text.call_args[0][0]
        call_args_2 = mock_ws_2.send_text.call_args[0][0]
        call_args_3 = mock_ws_3.send_text.call_args[0][0]

        assert "Response to call 1" in call_args_1
        assert "Response to call 2" in call_args_2
        assert "Response to call 3" in call_args_3

    @pytest.mark.asyncio
    async def test_multiple_conversations_independent_cleanup(self) -> None:
        """Test that cleaning up one WebSocket doesn't affect others."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Start three conversations
        channel._start_conversation("CALL_A", "profile_A")
        channel._start_conversation("CALL_B", "profile_B")
        channel._start_conversation("CALL_C", "profile_C")

        # Register websockets
        channel._websocket_manager.add_websocket("CALL_A", AsyncMock())
        channel._websocket_manager.add_websocket("CALL_B", AsyncMock())
        channel._websocket_manager.add_websocket("CALL_C", AsyncMock())

        # Verify initial state
        assert len(channel._conversations) == 3
        assert len(channel._websocket_manager) == 3

        # Clean up CALL_B WebSocket only
        await channel._cleanup_connection("CALL_B")

        # Verify CALL_B WebSocket is cleaned up but conversation still tracked
        assert not channel._websocket_manager.has_websocket("CALL_B")
        assert "CALL_B" in channel._conversations
        assert len(channel._conversations) == 3
        assert len(channel._websocket_manager) == 2

        # Verify CALL_A and CALL_C are still active
        assert "CALL_A" in channel._conversations
        assert "CALL_C" in channel._conversations
        assert channel._websocket_manager.has_websocket("CALL_A")
        assert channel._websocket_manager.has_websocket("CALL_C")

        # Clean up remaining WebSockets
        await channel._cleanup_connection("CALL_A")
        await channel._cleanup_connection("CALL_C")

        # Verify WebSockets cleaned up but conversations still tracked
        assert len(channel._websocket_manager) == 0
        assert len(channel._conversations) == 3

    @pytest.mark.asyncio
    async def test_websocket_manager_get_all_conversation_ids(self) -> None:
        """Test WebSocketManager returns all active conversation IDs."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Initially empty
        assert channel._websocket_manager.get_all_conversation_ids() == []

        # Add multiple websockets
        channel._websocket_manager.add_websocket("CONV_1", AsyncMock())
        channel._websocket_manager.add_websocket("CONV_2", AsyncMock())
        channel._websocket_manager.add_websocket("CONV_3", AsyncMock())

        # Get all conversation IDs
        conv_ids = channel._websocket_manager.get_all_conversation_ids()

        # Verify all IDs are returned
        assert len(conv_ids) == 3
        assert "CONV_1" in conv_ids
        assert "CONV_2" in conv_ids
        assert "CONV_3" in conv_ids

    @pytest.mark.asyncio
    async def test_concurrent_responses_correct_routing(self) -> None:
        """Test that concurrent responses are routed to correct websockets."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Setup two conversations
        channel._start_conversation("CONV_X", "profile_X")
        channel._start_conversation("CONV_Y", "profile_Y")

        # Create distinct mock websockets
        mock_ws_x = AsyncMock()
        mock_ws_y = AsyncMock()

        channel._websocket_manager.add_websocket("CONV_X", mock_ws_x)
        channel._websocket_manager.add_websocket("CONV_Y", mock_ws_y)

        # Send multiple messages to each conversation
        await channel.send_response("CONV_X", "Message 1 to X")
        await channel.send_response("CONV_Y", "Message 1 to Y")
        await channel.send_response("CONV_X", "Message 2 to X")
        await channel.send_response("CONV_Y", "Message 2 to Y")
        await channel.send_response("CONV_X", "Message 3 to X")

        # Verify correct call counts
        assert mock_ws_x.send_text.call_count == 3
        assert mock_ws_y.send_text.call_count == 2

        # Verify CONV_X received only X messages
        x_calls = [call[0][0] for call in mock_ws_x.send_text.call_args_list]
        assert all("to X" in call for call in x_calls)
        assert not any("to Y" in call for call in x_calls)

        # Verify CONV_Y received only Y messages
        y_calls = [call[0][0] for call in mock_ws_y.send_text.call_args_list]
        assert all("to Y" in call for call in y_calls)
        assert not any("to X" in call for call in y_calls)

    @pytest.mark.asyncio
    async def test_websocket_removal_idempotent(self) -> None:
        """Test that removing a websocket multiple times is safe."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Add a websocket
        channel._websocket_manager.add_websocket("CONV_Z", AsyncMock())
        assert channel._websocket_manager.has_websocket("CONV_Z")

        # Remove it once
        channel._websocket_manager.remove_websocket("CONV_Z")
        assert not channel._websocket_manager.has_websocket("CONV_Z")

        # Remove it again (should not raise error)
        channel._websocket_manager.remove_websocket("CONV_Z")
        assert not channel._websocket_manager.has_websocket("CONV_Z")

        # Remove non-existent websocket (should not raise error)
        channel._websocket_manager.remove_websocket("NON_EXISTENT")

    @pytest.mark.asyncio
    async def test_websocket_replacement(self) -> None:
        """Test that adding a websocket with same conversation ID replaces the old one."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Add first websocket
        first_ws = AsyncMock()
        channel._websocket_manager.add_websocket("CONV_REPLACE", first_ws)

        # Verify first websocket is registered
        retrieved_ws = channel._websocket_manager.get_websocket("CONV_REPLACE")
        assert retrieved_ws is first_ws

        # Add second websocket with same conversation ID
        second_ws = AsyncMock()
        channel._websocket_manager.add_websocket("CONV_REPLACE", second_ws)

        # Verify second websocket replaced the first
        retrieved_ws = channel._websocket_manager.get_websocket("CONV_REPLACE")
        assert retrieved_ws is second_ws
        assert retrieved_ws is not first_ws

        # Verify still only one websocket tracked
        assert len(channel._websocket_manager) == 1

    @pytest.mark.asyncio
    async def test_send_response_with_invalid_type_raises_error(self) -> None:
        """Test that send_response raises TypeError for invalid response types."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Start conversation
        channel._start_conversation("CALL_INVALID", "profile_test")

        # Mock websocket
        mock_websocket = AsyncMock()
        channel._websocket_manager.add_websocket("CALL_INVALID", mock_websocket)

        # Test with integer (invalid type)
        with pytest.raises(TypeError, match="Voice channel requires string or async generator"):
            await channel.send_response("CALL_INVALID", 123)  # type: ignore[arg-type]

        # Test with dict (invalid type)
        with pytest.raises(TypeError, match="Voice channel requires string or async generator"):
            await channel.send_response("CALL_INVALID", {"message": "test"})  # type: ignore[arg-type]

        # Test with list (invalid type)
        with pytest.raises(TypeError, match="Voice channel requires string or async generator"):
            await channel.send_response("CALL_INVALID", ["hello"])  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_send_response_with_async_generator(self) -> None:
        """Test that send_response correctly handles async generators (streaming)."""
        from collections.abc import AsyncGenerator

        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Start conversation
        channel._start_conversation("CALL_STREAM", "profile_test")

        # Mock websocket
        mock_websocket = AsyncMock()
        channel._websocket_manager.add_websocket("CALL_STREAM", mock_websocket)

        # Create async generator
        async def stream_response() -> AsyncGenerator[str, None]:
            yield "Hello "
            yield "world"

        # Send streaming response
        await channel.send_response("CALL_STREAM", stream_response())

        # Verify websocket.send_text was called for each chunk + final marker
        # 3 calls: "Hello ", "world", and final {"last": True}
        assert mock_websocket.send_text.call_count == 3

    @pytest.mark.asyncio
    async def test_conversation_ended_callback_fires_on_cleanup(self) -> None:
        """Voice _cleanup_connection does NOT trigger on_conversation_ended (webhook does)."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        captured: list[ConversationSession] = []

        def handler(ctx: ConversationSession) -> None:
            captured.append(ctx)

        tac.on_conversation_ended(handler)

        # Start conversation and add a mock websocket
        channel._start_conversation("CALL_CB1", "prof_cb1")
        mock_ws = MagicMock()
        channel._websocket_manager.add_websocket("CALL_CB1", mock_ws)

        await channel._cleanup_connection("CALL_CB1")

        # Callback should NOT be called (conversation still tracked for webhook)
        assert len(captured) == 0
        # Conversation should still be tracked
        assert "CALL_CB1" in channel._conversations
        # WebSocket should be removed
        assert not channel._websocket_manager.has_websocket("CALL_CB1")

    @pytest.mark.asyncio
    async def test_cleanup_connection_removes_websocket_only(self) -> None:
        """_cleanup_connection removes WebSocket but keeps conversation tracked."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        channel._start_conversation("CALL_CB2", "prof_cb2")
        mock_ws = MagicMock()
        channel._websocket_manager.add_websocket("CALL_CB2", mock_ws)

        await channel._cleanup_connection("CALL_CB2")

        # Conversation still tracked (waiting for webhook)
        assert "CALL_CB2" in channel._conversations
        # WebSocket removed
        assert not channel._websocket_manager.has_websocket("CALL_CB2")

    @pytest.mark.asyncio
    async def test_webhook_triggers_conversation_ended_callback(self) -> None:
        """Webhook with CLOSED status triggers on_conversation_ended callback."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        captured: list[ConversationSession] = []

        async def async_handler(ctx: ConversationSession) -> None:
            captured.append(ctx)

        tac.on_conversation_ended(async_handler)

        channel._start_conversation("CALL_ASYNC1", "prof_async1")

        # Process webhook with CLOSED status
        webhook_data = {
            "eventType": "CONVERSATION_UPDATED",
            "data": {"id": "CALL_ASYNC1", "status": "CLOSED"},
        }
        await channel.process_webhook(webhook_data)

        # Callback should be triggered
        assert len(captured) == 1
        assert captured[0].conversation_id == "CALL_ASYNC1"
        assert captured[0].channel == "VOICE"
        # Conversation should be removed
        assert "CALL_ASYNC1" not in channel._conversations

    @pytest.mark.asyncio
    async def test_cleanup_connection_idempotent(self) -> None:
        """Calling _cleanup_connection twice is safe (idempotent)."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        channel._start_conversation("CALL_NOCB", "prof_nocb")
        mock_ws = MagicMock()
        channel._websocket_manager.add_websocket("CALL_NOCB", mock_ws)

        await channel._cleanup_connection("CALL_NOCB")
        # Second cleanup should be a no-op (websocket already removed)
        await channel._cleanup_connection("CALL_NOCB")

        # Conversation still tracked (only webhook removes it)
        assert "CALL_NOCB" in channel._conversations
        assert not channel._websocket_manager.has_websocket("CALL_NOCB")

    @pytest.mark.asyncio
    async def test_task_cancellation_with_unified_workflow(self) -> None:
        """Test that task cancellation still works with unified workflow.

        Tests streaming via callback pattern.
        """
        from collections.abc import AsyncGenerator

        from tac.session import ThreadSafeSessionManager

        tac = TAC(get_test_config())

        # Track cancellation
        stream_started = False
        stream_cancelled = False
        chunks_sent = 0

        async def user_callback(
            user_message: str,
            context: ConversationSession,
            memory_response: TACMemoryResponse | None,
        ) -> None:
            """User callback that generates streaming response."""
            nonlocal stream_started, stream_cancelled, chunks_sent

            # Create async generator (simulates OpenAI streaming)
            async def stream_response() -> AsyncGenerator[str, None]:
                nonlocal stream_started, stream_cancelled, chunks_sent
                stream_started = True
                try:
                    for i in range(100):
                        await asyncio.sleep(0.01)  # Simulate slow streaming
                        chunks_sent += 1
                        yield f"chunk_{i}"
                except asyncio.CancelledError:
                    stream_cancelled = True
                    raise

            # Send streaming response via voice channel
            await voice_channel.send_response(context.conversation_id, stream_response())

        tac.on_message_ready(user_callback)

        # Create session manager and voice channel
        session_manager = ThreadSafeSessionManager()
        voice_channel = VoiceChannel(
            tac=tac, config={"session_manager": session_manager, "memory_mode": "never"}
        )

        # Setup conversation
        voice_channel._start_conversation("CONV_CANCEL_TEST", None)

        # Mock websocket
        mock_websocket = AsyncMock()
        voice_channel._websocket_manager.add_websocket("CONV_CANCEL_TEST", mock_websocket)

        # Create prompt message
        prompt_data = {
            "type": "prompt",
            "conversationId": "CONV_CANCEL_TEST",
            "voicePrompt": "Tell me a long story",
        }

        # Get session state
        session_state = session_manager.get_or_create_session("CONV_CANCEL_TEST")

        # Start processing prompt (creates task but doesn't await it)
        await voice_channel._handle_prompt_async("CONV_CANCEL_TEST", prompt_data, session_state)

        # Verify task was created
        assert session_state.stream_task is not None
        assert not session_state.stream_task.done()

        # Give task time to start and stream some chunks
        await asyncio.sleep(0.05)
        assert stream_started, "Stream should have started"

        # Simulate new prompt arriving (should cancel previous task)
        prompt_data2 = {
            "type": "prompt",
            "conversationId": "CONV_CANCEL_TEST",
            "voicePrompt": "Actually, never mind",
        }

        # Manually cancel the old task (simulating what _handle_prompt_async does)
        old_task = session_state.stream_task
        session_state.stream_task.cancel()

        # Wait for cancellation to complete
        try:
            await asyncio.wait_for(old_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass  # Expected

        # Verify cancellation happened
        assert stream_cancelled, "Stream should have been cancelled"
        assert chunks_sent < 100, f"Should not have sent all chunks (sent {chunks_sent})"
        assert chunks_sent > 0, "Should have sent some chunks before cancellation"

        # Now process new prompt
        await voice_channel._handle_prompt_async("CONV_CANCEL_TEST", prompt_data2, session_state)

        # Verify new task was created
        assert session_state.stream_task is not None
        assert session_state.stream_task != old_task, "Should have created new task"

        # Clean up new task
        if session_state.stream_task and not session_state.stream_task.done():
            session_state.stream_task.cancel()
            try:
                await session_state.stream_task
            except (asyncio.CancelledError, Exception):
                # Ignore exceptions during cleanup to avoid masking earlier test results
                pass

    def test_generate_twiml_minimal(self) -> None:
        """Test TwiML generation with only websocket URL."""
        twiml = generate_twiml("wss://example.com/voice")

        assert '<?xml version="1.0" encoding="UTF-8"?>' in twiml
        assert "<Response>" in twiml
        assert '<ConversationRelay url="wss://example.com/voice" />' in twiml
        assert "</Connect>" in twiml
        assert "</Response>" in twiml
        # Should NOT have greeting or action
        assert "welcomeGreeting" not in twiml
        assert "action=" not in twiml

    def test_generate_twiml_url_from_options_only(self) -> None:
        """URL supplied only via options.websocket_url (no positional arg) is
        emitted on <ConversationRelay> — the channel-less caller path."""
        twiml = generate_twiml(
            options=TwiMLOptions(
                websocket_url="wss://example.com/ws?agent_session_id=CA1",
                conversation_configuration="cc",
            )
        )

        assert 'url="wss://example.com/ws?agent_session_id=CA1"' in twiml
        assert 'conversationConfiguration="cc"' in twiml

    def test_generate_twiml_positional_url_wins_over_options(self) -> None:
        """When both are given, the positional websocket_url wins."""
        twiml = generate_twiml(
            "wss://positional.example.com/ws",
            TwiMLOptions(websocket_url="wss://options.example.com/ws"),
        )

        assert 'url="wss://positional.example.com/ws"' in twiml
        assert "options.example.com" not in twiml

    def test_generate_twiml_requires_a_url(self) -> None:
        """No URL via either source raises ValueError."""
        with pytest.raises(ValueError, match="requires a WebSocket URL"):
            generate_twiml(options=TwiMLOptions(welcome_greeting="hi"))

    def test_twiml_options_rejects_empty_websocket_url(self) -> None:
        """An empty/whitespace websocket_url is a misconfiguration — reject it
        at the model rather than silently emitting <ConversationRelay url="">."""
        with pytest.raises(ValueError, match="websocket_url cannot be empty"):
            TwiMLOptions(websocket_url="")
        with pytest.raises(ValueError, match="websocket_url cannot be empty"):
            TwiMLOptions(websocket_url="   ")
        # None (the default) is fine — falls through to the derived URL.
        assert TwiMLOptions(websocket_url=None).websocket_url is None

    def test_generate_twiml_with_welcome_greeting(self) -> None:
        """Test TwiML generation with welcome greeting."""
        twiml = generate_twiml(
            "wss://example.com/voice",
            TwiMLOptions(
                welcome_greeting="Hello! How can I help you?",
            ),
        )

        assert 'welcomeGreeting="Hello! How can I help you?"' in twiml

    def test_generate_twiml_with_action_url(self) -> None:
        """Test TwiML generation with action URL."""
        twiml = generate_twiml(
            "wss://example.com/voice",
            TwiMLOptions(
                action_url="https://example.com/callback",
            ),
        )

        assert '<Connect action="https://example.com/callback">' in twiml

    def test_generate_twiml_with_standard_custom_parameters(self) -> None:
        """Test TwiML generation with standard TAC custom parameters."""
        twiml = generate_twiml(
            "wss://example.com/voice",
            TwiMLOptions(
                custom_parameters={
                    "conversationId": "CH123",
                    "profileId": "mem_profile_123",
                    "customerParticipantId": "PA_cust",
                    "aiAgentParticipantId": "PA_agent",
                },
            ),
        )

        assert '<Parameter name="conversationId" value="CH123" />' in twiml
        assert '<Parameter name="profileId" value="mem_profile_123" />' in twiml
        assert '<Parameter name="customerParticipantId" value="PA_cust" />' in twiml
        assert '<Parameter name="aiAgentParticipantId" value="PA_agent" />' in twiml

    def test_generate_twiml_with_arbitrary_custom_parameters(self) -> None:
        """Test TwiML generation with arbitrary custom parameters."""
        twiml = generate_twiml(
            "wss://example.com/voice",
            TwiMLOptions(
                custom_parameters={
                    "custom_field_1": "value1",
                    "custom_field_2": "value2",
                    "session_id": "sess_123",
                },
            ),
        )

        assert '<Parameter name="custom_field_1" value="value1" />' in twiml
        assert '<Parameter name="custom_field_2" value="value2" />' in twiml
        assert '<Parameter name="session_id" value="sess_123" />' in twiml

    def test_generate_twiml_filters_none_values(self) -> None:
        """Test that None values are excluded from parameters."""
        twiml = generate_twiml(
            "wss://example.com/voice",
            TwiMLOptions(
                custom_parameters={
                    "field1": "value1",
                    "field2": None,
                    "field3": "value3",
                },
            ),
        )

        assert '<Parameter name="field1" value="value1" />' in twiml
        assert "field2" not in twiml  # None should be filtered
        assert '<Parameter name="field3" value="value3" />' in twiml

    def test_generate_twiml_complete_example(self) -> None:
        """Test complete TwiML generation with all options."""
        twiml = generate_twiml(
            "wss://example.ngrok.io/voice",
            TwiMLOptions(
                custom_parameters={
                    "conversationId": "CH_abc123",
                    "profileId": "mem_profile_xyz",
                    "customField": "customValue",
                },
                welcome_greeting="Welcome to our support line!",
                action_url="https://example.com/call-ended",
            ),
        )

        # Verify all components present
        assert '<?xml version="1.0" encoding="UTF-8"?>' in twiml
        assert '<Connect action="https://example.com/call-ended">' in twiml
        assert 'url="wss://example.ngrok.io/voice"' in twiml
        assert 'welcomeGreeting="Welcome to our support line!"' in twiml
        assert '<Parameter name="conversationId" value="CH_abc123" />' in twiml
        assert '<Parameter name="profileId" value="mem_profile_xyz" />' in twiml
        assert '<Parameter name="customField" value="customValue" />' in twiml

    def test_generate_twiml_with_conversation_configuration(self) -> None:
        """Test TwiML generation with conversation_configuration."""
        twiml = generate_twiml(
            "wss://example.com/voice",
            TwiMLOptions(
                conversation_configuration="conv_configuration_test_service_123",
            ),
        )

        assert 'conversationConfiguration="conv_configuration_test_service_123"' in twiml
        assert 'url="wss://example.com/voice"' in twiml

    def test_generate_twiml_without_conversation_configuration(self) -> None:
        """Test TwiML generation without conversation_configuration."""
        twiml = generate_twiml("wss://example.com/voice", TwiMLOptions())

        # Should not have conversation_configuration in output
        assert "conversationConfiguration" not in twiml


class TestGenerateTwiMLConversationRelayAttrs:
    """The widened TwiMLOptions surface should emit every documented attribute."""

    def test_voice_and_language_attrs(self) -> None:
        twiml = generate_twiml(
            "wss://example.com/ws",
            TwiMLOptions(
                voice="en-US-Journey-D",
                language="en-US",
                transcription_provider="deepgram",
                tts_provider="elevenlabs",
            ),
        )
        assert 'voice="en-US-Journey-D"' in twiml
        assert 'language="en-US"' in twiml
        assert 'transcriptionProvider="deepgram"' in twiml
        assert 'ttsProvider="elevenlabs"' in twiml

    def test_interruptible_dtmf_debug(self) -> None:
        twiml = generate_twiml(
            "wss://example.com/ws",
            TwiMLOptions(
                interruptible="speech",
                dtmf_detection=True,
                debug="speaker-events",
            ),
        )
        assert 'interruptible="speech"' in twiml
        assert 'dtmfDetection="true"' in twiml
        assert 'debug="speaker-events"' in twiml

    def test_interruptible_bool_normalized_to_enum(self) -> None:
        """Twilio accepts True/False on interruptible for backward-compat,
        but the documented enum values are 'any'/'none'. Normalize."""
        twiml_true = generate_twiml("wss://example.com/ws", TwiMLOptions(interruptible=True))
        twiml_false = generate_twiml("wss://example.com/ws", TwiMLOptions(interruptible=False))
        assert 'interruptible="any"' in twiml_true
        assert 'interruptible="none"' in twiml_false

    def test_language_children_emitted(self) -> None:
        from tac.models.voice import LanguageConfig

        twiml = generate_twiml(
            "wss://example.com/ws",
            TwiMLOptions(
                languages=[
                    LanguageConfig(
                        code="es-MX",
                        voice="es-MX-Neural2-A",
                        tts_provider="google",
                        transcription_provider="google",
                        speech_model="long",
                    ),
                    LanguageConfig(code="fr-FR"),
                ],
            ),
        )
        assert '<Language code="es-MX"' in twiml
        assert 'voice="es-MX-Neural2-A"' in twiml
        assert 'ttsProvider="google"' in twiml
        assert 'speechModel="long"' in twiml
        assert '<Language code="fr-FR" />' in twiml

    def test_welcome_greeting_interruptible(self) -> None:
        twiml = generate_twiml(
            "wss://example.com/ws",
            TwiMLOptions(
                welcome_greeting="Hi",
                welcome_greeting_interruptible="dtmf",
            ),
        )
        assert 'welcomeGreetingInterruptible="dtmf"' in twiml

    def test_language_override_attrs(self) -> None:
        twiml = generate_twiml(
            "wss://example.com/ws",
            TwiMLOptions(
                tts_language="en-US",
                transcription_language="fr-FR",
            ),
        )
        assert 'ttsLanguage="en-US"' in twiml
        assert 'transcriptionLanguage="fr-FR"' in twiml

    def test_speech_model_and_elevenlabs_normalization(self) -> None:
        twiml = generate_twiml(
            "wss://example.com/ws",
            TwiMLOptions(
                speech_model="nova-3-general",
                elevenlabs_text_normalization="on",
            ),
        )
        assert 'speechModel="nova-3-general"' in twiml
        assert 'elevenlabsTextNormalization="on"' in twiml

    def test_turn_detection_attrs(self) -> None:
        twiml = generate_twiml(
            "wss://example.com/ws",
            TwiMLOptions(
                eot_threshold=0.75,
                partial_prompts=True,
                deepgram_smart_format=False,
                speech_timeout=1500,
            ),
        )
        assert 'eotThreshold="0.75"' in twiml
        assert 'partialPrompts="true"' in twiml
        assert 'deepgramSmartFormat="false"' in twiml
        assert 'speechTimeout="1500"' in twiml

    def test_interrupt_sensitivity_and_report_input(self) -> None:
        twiml = generate_twiml(
            "wss://example.com/ws",
            TwiMLOptions(
                interrupt_sensitivity="medium",
                report_input_during_agent_speech="speech",
            ),
        )
        assert 'interruptSensitivity="medium"' in twiml
        assert 'reportInputDuringAgentSpeech="speech"' in twiml

    def test_ignore_backchannel_and_preemptible(self) -> None:
        twiml = generate_twiml(
            "wss://example.com/ws",
            TwiMLOptions(ignore_backchannel=True, preemptible=True),
        )
        assert 'ignoreBackchannel="true"' in twiml
        assert 'preemptible="true"' in twiml

    def test_hints_events_intelligence_service(self) -> None:
        twiml = generate_twiml(
            "wss://example.com/ws",
            TwiMLOptions(
                hints="TwiML,ConversationRelay",
                events="speaker-events tokens-played",
                intelligence_service="GAaabbcc",
            ),
        )
        assert 'hints="TwiML,ConversationRelay"' in twiml
        assert 'events="speaker-events tokens-played"' in twiml
        assert 'intelligenceService="GAaabbcc"' in twiml

    def test_speech_timeout_accepts_auto(self) -> None:
        opts = TwiMLOptions(speech_timeout="auto")
        assert opts.speech_timeout == "auto"
        twiml = generate_twiml("wss://example.com/ws", opts)
        assert 'speechTimeout="auto"' in twiml

    def test_speech_timeout_rejects_other_strings(self) -> None:
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TwiMLOptions(speech_timeout="fast")  # type: ignore[arg-type]

    def test_omitted_fields_absent_from_output(self) -> None:
        twiml = generate_twiml("wss://example.com/ws")
        for attr in (
            "voice=",
            "language=",
            "transcriptionProvider=",
            "ttsProvider=",
            "interruptible=",
            "dtmfDetection=",
            "debug=",
            "welcomeGreetingInterruptible=",
            "ttsLanguage=",
            "transcriptionLanguage=",
            "speechModel=",
            "elevenlabsTextNormalization=",
            "eotThreshold=",
            "partialPrompts=",
            "deepgramSmartFormat=",
            "speechTimeout=",
            "interruptSensitivity=",
            "reportInputDuringAgentSpeech=",
            "ignoreBackchannel=",
            "preemptible=",
            "hints=",
            "events=",
            "intelligenceService=",
            "<Language",
        ):
            assert attr not in twiml


class TestTwiMLOptionsExtra:
    """`extra` lets users pass through ConversationRelay attributes not yet typed."""

    def test_extra_attrs_emitted(self) -> None:
        twiml = generate_twiml(
            "wss://example.com/ws",
            TwiMLOptions(extra={"future_feature": "on", "another_attr": True}),
        )
        # Twilio SDK snake_case → camelCase
        assert 'futureFeature="on"' in twiml
        assert 'anotherAttr="true"' in twiml

    def test_extra_shadowing_typed_field_raises(self) -> None:
        """A typed-field name in ``extra`` raises at construction. Silent
        drop-and-warn would be a footgun: the user explicitly set the field
        via ``extra`` and got nothing back."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="shadow typed fields"):
            TwiMLOptions(voice="en-US-Journey-D", extra={"voice": "should-not-appear"})

        with pytest.raises(ValidationError, match="shadow typed fields"):
            # Even when the typed field is unset, a shadow key must use the
            # typed field directly so validators / type coercion run.
            TwiMLOptions(extra={"speech_timeout": 800})

    def test_extra_none_emits_nothing(self) -> None:
        twiml = generate_twiml("wss://example.com/ws", TwiMLOptions())
        # Sanity: no trailing garbage from extra handling when it's unset.
        assert "<ConversationRelay url=" in twiml


class TestHandleIncomingCallMerge:
    """Merge layers: customizer → static twiml_options → TAC defaults."""

    @pytest.mark.asyncio
    async def test_tac_defaults_applied(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        twiml = await channel.handle_incoming_call()
        assert 'welcomeGreeting="Hello! How can I assist you today?"' in twiml
        assert 'conversationConfiguration="conv_configuration_test123"' in twiml

    @pytest.mark.asyncio
    async def test_static_options_override_conversation_configuration(self) -> None:
        from tac.channels.voice import VoiceChannelConfig

        tac = TAC(get_test_config())
        channel = VoiceChannel(
            tac,
            config=VoiceChannelConfig(
                default_twiml_options=TwiMLOptions(
                    conversation_configuration="conv_configuration_custom"
                ),
            ),
        )
        twiml = await channel.handle_incoming_call()
        assert 'conversationConfiguration="conv_configuration_custom"' in twiml
        assert "conv_configuration_test123" not in twiml

    @pytest.mark.asyncio
    async def test_studio_handoff_used_when_flow_sid_set(self) -> None:
        """Studio handoff URL is used when configured and no higher layer set action_url."""
        flow_sid = "FW" + "a" * 32
        tac = TAC({**get_test_config(), "studio_handoff_flow_sid": flow_sid})
        channel = VoiceChannel(tac)
        twiml = await channel.handle_incoming_call()
        expected = (
            f'action="https://webhooks.twilio.com/v1/Accounts/ACtest123'
            f'/Flows/{flow_sid}?Trigger=incomingCall"'
        )
        assert expected in twiml

    @pytest.mark.asyncio
    async def test_studio_handoff_beats_default_action_url(self) -> None:
        """Studio handoff is a user-expressed intent (explicit config) and
        wins over the derived cleanup URL. Setting Studio handoff is the
        signal that the user wants Studio's cleanup, not the SDK's default."""
        flow_sid = "FW" + "a" * 32
        tac = TAC({**get_test_config(), "studio_handoff_flow_sid": flow_sid})
        channel = VoiceChannel(tac)
        twiml = await channel.handle_incoming_call()
        expected = (
            f'action="https://webhooks.twilio.com/v1/Accounts/ACtest123'
            f'/Flows/{flow_sid}?Trigger=incomingCall"'
        )
        assert expected in twiml
        # Default cleanup URL must not also appear.
        assert "conversation-relay-callback" not in twiml

    @pytest.mark.asyncio
    async def test_action_url_falls_back_to_derived_default(self) -> None:
        """With nothing else configured, action_url is derived from
        TACConfig.voice_public_domain + voice_action_path."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        twiml = await channel.handle_incoming_call()
        assert 'action="https://example.com/conversation-relay-callback"' in twiml

    @pytest.mark.asyncio
    async def test_static_options_action_url_beats_derived_default(self) -> None:
        """A static action_url on default_twiml_options is an explicit user
        choice and wins over the derived cleanup URL."""
        from tac.channels.voice import VoiceChannelConfig

        tac = TAC(get_test_config())
        channel = VoiceChannel(
            tac,
            config=VoiceChannelConfig(
                default_twiml_options=TwiMLOptions(action_url="https://static.example.com/end"),
            ),
        )
        twiml = await channel.handle_incoming_call()
        assert 'action="https://static.example.com/end"' in twiml
        assert "conversation-relay-callback" not in twiml

    @pytest.mark.asyncio
    async def test_action_url_three_layer_resolution(self) -> None:
        """Customizer setting only `voice` (no action_url) must not clobber a
        default_twiml_options.action_url. Verifies the _overlay_fields skip
        invariant — every layer's action_url is funneled through
        _resolve_action_url, never via the field overlay."""
        from tac.channels.voice import VoiceChannelConfig
        from tac.models.voice import TwiMLRequest

        async def customizer(req: TwiMLRequest) -> TwiMLOptions:
            return TwiMLOptions(voice="en-US-Journey-D")  # no action_url

        tac = TAC(get_test_config())
        channel = VoiceChannel(
            tac,
            config=VoiceChannelConfig(
                default_twiml_options=TwiMLOptions(action_url="https://static.example.com/end"),
            ),
        )
        channel.on_inbound_call_twiml(customizer)
        twiml = await channel.handle_incoming_call(twiml_request=TwiMLRequest())
        assert 'action="https://static.example.com/end"' in twiml
        assert 'voice="en-US-Journey-D"' in twiml

    @pytest.mark.asyncio
    async def test_customizer_action_url_none_suppresses(self) -> None:
        """Explicit action_url=None on the customizer suppresses the
        <Connect action=...> attribute, even if Studio handoff or a channel
        default would otherwise populate it."""
        from tac.channels.voice import VoiceChannelConfig
        from tac.models.voice import TwiMLRequest

        async def customizer(req: TwiMLRequest) -> TwiMLOptions:
            return TwiMLOptions(action_url=None)

        flow_sid = "FW" + "a" * 32
        tac = TAC({**get_test_config(), "studio_handoff_flow_sid": flow_sid})
        channel = VoiceChannel(
            tac,
            config=VoiceChannelConfig(
                default_twiml_options=TwiMLOptions(action_url="https://static.example.com/end"),
            ),
        )
        channel.on_inbound_call_twiml(customizer)
        twiml = await channel.handle_incoming_call(twiml_request=TwiMLRequest())
        assert "action=" not in twiml

    @pytest.mark.asyncio
    async def test_default_options_action_url_none_suppresses(self) -> None:
        """Explicit action_url=None on default_twiml_options suppresses
        <Connect action=...> channel-wide, even with Studio handoff configured."""
        from tac.channels.voice import VoiceChannelConfig

        flow_sid = "FW" + "a" * 32
        tac = TAC({**get_test_config(), "studio_handoff_flow_sid": flow_sid})
        channel = VoiceChannel(
            tac,
            config=VoiceChannelConfig(
                default_twiml_options=TwiMLOptions(action_url=None),
            ),
        )
        twiml = await channel.handle_incoming_call()
        assert "action=" not in twiml

    @pytest.mark.asyncio
    async def test_host_twiml_options_sets_per_call_websocket_url(self) -> None:
        """A custom in-process host passes per-call transport facts via
        host_twiml_options — e.g. an affinity URL — without registering a
        customizer. This is the TACHostedAgentsApp use case."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        affinity_url = "wss://example.com/ws?agent_session_id=CA123"
        twiml = await channel.handle_incoming_call(
            host_twiml_options=TwiMLOptions(
                websocket_url=affinity_url,
                custom_parameters={"agent_session_id": "CA123"},
            ),
        )

        assert f'url="{affinity_url}"' in twiml
        assert 'url="wss://example.com/ws"' not in twiml  # not the derived URL
        # conversation_configuration still populated from TACConfig (not clobbered).
        assert 'conversationConfiguration="conv_configuration_test123"' in twiml

    @pytest.mark.asyncio
    async def test_app_customizer_beats_host_twiml_options(self) -> None:
        """Precedence: the application's on_inbound_call_twiml customizer sits
        ABOVE host_twiml_options — a developer's explicit choice wins over the
        host's per-call values (for fields the dev sets)."""
        from tac.channels.voice import VoiceChannelConfig
        from tac.models.voice import TwiMLRequest

        async def app_customizer(req: TwiMLRequest) -> TwiMLOptions:
            return TwiMLOptions(welcome_greeting="App wins")

        tac = TAC(get_test_config())
        channel = VoiceChannel(tac, config=VoiceChannelConfig())
        channel.on_inbound_call_twiml(app_customizer)

        twiml = await channel.handle_incoming_call(
            twiml_request=TwiMLRequest(),
            host_twiml_options=TwiMLOptions(
                websocket_url="wss://example.com/ws?agent_session_id=CA1",
                welcome_greeting="Host loses",
            ),
        )

        # Dev's greeting wins the contested field...
        assert 'welcomeGreeting="App wins"' in twiml
        assert "Host loses" not in twiml
        # ...but the host's websocket_url (dev didn't set it) still applies.
        assert 'url="wss://example.com/ws?agent_session_id=CA1"' in twiml

    @pytest.mark.asyncio
    async def test_default_options_beats_host_twiml_options(self) -> None:
        """default_twiml_options sits above host_twiml_options."""
        from tac.channels.voice import VoiceChannelConfig

        tac = TAC(get_test_config())
        channel = VoiceChannel(
            tac,
            config=VoiceChannelConfig(
                default_twiml_options=TwiMLOptions(welcome_greeting="Channel default"),
            ),
        )

        twiml = await channel.handle_incoming_call(
            host_twiml_options=TwiMLOptions(welcome_greeting="Host override"),
        )

        assert 'welcomeGreeting="Channel default"' in twiml
        assert "Host override" not in twiml


class TestStaticTwiMLOptions:
    """VoiceChannelConfig.twiml_options applies to every call without a callback."""

    @pytest.mark.asyncio
    async def test_static_options_applied(self) -> None:
        from tac.channels.voice import VoiceChannelConfig

        tac = TAC(get_test_config())
        channel = VoiceChannel(
            tac,
            config=VoiceChannelConfig(
                default_twiml_options=TwiMLOptions(voice="en-US-Journey-D", language="en-US"),
            ),
        )
        twiml = await channel.handle_incoming_call()
        assert 'voice="en-US-Journey-D"' in twiml
        assert 'language="en-US"' in twiml

    @pytest.mark.asyncio
    async def test_welcome_greeting_via_twiml_options(self) -> None:
        from tac.channels.voice import VoiceChannelConfig

        tac = TAC(get_test_config())
        channel = VoiceChannel(
            tac,
            config=VoiceChannelConfig(
                default_twiml_options=TwiMLOptions(welcome_greeting="Bonjour!"),
            ),
        )
        twiml = await channel.handle_incoming_call()
        assert 'welcomeGreeting="Bonjour!"' in twiml


class TestCustomizeTwiMLOptions:
    """Per-call customizer runs on top of static options and TAC defaults."""

    @pytest.mark.asyncio
    async def test_customizer_skipped_without_twiml_request(self) -> None:
        from tac.models.voice import TwiMLRequest

        called = False

        async def customizer(ctx: TwiMLRequest) -> TwiMLOptions:
            nonlocal called
            called = True
            return TwiMLOptions(voice="en-US-Journey-D")

        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        channel.on_inbound_call_twiml(customizer)
        twiml = await channel.handle_incoming_call()
        assert called is False
        assert "voice=" not in twiml

    @pytest.mark.asyncio
    async def test_customizer_invoked_with_twiml_request(self) -> None:
        from tac.models.voice import TwiMLRequest

        seen: dict[str, TwiMLRequest] = {}

        async def customizer(ctx: TwiMLRequest) -> TwiMLOptions:
            seen["ctx"] = ctx
            return TwiMLOptions(voice="en-US-Journey-D", interruptible="speech")

        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        channel.on_inbound_call_twiml(customizer)
        ctx = TwiMLRequest(from_number="+14155551234", caller_country="US")
        twiml = await channel.handle_incoming_call(
            twiml_request=ctx,
        )
        assert seen["ctx"] is ctx
        assert 'voice="en-US-Journey-D"' in twiml
        assert 'interruptible="speech"' in twiml

    @pytest.mark.asyncio
    async def test_customizer_output_beats_static(self) -> None:
        from tac.channels.voice import VoiceChannelConfig
        from tac.models.voice import TwiMLRequest

        async def customizer(ctx: TwiMLRequest) -> TwiMLOptions:
            return TwiMLOptions(voice="es-MX-Neural2-A")

        tac = TAC(get_test_config())
        channel = VoiceChannel(
            tac,
            config=VoiceChannelConfig(
                default_twiml_options=TwiMLOptions(voice="en-US-Journey-D"),
            ),
        )
        channel.on_inbound_call_twiml(customizer)
        twiml = await channel.handle_incoming_call(
            twiml_request=TwiMLRequest(),
        )
        assert 'voice="es-MX-Neural2-A"' in twiml

    @pytest.mark.asyncio
    async def test_customizer_unset_fields_keep_lower_layers(self) -> None:
        from tac.channels.voice import VoiceChannelConfig
        from tac.models.voice import TwiMLRequest

        async def customizer(ctx: TwiMLRequest) -> TwiMLOptions:
            return TwiMLOptions(voice="en-US-Journey-D")

        tac = TAC(get_test_config())
        channel = VoiceChannel(
            tac,
            config=VoiceChannelConfig(
                default_twiml_options=TwiMLOptions(welcome_greeting="Channel default"),
            ),
        )
        channel.on_inbound_call_twiml(customizer)
        twiml = await channel.handle_incoming_call(
            twiml_request=TwiMLRequest(),
        )
        # Customizer didn't set welcome_greeting; channel default survives.
        assert 'welcomeGreeting="Channel default"' in twiml
        assert 'voice="en-US-Journey-D"' in twiml

    @pytest.mark.asyncio
    async def test_customizer_action_url_wins_over_studio_handoff(self) -> None:
        from tac.models.voice import TwiMLRequest

        flow_sid = "FW" + "a" * 32

        async def customizer(ctx: TwiMLRequest) -> TwiMLOptions:
            return TwiMLOptions(action_url="https://customizer.example.com/end")

        tac = TAC({**get_test_config(), "studio_handoff_flow_sid": flow_sid})
        channel = VoiceChannel(tac)

        channel.on_inbound_call_twiml(customizer)
        twiml = await channel.handle_incoming_call(
            twiml_request=TwiMLRequest(),
        )
        assert 'action="https://customizer.example.com/end"' in twiml


class TestConversationInitializationFlow:
    """Test new conversation initialization flow with ConversationRelay."""

    @pytest.mark.asyncio
    async def test_first_prompt_initializes_conversation_from_relay(self) -> None:
        """Test first prompt queries CO and initializes conversation via websocket flow."""
        from tac.channels.websocket_protocol import WebSocketDisconnectError
        from tac.models.conversation import ParticipantAddress, ParticipantResponse

        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Track conversation initialization via callback
        initialized_conversations = []
        observed_author_info = []

        async def on_message(user_message, context, memory_response):
            initialized_conversations.append(context.conversation_id)
            observed_author_info.append(context.author_info)

        tac.on_message_ready(on_message)

        # Mock Conversation Orchestrator to return a conversation created by ConversationRelay
        mock_conversation = ConversationResponse(
            id="CH_relay_123",
            accountId="ACtest123",
            configuration_id="conv_configuration_test123",
            status="ACTIVE",
        )
        co_client = tac.conversation_orchestrator_client
        co_client.list_conversations = AsyncMock(return_value=[mock_conversation])

        # Mock participants list with VOICE channel address
        mock_participant = ParticipantResponse(
            id="PA_customer",
            conversation_id="CH_relay_123",
            account_id="ACtest123",
            name="Customer",
            profile_id="profile_voice_123",
            addresses=[
                ParticipantAddress(channel="VOICE", address="+15551234567"),
            ],
        )
        co_client.list_participants = AsyncMock(return_value=[mock_participant])

        # Create mock websocket that sends: setup -> prompt -> disconnect
        mock_websocket = AsyncMock()
        setup_data = {"type": "setup", "callSid": "CA_test_call", "from": "+15551234567"}
        prompt_data = {"type": "prompt", "voicePrompt": "Hello"}

        mock_websocket.receive_json = AsyncMock(
            side_effect=[setup_data, prompt_data, WebSocketDisconnectError()]
        )

        # Drive the real websocket handler
        await channel.handle_websocket(mock_websocket)

        # Verify callback was called (conversation initialized successfully)
        assert initialized_conversations == ["CH_relay_123"]

        # Verify CO was queried with correct parameters
        co_client.list_conversations.assert_called_once_with(
            channel_id="CA_test_call",
            status=["ACTIVE"],
        )
        co_client.list_participants.assert_called_once_with("CH_relay_123")

        # author_info is populated from setup's `from` — parity with SMS so that
        # memory lookup-by-address and Studio handoff's `To` both work on voice.
        assert observed_author_info[0] is not None
        assert observed_author_info[0].address == "+15551234567"

    @pytest.mark.asyncio
    async def test_profile_id_retrieval_filters_by_voice_channel(self) -> None:
        """Test that profile_id is retrieved by filtering on VOICE channel and from_number."""
        from tac.models.conversation import ParticipantAddress, ParticipantResponse

        tac = TAC(get_test_config())
        from_number = "+15551234567"

        # Mock participants with multiple addresses, only one matches VOICE channel
        mock_participants = [
            ParticipantResponse(
                id="PA_sms",
                conversation_id="CH_test",
                account_id="ACtest123",
                name="SMS Participant",
                profile_id="profile_sms_wrong",
                addresses=[
                    ParticipantAddress(channel="SMS", address="+15551234567"),
                ],
            ),
            ParticipantResponse(
                id="PA_voice",
                conversation_id="CH_test",
                account_id="ACtest123",
                name="Voice Participant",
                profile_id="profile_voice_correct",
                addresses=[
                    ParticipantAddress(channel="VOICE", address="+15551234567"),
                ],
            ),
            ParticipantResponse(
                id="PA_other",
                conversation_id="CH_test",
                account_id="ACtest123",
                name="Other Participant",
                profile_id="profile_other_wrong",
                addresses=[
                    ParticipantAddress(channel="VOICE", address="+15559999999"),
                ],
            ),
        ]
        co_client = tac.conversation_orchestrator_client
        co_client.list_participants = AsyncMock(return_value=mock_participants)

        # Simulate profile_id retrieval logic
        participants = await co_client.list_participants("CH_test")
        profile_id = None
        for participant in participants:
            if from_number and participant.addresses:
                for address in participant.addresses:
                    if (
                        address.channel == "VOICE"
                        and address.address == from_number
                        and participant.profile_id
                    ):
                        profile_id = participant.profile_id
                        break
            if profile_id:
                break

        # Verify correct profile_id was selected
        assert profile_id == "profile_voice_correct"

    @pytest.mark.asyncio
    @patch("tac.channels.voice.channel._POLL_BASE_DELAY", 0)
    async def test_error_when_no_conversations_found(self, capsys: pytest.CaptureFixture) -> None:
        """Test RuntimeError when ConversationRelay creates 0 conversations."""
        from tac.channels.websocket_protocol import WebSocketDisconnectError

        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Mock Conversation Orchestrator to return no conversations
        tac.conversation_orchestrator_client.list_conversations = AsyncMock(return_value=[])

        # Create mock websocket: setup -> prompt (triggers error)
        mock_websocket = AsyncMock()
        setup_data = {"type": "setup", "callSid": "CA_test_call", "from": "+15551234567"}
        prompt_data = {"type": "prompt", "voicePrompt": "Hello"}

        mock_websocket.receive_json = AsyncMock(
            side_effect=[setup_data, prompt_data, WebSocketDisconnectError()]
        )

        # Drive the real websocket handler - error will be caught and logged
        await channel.handle_websocket(mock_websocket)

        # Capture output
        captured = capsys.readouterr()

        # Verify error was logged with correct message (now includes poll attempt count)
        assert "Expected exactly 1 conversation" in captured.out
        assert "but found 0" in captured.out

        # Verify Conversation Orchestrator was polled (up to _POLL_ATTEMPTS)
        assert tac.conversation_orchestrator_client.list_conversations.call_count == 10

        # Verify no conversation was initialized
        assert len(channel._conversations) == 0

    @pytest.mark.asyncio
    @patch("tac.channels.voice.channel._POLL_BASE_DELAY", 0)
    async def test_error_when_multiple_conversations_found(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """Test RuntimeError when ConversationRelay creates 2+ conversations."""
        from tac.channels.websocket_protocol import WebSocketDisconnectError

        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Mock Conversation Orchestrator to return multiple conversations
        mock_conversations = [
            ConversationResponse(
                id="CH_relay_1",
                accountId="ACtest123",
                configuration_id="conv_configuration_test123",
                status="ACTIVE",
            ),
            ConversationResponse(
                id="CH_relay_2",
                accountId="ACtest123",
                configuration_id="conv_configuration_test123",
                status="ACTIVE",
            ),
        ]
        co_client = tac.conversation_orchestrator_client
        co_client.list_conversations = AsyncMock(return_value=mock_conversations)

        # Create mock websocket: setup -> prompt (triggers error)
        mock_websocket = AsyncMock()
        setup_data = {"type": "setup", "callSid": "CA_test_call", "from": "+15551234567"}
        prompt_data = {"type": "prompt", "voicePrompt": "Hello"}

        mock_websocket.receive_json = AsyncMock(
            side_effect=[setup_data, prompt_data, WebSocketDisconnectError()]
        )

        # Drive the real websocket handler - error will be caught and logged
        await channel.handle_websocket(mock_websocket)

        # Capture output
        captured = capsys.readouterr()

        # Verify error was logged with correct message
        assert "Expected exactly 1 conversation" in captured.out
        assert "but found 2" in captured.out

        # Verify CO was polled (up to _POLL_ATTEMPTS since count != 1)
        assert co_client.list_conversations.call_count == 10

        # Verify no conversation was initialized
        assert len(channel._conversations) == 0

    @pytest.mark.asyncio
    async def test_setup_message_starts_background_conversation_init(self) -> None:
        """Setup kicks off the CO lookup immediately in the background — it
        doesn't wait for the first prompt — so it overlaps with the wait for
        the caller's first utterance instead of adding to it.

        If the call disconnects before any prompt arrives to claim the
        result, the websocket the lookup already registered (as a side
        effect of `_initialize_conversation`) must still be cleaned up, not
        leaked. (The conversation itself intentionally stays in
        `_conversations` until CO's CLOSED webhook, same as any other
        orchestrator-mode call — see `_cleanup_connection`.)
        """
        from tac.channels.websocket_protocol import WebSocketDisconnectError
        from tac.models.conversation import ParticipantAddress, ParticipantResponse

        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        mock_conversation = ConversationResponse(
            id="CH_setup_test",
            accountId="ACtest123",
            configuration_id="conv_configuration_test123",
            status="ACTIVE",
        )
        co_client = tac.conversation_orchestrator_client
        co_client.list_conversations = AsyncMock(return_value=[mock_conversation])
        mock_participant = ParticipantResponse(
            id="PA_test",
            conversation_id="CH_setup_test",
            account_id="ACtest123",
            name="Test Participant",
            profile_id="profile_setup",
            addresses=[ParticipantAddress(channel="VOICE", address="+15551234567")],
        )
        co_client.list_participants = AsyncMock(return_value=[mock_participant])

        # Create mock websocket: setup -> disconnect (no prompt ever arrives).
        # receive_json yields to the event loop a couple of times before
        # raising, so the background init task gets a chance to actually run
        # (rather than being cancelled before it starts) — this is what makes
        # the "already registered, must be cleaned up" path deterministic.
        mock_websocket = AsyncMock()
        setup_data = {"type": "setup", "callSid": "CA_setup_test", "from": "+15551234567"}
        call_count = 0

        async def fake_receive_json() -> dict[str, str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return setup_data
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            raise WebSocketDisconnectError()

        mock_websocket.receive_json = fake_receive_json

        # Drive handle_websocket - should start CO lookup on setup and clean
        # up properly when the call ends before any prompt claims it.
        await channel.handle_websocket(mock_websocket)

        # The background lookup ran to completion (not cancelled mid-flight).
        co_client.list_conversations.assert_called_once_with(
            channel_id="CA_setup_test",
            status=["ACTIVE"],
        )
        co_client.list_participants.assert_called_once_with("CH_setup_test")

        # The websocket registration is cleaned up (not leaked) even though
        # no prompt ever arrived to claim conv_id itself.
        assert not channel._websocket_manager.has_websocket("CH_setup_test")
        # The conversation entry legitimately stays until CO's CLOSED
        # webhook — same as any other orchestrator-mode call.
        assert list(channel._conversations.keys()) == ["CH_setup_test"]

    @pytest.mark.asyncio
    async def test_subsequent_prompts_reuse_conversation(self) -> None:
        """Test second/third prompts use already-initialized conversation via websocket flow."""
        from tac.channels.websocket_protocol import WebSocketDisconnectError
        from tac.models.conversation import ParticipantAddress, ParticipantResponse

        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Track message callbacks
        messages_processed = []

        async def on_message(user_message, context, memory_response):
            messages_processed.append(user_message)

        tac.on_message_ready(on_message)

        # Mock Conversation Orchestrator
        mock_conversation = ConversationResponse(
            id="CH_reuse_test",
            accountId="ACtest123",
            configuration_id="conv_configuration_test123",
            status="ACTIVE",
        )
        co_client = tac.conversation_orchestrator_client
        co_client.list_conversations = AsyncMock(return_value=[mock_conversation])
        mock_participant = ParticipantResponse(
            id="PA_test",
            conversation_id="CH_reuse_test",
            account_id="ACtest123",
            name="Test Participant",
            profile_id="profile_reuse",
            addresses=[
                ParticipantAddress(channel="VOICE", address="+15551234567"),
            ],
        )
        co_client.list_participants = AsyncMock(return_value=[mock_participant])

        # Create mock websocket: setup -> prompt1 -> prompt2 -> prompt3 -> disconnect
        mock_websocket = AsyncMock()
        setup_data = {"type": "setup", "callSid": "CA_reuse_test", "from": "+15551234567"}
        prompt1_data = {"type": "prompt", "voicePrompt": "First message"}
        prompt2_data = {"type": "prompt", "voicePrompt": "Second message"}
        prompt3_data = {"type": "prompt", "voicePrompt": "Third message"}

        mock_websocket.receive_json = AsyncMock(
            side_effect=[
                setup_data,
                prompt1_data,
                prompt2_data,
                prompt3_data,
                WebSocketDisconnectError(),
            ]
        )

        # Drive the real websocket handler
        await channel.handle_websocket(mock_websocket)

        # Verify all 3 messages were processed
        assert len(messages_processed) == 3
        assert messages_processed == ["First message", "Second message", "Third message"]

        # Verify CO was called ONLY ONCE for initialization (on first prompt)
        assert co_client.list_conversations.call_count == 1
        assert co_client.list_participants.call_count == 1

        # Verify the calls used correct parameters
        co_client.list_conversations.assert_called_once_with(
            channel_id="CA_reuse_test",
            status=["ACTIVE"],
        )
        co_client.list_participants.assert_called_once_with("CH_reuse_test")


class TestSessionManagerDefaults:
    """Test session_manager default behavior in VoiceChannelConfig."""

    def test_default_session_manager_is_created(self) -> None:
        """Test that VoiceChannel creates a session_manager by default."""
        from tac.session import ThreadSafeSessionManager

        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Verify session_manager is created by default
        assert channel.session_manager is not None
        assert isinstance(channel.session_manager, ThreadSafeSessionManager)

    def test_session_manager_can_be_set_to_none(self) -> None:
        """Test that session_manager can be explicitly disabled."""
        from tac.channels.voice import VoiceChannelConfig

        tac = TAC(get_test_config())
        config = VoiceChannelConfig(session_manager=None)
        channel = VoiceChannel(tac, config=config)

        # Verify session_manager is None when explicitly disabled
        assert channel.session_manager is None

    def test_session_manager_can_be_dict_none(self) -> None:
        """Test that session_manager can be disabled via config dict."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac, config={"session_manager": None})

        # Verify session_manager is None when explicitly disabled via dict
        assert channel.session_manager is None

    @pytest.mark.asyncio
    async def test_cleanup_cancels_running_task(self) -> None:
        """Test that cleanup cancels in-flight tasks (user hung up, no point continuing)."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Track if task was cancelled
        task_cancelled = []

        async def running_callback(message, context, memory):
            try:
                await asyncio.sleep(1.0)  # Simulate work
                return "Should not reach here"
            except asyncio.CancelledError:
                task_cancelled.append(True)
                raise

        tac.on_message_ready(running_callback)

        # Start conversation
        channel._start_conversation("CONV_CLEANUP_TEST", None)
        mock_websocket = AsyncMock()
        channel._websocket_manager.add_websocket("CONV_CLEANUP_TEST", mock_websocket)

        # Create and start a task
        session_state = channel.session_manager.get_or_create_session("CONV_CLEANUP_TEST")
        prompt_data = {
            "type": "prompt",
            "conversationId": "CONV_CLEANUP_TEST",
            "voicePrompt": "Test",
        }
        await channel._handle_prompt_async("CONV_CLEANUP_TEST", prompt_data, session_state)

        # Give task time to start
        await asyncio.sleep(0.05)
        assert session_state.stream_task is not None
        assert not session_state.stream_task.done()

        # Cleanup should cancel the task
        await channel._cleanup_connection("CONV_CLEANUP_TEST")

        # Verify task was cancelled
        assert task_cancelled == [True]
        assert session_state.stream_task.done()


class TestVoicePublicDomainNormalization:
    """TACConfig.voice_public_domain strips schemes/whitespace/trailing slashes."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("https://example.ngrok.app", "example.ngrok.app"),
            ("http://example.ngrok.app", "example.ngrok.app"),
            ("wss://example.ngrok.app", "example.ngrok.app"),
            ("example.ngrok.app/", "example.ngrok.app"),
            ("https://example.ngrok.app/", "example.ngrok.app"),
            ("  example.ngrok.app  ", "example.ngrok.app"),
            ("example.ngrok.app", "example.ngrok.app"),
            ("", None),
        ],
    )
    def test_normalizes(self, raw: str, expected: str | None) -> None:
        tac = TAC({**get_test_config(), "voice_public_domain": raw})
        assert tac.config.voice_public_domain == expected

    @pytest.mark.asyncio
    async def test_normalized_value_produces_valid_twiml(self) -> None:
        """Regression: a sloppy https://...  / value used to concatenate to
        wss://https://example.ngrok.app//ws."""
        tac = TAC({**get_test_config(), "voice_public_domain": "https://example.ngrok.app/"})
        channel = VoiceChannel(tac)
        twiml = await channel.handle_incoming_call()
        assert 'url="wss://example.ngrok.app/ws"' in twiml


class TestVoicePathsOnTACConfig:
    """One source of truth for channel URL construction and route registration."""

    def test_default_paths(self) -> None:
        tac = TAC(get_test_config())
        assert tac.config.voice_websocket_path == "/ws"
        assert tac.config.voice_action_path == "/conversation-relay-callback"
        assert tac.config.voice_call_event_path == "/twilio/call-events"

    def test_call_event_path_per_kind(self) -> None:
        """One helper builds all three, so channel and server can't drift."""
        tac = TAC(get_test_config())
        assert tac.config.call_event_path("status") == "/twilio/call-events/status"
        assert tac.config.call_event_path("amd") == "/twilio/call-events/amd"
        assert tac.config.call_event_path("recording") == "/twilio/call-events/recording"

    def test_call_event_path_normalizes_trailing_slash(self) -> None:
        tac = TAC({**get_test_config(), "voice_call_event_path": "/hooks/calls/"})
        assert tac.config.call_event_path("amd") == "/hooks/calls/amd"

    def test_call_event_url_requires_domain(self) -> None:
        config = {**get_test_config()}
        config.pop("voice_public_domain")
        tac = TAC(config)
        assert tac.config.call_event_url("status") is None

    def test_call_event_url_with_domain(self) -> None:
        tac = TAC({**get_test_config(), "voice_public_domain": "example.ngrok.app"})
        assert (
            tac.config.call_event_url("status")
            == "https://example.ngrok.app/twilio/call-events/status"
        )

    def test_call_event_kinds_covers_every_kind(self) -> None:
        """CALL_EVENT_KINDS is what the server iterates to validate its paths.

        The routes themselves are three explicit decorators, so a new kind needs
        a route added by hand — this pins the set they have to stay in step with.
        """
        from tac.core.config import CALL_EVENT_KINDS

        assert set(CALL_EVENT_KINDS) == {"status", "amd", "recording"}


class TestCallEventPredicates:
    """Keep mode-specific string matching out of application code."""

    @pytest.mark.parametrize(
        "answered_by",
        ["machine_start", "machine_end_beep", "machine_end_silence", "machine_end_other"],
    )
    def test_is_machine_true_for_every_machine_value(self, answered_by: str) -> None:
        from tac.models.voice import AmdEvent

        assert AmdEvent(call_sid="CA1", answered_by=answered_by).is_machine is True

    @pytest.mark.parametrize("answered_by", ["human", "fax", "unknown", None, ""])
    def test_is_machine_false_otherwise(self, answered_by: str | None) -> None:
        """'unknown' means detection timed out — never hang up on a guess."""
        from tac.models.voice import AmdEvent

        assert AmdEvent(call_sid="CA1", answered_by=answered_by).is_machine is False

    @pytest.mark.parametrize("status", ["busy", "no-answer", "failed", "canceled"])
    def test_is_unreached_true_for_dispositions(self, status: str) -> None:
        from tac.models.voice import CallStatusEvent

        assert CallStatusEvent(call_sid="CA1", call_status=status).is_unreached is True

    @pytest.mark.parametrize("status", ["completed", "in-progress", "ringing", None])
    def test_is_unreached_false_otherwise(self, status: str | None) -> None:
        from tac.models.voice import CallStatusEvent

        assert CallStatusEvent(call_sid="CA1", call_status=status).is_unreached is False


class TestCallEventModels:
    """Each model parses its own fields; everything else goes to ``extra``."""

    def test_amd_event(self) -> None:
        from tac.models.voice import AmdEvent

        event = AmdEvent.from_form(
            {
                "CallSid": "CA1",
                "AccountSid": "ACtest123",
                "AnsweredBy": "machine_end_beep",
                "MachineDetectionDuration": "3200",
            }
        )
        assert event.call_sid == "CA1"
        assert event.answered_by == "machine_end_beep"
        assert event.machine_detection_duration == "3200"

    def test_recording_event(self) -> None:
        from tac.models.voice import RecordingEvent

        event = RecordingEvent.from_form(
            {
                "CallSid": "CA1",
                "RecordingSid": "RE1",
                "RecordingUrl": "https://x/r",
                "RecordingStatus": "completed",
                "RecordingDuration": "12",
            }
        )
        assert event.recording_sid == "RE1"
        assert event.recording_url == "https://x/r"
        assert event.recording_duration == "12"

    def test_status_event(self) -> None:
        from tac.models.voice import CallStatusEvent

        event = CallStatusEvent.from_form(
            {
                "CallSid": "CA1",
                "CallStatus": "no-answer",
                "CallDuration": "0",
                "SipResponseCode": "480",
            }
        )
        assert event.call_status == "no-answer"
        assert event.call_duration == "0"
        assert event.sip_response_code == "480"

    def test_keeps_extra_form(self) -> None:
        from tac.models.voice import CallStatusEvent

        event = CallStatusEvent.from_form(
            {"CallSid": "CA1", "CallStatus": "completed", "Custom": "x"}
        )
        assert event.extra["Custom"] == "x"
        # Fields belonging to another event type land in extra, not dropped.
        assert "RecordingSid" not in event.extra


class TestHandleCallEvents:
    @pytest.mark.asyncio
    async def test_status_fires_registered_handler(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        received = []

        async def handler(event: object) -> None:
            received.append(event)

        channel.on_call_status(handler)
        await channel.handle_call_status_event(
            {"CallSid": "CA1", "AccountSid": "ACtest123", "CallStatus": "no-answer"}
        )
        assert len(received) == 1
        assert received[0].call_status == "no-answer"

    @pytest.mark.asyncio
    async def test_amd_fires_registered_handler(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        received = []

        async def handler(event: object) -> None:
            received.append(event)

        channel.on_amd(handler)
        await channel.handle_amd_event(
            {"CallSid": "CA1", "AccountSid": "ACtest123", "AnsweredBy": "human"}
        )
        assert received[0].answered_by == "human"

    @pytest.mark.asyncio
    async def test_recording_fires_registered_handler(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        received = []

        async def handler(event: object) -> None:
            received.append(event)

        channel.on_recording(handler)
        await channel.handle_recording_event(
            {"CallSid": "CA1", "AccountSid": "ACtest123", "RecordingStatus": "completed"}
        )
        assert received[0].recording_status == "completed"

    @pytest.mark.asyncio
    async def test_handlers_are_independently_optional(self) -> None:
        """Only on_amd registered: amd fires, status/recording no-op silently."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        received = []

        async def handler(event: object) -> None:
            received.append(event)

        channel.on_amd(handler)
        await channel.handle_call_status_event({"CallSid": "CA1", "CallStatus": "completed"})
        await channel.handle_recording_event({"CallSid": "CA1", "RecordingStatus": "completed"})
        await channel.handle_amd_event({"CallSid": "CA1", "AnsweredBy": "human"})
        assert len(received) == 1
        assert received[0].answered_by == "human"

    @pytest.mark.asyncio
    async def test_noop_without_handler(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        # Should not raise despite no handler registered.
        await channel.handle_call_status_event({"CallSid": "CA1", "CallStatus": "completed"})

    @pytest.mark.asyncio
    async def test_ignores_account_sid_mismatch(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        received = []

        async def handler(event: object) -> None:
            received.append(event)

        channel.on_call_status(handler)
        await channel.handle_call_status_event(
            {"CallSid": "CA1", "AccountSid": "ACwrong", "CallStatus": "completed"}
        )
        assert received == []


class TestEndCall:
    @pytest.mark.asyncio
    async def test_hangs_up_via_twilio(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        mock_client = MagicMock()
        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            assert await channel.end_call("CA1") is True

        mock_client.calls.assert_called_once_with("CA1")
        mock_client.calls().update.assert_called_with(status="completed")

    @pytest.mark.asyncio
    async def test_cleans_up_session_via_call_sid(self) -> None:
        """conv_id != call_sid in orchestrator mode; resolved by scanning sessions."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Simulate an active orchestrator-mode session (conv id != call sid).
        session = channel._start_conversation("conv_abc")
        session.call_sid = "CA1"
        channel._end_conversation = AsyncMock()  # type: ignore[method-assign]

        mock_client = MagicMock()
        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            await channel.end_call("CA1")

        channel._end_conversation.assert_awaited_once_with("conv_abc")

    @pytest.mark.asyncio
    async def test_hangup_failure_returns_false_without_raising(self) -> None:
        """Already-ended calls are routine: reports rather than raises."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        mock_client = MagicMock()
        mock_client.calls("CA1").update.side_effect = RuntimeError("Twilio 400")
        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            assert await channel.end_call("CA1") is False

    @pytest.mark.asyncio
    async def test_session_cleanup_runs_even_when_hangup_fails(self) -> None:
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        session = channel._start_conversation("conv_abc")
        session.call_sid = "CA1"
        channel._end_conversation = AsyncMock()  # type: ignore[method-assign]

        mock_client = MagicMock()
        mock_client.calls("CA1").update.side_effect = RuntimeError("Twilio 400")
        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            assert await channel.end_call("CA1") is False

        channel._end_conversation.assert_awaited_once_with("conv_abc")

    @pytest.mark.asyncio
    async def test_hangup_works_without_tracked_session(self) -> None:
        """A machine may never prompt (no session); the hangup must still work."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)
        channel._end_conversation = AsyncMock()  # type: ignore[method-assign]

        mock_client = MagicMock()
        with patch.object(channel, "_get_twilio_client", return_value=mock_client):
            assert await channel.end_call("CA_unknown") is True

        mock_client.calls().update.assert_called_with(status="completed")
        channel._end_conversation.assert_not_awaited()


class TestGetConversationSessionByCallSid:
    """Call events carry only the CallSid; session methods are keyed by conv id."""

    def test_resolves_orchestrator_mode_session(self) -> None:
        """conv_id is the Orchestrator id, so the CallSid needs a lookup."""
        channel = VoiceChannel(TAC(get_test_config()))
        session = channel._start_conversation("conv_abc")
        session.call_sid = "CA1"

        found = channel.get_conversation_session_by_call_sid("CA1")
        assert found is session
        assert found.conversation_id == "conv_abc"

    def test_resolves_relay_only_session(self) -> None:
        """conv_id == call_sid here, but the lookup shouldn't assume it."""
        channel = VoiceChannel(TAC(get_test_config()))
        session = channel._start_conversation("CA1")
        session.call_sid = "CA1"

        assert channel.get_conversation_session_by_call_sid("CA1") is session

    def test_returns_none_for_unknown_call_sid(self) -> None:
        channel = VoiceChannel(TAC(get_test_config()))
        session = channel._start_conversation("conv_abc")
        session.call_sid = "CA1"

        assert channel.get_conversation_session_by_call_sid("CA_other") is None

    def test_ignores_sessions_without_a_call_sid(self) -> None:
        """Messaging sessions leave call_sid None — must not match on None."""
        channel = VoiceChannel(TAC(get_test_config()))
        channel._start_conversation("conv_no_sid")

        assert channel.get_conversation_session_by_call_sid("CA1") is None

    def test_picks_the_matching_session_among_several(self) -> None:
        channel = VoiceChannel(TAC(get_test_config()))
        for conv_id, call_sid in [("c1", "CA1"), ("c2", "CA2"), ("c3", "CA3")]:
            channel._start_conversation(conv_id).call_sid = call_sid

        found = channel.get_conversation_session_by_call_sid("CA2")
        assert found is not None
        assert found.conversation_id == "c2"

    def test_returns_none_before_the_first_prompt(self) -> None:
        """Sessions start on the first prompt, so a connected-but-silent call has none.

        This is what an on_amd handler sees under machine_detection="Enable":
        AMD resolves before the callee has said anything.
        """
        channel = VoiceChannel(TAC(get_test_config()))

        assert channel.get_conversation_session_by_call_sid("CA1") is None

    @pytest.mark.asyncio
    async def test_reaches_the_live_agent_mid_conversation(self) -> None:
        """Once the caller has prompted, out-of-band code can reach the session."""
        channel = VoiceChannel(TAC(get_test_config()))
        session = channel._start_conversation("conv_abc")
        session.call_sid = "CA1"
        channel.send_response = AsyncMock()  # type: ignore[method-assign]

        async def on_amd(event: object) -> None:
            found = channel.get_conversation_session_by_call_sid(event.call_sid)
            assert found is not None
            found.metadata["reached_voicemail"] = True
            await channel.send_response(found.conversation_id, "We'll try again.")

        channel.on_amd(on_amd)
        await channel.handle_amd_event({"CallSid": "CA1", "AnsweredBy": "machine_start"})

        assert session.metadata["reached_voicemail"] is True
        channel.send_response.assert_awaited_once_with("conv_abc", "We'll try again.")
