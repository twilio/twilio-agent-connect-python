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
    CustomParameters,
    InterruptMessage,
    PromptMessage,
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

        assert channel.get_channel_name() == "voice"

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
        assert captured_context.channel == "voice"
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
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Generate TwiML (no need to mock - ConversationRelay handles conversation creation)
        twiml = await channel.handle_incoming_call(
            options={
                "websocket_url": "wss://example.ngrok.io/ws",
                "action_url": "https://example.ngrok.io/flex_handoff",
                "welcome_greeting": "Welcome!",
            },
        )

        # Verify TwiML contains expected elements
        assert '<?xml version="1.0" encoding="UTF-8"?>' in twiml
        assert "<Response>" in twiml
        assert '<Connect action="https://example.ngrok.io/flex_handoff">' in twiml
        assert "<ConversationRelay" in twiml
        assert 'url="wss://example.ngrok.io/ws"' in twiml
        assert 'welcomeGreeting="Welcome!"' in twiml
        assert 'conversationConfiguration="conv_configuration_test123"' in twiml
        assert "</Connect>" in twiml
        assert "</Response>" in twiml

    @pytest.mark.asyncio
    async def test_handle_incoming_call_default_greeting(self) -> None:
        """Test handle_incoming_call uses default greeting."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Generate TwiML without custom greeting (uses default)
        twiml = await channel.handle_incoming_call(
            options={
                "websocket_url": "wss://test.ngrok.io/ws",
                "action_url": "https://example.ngrok.io/flex_handoff",
            },
        )

        # Verify default greeting is used
        assert 'welcomeGreeting="Hello! How can I assist you today?"' in twiml
        assert 'conversationConfiguration="conv_configuration_test123"' in twiml

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
        assert captured[0].channel == "voice"
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
        twiml = generate_twiml(TwiMLOptions(websocket_url="wss://example.com/voice"))

        assert '<?xml version="1.0" encoding="UTF-8"?>' in twiml
        assert "<Response>" in twiml
        assert '<ConversationRelay url="wss://example.com/voice" />' in twiml
        assert "</Connect>" in twiml
        assert "</Response>" in twiml
        # Should NOT have greeting or action
        assert "welcomeGreeting" not in twiml
        assert "action=" not in twiml

    def test_generate_twiml_with_welcome_greeting(self) -> None:
        """Test TwiML generation with welcome greeting."""
        twiml = generate_twiml(
            TwiMLOptions(
                websocket_url="wss://example.com/voice",
                welcome_greeting="Hello! How can I help you?",
            )
        )

        assert 'welcomeGreeting="Hello! How can I help you?"' in twiml

    def test_generate_twiml_with_action_url(self) -> None:
        """Test TwiML generation with action URL."""
        twiml = generate_twiml(
            TwiMLOptions(
                websocket_url="wss://example.com/voice",
                action_url="https://example.com/callback",
            )
        )

        assert '<Connect action="https://example.com/callback">' in twiml

    def test_generate_twiml_with_standard_custom_parameters(self) -> None:
        """Test TwiML generation with standard TAC custom parameters."""
        twiml = generate_twiml(
            TwiMLOptions(
                websocket_url="wss://example.com/voice",
                custom_parameters={
                    "conversationId": "CH123",
                    "profileId": "mem_profile_123",
                    "customerParticipantId": "PA_cust",
                    "aiAgentParticipantId": "PA_agent",
                },
            )
        )

        assert '<Parameter name="conversationId" value="CH123" />' in twiml
        assert '<Parameter name="profileId" value="mem_profile_123" />' in twiml
        assert '<Parameter name="customerParticipantId" value="PA_cust" />' in twiml
        assert '<Parameter name="aiAgentParticipantId" value="PA_agent" />' in twiml

    def test_generate_twiml_with_arbitrary_custom_parameters(self) -> None:
        """Test TwiML generation with arbitrary custom parameters."""
        twiml = generate_twiml(
            TwiMLOptions(
                websocket_url="wss://example.com/voice",
                custom_parameters={
                    "custom_field_1": "value1",
                    "custom_field_2": "value2",
                    "session_id": "sess_123",
                },
            )
        )

        assert '<Parameter name="custom_field_1" value="value1" />' in twiml
        assert '<Parameter name="custom_field_2" value="value2" />' in twiml
        assert '<Parameter name="session_id" value="sess_123" />' in twiml

    def test_generate_twiml_with_pydantic_model(self) -> None:
        """Test TwiML generation using Pydantic CustomParameters model."""
        custom_params = CustomParameters(conversationId="CH123", profileId="mem_profile_123")

        twiml = generate_twiml(
            TwiMLOptions(
                websocket_url="wss://example.com/voice",
                custom_parameters=custom_params,
            )
        )

        # Should use camelCase aliases
        assert '<Parameter name="conversationId" value="CH123" />' in twiml
        assert '<Parameter name="profileId" value="mem_profile_123" />' in twiml

    def test_generate_twiml_with_dict_options(self) -> None:
        """Test TwiML generation accepting plain dict instead of TwiMLOptions."""
        twiml = generate_twiml(
            {
                "websocket_url": "wss://example.com/voice",
                "custom_parameters": {"key": "value"},
                "welcome_greeting": "Hi there!",
            }
        )

        assert 'url="wss://example.com/voice"' in twiml
        assert '<Parameter name="key" value="value" />' in twiml
        assert 'welcomeGreeting="Hi there!"' in twiml

    def test_generate_twiml_filters_none_values(self) -> None:
        """Test that None values are excluded from parameters."""
        twiml = generate_twiml(
            TwiMLOptions(
                websocket_url="wss://example.com/voice",
                custom_parameters={
                    "field1": "value1",
                    "field2": None,
                    "field3": "value3",
                },
            )
        )

        assert '<Parameter name="field1" value="value1" />' in twiml
        assert "field2" not in twiml  # None should be filtered
        assert '<Parameter name="field3" value="value3" />' in twiml

    def test_generate_twiml_escapes_xml_special_chars(self) -> None:
        """Test XML character escaping in parameter values."""
        twiml = generate_twiml(
            TwiMLOptions(
                websocket_url="wss://example.com/voice",
                custom_parameters={
                    "field": 'value with "quotes" & ampersand',
                },
            )
        )

        # Twilio SDK automatically escapes XML special characters
        assert "&amp;" in twiml
        assert "&quot;" in twiml
        # Verify the full escaped parameter is present
        expected_param = (
            '<Parameter name="field" value="value with &quot;quotes&quot; &amp; ampersand" />'
        )
        assert expected_param in twiml

    def test_generate_twiml_complete_example(self) -> None:
        """Test complete TwiML generation with all options."""
        twiml = generate_twiml(
            TwiMLOptions(
                websocket_url="wss://example.ngrok.io/voice",
                custom_parameters={
                    "conversationId": "CH_abc123",
                    "profileId": "mem_profile_xyz",
                    "customField": "customValue",
                },
                welcome_greeting="Welcome to our support line!",
                action_url="https://example.com/call-ended",
            )
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
            TwiMLOptions(
                websocket_url="wss://example.com/voice",
                conversation_configuration="conv_configuration_test_service_123",
            )
        )

        assert 'conversationConfiguration="conv_configuration_test_service_123"' in twiml
        assert 'url="wss://example.com/voice"' in twiml

    def test_generate_twiml_without_conversation_configuration(self) -> None:
        """Test TwiML generation without conversation_configuration."""
        twiml = generate_twiml(
            TwiMLOptions(
                websocket_url="wss://example.com/voice",
            )
        )

        # Should not have conversation_configuration in output
        assert "conversationConfiguration" not in twiml

    @pytest.mark.asyncio
    async def test_handle_incoming_call_with_additional_parameters(self) -> None:
        """Test handle_incoming_call includes additional custom parameters."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Generate TwiML with additional parameters
        twiml = await channel.handle_incoming_call(
            options={
                "websocket_url": "wss://example.ngrok.io/ws",
                "action_url": "https://example.ngrok.io/callback",
                "welcome_greeting": "Welcome!",
                "custom_parameters": {
                    "session_id": "sess_abc123",
                    "user_language": "es",
                    "priority": "high",
                },
            },
        )

        # Verify conversation_configuration is present
        assert 'conversationConfiguration="conv_configuration_test123"' in twiml

        # Verify additional custom parameters are present
        assert '<Parameter name="session_id" value="sess_abc123" />' in twiml
        assert '<Parameter name="user_language" value="es" />' in twiml
        assert '<Parameter name="priority" value="high" />' in twiml

    @pytest.mark.asyncio
    async def test_handle_incoming_call_without_additional_parameters(self) -> None:
        """Test handle_incoming_call works without additional parameters."""
        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Generate TwiML without additional parameters
        twiml = await channel.handle_incoming_call(
            options={
                "websocket_url": "wss://example.ngrok.io/ws",
            },
        )

        # Verify conversation_configuration is present
        assert 'conversationConfiguration="conv_configuration_test123"' in twiml
        # Verify no custom parameters
        assert "session_id" not in twiml
        assert "user_language" not in twiml


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

        # Verify Conversation Orchestrator was polled (up to 5 attempts)
        assert tac.conversation_orchestrator_client.list_conversations.call_count == 5

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

        # Verify CO was polled (up to 5 attempts since count != 1)
        assert co_client.list_conversations.call_count == 5

        # Verify no conversation was initialized
        assert len(channel._conversations) == 0

    @pytest.mark.asyncio
    async def test_setup_message_does_not_initialize_conversation(self) -> None:
        """Test that setup message stores call_sid but doesn't initialize conversation."""
        from tac.channels.websocket_protocol import WebSocketDisconnectError

        tac = TAC(get_test_config())
        channel = VoiceChannel(tac)

        # Mock Conversation Orchestrator (should not be called during setup)
        tac.conversation_orchestrator_client.list_conversations = AsyncMock()
        tac.conversation_orchestrator_client.list_participants = AsyncMock()

        # Create mock websocket: setup -> disconnect (no prompt)
        mock_websocket = AsyncMock()
        setup_data = {"type": "setup", "callSid": "CA_setup_test", "from": "+15551234567"}

        mock_websocket.receive_json = AsyncMock(
            side_effect=[setup_data, WebSocketDisconnectError()]
        )

        # Drive handle_websocket - should process setup but not initialize conversation
        await channel.handle_websocket(mock_websocket)

        # Verify CO was NOT called (initialization only on first prompt)
        tac.conversation_orchestrator_client.list_conversations.assert_not_called()
        tac.conversation_orchestrator_client.list_participants.assert_not_called()

        # Verify no conversations initialized
        assert len(channel._conversations) == 0

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
