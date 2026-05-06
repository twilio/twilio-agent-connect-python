"""Tests for OpenAI adapter with memory injection."""

import builtins
import copy
import importlib
import sys
from unittest.mock import AsyncMock, Mock

import pytest

from tac.adapters.openai import with_tac_memory
from tac.adapters.options import AdapterOptions
from tac.models.memory import (
    MemoryCommunication,
    MemoryCommunicationContent,
    MemoryParticipant,
    MemoryRetrievalResponse,
    ObservationInfo,
    ProfileResponse,
    SummaryInfo,
)
from tac.models.session import ConversationSession
from tac.models.tac import (
    TACMemoryResponse,
)


@pytest.fixture
def mock_openai_client() -> Mock:
    """Create a mock OpenAI client."""
    client = Mock()
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = Mock(
        return_value={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Test response"},
                }
            ]
        }
    )
    client.chat.completions.stream = Mock(return_value=Mock())
    # Add Responses API
    client.responses = Mock()
    client.responses.create = Mock(
        return_value=Mock(output_text="Test response from Responses API")
    )
    # Add other attributes that should be proxied
    client.embeddings = Mock()
    client.images = Mock()
    return client


@pytest.fixture
def mock_async_openai_client() -> Mock:
    """Create a mock AsyncOpenAI client."""
    from openai import AsyncOpenAI

    client = Mock(spec=AsyncOpenAI)
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Test async response"},
                }
            ]
        }
    )
    client.chat.completions.stream = Mock(return_value=Mock())
    # Add Responses API
    client.responses = Mock()
    client.responses.create = AsyncMock(
        return_value=Mock(output_text="Test async response from Responses API")
    )
    # Add other attributes that should be proxied
    client.embeddings = Mock()
    client.images = Mock()
    return client


@pytest.fixture
def sample_memory_response() -> TACMemoryResponse:
    """Create a sample memory response."""
    memory_data = MemoryRetrievalResponse(
        observations=[
            ObservationInfo(
                id="obs1",
                content="Customer prefers email communication",
                source="conversation",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
                occurred_at="2024-01-01T00:00:00Z",
            )
        ],
        summaries=[
            SummaryInfo(
                id="sum1",
                content="Previous discussion about billing issues",
                conversation_id="conv123",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
                occurred_at="2024-01-01T00:00:00Z",
            )
        ],
        communications=[
            MemoryCommunication(
                id="comm1",
                content=MemoryCommunicationContent(text="Hello, I need help"),
                author=MemoryParticipant(
                    id="part1",
                    name="John Doe",
                    address="+1234567890",
                    channel="SMS",
                    type="CUSTOMER",
                ),
                recipients=[],
                timestamp="2024-01-01T00:00:00Z",
                created_at="2024-01-01T00:00:00Z",
            )
        ],
    )
    return TACMemoryResponse(memory_data)


@pytest.fixture
def sample_context() -> ConversationSession:
    """Create a sample conversation context with profile."""
    return ConversationSession(
        conversation_id="conv123",
        profile_id="prof456",
        channel="sms",
        profile=ProfileResponse(
            id="prof456",
            createdAt="2024-01-01T00:00:00Z",
            traits={"Contact": {"name": "John Doe", "email": "john@example.com"}},
        ),
    )


def test_with_tac_memory_returns_wrapper(mock_openai_client: Mock) -> None:
    """Test that with_tac_memory returns a wrapper, not the original client."""
    wrapper = with_tac_memory(mock_openai_client)

    # Should return a wrapper, not the original client
    assert wrapper is not mock_openai_client
    # Wrapper should have access to the original client
    assert hasattr(wrapper, "_client")
    assert wrapper._client is mock_openai_client


def test_wrapper_does_not_mutate_original_client(mock_openai_client: Mock) -> None:
    """Test that wrapping a client does not mutate the original."""
    original_chat = mock_openai_client.chat
    original_completions = mock_openai_client.chat.completions

    # Create wrapper
    wrapper = with_tac_memory(mock_openai_client)

    # Original client should be unchanged
    assert mock_openai_client.chat is original_chat
    assert mock_openai_client.chat.completions is original_completions

    # Wrapper should have different chat namespace
    assert wrapper.chat is not original_chat


def test_memory_injection_with_observations(
    mock_openai_client: Mock,
    sample_memory_response: TACMemoryResponse,
    sample_context: ConversationSession,
) -> None:
    """Test that observations are injected into messages."""
    wrapper = with_tac_memory(mock_openai_client, sample_memory_response, sample_context)

    messages = [{"role": "user", "content": "Hello"}]
    wrapper.chat.completions.create(model="gpt-4", messages=messages)

    # Check that create was called with enhanced messages
    call_args = mock_openai_client.chat.completions.create.call_args
    enhanced_messages = call_args[1]["messages"]

    # Should have original message plus memory system message
    assert len(enhanced_messages) == 2
    assert enhanced_messages[0]["role"] == "system"
    assert "Customer prefers email communication" in enhanced_messages[0]["content"]
    assert enhanced_messages[1] == messages[0]


def test_memory_injection_with_summaries(
    mock_openai_client: Mock,
    sample_memory_response: TACMemoryResponse,
    sample_context: ConversationSession,
) -> None:
    """Test that summaries are injected into messages."""
    wrapper = with_tac_memory(mock_openai_client, sample_memory_response, sample_context)

    messages = [{"role": "user", "content": "Hello"}]
    wrapper.chat.completions.create(model="gpt-4", messages=messages)

    call_args = mock_openai_client.chat.completions.create.call_args
    enhanced_messages = call_args[1]["messages"]

    # Check summaries are in the system message
    system_content = enhanced_messages[0]["content"]
    assert "Previous discussion about billing issues" in system_content


def test_memory_injection_with_communications(
    mock_openai_client: Mock,
    sample_memory_response: TACMemoryResponse,
    sample_context: ConversationSession,
) -> None:
    """Test that communications are injected into messages."""
    wrapper = with_tac_memory(mock_openai_client, sample_memory_response, sample_context)

    messages = [{"role": "user", "content": "Hello"}]
    wrapper.chat.completions.create(model="gpt-4", messages=messages)

    call_args = mock_openai_client.chat.completions.create.call_args
    enhanced_messages = call_args[1]["messages"]

    # Check communications are in the system message
    system_content = enhanced_messages[0]["content"]
    assert "Hello, I need help" in system_content


def test_profile_injection_with_traits(
    mock_openai_client: Mock, sample_context: ConversationSession
) -> None:
    """Test that profile traits are injected into messages."""
    wrapper = with_tac_memory(mock_openai_client, context=sample_context)

    messages = [{"role": "user", "content": "Hello"}]
    wrapper.chat.completions.create(model="gpt-4", messages=messages)

    call_args = mock_openai_client.chat.completions.create.call_args
    enhanced_messages = call_args[1]["messages"]

    # Check profile traits are in the system message
    system_content = enhanced_messages[0]["content"]
    assert "John Doe" in system_content
    assert "john@example.com" in system_content


def test_profile_trait_filtering_with_options(
    mock_openai_client: Mock, sample_context: ConversationSession
) -> None:
    """Test that profile traits are filtered based on AdapterOptions."""
    # Only include "Contact" trait group
    options = AdapterOptions(profile_traits=["Contact"])
    wrapper = with_tac_memory(mock_openai_client, context=sample_context, options=options)

    messages = [{"role": "user", "content": "Hello"}]
    wrapper.chat.completions.create(model="gpt-4", messages=messages)

    call_args = mock_openai_client.chat.completions.create.call_args
    enhanced_messages = call_args[1]["messages"]

    system_content = enhanced_messages[0]["content"]
    assert "Contact" in system_content


def test_profile_trait_exclusion_with_empty_list(
    mock_openai_client: Mock, sample_context: ConversationSession
) -> None:
    """Test that empty trait list excludes all profile traits."""
    options = AdapterOptions(profile_traits=[])
    wrapper = with_tac_memory(mock_openai_client, context=sample_context, options=options)

    messages = [{"role": "user", "content": "Hello"}]
    wrapper.chat.completions.create(model="gpt-4", messages=messages)

    call_args = mock_openai_client.chat.completions.create.call_args
    enhanced_messages = call_args[1]["messages"]

    # Should only have user message (no memory system message since no traits)
    assert len(enhanced_messages) == 1
    assert enhanced_messages == messages


def test_no_injection_without_memory_or_profile(mock_openai_client: Mock) -> None:
    """Test that no injection happens when memory and profile are absent."""
    wrapper = with_tac_memory(mock_openai_client)

    messages = [{"role": "user", "content": "Hello"}]
    wrapper.chat.completions.create(model="gpt-4", messages=messages)

    call_args = mock_openai_client.chat.completions.create.call_args
    enhanced_messages = call_args[1]["messages"]

    # Should be unchanged (no system message added)
    assert enhanced_messages == messages


def test_multiple_wrappers_no_memory_bleeding(
    mock_openai_client: Mock,
    sample_memory_response: TACMemoryResponse,
    sample_context: ConversationSession,
) -> None:
    """Test that multiple wrappers on same client don't interfere with each other."""
    # Create two wrappers with different memory
    memory_1 = sample_memory_response
    memory_data_2 = MemoryRetrievalResponse(
        observations=[
            ObservationInfo(
                id="obs2",
                content="Different observation for conversation 2",
                source="conversation",
                created_at="2024-01-02T00:00:00Z",
                updated_at="2024-01-02T00:00:00Z",
                occurred_at="2024-01-02T00:00:00Z",
            )
        ],
        summaries=[],
        communications=[],
    )
    memory_2 = TACMemoryResponse(memory_data_2)

    context_1 = sample_context
    context_2 = ConversationSession(
        conversation_id="conv456", profile_id="prof789", channel="voice"
    )

    wrapper_1 = with_tac_memory(mock_openai_client, memory_1, context_1)
    wrapper_2 = with_tac_memory(mock_openai_client, memory_2, context_2)

    # Call wrapper 1
    messages_1 = [{"role": "user", "content": "Message 1"}]
    wrapper_1.chat.completions.create(model="gpt-4", messages=messages_1)

    call_1_args = mock_openai_client.chat.completions.create.call_args
    enhanced_messages_1 = call_1_args[1]["messages"]
    system_content_1 = enhanced_messages_1[0]["content"]

    # Call wrapper 2
    messages_2 = [{"role": "user", "content": "Message 2"}]
    wrapper_2.chat.completions.create(model="gpt-4", messages=messages_2)

    call_2_args = mock_openai_client.chat.completions.create.call_args
    enhanced_messages_2 = call_2_args[1]["messages"]
    system_content_2 = enhanced_messages_2[0]["content"]

    # Verify each wrapper injected its own memory
    assert "Customer prefers email communication" in system_content_1
    assert "Different observation for conversation 2" in system_content_2

    # Verify no memory bleeding
    assert "Different observation for conversation 2" not in system_content_1
    assert "Customer prefers email communication" not in system_content_2


def test_original_client_unchanged_after_wrapper_calls(
    mock_openai_client: Mock, sample_memory_response: TACMemoryResponse
) -> None:
    """Test that original client can still be used without memory injection."""
    wrapper = with_tac_memory(mock_openai_client, sample_memory_response)

    # Call through wrapper
    wrapper.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": "Hello"}])

    # Call original client directly
    messages_original = [{"role": "user", "content": "Direct call"}]
    mock_openai_client.chat.completions.create(model="gpt-4", messages=messages_original)

    # Get the last call (direct call to original client)
    call_args = mock_openai_client.chat.completions.create.call_args
    direct_messages = call_args[1]["messages"]

    # Should be unchanged (no system message)
    assert direct_messages == messages_original


def test_wrapper_proxies_other_attributes(mock_openai_client: Mock) -> None:
    """Test that wrapper proxies non-chat attributes correctly."""
    wrapper = with_tac_memory(mock_openai_client)

    # Access other OpenAI client attributes through wrapper
    embeddings = wrapper.embeddings
    images = wrapper.images

    # Should proxy to original client
    assert embeddings is mock_openai_client.embeddings
    assert images is mock_openai_client.images


def test_wrapper_preserves_original_message_order(
    mock_openai_client: Mock, sample_memory_response: TACMemoryResponse
) -> None:
    """Test that wrapper preserves the order of original messages."""
    wrapper = with_tac_memory(mock_openai_client, sample_memory_response)

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]

    wrapper.chat.completions.create(model="gpt-4", messages=messages)

    call_args = mock_openai_client.chat.completions.create.call_args
    enhanced_messages = call_args[1]["messages"]

    # Memory system message should be first, followed by original messages in order
    assert enhanced_messages[0]["role"] == "system"
    assert "Customer Context" in enhanced_messages[0]["content"]
    assert enhanced_messages[1:] == messages


def test_wrapper_deep_copies_messages(
    mock_openai_client: Mock, sample_memory_response: TACMemoryResponse
) -> None:
    """Test that wrapper doesn't mutate original messages list."""
    wrapper = with_tac_memory(mock_openai_client, sample_memory_response)

    original_messages = [{"role": "user", "content": "Hello"}]
    original_messages_copy = copy.deepcopy(original_messages)

    wrapper.chat.completions.create(model="gpt-4", messages=original_messages)

    # Original messages should be unchanged
    assert original_messages == original_messages_copy


def test_wrapper_passes_through_other_kwargs(mock_openai_client: Mock) -> None:
    """Test that wrapper passes through other kwargs to create()."""
    wrapper = with_tac_memory(mock_openai_client)

    wrapper.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
    )

    call_args = mock_openai_client.chat.completions.create.call_args
    kwargs = call_args[1]

    # Check that kwargs are passed through
    assert kwargs["model"] == "gpt-4"
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 100
    assert kwargs["top_p"] == 0.9


# ========== Stream Tests ==========


def test_stream_memory_injection(
    mock_openai_client: Mock,
    sample_memory_response: TACMemoryResponse,
    sample_context: ConversationSession,
) -> None:
    """Test that stream() injects memory into messages."""
    wrapper = with_tac_memory(mock_openai_client, sample_memory_response, sample_context)

    messages = [{"role": "user", "content": "Hello"}]
    wrapper.chat.completions.stream(model="gpt-4", messages=messages)

    # Check that stream was called with enhanced messages
    call_args = mock_openai_client.chat.completions.stream.call_args
    enhanced_messages = call_args[1]["messages"]

    # Should have original message plus memory system message
    assert len(enhanced_messages) == 2
    assert enhanced_messages[0]["role"] == "system"
    assert "Customer prefers email communication" in enhanced_messages[0]["content"]
    assert enhanced_messages[1] == messages[0]


def test_stream_no_injection_without_memory(mock_openai_client: Mock) -> None:
    """Test that stream() doesn't inject when memory is absent."""
    wrapper = with_tac_memory(mock_openai_client)

    messages = [{"role": "user", "content": "Hello"}]
    wrapper.chat.completions.stream(model="gpt-4", messages=messages)

    call_args = mock_openai_client.chat.completions.stream.call_args
    enhanced_messages = call_args[1]["messages"]

    # Should be unchanged
    assert enhanced_messages == messages


def test_stream_passes_through_kwargs(mock_openai_client: Mock) -> None:
    """Test that stream() passes through other kwargs."""
    wrapper = with_tac_memory(mock_openai_client)

    wrapper.chat.completions.stream(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.8,
        max_tokens=150,
    )

    call_args = mock_openai_client.chat.completions.stream.call_args
    kwargs = call_args[1]

    assert kwargs["model"] == "gpt-4"
    assert kwargs["temperature"] == 0.8
    assert kwargs["max_tokens"] == 150


# ========== Responses API Tests ==========


def test_responses_api_memory_injection(
    mock_openai_client: Mock,
    sample_memory_response: TACMemoryResponse,
    sample_context: ConversationSession,
) -> None:
    """Test that Responses API injects memory into instructions."""
    wrapper = with_tac_memory(mock_openai_client, sample_memory_response, sample_context)

    wrapper.responses.create(
        model="gpt-5.4-mini",
        instructions="You are a helpful assistant.",
        input=[{"role": "user", "content": "Hello"}],
    )

    # Check that create was called with enhanced instructions
    call_args = mock_openai_client.responses.create.call_args
    enhanced_instructions = call_args[1]["instructions"]

    # Should have memory appended to instructions (instructions first, then memory)
    assert "Customer prefers email communication" in enhanced_instructions
    assert "You are a helpful assistant." in enhanced_instructions
    # Verify ordering: instructions should appear before memory content
    instructions_pos = enhanced_instructions.find("You are a helpful assistant.")
    memory_pos = enhanced_instructions.find("Customer prefers email communication")
    assert instructions_pos < memory_pos, "Instructions should come before memory"


def test_responses_api_no_injection_without_memory(mock_openai_client: Mock) -> None:
    """Test that Responses API doesn't inject when memory is absent."""
    wrapper = with_tac_memory(mock_openai_client)

    original_instructions = "You are a helpful assistant."
    wrapper.responses.create(
        model="gpt-5.4-mini",
        instructions=original_instructions,
        input=[{"role": "user", "content": "Hello"}],
    )

    call_args = mock_openai_client.responses.create.call_args
    enhanced_instructions = call_args[1]["instructions"]

    # Should be unchanged
    assert enhanced_instructions == original_instructions


def test_responses_api_with_profile(
    mock_openai_client: Mock, sample_context: ConversationSession
) -> None:
    """Test that Responses API injects profile traits into instructions."""
    wrapper = with_tac_memory(mock_openai_client, context=sample_context)

    wrapper.responses.create(
        model="gpt-5.4-mini",
        instructions="You are a helpful assistant.",
        input=[{"role": "user", "content": "Hello"}],
    )

    call_args = mock_openai_client.responses.create.call_args
    enhanced_instructions = call_args[1]["instructions"]

    # Check profile traits are in the instructions
    assert "John Doe" in enhanced_instructions
    assert "john@example.com" in enhanced_instructions


def test_responses_api_without_instructions(
    mock_openai_client: Mock,
    sample_memory_response: TACMemoryResponse,
    sample_context: ConversationSession,
) -> None:
    """Test that Responses API can inject memory even without original instructions."""
    wrapper = with_tac_memory(mock_openai_client, sample_memory_response, sample_context)

    wrapper.responses.create(
        model="gpt-5.4-mini",
        input=[{"role": "user", "content": "Hello"}],
    )

    call_args = mock_openai_client.responses.create.call_args
    enhanced_instructions = call_args[1]["instructions"]

    # Should have memory as the instructions
    assert "Customer prefers email communication" in enhanced_instructions


def test_responses_api_passes_through_kwargs(mock_openai_client: Mock) -> None:
    """Test that Responses API passes through other kwargs."""
    wrapper = with_tac_memory(mock_openai_client)

    wrapper.responses.create(
        model="gpt-5.4-mini",
        instructions="You are a helpful assistant.",
        input=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
        max_tokens=100,
    )

    call_args = mock_openai_client.responses.create.call_args
    kwargs = call_args[1]

    assert kwargs["model"] == "gpt-5.4-mini"
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 100


# ========== Async Tests ==========


@pytest.mark.asyncio
async def test_async_client_wrapper(
    mock_async_openai_client: Mock,
    sample_memory_response: TACMemoryResponse,
    sample_context: ConversationSession,
) -> None:
    """Test that AsyncOpenAI client is wrapped correctly."""
    wrapper = with_tac_memory(mock_async_openai_client, sample_memory_response, sample_context)

    # Should return AsyncTACOpenAIClient
    assert wrapper is not mock_async_openai_client
    assert hasattr(wrapper, "_client")
    assert wrapper._client is mock_async_openai_client


@pytest.mark.asyncio
async def test_async_create_memory_injection(
    mock_async_openai_client: Mock,
    sample_memory_response: TACMemoryResponse,
    sample_context: ConversationSession,
) -> None:
    """Test that async create() injects memory into messages."""
    wrapper = with_tac_memory(mock_async_openai_client, sample_memory_response, sample_context)

    messages = [{"role": "user", "content": "Hello"}]
    await wrapper.chat.completions.create(model="gpt-4", messages=messages)

    # Check that create was called with enhanced messages
    call_args = mock_async_openai_client.chat.completions.create.call_args
    enhanced_messages = call_args[1]["messages"]

    # Should have original message plus memory system message
    assert len(enhanced_messages) == 2
    assert enhanced_messages[0]["role"] == "system"
    assert "Customer prefers email communication" in enhanced_messages[0]["content"]
    assert enhanced_messages[1] == messages[0]


@pytest.mark.asyncio
async def test_async_create_no_injection_without_memory(mock_async_openai_client: Mock) -> None:
    """Test that async create() doesn't inject when memory is absent."""
    wrapper = with_tac_memory(mock_async_openai_client)

    messages = [{"role": "user", "content": "Hello"}]
    await wrapper.chat.completions.create(model="gpt-4", messages=messages)

    call_args = mock_async_openai_client.chat.completions.create.call_args
    enhanced_messages = call_args[1]["messages"]

    # Should be unchanged
    assert enhanced_messages == messages


@pytest.mark.asyncio
async def test_async_stream_memory_injection(
    mock_async_openai_client: Mock,
    sample_memory_response: TACMemoryResponse,
    sample_context: ConversationSession,
) -> None:
    """Test that async stream() injects memory into messages."""
    wrapper = with_tac_memory(mock_async_openai_client, sample_memory_response, sample_context)

    messages = [{"role": "user", "content": "Hello"}]
    wrapper.chat.completions.stream(model="gpt-4", messages=messages)

    # Check that stream was called with enhanced messages
    call_args = mock_async_openai_client.chat.completions.stream.call_args
    enhanced_messages = call_args[1]["messages"]

    # Should have original message plus memory system message
    assert len(enhanced_messages) == 2
    assert enhanced_messages[0]["role"] == "system"
    assert "Customer prefers email communication" in enhanced_messages[0]["content"]
    assert enhanced_messages[1] == messages[0]


@pytest.mark.asyncio
async def test_async_wrapper_proxies_other_attributes(mock_async_openai_client: Mock) -> None:
    """Test that async wrapper proxies non-chat attributes correctly."""
    wrapper = with_tac_memory(mock_async_openai_client)

    # Access other AsyncOpenAI client attributes through wrapper
    embeddings = wrapper.embeddings
    images = wrapper.images

    # Should proxy to original client
    assert embeddings is mock_async_openai_client.embeddings
    assert images is mock_async_openai_client.images


# ========== Async Responses API Tests ==========


@pytest.mark.asyncio
async def test_async_responses_api_memory_injection(
    mock_async_openai_client: Mock,
    sample_memory_response: TACMemoryResponse,
    sample_context: ConversationSession,
) -> None:
    """Test that async Responses API injects memory into instructions."""
    wrapper = with_tac_memory(mock_async_openai_client, sample_memory_response, sample_context)

    await wrapper.responses.create(
        model="gpt-5.4-mini",
        instructions="You are a helpful assistant.",
        input=[{"role": "user", "content": "Hello"}],
    )

    # Check that create was called with enhanced instructions
    call_args = mock_async_openai_client.responses.create.call_args
    enhanced_instructions = call_args[1]["instructions"]

    # Should have memory appended to instructions (instructions first, then memory)
    assert "Customer prefers email communication" in enhanced_instructions
    assert "You are a helpful assistant." in enhanced_instructions
    # Verify ordering: instructions should appear before memory content
    instructions_pos = enhanced_instructions.find("You are a helpful assistant.")
    memory_pos = enhanced_instructions.find("Customer prefers email communication")
    assert instructions_pos < memory_pos, "Instructions should come before memory"


@pytest.mark.asyncio
async def test_async_responses_api_no_injection_without_memory(
    mock_async_openai_client: Mock,
) -> None:
    """Test that async Responses API doesn't inject when memory is absent."""
    wrapper = with_tac_memory(mock_async_openai_client)

    original_instructions = "You are a helpful assistant."
    await wrapper.responses.create(
        model="gpt-5.4-mini",
        instructions=original_instructions,
        input=[{"role": "user", "content": "Hello"}],
    )

    call_args = mock_async_openai_client.responses.create.call_args
    enhanced_instructions = call_args[1]["instructions"]

    # Should be unchanged
    assert enhanced_instructions == original_instructions


class TestOpenAISoftDependency:
    """Regression tests for the openai soft-dependency contract.

    Importing `tac.adapters.openai` must not require `openai` to be installed;
    only calling `with_tac_memory()` should raise if it's missing.
    """

    def test_module_imports_without_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Importing the adapter module should not require `openai`."""
        # Drop any cached openai/adapter modules so reimport is forced
        for mod_name in list(sys.modules):
            if mod_name == "openai" or mod_name.startswith("openai."):
                monkeypatch.delitem(sys.modules, mod_name, raising=False)
            if mod_name == "tac.adapters.openai" or mod_name.startswith("tac.adapters.openai."):
                monkeypatch.delitem(sys.modules, mod_name, raising=False)

        real_import = builtins.__import__

        def blocked_import(
            name: str,
            globals: object = None,
            locals: object = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "openai" or name.startswith("openai."):
                raise ImportError(f"No module named '{name}' (blocked for test)")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", blocked_import)

        # Should succeed — top-level openai imports are under TYPE_CHECKING
        adapter_mod = importlib.import_module("tac.adapters.openai.adapter")
        assert hasattr(adapter_mod, "with_tac_memory")
        assert "openai" not in sys.modules

    def test_with_tac_memory_raises_helpful_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Calling with_tac_memory() without openai installed should raise
        ImportError with a pointer to `pip install openai`."""
        monkeypatch.delitem(sys.modules, "openai", raising=False)

        real_import = builtins.__import__

        def blocked_import(
            name: str,
            globals: object = None,
            locals: object = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "openai" or name.startswith("openai."):
                raise ImportError(f"No module named '{name}' (blocked for test)")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", blocked_import)

        with pytest.raises(ImportError, match="pip install openai"):
            with_tac_memory(Mock())
