"""Tests for MemoryPromptBuilder."""

import pytest

from tac.adapters.prompt_builder import MemoryPromptBuilder
from tac.models.memory import (
    MemoryRetrievalResponse,
    ObservationInfo,
    ProfileResponse,
)
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse


@pytest.fixture
def sample_memory_response() -> TACMemoryResponse:
    """Create a sample memory response with observations."""
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
        summaries=[],
        communications=[],
    )
    return TACMemoryResponse(memory_data)


@pytest.fixture
def sample_context() -> ConversationSession:
    """Create a sample conversation context with profile."""
    profile = ProfileResponse(
        id="profile1",
        created_at="2024-01-01T00:00:00Z",
        traits={"Contact": {"email": "test@example.com"}},
    )
    return ConversationSession(
        conversation_id="conv1",
        profile_id="profile1",
        channel="sms",
        profile=profile,
        phone_number="+1234567890",
    )


class TestMemoryPromptBuilderBuild:
    """Tests for MemoryPromptBuilder.build() method."""

    def test_build_with_memory_and_profile(
        self, sample_memory_response: TACMemoryResponse, sample_context: ConversationSession
    ) -> None:
        """Should return formatted prompt with both memory and profile."""
        result = MemoryPromptBuilder.build(sample_memory_response, sample_context)

        assert result is not None
        assert "# Customer Context" in result
        assert "## Customer Profile" in result
        assert "test@example.com" in result
        assert "## Key Observations" in result
        assert "Customer prefers email communication" in result

    def test_build_with_memory_only(self, sample_memory_response: TACMemoryResponse) -> None:
        """Should return formatted prompt with only memory."""
        result = MemoryPromptBuilder.build(sample_memory_response, None)

        assert result is not None
        assert "# Customer Context" in result
        assert "## Key Observations" in result
        assert "Customer prefers email communication" in result
        assert "## Customer Profile" not in result

    def test_build_with_profile_only(self, sample_context: ConversationSession) -> None:
        """Should return formatted prompt with only profile."""
        result = MemoryPromptBuilder.build(None, sample_context)

        assert result is not None
        assert "# Customer Context" in result
        assert "## Customer Profile" in result
        assert "test@example.com" in result
        assert "## Key Observations" not in result

    def test_build_with_no_data(self) -> None:
        """Should return None when no memory or profile data is available."""
        result = MemoryPromptBuilder.build(None, None)
        assert result is None

    def test_build_with_empty_memory(self) -> None:
        """Should return None when memory response has no data."""
        empty_memory = TACMemoryResponse(
            MemoryRetrievalResponse(
                observations=[],
                summaries=[],
                communications=[],
            )
        )
        result = MemoryPromptBuilder.build(empty_memory, None)
        assert result is None


class TestMemoryPromptBuilderCompose:
    """Tests for MemoryPromptBuilder.compose() method."""

    def test_compose_with_memory(
        self, sample_memory_response: TACMemoryResponse, sample_context: ConversationSession
    ) -> None:
        """Should compose system prompt with memory appended."""
        base_prompt = "You are a helpful assistant."
        result = MemoryPromptBuilder.compose(base_prompt, sample_memory_response, sample_context)

        assert result.startswith(base_prompt)
        assert "\n\n# Customer Context" in result
        assert "Customer prefers email communication" in result
        assert "test@example.com" in result

    def test_compose_with_no_memory(self) -> None:
        """Should return base prompt unchanged when no memory is available."""
        base_prompt = "You are a helpful assistant."
        result = MemoryPromptBuilder.compose(base_prompt, None, None)

        assert result == base_prompt
        assert "# Customer Context" not in result

    def test_compose_with_empty_memory(self) -> None:
        """Should return base prompt unchanged when memory is empty."""
        base_prompt = "You are a helpful assistant."
        empty_memory = TACMemoryResponse(
            MemoryRetrievalResponse(
                observations=[],
                summaries=[],
                communications=[],
            )
        )
        result = MemoryPromptBuilder.compose(base_prompt, empty_memory, None)

        assert result == base_prompt
        assert "# Customer Context" not in result

    def test_compose_with_none_system_prompt_and_memory(
        self, sample_memory_response: TACMemoryResponse
    ) -> None:
        """Should return just memory when system_prompt is None but memory exists."""
        result = MemoryPromptBuilder.compose(None, sample_memory_response, None)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "# Customer Context" in result
        assert "Customer prefers email communication" in result

    def test_compose_with_none_system_prompt_and_no_memory(self) -> None:
        """Should return empty string when both system_prompt and memory are None."""
        result = MemoryPromptBuilder.compose(None, None, None)
        assert result == ""
        assert isinstance(result, str)

    def test_compose_with_system_prompt_only(self) -> None:
        """Should return system_prompt unchanged when no memory exists."""
        base_prompt = "You are a helpful assistant."
        result = MemoryPromptBuilder.compose(base_prompt, None, None)

        assert result == base_prompt

    def test_compose_handles_all_cases(self, sample_memory_response: TACMemoryResponse) -> None:
        """Verify compose() handles all None/value combinations correctly."""
        base_prompt = "You are a helpful assistant."

        # Case 1: Both None → empty string
        result1 = MemoryPromptBuilder.compose(None, None, None)
        assert result1 == ""
        assert isinstance(result1, str)

        # Case 2: Only system_prompt → system_prompt
        result2 = MemoryPromptBuilder.compose(base_prompt, None, None)
        assert result2 == base_prompt

        # Case 3: Only memory → memory
        result3 = MemoryPromptBuilder.compose(None, sample_memory_response, None)
        assert isinstance(result3, str)
        assert len(result3) > 0
        assert "Customer prefers email communication" in result3

        # Case 4: Both → composed
        result4 = MemoryPromptBuilder.compose(base_prompt, sample_memory_response, None)
        assert isinstance(result4, str)
        assert base_prompt in result4
        assert "Customer prefers email communication" in result4

    def test_compose_with_multiline_base_prompt(
        self, sample_memory_response: TACMemoryResponse
    ) -> None:
        """Should handle multiline base prompts correctly."""
        base_prompt = """You are a helpful assistant.
Keep responses short and conversational.
Do not use markdown."""

        result = MemoryPromptBuilder.compose(base_prompt, sample_memory_response, None)

        assert result.startswith(base_prompt)
        assert "\n\n# Customer Context" in result
        assert "Customer prefers email communication" in result

    def test_compose_eliminates_if_else_pattern(
        self, sample_memory_response: TACMemoryResponse
    ) -> None:
        """
        Demonstrate that compose() eliminates the need for if/else pattern.

        This test shows the before/after comparison mentioned in the docstring.
        """
        base_prompt = "You are a helpful assistant."

        # Old pattern (what users had to write before)
        memory_context = MemoryPromptBuilder.build(sample_memory_response, None)
        if memory_context:
            old_result = f"{base_prompt}\n\n{memory_context}"
        else:
            old_result = base_prompt

        # New pattern (what users write now)
        new_result = MemoryPromptBuilder.compose(base_prompt, sample_memory_response, None)

        # Both should produce the same output
        assert new_result == old_result

    def test_compose_with_empty_base_prompt(
        self, sample_memory_response: TACMemoryResponse
    ) -> None:
        """Should handle empty base prompt by composing with memory."""
        result = MemoryPromptBuilder.compose("", sample_memory_response, None)

        # Empty string is falsy in Python, so compose() treats "" like None:
        # it returns the memory content directly without concatenation
        assert "# Customer Context" in result
        assert "Customer prefers email communication" in result
        assert result.startswith("# Customer Context")  # Memory content only, no prefix
