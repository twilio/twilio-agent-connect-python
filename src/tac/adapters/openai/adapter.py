"""OpenAI adapter for automatic memory injection using wrapper approach."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, overload

from tac.adapters.options import AdapterOptions
from tac.adapters.prompt_builder import MemoryPromptBuilder
from tac.core.logging import get_logger
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse

logger = get_logger(__name__)

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI
    from openai.types.chat import ChatCompletionMessageParam


class _BaseCompletionsNamespace:
    """Base class for completions namespace wrappers with shared logic."""

    def __init__(
        self,
        completions: Any,
        memory_response: TACMemoryResponse | None,
        context: ConversationSession | None,
        options: AdapterOptions | None,
    ):
        self._completions = completions
        self._memory_response = memory_response
        self._context = context
        self._options = options

    def _enhance_messages(
        self, messages: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        """Enhance messages with memory injection."""
        return _inject_memory(messages, self._memory_response, self._context, self._options)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class TACCompletionsNamespace(_BaseCompletionsNamespace):
    """Sync wrapper for OpenAI chat.completions namespace with memory injection."""

    def create(self, *args: Any, messages: list[ChatCompletionMessageParam], **kwargs: Any) -> Any:
        """Intercepts create() calls to inject memory automatically."""
        return self._completions.create(*args, messages=self._enhance_messages(messages), **kwargs)

    def stream(self, *args: Any, messages: list[ChatCompletionMessageParam], **kwargs: Any) -> Any:
        """Intercepts stream() calls to inject memory automatically."""
        return self._completions.stream(*args, messages=self._enhance_messages(messages), **kwargs)


class AsyncTACCompletionsNamespace(_BaseCompletionsNamespace):
    """Async wrapper for OpenAI chat.completions namespace with memory injection."""

    async def create(
        self, *args: Any, messages: list[ChatCompletionMessageParam], **kwargs: Any
    ) -> Any:
        """Intercepts async create() calls to inject memory automatically."""
        return await self._completions.create(
            *args, messages=self._enhance_messages(messages), **kwargs
        )

    def stream(self, *args: Any, messages: list[ChatCompletionMessageParam], **kwargs: Any) -> Any:
        """Intercepts async stream() calls to inject memory automatically."""
        return self._completions.stream(*args, messages=self._enhance_messages(messages), **kwargs)


class _BaseResponsesNamespace:
    """Base class for responses namespace wrappers with shared logic."""

    def __init__(
        self,
        responses: Any,
        memory_response: TACMemoryResponse | None,
        context: ConversationSession | None,
        options: AdapterOptions | None,
    ):
        self._responses = responses
        self._memory_response = memory_response
        self._context = context
        self._options = options

    def _enhance_instructions(self, instructions: str | None) -> str:
        """Enhance instructions with memory injection. Always returns a string."""
        return MemoryPromptBuilder.compose(
            instructions, self._memory_response, self._context, self._options
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._responses, name)


class TACResponsesNamespace(_BaseResponsesNamespace):
    """Sync wrapper for OpenAI responses namespace with memory injection."""

    def create(self, *args: Any, instructions: str | None = None, **kwargs: Any) -> Any:
        """Intercepts create() calls to inject memory automatically."""
        return self._responses.create(
            *args, instructions=self._enhance_instructions(instructions), **kwargs
        )


class AsyncTACResponsesNamespace(_BaseResponsesNamespace):
    """Async wrapper for OpenAI responses namespace with memory injection."""

    async def create(self, *args: Any, instructions: str | None = None, **kwargs: Any) -> Any:
        """Intercepts async create() calls to inject memory automatically."""
        return await self._responses.create(
            *args, instructions=self._enhance_instructions(instructions), **kwargs
        )


class _BaseChatNamespace:
    """Base class for chat namespace wrappers with shared logic."""

    def __init__(
        self,
        chat: Any,
        memory_response: TACMemoryResponse | None,
        context: ConversationSession | None,
        options: AdapterOptions | None,
    ):
        self._chat = chat
        self._memory_response = memory_response
        self._context = context
        self._options = options

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class TACChatNamespace(_BaseChatNamespace):
    """Sync wrapper for OpenAI chat namespace."""

    @property
    def completions(self) -> TACCompletionsNamespace:
        return TACCompletionsNamespace(
            self._chat.completions, self._memory_response, self._context, self._options
        )


class AsyncTACChatNamespace(_BaseChatNamespace):
    """Async wrapper for OpenAI chat namespace."""

    @property
    def completions(self) -> AsyncTACCompletionsNamespace:
        return AsyncTACCompletionsNamespace(
            self._chat.completions, self._memory_response, self._context, self._options
        )


class _BaseOpenAIClient:
    """Base class for OpenAI client wrappers with shared logic."""

    def __init__(
        self,
        client: OpenAI | AsyncOpenAI,
        memory_response: TACMemoryResponse | None,
        context: ConversationSession | None,
        options: AdapterOptions | None,
    ):
        self._client = client
        self._memory_response = memory_response
        self._context = context
        self._options = options

    def __getattr__(self, name: str) -> Any:
        """Proxy all other OpenAI client features (embeddings, images, audio, etc)."""
        return getattr(self._client, name)


class TACOpenAIClient(_BaseOpenAIClient):
    """
    Sync wrapper for OpenAI client that automatically injects TAC memory.

    Does NOT mutate the original client. Safe for global clients and concurrent conversations.
    """

    @property
    def chat(self) -> TACChatNamespace:
        return TACChatNamespace(
            self._client.chat, self._memory_response, self._context, self._options
        )

    @property
    def responses(self) -> TACResponsesNamespace:
        return TACResponsesNamespace(
            self._client.responses, self._memory_response, self._context, self._options
        )


class AsyncTACOpenAIClient(_BaseOpenAIClient):
    """
    Async wrapper for AsyncOpenAI client that automatically injects TAC memory.

    Does NOT mutate the original client. Safe for global clients and concurrent conversations.
    """

    @property
    def chat(self) -> AsyncTACChatNamespace:
        return AsyncTACChatNamespace(
            self._client.chat, self._memory_response, self._context, self._options
        )

    @property
    def responses(self) -> AsyncTACResponsesNamespace:
        return AsyncTACResponsesNamespace(
            self._client.responses, self._memory_response, self._context, self._options
        )


@overload
def with_tac_memory(
    openai_client: OpenAI,
    memory_response: TACMemoryResponse | None = None,
    context: ConversationSession | None = None,
    options: AdapterOptions | None = None,
) -> TACOpenAIClient: ...


@overload
def with_tac_memory(
    openai_client: AsyncOpenAI,
    memory_response: TACMemoryResponse | None = None,
    context: ConversationSession | None = None,
    options: AdapterOptions | None = None,
) -> AsyncTACOpenAIClient: ...


def with_tac_memory(
    openai_client: OpenAI | AsyncOpenAI,
    memory_response: TACMemoryResponse | None = None,
    context: ConversationSession | None = None,
    options: AdapterOptions | None = None,
) -> TACOpenAIClient | AsyncTACOpenAIClient:
    """
    Wraps an OpenAI or AsyncOpenAI client with automatic Twilio memory injection.

    Does NOT mutate the original client. Returns a new wrapper object that
    intercepts chat.completions.create() and stream() calls and injects memory automatically.

    Supports both synchronous and asynchronous clients.

    Args:
        openai_client: The OpenAI or AsyncOpenAI client instance to wrap
        memory_response: Optional memory response from TAC.retrieve_memory()
        context: Optional conversation session context with profile data
        options: Optional adapter options for controlling memory injection

    Returns:
        Wrapped OpenAI client with memory injection (TACOpenAIClient or AsyncTACOpenAIClient)

    Examples:
        Sync usage:
        >>> client = with_tac_memory(openai_client, memory_response, context)
        >>> response = client.chat.completions.create(
        ...     model="gpt-4", messages=[{"role": "user", "content": "Hello"}]
        ... )

        Async usage:
        >>> async_client = with_tac_memory(async_openai_client, memory_response, context)
        >>> response = await async_client.chat.completions.create(
        ...     model="gpt-4", messages=[{"role": "user", "content": "Hello"}]
        ... )

        Streaming:
        >>> with client.chat.completions.stream(
        ...     model="gpt-4", messages=[{"role": "user", "content": "Hello"}]
        ... ) as stream:
        ...     for event in stream:
        ...         print(event.content)
    """
    try:
        from openai import AsyncOpenAI
    except ImportError as e:
        raise ImportError(
            "with_tac_memory() requires the openai package. Install with: pip install openai"
        ) from e

    if isinstance(openai_client, AsyncOpenAI):
        return AsyncTACOpenAIClient(openai_client, memory_response, context, options)
    return TACOpenAIClient(openai_client, memory_response, context, options)


def _inject_memory(
    messages: list[ChatCompletionMessageParam],
    memory_response: TACMemoryResponse | None,
    context: ConversationSession | None,
    options: AdapterOptions | None,
) -> list[ChatCompletionMessageParam]:
    """
    Inject TAC memory and profile into OpenAI messages.

    Uses MemoryPromptBuilder to create a memory prompt, then inserts it
    as a system message at the beginning of the conversation.

    Args:
        messages: Original OpenAI chat messages
        memory_response: Memory data from TAC.retrieve_memory()
        context: Conversation session with profile data
        options: Adapter options for trait filtering

    Returns:
        Enhanced messages with memory injected as first system message,
        or original messages if no memory data is available.
    """
    # Build memory prompt using shared builder
    memory_content = MemoryPromptBuilder.build(memory_response, context, options)

    # No memory to inject
    if not memory_content:
        return messages

    logger.debug("[ADAPTER:OPENAI] Injecting memory context")

    # Create a copy to avoid mutating original messages
    enhanced_messages = copy.deepcopy(messages)

    # Insert memory as system message at the start
    memory_message: ChatCompletionMessageParam = {
        "role": "system",
        "content": memory_content,
    }
    enhanced_messages.insert(0, memory_message)

    return enhanced_messages
