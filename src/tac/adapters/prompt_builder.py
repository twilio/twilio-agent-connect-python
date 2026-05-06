"""
Memory prompt builder for TAC adapters.

This module provides a clean class-based API for building LLM prompts from TAC memory
data (observations, summaries, communications) and customer profile information.

All adapters (OpenAI, Anthropic, Bedrock, LangChain, etc.) should use MemoryPromptBuilder
to ensure consistent memory presentation across different LLM providers.
"""

from tac.adapters.options import AdapterOptions
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse


class MemoryPromptBuilder:
    """
    Builds LLM prompts from TAC memory and profile data.

    This class orchestrates prompt building by calling helper methods on
    TACMemoryResponse and ConversationSession models, then assembles the
    sections into a complete prompt.

    Example:
        >>> prompt = MemoryPromptBuilder.build(memory_response, context, options)
        >>> if prompt:
        ...     # Inject into your LLM messages
        ...     messages.insert(0, {"role": "system", "content": prompt})
    """

    @staticmethod
    def build(
        memory_response: TACMemoryResponse | None = None,
        context: ConversationSession | None = None,
        options: AdapterOptions | None = None,
    ) -> str | None:
        """
        Build a complete memory prompt from TAC data.

        This is the main entry point. Delegates formatting to model helper methods,
        then assembles sections into a complete prompt.

        Args:
            memory_response: Memory data from TAC.retrieve_memory()
            context: Conversation session with profile data
            options: Adapter options for trait filtering

        Returns:
            Formatted prompt string ready for LLM injection, or None if
            no memory/profile data is available.

        Example:
            >>> prompt = MemoryPromptBuilder.build(
            ...     memory_response=memory_response,
            ...     context=context,
            ...     options=AdapterOptions(profile_traits=["Contact"]),
            ... )
            >>> print(prompt)
            # Customer Context
            You have access to the following information about this customer
            from previous interactions:

            ## Customer Profile
            Information about this customer:
            - Contact: {"name": "John Doe", "email": "john@example.com"}

            ## Key Observations
            Important notes about the customer from previous interactions:
            - Customer prefers email communication
        """
        # Early return if no data to format
        if not memory_response and not (context and context.profile):
            return None

        sections = []

        # Get profile prompt section from ConversationSession model
        if context:
            trait_groups = options.get_profile_traits() if options else None
            profile_section = context.build_profile_prompt(trait_groups)
            if profile_section:
                sections.append(profile_section)

        # Get memory prompt sections from TACMemoryResponse model
        if memory_response:
            memory_sections = memory_response.build_memory_prompts()
            sections.extend(memory_sections)

        # No sections means no data to show
        if not sections:
            return None

        return MemoryPromptBuilder._assemble_prompt(sections)

    @staticmethod
    def compose(
        system_prompt: str | None = None,
        memory_response: TACMemoryResponse | None = None,
        context: ConversationSession | None = None,
        options: AdapterOptions | None = None,
    ) -> str:
        """
        Compose system prompt with memory context.

        Appends memory to system_prompt if available. Always returns a string.

        Example:
            >>> prompt = MemoryPromptBuilder.compose(
            ...     "You are a helpful assistant", memory_response, context
            ... )
        """
        memory_content = MemoryPromptBuilder.build(memory_response, context, options)

        if not system_prompt and not memory_content:
            return ""

        if not system_prompt:
            return memory_content or ""

        if not memory_content:
            return system_prompt

        return f"{system_prompt}\n\n{memory_content}"

    @staticmethod
    def _assemble_prompt(sections: list[str]) -> str:
        """
        Assemble sections into final prompt with header.

        Args:
            sections: List of formatted section strings

        Returns:
            Complete prompt with header and all sections joined.
        """
        header = [
            "# Customer Context",
            (
                "You have access to the following information about this customer "
                "from previous interactions:"
            ),
            "",  # Blank line after header
        ]

        # Join sections with blank lines between them
        body = "\n\n".join(sections)

        # Combine header and body, ensuring a blank line after the header
        return "\n".join(header) + "\n" + body
