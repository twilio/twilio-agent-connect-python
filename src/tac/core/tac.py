import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError

from tac.context.conversation import ConversationClient
from tac.context.knowledge import KnowledgeClient
from tac.context.memory import MemoryClient
from tac.core.config import TACConfig
from tac.core.logging import get_logger, setup_logging
from tac.intelligence.operator_result_processor import OperatorResultProcessor
from tac.models.intelligence import OperatorProcessingResult
from tac.models.memory import ProfileLookupResponse
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse


class TAC:
    """
    Main Twilio Agent Connect class for processing webhook events with configuration.

    This class accepts configuration and provides methods to process webhook events.
    """

    def __init__(self, config: TACConfig | dict[str, Any]):
        """Initialize TAC instance with configuration.

        Args:
            config: TACConfig instance or dictionary with configuration parameters.
        """
        if isinstance(config, dict):
            try:
                self.config = TACConfig(**config)
            except ValidationError as e:
                raise ValueError(f"Invalid configuration: {e}") from e
        elif isinstance(config, TACConfig):
            self.config = config
        else:
            raise ValueError("Config must be TACConfig instance or dictionary")

        setup_logging(log_level=self.config.log_level, log_format="console")
        self.logger = get_logger(__name__)

        self.conversation_orchestrator_client = ConversationClient(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            configuration_id=self.config.conversation_configuration_id,
            region=self.config.region,
        )

        try:
            configuration = self.conversation_orchestrator_client.get_configuration(
                self.config.conversation_configuration_id
            )
        except Exception as e:
            raise ValueError(
                f"Failed to fetch Conversation Orchestrator configuration: {e}. "
                "TAC initialization requires a valid Conversation Orchestrator configuration. "
                "Please check your conversation_configuration_id and credentials."
            ) from e

        # TODO(maestro): Remove once the Actions API resolves the V1 Chat service SID
        # server-side. Maestro team confirmed this should not be required client-side;
        # until they ship the fix, CHAT sends fail with
        #   "chatService attribute is required for CHAT channel"
        # unless we pass it on channelSettings.chatService. We source it from the
        # Configuration's conversationsV1Bridge since the inbound webhook's serviceId
        # is the literal "unused" for CHAT. When the server-side fix lands, drop this
        # attribute plus ActionChannelSettings.chat_service and the chat channel's
        # chat_service_sid plumbing.
        self.conversations_v1_service_sid: str | None = (
            configuration.conversations_v1_bridge.service_id
            if configuration.conversations_v1_bridge
            else None
        )

        self.conversation_memory_client = MemoryClient(
            store_id=configuration.memory_store_id,
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            region=self.config.region,
        )

        self.knowledge_client: KnowledgeClient | None = None
        if self.config.knowledge_base_id:
            self.knowledge_client = KnowledgeClient(
                api_key=self.config.api_key,
                api_secret=self.config.api_secret,
                region=self.config.region,
            )

        self.ci_processor: OperatorResultProcessor | None = None
        if self.config.conversation_intelligence_config:
            self.ci_processor = OperatorResultProcessor(
                conversation_memory_client=self.conversation_memory_client,
                config=self.config.conversation_intelligence_config,
            )
            self.logger.info("Conversation Intelligence processor initialized")

        self._message_ready_callback: (
            Callable[[str, ConversationSession, TACMemoryResponse | None], str | None]
            | Callable[[str, ConversationSession, TACMemoryResponse | None], Awaitable[str | None]]
            | None
        ) = None

        self._interrupt_callback: (
            Callable[[ConversationSession, Any], None]
            | Callable[[ConversationSession, Any], Awaitable[None]]
            | None
        ) = None

        self._conversation_ended_callback: (
            Callable[[ConversationSession], None]
            | Callable[[ConversationSession], Awaitable[None]]
            | None
        ) = None

    async def retrieve_memory(
        self,
        conversation_context: ConversationSession,
        query: str | None = None,
    ) -> TACMemoryResponse:
        """Retrieve memories from Memory Store with fallback to Conversation Orchestrator.

        Args:
            conversation_context: Session containing conversation and profile information.
            query: Optional search query to filter memories.

        Returns:
            Memory response containing conversation history and profile data.
        """
        try:
            if not conversation_context.profile_id:
                self.logger.debug(
                    "profile_id not found, attempting to lookup profile using address"
                )

                if conversation_context.author_info and conversation_context.author_info.address:
                    address = conversation_context.author_info.address
                    id_type = "email" if "@" in address else "phone"
                    lookup_response: ProfileLookupResponse = (
                        await self.conversation_memory_client.lookup_profile(
                            id_type=id_type,
                            value=address,
                        )
                    )

                    if lookup_response.profiles:
                        conversation_context.profile_id = lookup_response.profiles[0]
                        self.logger.debug(f"Found profile_id: {conversation_context.profile_id}")
                    else:
                        self.logger.debug(f"No profile found for address {address}")
                        raise ValueError("No profile found for address")
                else:
                    self.logger.debug(
                        "profile_id not found and author_info.address not available for lookup"
                    )
                    raise ValueError("No profile_id or author_info available")

            if conversation_context.profile_id and not conversation_context.profile:
                try:
                    profile_response = await self.conversation_memory_client.get_profile(
                        profile_id=conversation_context.profile_id,
                        trait_groups=self.config.memory_config.trait_groups,
                    )
                    conversation_context.profile = profile_response
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.warning(
                        f"Failed to fetch profile for {conversation_context.profile_id}: {e}. "
                        "Continuing without profile data.",
                        exc_info=True,
                    )

            # Get memory retrieval configuration
            cfg = self.config.memory_config
            memory_response = await self.conversation_memory_client.retrieve_memory(
                profile_id=conversation_context.profile_id,
                conversation_id=conversation_context.conversation_id,
                query=query,
                observations_limit=cfg.observations_limit,
                summaries_limit=cfg.summaries_limit,
                communications_limit=cfg.communications_limit,
                relevance_threshold=cfg.relevance_threshold,
            )
            return TACMemoryResponse(memory_response)

        except Exception as e:
            self.logger.warning(
                f"Memory retrieval failed: {e}. "
                "Falling back to Conversation Orchestrator Communications API."
            )
            communications = await self.conversation_orchestrator_client.list_communications(
                conversation_id=conversation_context.conversation_id
            )
            return TACMemoryResponse(communications)

    async def process_cintel_event(
        self,
        payload: dict[str, Any],
    ) -> OperatorProcessingResult:
        """Process Conversation Intelligence webhook and create observations/summaries in Memory.

        Args:
            payload: Webhook payload from Conversation Intelligence service.

        Returns:
            Processing result with created observations and summaries.
        """
        if not self.ci_processor:
            raise ValueError(
                "Conversation Intelligence processor is not initialized. "
                "Ensure conversation_intelligence_config is provided when creating TACConfig."
            )

        return await self.ci_processor.process_event(payload)

    def on_message_ready(
        self,
        callback: (
            Callable[[str, ConversationSession, TACMemoryResponse | None], str | None]
            | Callable[[str, ConversationSession, TACMemoryResponse | None], Awaitable[str | None]]
        ),
    ) -> None:
        """Register callback invoked when a message is ready.

        Callback can return a string (TAC auto-sends to channel) or None (manual handling).

        Example:
            ```python
            async def handle_message(
                message: str, context: ConversationSession, memory: TACMemoryResponse | None
            ) -> str:
                response = await openai_client.responses.create(...)
                return response.output_text  # TAC routes to appropriate channel


            tac.on_message_ready(handle_message)
            ```

        Args:
            callback: Function with (message, context, memory). Returns str or None.
        """
        self._message_ready_callback = callback

    def on_interrupt(
        self,
        callback: (
            Callable[[ConversationSession, Any], None]
            | Callable[[ConversationSession, Any], Awaitable[None]]
        ),
    ) -> None:
        """Register callback invoked on user interrupt.

        Example:
            ```python
            def handle_interrupt(context: ConversationSession, interrupt_data: Any):
                # Handle user interrupt...
                pass


            tac.on_interrupt(handle_interrupt)
            ```

        Args:
            callback: Function to call with (context, interrupt_data). Supports sync and async.
        """
        self._interrupt_callback = callback

    def on_conversation_ended(
        self,
        callback: (
            Callable[[ConversationSession], None] | Callable[[ConversationSession], Awaitable[None]]
        ),
    ) -> None:
        """Register callback invoked when conversation ends.

        Example:
            ```python
            def handle_conversation_ended(context: ConversationSession):
                # Clean up conversation...
                pass


            tac.on_conversation_ended(handle_conversation_ended)
            ```

        Args:
            callback: Function to call with conversation context. Supports sync and async.
        """
        self._conversation_ended_callback = callback

    async def trigger_message_ready(
        self,
        user_message: str,
        conversation_context: ConversationSession,
        memory_response: TACMemoryResponse | None = None,
    ) -> str | None:
        """Trigger the registered message ready callback.

        Args:
            user_message: User's message text.
            conversation_context: Session containing conversation information.
            memory_response: Optional memory data to pass to callback.

        Returns:
            Response string if callback returns one (for auto-send), None otherwise.

        Raises:
            TypeError: If callback returns a value that is neither None nor str.
        """
        if self._message_ready_callback:
            result = self._message_ready_callback(
                user_message, conversation_context, memory_response
            )
            if inspect.isawaitable(result):
                result = await result

            # Validate callback return type (must be str or None)
            if result is not None and not isinstance(result, str):
                raise TypeError(
                    f"on_message_ready callback must return str or None, "
                    f"got {type(result).__name__}. "
                    f"To send responses manually, return None and call channel.send_response()."
                )
            return result
        return None

    def trigger_interrupt(
        self,
        conversation_context: ConversationSession,
        interrupt_data: Any,
    ) -> None:
        """Trigger the registered interrupt callback.

        Args:
            conversation_context: Session containing conversation information.
            interrupt_data: Interrupt event data from voice channel.
        """
        if self._interrupt_callback:
            result = self._interrupt_callback(conversation_context, interrupt_data)
            if inspect.isawaitable(result):
                try:
                    asyncio.ensure_future(result)
                except RuntimeError:
                    # Close the coroutine to prevent "was never awaited" warning
                    if inspect.iscoroutine(result):
                        result.close()
                    self.logger.warning(
                        "Async interrupt callback registered but no event loop running. "
                        "Callback will not be executed."
                    )

    async def trigger_conversation_ended(
        self,
        conversation_context: ConversationSession,
    ) -> None:
        """Trigger the registered conversation ended callback.

        Args:
            conversation_context: Session containing conversation information.
        """
        if self._conversation_ended_callback:
            result = self._conversation_ended_callback(conversation_context)
            if inspect.isawaitable(result):
                await result
