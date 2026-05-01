from typing import Any, Literal

import httpx
from pydantic import ValidationError

from tac.context.base import BaseAPIClient
from tac.models.conversation import (
    ActionResponse,
    Communication,
    CommunicationRequest,
    CommunicationsListResponse,
    ConversationConfiguration,
    ConversationRequest,
    ConversationResponse,
    ConversationsListResponse,
    ParticipantAddress,
    ParticipantRequest,
    ParticipantResponse,
    SendMessageActionRequest,
    UpdateConversationRequest,
)

_HEADER_CONFLICTING_RESOURCE_ID = "x-conflicting-resource-id"


class ConversationClient(BaseAPIClient):
    """Client for interacting with Conversation Orchestrator API."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        configuration_id: str,
        region: str | None = None,
    ) -> None:
        """
        Initialize the Conversation client.

        Args:
            api_key: Twilio API Key SID for authentication
            api_secret: Twilio API Key Secret for authentication
            configuration_id: Conversation Configuration ID for API requests
            region: Optional Twilio region (e.g., 'au1', 'ie1')
        """
        super().__init__(api_key, api_secret, region)
        self.configuration_id = configuration_id
        self.base_url = self._build_base_url("conversations", self.region)

    async def list_conversations(
        self,
        status: list[Literal["ACTIVE", "INACTIVE", "CLOSED"]] | None = None,
        channel_id: str | None = None,
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> list[ConversationResponse]:
        """
        List conversations with optional filtering and pagination.

        Args:
            status: Optional list of statuses to filter conversations
                   ("ACTIVE", "INACTIVE", "CLOSED")
            channel_id: Optional resource ID (call ID, message ID, etc.) to filter conversations
            page_size: Maximum number of items to return (1-1000)
            page_token: Token for pagination

        Returns:
            List of ConversationResponse objects

        Raises:
            httpx.HTTPError: If the API request fails
        """
        url = f"{self.base_url}/v2/Conversations"

        # Build query parameters
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        if channel_id:
            params["channelId"] = channel_id
        if page_size:
            params["pageSize"] = page_size
        if page_token:
            params["pageToken"] = page_token

        try:
            async with self._get_client() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                conversations_list = ConversationsListResponse(**response.json())
                return conversations_list.conversations

        except httpx.HTTPError as e:
            response_text = (
                getattr(e.response, "text", "No response body")
                if hasattr(e, "response")
                else "No response"
            )
            self.logger.error(
                f"Failed to list conversations: {e}\n"
                f"URL: {url}\n"
                f"Query params: {params}\n"
                f"Response: {response_text}"
            )
            raise

    async def add_participant(
        self,
        conversation_id: str,
        addresses: list[ParticipantAddress] | None = None,
        participant_type: Literal["HUMAN_AGENT", "CUSTOMER", "AI_AGENT", "AGENT", "UNKNOWN"]
        | None = None,
    ) -> ParticipantResponse:
        """
        Add a new participant to a conversation.

        Used by `_reconcile_participants` when Maestro's v1-bridge emits only
        the customer participant on an inbound SMS/chat — TAC adds itself as
        `AI_AGENT` before replying.

        Args:
            conversation_id: The conversation ID to add participant to
            addresses: List of communication addresses for the participant (optional)
            participant_type: Type of participant (e.g., "CUSTOMER", "AI_AGENT"). Optional.

        Returns:
            ParticipantResponse object containing the created participant details

        Raises:
            httpx.HTTPError: If the API request fails
        """
        url = f"{self.base_url}/v2/Conversations/{conversation_id}/Participants"

        request_data = ParticipantRequest(addresses=addresses, type=participant_type)
        request_payload = request_data.model_dump(by_alias=True, exclude_none=True)

        try:
            async with self._get_client() as client:
                response = await client.post(url, json=request_payload)
                response.raise_for_status()
                return ParticipantResponse(**response.json())

        except httpx.HTTPError as e:
            response_text = (
                getattr(e.response, "text", "No response body")
                if hasattr(e, "response")
                else "No response"
            )
            self.logger.error(
                f"Failed to add participant: {e}\n"
                f"URL: {url}\n"
                f"Request body: {request_payload}\n"
                f"Response: {response_text}"
            )
            raise

    async def update_participant(
        self,
        conversation_id: str,
        participant_id: str,
        participant_type: Literal["HUMAN_AGENT", "CUSTOMER", "AI_AGENT", "AGENT"],
        addresses: list[ParticipantAddress],
        name: str | None = None,
        profile_id: str | None = None,
    ) -> ParticipantResponse:
        """
        Replace an existing participant.

        PUT is a full resource replacement per the Maestro spec — any field
        omitted from the body is cleared on the server. Callers must pass the
        current `addresses` (and `name` if set) to preserve them; pass a new
        `profile_id` to attach a profile during reconciliation.

        Args:
            conversation_id: Conversation ID containing the participant
            participant_id: Participant ID to update
            participant_type: New participant type
            addresses: Current participant addresses (required to avoid wiping)
            name: Current participant display name (optional)
            profile_id: Memora profile ID to attach (optional)

        Returns:
            ParticipantResponse reflecting the updated participant.

        Raises:
            httpx.HTTPError: If the API request fails.
        """
        url = f"{self.base_url}/v2/Conversations/{conversation_id}/Participants/{participant_id}"

        request_data = ParticipantRequest(
            name=name,
            type=participant_type,
            profile_id=profile_id,
            addresses=addresses,
        )
        request_payload = request_data.model_dump(by_alias=True, exclude_none=True)

        try:
            async with self._get_client() as client:
                response = await client.put(
                    url,
                    json=request_payload,
                )
                response.raise_for_status()
                return ParticipantResponse(**response.json())

        except httpx.HTTPError as e:
            response_text = (
                getattr(e.response, "text", "No response body")
                if hasattr(e, "response")
                else "No response"
            )
            self.logger.error(
                f"Failed to update participant: {e}\n"
                f"URL: {url}\n"
                f"Request body: {request_payload}\n"
                f"Response: {response_text}"
            )
            raise

    async def list_participants(self, conversation_id: str) -> list[ParticipantResponse]:
        url = f"{self.base_url}/v2/Conversations/{conversation_id}/Participants"

        try:
            async with self._get_client() as client:
                response = await client.get(url)
                response.raise_for_status()
                participants = response.json().get("participants", [])
                return [ParticipantResponse(**p) for p in participants]
        except httpx.TimeoutException:
            self.logger.error(f"Timeout listing participants for conversation {conversation_id}")
            return []
        except httpx.HTTPError as e:
            self.logger.error(
                f"HTTP error listing participants for conversation {conversation_id}: {e}"
            )
            return []
        except ValueError as e:
            self.logger.error(f"Invalid JSON format when listing participants: {e}")
            return []

    async def create_conversation(
        self,
        name: str | None = None,
        participants: list[ParticipantRequest] | None = None,
    ) -> ConversationResponse:
        """
        Create a new conversation, optionally with inline participants.

        When participants are provided, CO creates them atomically with the
        conversation. If an active conversation with the same participant
        addresses already exists (respecting the configuration's group-by
        rules), CO returns 409 with a pointer to the existing conversation.

        Args:
            name: Conversation name (optional)
            participants: Optional list of participants to create with the conversation

        Returns:
            ConversationResponse object containing the created conversation details

        Raises:
            httpx.HTTPError: If the API request fails
        """
        url = f"{self.base_url}/v2/Conversations"

        request_data = ConversationRequest(
            configuration_id=self.configuration_id, name=name, participants=participants
        )
        request_payload = request_data.model_dump(by_alias=True, exclude_none=True)

        try:
            async with self._get_client() as client:
                response = await client.post(
                    url,
                    json=request_payload,
                )
                response.raise_for_status()
                conversation = ConversationResponse(**response.json())
                return conversation

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                self.logger.info(f"Conversation creation returned 409 (dedup): {e}\nURL: {url}")
            else:
                self.logger.error(
                    f"Failed to create conversation: {e}\n"
                    f"URL: {url}\n"
                    f"Request body: {request_payload}\n"
                    f"Response: {e.response.text}"
                )
            raise
        except httpx.HTTPError as e:
            response_text = (
                getattr(e.response, "text", "No response body")
                if hasattr(e, "response")
                else "No response"
            )
            self.logger.error(
                f"Failed to create conversation: {e}\n"
                f"URL: {url}\n"
                f"Request body: {request_payload}\n"
                f"Response: {response_text}"
            )
            raise

    async def create_or_reuse_conversation(
        self,
        participants: list[ParticipantRequest],
    ) -> tuple[str, bool]:
        """Create a conversation with inline participants, reusing an existing one on 409.

        On 409 CO returns the existing conversation ID in the
        X-Conflicting-Resource-Id response header.

        Returns:
            Tuple of (conversation_id, reused) where reused is True if an
            existing conversation was found via 409 dedup.

        Raises:
            httpx.HTTPStatusError: If the API returns a non-409 error
            RuntimeError: If 409 is returned without X-Conflicting-Resource-Id header
        """
        try:
            conversation = await self.create_conversation(participants=participants)
            return conversation.id, False
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 409:
                raise
            conflict_id = e.response.headers.get(_HEADER_CONFLICTING_RESOURCE_ID)
            if not conflict_id:
                raise RuntimeError(
                    "Received 409 from create_conversation but "
                    f"{_HEADER_CONFLICTING_RESOURCE_ID} header is missing"
                ) from e
            self.logger.info(f"Reusing existing active conversation (409 dedup): {conflict_id}")
            return conflict_id, True

    async def update_conversation(
        self,
        conversation_id: str,
        status: Literal["ACTIVE", "INACTIVE", "CLOSED"],
        name: str | None = None,
    ) -> ConversationResponse:
        """
        Update an existing conversation.

        Args:
            conversation_id: The conversation ID to update
            status: Conversation status to update ("ACTIVE", "INACTIVE", "CLOSED") - required
            name: Optional conversation name to update

        Returns:
            ConversationResponse object containing the updated conversation details

        Raises:
            httpx.HTTPError: If the API request fails
        """
        url = f"{self.base_url}/v2/Conversations/{conversation_id}"

        request_data = UpdateConversationRequest(status=status, name=name)
        request_payload = request_data.model_dump(by_alias=True, exclude_none=True)

        try:
            async with self._get_client() as client:
                response = await client.put(
                    url,
                    json=request_payload,
                )
                response.raise_for_status()
                conversation = ConversationResponse(**response.json())
                return conversation

        except httpx.HTTPError as e:
            response_text = (
                getattr(e.response, "text", "No response body")
                if hasattr(e, "response")
                else "No response"
            )
            self.logger.error(
                f"Failed to update conversation: {e}\n"
                f"URL: {url}\n"
                f"Request body: {request_payload}\n"
                f"Response: {response_text}"
            )
            raise

    async def clear_status_callbacks(
        self,
        conversation_id: str,
    ) -> None:
        """
        Clear statusCallbacks on a conversation's instance configuration.

        This stops the conversation from sending webhook events to TAC,
        which is needed during handoff so the receiving system can take over.

        Args:
            conversation_id: The conversation ID to update

        Raises:
            httpx.HTTPError: If the API request fails
        """
        url = f"{self.base_url}/v2/Conversations/{conversation_id}"
        request_payload: dict[str, Any] = {"configuration": {"statusCallbacks": []}}

        try:
            async with self._get_client() as client:
                response = await client.patch(
                    url,
                    json=request_payload,
                )
                response.raise_for_status()

        except httpx.HTTPError as e:
            response_text = (
                getattr(e.response, "text", "No response body")
                if hasattr(e, "response")
                else "No response"
            )
            self.logger.error(
                f"Failed to clear status callbacks: {e}\n"
                f"URL: {url}\n"
                f"Request body: {request_payload}\n"
                f"Response: {response_text}"
            )
            raise

    async def create_communication(
        self,
        conversation_id: str,
        communication_request: CommunicationRequest,
    ) -> Communication:
        """
        Create a new communication for a conversation.

        Args:
            conversation_id: The conversation ID to create communication for
            communication_request: CommunicationRequest object with author, content, and recipients

        Returns:
            Communication object containing the created communication details

        Raises:
            httpx.HTTPError: If the API request fails
        """
        url = f"{self.base_url}/v2/Conversations/{conversation_id}/Communications"

        request_payload = communication_request.model_dump(by_alias=True, exclude_none=True)

        try:
            async with self._get_client() as client:
                response = await client.post(
                    url,
                    json=request_payload,
                )
                response.raise_for_status()
                communication = Communication(**response.json())
                return communication

        except httpx.HTTPError as e:
            response_text = (
                getattr(e.response, "text", "No response body")
                if hasattr(e, "response")
                else "No response"
            )
            self.logger.error(
                f"Failed to add communication: {e}\n"
                f"URL: {url}\n"
                f"Request body: {request_payload}\n"
                f"Response: {response_text}"
            )
            raise

    async def list_communications(
        self,
        conversation_id: str,
        channel_id: str | None = None,
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> list[Communication]:
        """
        List communications for a conversation.

        Args:
            conversation_id: The conversation ID to list communications for
            channel_id: Optional channel ID filter (call ID, message ID, etc.)
            page_size: Maximum number of items to return (1-1000)
            page_token: Token for pagination

        Returns:
            List of Communication objects

        Raises:
            httpx.HTTPError: If the API request fails
        """
        url = f"{self.base_url}/v2/Conversations/{conversation_id}/Communications"

        # Build query parameters
        params: dict[str, Any] = {}
        if channel_id:
            params["channelId"] = channel_id
        if page_size:
            params["pageSize"] = page_size
        if page_token:
            params["pageToken"] = page_token

        try:
            async with self._get_client() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                communications_list = CommunicationsListResponse(**response.json())
                return communications_list.communications

        except httpx.HTTPError as e:
            response_text = (
                getattr(e.response, "text", "No response body")
                if hasattr(e, "response")
                else "No response"
            )
            self.logger.error(
                f"Failed to list communications: {e}\n"
                f"URL: {url}\n"
                f"Query params: {params}\n"
                f"Response: {response_text}"
            )
            raise

    async def create_action(
        self,
        conversation_id: str,
        request: SendMessageActionRequest,
    ) -> ActionResponse:
        """
        Create an action via POST /v2/Conversations/{conversationId}/Actions.

        Currently supports SEND_MESSAGE actions. Returns 202 Accepted; the action is
        processed asynchronously and its status can be polled via getAction.

        Args:
            conversation_id: The conversation ID to create the action in
            request: SendMessageActionRequest with `from`, `to`, and content

        Returns:
            ActionResponse with id, type, status, and conversationId

        Raises:
            httpx.HTTPError: If the API request fails
        """
        url = f"{self.base_url}/v2/Conversations/{conversation_id}/Actions"
        request_payload = request.model_dump(by_alias=True, exclude_none=True)

        try:
            async with self._get_client() as client:
                response = await client.post(url, json=request_payload)
                response.raise_for_status()

                # Expect 202 Accepted, warn on other 2xx
                if response.status_code != 202:
                    self.logger.warning(
                        f"Expected 202 Accepted, got {response.status_code}",
                        url=url,
                        status_code=response.status_code,
                    )

                return ActionResponse(**response.json())

        except httpx.HTTPError as e:
            response_text = (
                getattr(e.response, "text", "No response body")
                if hasattr(e, "response")
                else "No response"
            )
            self.logger.error(
                f"Failed to create action: {e}\n"
                f"URL: {url}\n"
                f"ConversationId: {conversation_id}\n"
                f"Request body: {request_payload}\n"
                f"Response: {response_text}"
            )
            raise

    def get_configuration(self, configuration_id: str) -> ConversationConfiguration:
        """
        Retrieve the details for a single configuration.

        Args:
            configuration_id: The configuration ID to retrieve

        Returns:
            ConversationConfiguration object containing the configuration details

        Raises:
            httpx.HTTPError: If the API request fails
            ValueError: If the response schema is invalid
        """
        url = f"{self.base_url}/v2/ControlPlane/Configurations/{configuration_id}"

        try:
            with self._get_sync_client() as client:
                response = client.get(url)
                response.raise_for_status()

                try:
                    configuration = ConversationConfiguration(**response.json())
                    return configuration
                except ValidationError as e:
                    self.logger.error(
                        f"Failed to parse configuration response: {e}\n"
                        f"URL: {url}\nResponse: {response.text}"
                    )
                    raise ValueError(f"Invalid configuration response schema: {e}") from e

        except httpx.HTTPError as e:
            response_text = (
                getattr(e.response, "text", "No response body")
                if hasattr(e, "response")
                else "No response"
            )
            self.logger.error(
                f"Failed to get configuration: {e}\nURL: {url}\nResponse: {response_text}"
            )
            raise
