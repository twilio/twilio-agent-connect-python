from typing import Any

import httpx

from tac.context.base import BaseAPIClient
from tac.models.memory import (
    MemoryRetrievalRequest,
    MemoryRetrievalResponse,
    ProfileLookupRequest,
    ProfileLookupResponse,
    ProfileResponse,
)


class MemoryClient(BaseAPIClient):
    """Client for interacting with Twilio Conversation Memory data plane API."""

    def __init__(
        self,
        store_id: str,
        api_key: str,
        api_secret: str,
        region: str | None = None,
    ) -> None:
        """
        Initialize the Memory client.

        Args:
            store_id: Memory store ID (starts with mem_store_).
            api_key: API Key for Conversation Memory authentication.
            api_secret: API Secret for Conversation Memory authentication.
            region: Optional Twilio region (e.g., 'au1', 'ie1')
        """
        super().__init__(api_key, api_secret, region)
        self.store_id = store_id
        self.base_url = self._build_base_url("memory", self.region)

    async def retrieve_memory(
        self,
        profile_id: str,
        conversation_id: str | None = None,
        query: str | None = None,
        observations_limit: int | None = None,
        summaries_limit: int | None = None,
        communications_limit: int | None = None,
        relevance_threshold: float | None = None,
    ) -> MemoryRetrievalResponse:
        """
        Retrieve conversation memories with semantic search and configurable limits.

        Args:
            profile_id: Profile ID (TTID format)
            conversation_id: Optional conversation ID (TTID format)
            query: Optional semantic search query (1-1024 characters)
            observations_limit: Max observations to return (0-100)
            summaries_limit: Max summaries to return (0-100)
            communications_limit: Max communications to return (0-100)
            relevance_threshold: Min relevance score (0.0-1.0)

        Returns:
            MemoryRetrievalResponse with observations, summaries, communications, and metadata.
            Returns empty MemoryRetrievalResponse() if API request fails or response cannot be
            parsed.
        """

        endpoint = f"/v1/Stores/{self.store_id}/Profiles/{profile_id}/Recall"
        url = f"{self.base_url}{endpoint}"

        request_data = MemoryRetrievalRequest(
            conversation_id=conversation_id,
            query=query,
            observations_limit=observations_limit,
            summaries_limit=summaries_limit,
            communications_limit=communications_limit,
            relevance_threshold=relevance_threshold,
        )
        request_payload = request_data.model_dump(by_alias=True, exclude_none=True)

        try:
            # POST request with JSON body as per API spec
            async with self._get_client() as client:
                response = await client.post(
                    url,
                    json=request_payload,
                )

                response.raise_for_status()

                # Parse the response according to the API spec
                data = response.json()
                memory_response = MemoryRetrievalResponse(**data)

                # Return full response with observations, summaries, sessions, and metadata
                return memory_response

        except httpx.HTTPError as e:
            response_text = (
                getattr(e.response, "text", "No response body")
                if hasattr(e, "response")
                else "No response"
            )
            self.logger.error(
                f"Failed to retrieve context from Conversation Memory: {e}\n"
                f"URL: {url}\n"
                f"Request body: {request_payload}\n"
                f"Response: {response_text}"
            )
            # Return empty response on API errors
            return MemoryRetrievalResponse()

        except Exception as e:
            self.logger.error(f"Failed to parse Conversation Memory response: {e}")
            # Return empty response on parsing errors
            return MemoryRetrievalResponse()

    async def get_profile(
        self,
        profile_id: str,
        trait_groups: list[str] | None = None,
    ) -> ProfileResponse:
        """
        Retrieve a profile by ID with optional trait group selection.

        Args:
            profile_id: Profile ID using Twilio Type ID (TTID) format
            trait_groups: Optional list of trait group names to include in the response

        Returns:
            ProfileResponse containing profile ID, creation timestamp, and traits

        Raises:
            httpx.HTTPError: If the API request fails
            ValueError: If the response cannot be parsed
        """
        endpoint = f"/v1/Stores/{self.store_id}/Profiles/{profile_id}"
        url = f"{self.base_url}{endpoint}"

        params = {}
        if trait_groups:
            params["traitGroups"] = ",".join(trait_groups)

        try:
            async with self._get_client() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                profile_response = ProfileResponse(**data)

                return profile_response
        except httpx.HTTPError as e:
            response_text = (
                getattr(e.response, "text", "No response body")
                if hasattr(e, "response")
                else "No response"
            )
            self.logger.error(
                f"Failed to retrieve profile from Conversation Memory: {e}\n"
                f"URL: {url}\n"
                f"Query params: {params}\n"
                f"Response: {response_text}"
            )
            raise
        except Exception as e:
            self.logger.error(f"Failed to generate Conversation Memory profile response: {e}")
            raise

    async def lookup_profile(
        self,
        id_type: str,
        value: str,
    ) -> ProfileLookupResponse:
        """
        Find profiles that contain a specific identifier value.

        Submit an identifier object specifying the idType and value.
        The value is normalized using the configured identity resolution settings
        (such as phone number formatting) prior to matching. Multiple matches are
        returned if more than one profile is associated with the identifier.
        Returns canonical profile IDs (the earliest ID if profiles have been merged)
        along with the normalized value actually searched.

        Args:
            id_type: Identifier type as configured in the service's Identity Resolution Settings
                    (e.g., "phone", "email"). Must be 2-30 characters.
            value: Raw value captured for the identifier (e.g., "+13175556789").
                  The service normalizes this value according to the configured rules.

        Returns:
            ProfileLookupResponse containing normalized value and list of matching profile IDs

        Raises:
            httpx.HTTPError: If the API request fails
            ValueError: If the response cannot be parsed
        """
        endpoint = f"/v1/Stores/{self.store_id}/Profiles/Lookup"
        url = f"{self.base_url}{endpoint}"

        request_data = ProfileLookupRequest(id_type=id_type, value=value)
        request_payload = request_data.model_dump(by_alias=True, exclude_none=True)

        try:
            async with self._get_client() as client:
                response = await client.post(url, json=request_payload)
                response.raise_for_status()

                data = response.json()
                lookup_response = ProfileLookupResponse(**data)

                return lookup_response
        except httpx.HTTPError as e:
            response_text = (
                getattr(e.response, "text", "No response body")
                if hasattr(e, "response")
                else "No response"
            )
            self.logger.error(
                f"Failed to lookup profile from Conversation Memory: {e}\n"
                f"URL: {url}\n"
                f"Request body: {request_payload}\n"
                f"Response: {response_text}"
            )
            raise
        except Exception as e:
            self.logger.error(f"Failed to parse Conversation Memory lookup response: {e}")
            raise

    async def create_profile(
        self,
        traits: dict[str, dict[str, Any]],
    ) -> str:
        """
        Create a profile via identity resolution (upsert).

        Memora runs identity resolution on the submitted traits: if an
        identifier match is found the existing canonical profile ID is
        returned, otherwise a new profile is minted. The body must contain
        at least one trait promoted-to-identifier per the store's identity
        resolution settings, else resolution fails with 400.

        The write is queued (202 Accepted) — the canonical profile ID is
        returned synchronously in the response body, but downstream traits
        may not be fully persisted immediately.

        Args:
            traits: Trait-group → field → value mapping, e.g.
                `{"Contact": {"phone": "+13175551234"}}`. Max 50 groups × 99
                traits each.

        Returns:
            Canonical profile ID (`mem_profile_…`).

        Raises:
            httpx.HTTPError: If the API request fails.
            ValueError: If the response does not contain an `id` field.
        """
        endpoint = f"/v1/Stores/{self.store_id}/Profiles"
        url = f"{self.base_url}{endpoint}"

        payload: dict[str, Any] = {"traits": traits}

        try:
            async with self._get_client() as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as e:
            response_text = (
                getattr(e.response, "text", "No response body")
                if hasattr(e, "response")
                else "No response"
            )
            self.logger.error(
                f"Failed to create profile: {e}\n"
                f"URL: {url}\n"
                f"Request body: {payload}\n"
                f"Response: {response_text}"
            )
            raise

        profile_id = data.get("id") if isinstance(data, dict) else None
        if not isinstance(profile_id, str):
            raise ValueError(f"CreateProfile response missing 'id' field: {data!r}")
        return profile_id

    async def create_observation(
        self,
        profile_id: str,
        content: str,
        source: str = "conversation-intelligence",
        conversation_ids: list[str] | None = None,
        occurred_at: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a new observation in Conversation Memory.

        Args:
            profile_id: Profile ID to associate observation with
            content: Observation content (the summary text or extracted fact)
            source: Source system identifier (default: "conversation-intelligence")
            conversation_ids: List of conversation IDs this observation relates to
            occurred_at: Optional timestamp when observation occurred (ISO 8601 format)

        Returns:
            Dict with created observation details

        Raises:
            httpx.HTTPError: If the API request fails
        """
        endpoint = f"/v1/Stores/{self.store_id}/Profiles/{profile_id}/Observations"
        url = f"{self.base_url}{endpoint}"

        payload: dict[str, Any] = {
            "content": content,
            "source": source,
        }
        if conversation_ids:
            payload["conversationIds"] = conversation_ids
        if occurred_at:
            payload["occurredAt"] = occurred_at

        try:
            async with self._get_client() as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result: dict[str, Any] = response.json()
                return result

        except httpx.HTTPError as e:
            response_text = (
                getattr(e.response, "text", "No response body")
                if hasattr(e, "response")
                else "No response"
            )
            self.logger.error(
                f"Failed to create observation: {e}\n"
                f"URL: {url}\n"
                f"Profile ID: {profile_id}\n"
                f"Response: {response_text}"
            )
            raise

    async def create_conversation_summaries(
        self,
        profile_id: str,
        summaries: list[dict[str, Any]],
    ) -> dict[str, str]:
        """
        Create conversation summaries in Conversation Memory.

        Args:
            profile_id: Profile ID to associate summaries with
            summaries: List of summary objects, each containing:
                - content (str): The summary text
                - conversationId (str): The conversation ID
                - occurredAt (str): ISO 8601 timestamp when conversation occurred
                - source (str, optional): Source system identifier

        Returns:
            Response dict with message field (e.g., {"message": "Summaries creation accepted"})

        Raises:
            httpx.HTTPError: If the API request fails
        """
        endpoint = f"/v1/Stores/{self.store_id}/Profiles/{profile_id}/ConversationSummaries"
        url = f"{self.base_url}{endpoint}"

        payload: dict[str, Any] = {
            "summaries": summaries,
        }

        try:
            async with self._get_client() as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result: dict[str, str] = response.json()
                return result

        except httpx.HTTPError as e:
            response_text = (
                getattr(e.response, "text", "No response body")
                if hasattr(e, "response")
                else "No response"
            )
            self.logger.error(
                f"Failed to create conversation summaries: {e}\n"
                f"URL: {url}\n"
                f"Profile ID: {profile_id}\n"
                f"Summary count: {len(summaries)}\n"
                f"Response: {response_text}"
            )
            raise
