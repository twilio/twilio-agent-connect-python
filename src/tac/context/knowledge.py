from typing import Any

import httpx

from tac.context.base import BaseAPIClient
from tac.models.knowledge import KnowledgeBase, KnowledgeChunkResult


class KnowledgeClient(BaseAPIClient):
    """Client for interacting with Twilio Knowledge Base API.

    See [Enterprise Knowledge](https://www.twilio.com/docs/conversations/knowledge) for
    the product overview.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        region: str | None = None,
    ) -> None:
        """
        Initialize the Knowledge client.

        Args:
            api_key: API Key for Knowledge Base authentication.
            api_secret: API Secret for Knowledge Base authentication.
            region: Optional Twilio region (e.g., 'au1', 'ie1')
        """
        super().__init__(api_key, api_secret, region)
        self.base_url = self._build_base_url("knowledge", self.region)

    async def get_knowledge_base(self, knowledge_base_id: str) -> KnowledgeBase:
        """
        Fetch knowledge base metadata from the Knowledge Base API.

        Args:
            knowledge_base_id: The knowledge base ID to fetch (format: know_knowledgebase_*)

        Returns:
            KnowledgeBase object with metadata from the API

        Raises:
            httpx.HTTPError: If the API request fails
        """
        url = f"{self.base_url}/v2/ControlPlane/KnowledgeBases/{knowledge_base_id}"

        try:
            async with self._get_client() as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

                return KnowledgeBase(**data)
        except httpx.HTTPError as e:
            self.logger.error(f"Failed to fetch knowledge base: {e}")
            raise

    async def search_knowledge_base(
        self,
        knowledge_base_id: str,
        query: str,
        top_k: int = 5,
        knowledge_ids: list[str] | None = None,
    ) -> list[KnowledgeChunkResult]:
        """
        Search a knowledge base with the given query.

        Args:
            knowledge_base_id: The knowledge base ID to search (format: know_knowledgebase_*)
            query: The search query string (max 2048 characters)
            top_k: Number of knowledge chunks to return (default: 5, max: 20)
            knowledge_ids: Optional list of specific knowledge IDs to filter search results

        Returns:
            List of KnowledgeChunkResult objects with content and relevance scores

        Raises:
            httpx.HTTPError: If the API request fails
        """
        url = f"{self.base_url}/v2/KnowledgeBases/{knowledge_base_id}/Search"
        payload: dict[str, Any] = {
            "query": query,
            "top": top_k,
        }
        if knowledge_ids:
            payload["knowledgeIds"] = knowledge_ids

        try:
            async with self._get_client() as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()

                data = response.json()
                chunks = [KnowledgeChunkResult(**chunk) for chunk in data["chunks"]]
                return chunks

        except httpx.HTTPError as e:
            self.logger.error(f"Failed to search knowledge base: {e}")
            raise
