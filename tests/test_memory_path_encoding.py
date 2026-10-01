"""Tests that memory API path parameters cannot alter the request path."""

import httpx
import pytest

from tac.context.memory import MemoryClient


@pytest.mark.asyncio
async def test_profile_id_cannot_escape_its_path_segment() -> None:
    """A profile ID carrying separators must stay inside its own path segment."""
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json={})

    client = MemoryClient(
        api_key="SK00000000000000000000000000000000",
        api_secret="secret",
        store_id="mem_store_owned_by_the_application",
    )
    client._get_client = lambda: httpx.AsyncClient(  # type: ignore[method-assign]
        transport=httpx.MockTransport(handler)
    )

    await client.retrieve_memory(profile_id="../../mem_store_elsewhere/Profiles/mem_profile_other")

    assert len(seen) == 1
    # raw_path is what goes on the wire: the separators must be percent-encoded,
    # so the value stays one segment and cannot address another store.
    raw_path = seen[0].raw_path.decode()
    assert raw_path.startswith("/v1/Stores/mem_store_owned_by_the_application/Profiles/")
    assert raw_path.endswith("/Recall")
    assert raw_path.count("/") == 6
    assert "%2F" in raw_path
