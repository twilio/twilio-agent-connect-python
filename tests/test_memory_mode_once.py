"""Tests for `memory_mode="once"` — a Voice-only mode.

`"once"` primes a recall with no query and caches it on the session for the
rest of the call. It needs a session that outlives a single request, which
only voice has: a call is pinned to the instance holding its WebSocket. The
messaging channels hold nothing between webhooks, so they reject the mode at
construction rather than silently degrading to a cacheless, query-less recall.
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from tac import TAC
from tac.channels.chat import ChatChannel
from tac.channels.messaging import MessagingChannel
from tac.channels.rcs import RCSChannel
from tac.channels.sms import SMSChannel
from tac.channels.voice import VoiceChannel
from tac.channels.voice.conversation_relay import ConversationRelayProviderConfig
from tac.channels.whatsapp import WhatsAppChannel
from tac.context.memory import MemoryClient
from tac.models.conversation import ParticipantAddress
from tac.models.memory import MemoryRetrievalMeta, MemoryRetrievalResponse
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse


def get_test_config() -> dict[str, Any]:
    from tac.core.config import TwilioMemoryConfig

    return {
        "account_sid": "ACtest123",
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_token",
        "conversation_configuration_id": "conv_configuration_test123",
        "phone_number": "+15551234567",
        "rcs_sender_id": "rcs_sender_test",
        "whatsapp_number": "whatsapp:+15551234567",
        "memory_config": TwilioMemoryConfig(trait_groups=["Contact"]),
    }


def make_voice_channel() -> tuple[TAC, VoiceChannel, AsyncMock]:
    """A voice channel in `"once"` mode with the Memory recall call mocked."""
    tac = TAC(get_test_config())
    tac.conversation_memory_client = MemoryClient(
        store_id="MGtest123",
        api_key=tac.config.api_key,
        api_secret=tac.config.api_secret,
    )
    recall = AsyncMock(
        return_value=MemoryRetrievalResponse(
            observations=[],
            summaries=[],
            meta=MemoryRetrievalMeta(queryTime=0),
        )
    )
    tac.conversation_memory_client.retrieve_memory = recall
    tac.conversation_memory_client.get_profile = AsyncMock(side_effect=RuntimeError("no profile"))
    channel = VoiceChannel(tac, config=ConversationRelayProviderConfig(memory_mode="once"))
    return tac, channel, recall


class TestMessagingRejectsOnce:
    """Fail loudly at construction — a silent downgrade shows up in production
    as "the bot forgot things"."""

    @pytest.mark.parametrize("channel_cls", [SMSChannel, RCSChannel, WhatsAppChannel, ChatChannel])
    def test_config_rejects_once(self, channel_cls: type) -> None:
        tac = TAC(get_test_config())
        with pytest.raises(Exception) as exc_info:
            channel_cls(tac, config={"memory_mode": "once"})
        assert "once" in str(exc_info.value)

    def test_custom_subclass_passing_once_directly_is_rejected(self) -> None:
        """The config field's Literal can't see a subclass that bypasses it."""

        class CustomChannel(MessagingChannel):
            def get_channel_name(self) -> str:
                return "SMS"

            def is_default_agent_address(self, author_address: str) -> bool:
                return False

            def get_agent_address(self, session: ConversationSession) -> ParticipantAddress:
                return ParticipantAddress(channel="SMS", address="+15551234567")

        with pytest.raises(ValueError, match='does not support memory_mode="once"'):
            CustomChannel(TAC(get_test_config()), memory_mode="once")


class TestVoiceOnceMode:
    @pytest.mark.asyncio
    async def test_caches_after_the_first_retrieval(self) -> None:
        _tac, channel, recall = make_voice_channel()
        session = channel._start_conversation("CA_once", "mem_profile_1")

        for prompt in ("first", "second", "third"):
            response = await channel._retrieve_memory_if_enabled(session, prompt, "CA_once")
            assert isinstance(response, TACMemoryResponse)

        assert recall.call_count == 1
        assert session.cached_memory is not None

    @pytest.mark.asyncio
    async def test_primes_with_no_query_and_no_conversation_id(self) -> None:
        """No per-turn topic justifies Memory's server-side query expansion."""
        _tac, channel, recall = make_voice_channel()
        session = channel._start_conversation("CA_once", "mem_profile_1")

        await channel._retrieve_memory_if_enabled(session, "what's my balance?", "CA_once")

        assert recall.call_args.kwargs["query"] is None
        assert recall.call_args.kwargs["conversation_id"] is None

    @pytest.mark.asyncio
    async def test_inactive_webhook_invalidates_the_cache(self) -> None:
        """Conversation Orchestrator rewrites memory on the INACTIVE transition,
        so the cached copy is stale from then on."""
        _tac, channel, recall = make_voice_channel()
        session = channel._start_conversation("CH_once", "mem_profile_1")

        await channel._retrieve_memory_if_enabled(session, "first", "CH_once")
        assert session.cached_memory is not None

        await channel.process_webhook(
            {
                "eventType": "CONVERSATION_UPDATED",
                "data": {
                    "id": "CH_once",
                    "status": "INACTIVE",
                    "configurationId": "conv_configuration_test123",
                },
            }
        )
        assert session.cached_memory is None

        await channel._retrieve_memory_if_enabled(session, "second", "CH_once")
        assert recall.call_count == 2

    @pytest.mark.asyncio
    async def test_active_webhook_leaves_the_cache_alone(self) -> None:
        _tac, channel, recall = make_voice_channel()
        session = channel._start_conversation("CH_once", "mem_profile_1")
        await channel._retrieve_memory_if_enabled(session, "first", "CH_once")

        await channel.process_webhook(
            {
                "eventType": "CONVERSATION_UPDATED",
                "data": {
                    "id": "CH_once",
                    "status": "ACTIVE",
                    "configurationId": "conv_configuration_test123",
                },
            }
        )

        assert session.cached_memory is not None
        await channel._retrieve_memory_if_enabled(session, "second", "CH_once")
        assert recall.call_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_prompts_prime_the_cache_once(self) -> None:
        """Two utterances racing must not both issue the priming recall."""
        import asyncio

        _tac, channel, recall = make_voice_channel()
        session = channel._start_conversation("CA_race", "mem_profile_1")

        slow_recall_started = asyncio.Event()
        release = asyncio.Event()
        original = recall.side_effect

        async def slow_recall(**kwargs: Any) -> MemoryRetrievalResponse:
            slow_recall_started.set()
            await release.wait()
            return MemoryRetrievalResponse(
                observations=[], summaries=[], meta=MemoryRetrievalMeta(queryTime=0)
            )

        recall.side_effect = slow_recall
        try:
            first = asyncio.create_task(
                channel._retrieve_memory_if_enabled(session, "a", "CA_race")
            )
            await slow_recall_started.wait()
            second = asyncio.create_task(
                channel._retrieve_memory_if_enabled(session, "b", "CA_race")
            )
            await asyncio.sleep(0)
            release.set()
            await asyncio.gather(first, second)
        finally:
            recall.side_effect = original

        assert recall.call_count == 1
