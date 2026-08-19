"""Tests for BaseChannel's background conversation sweeper.

The sweeper reconciles a channel's instance-local `_conversations` against
Conversation Orchestrator, so a `CONVERSATION_UPDATED`/`CLOSED` webhook that
routed to a different instance doesn't leave this one leaking the session.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from tac import TAC
from tac.channels.sms import SMSChannel
from tac.core.config import DEFAULT_CONVERSATION_SWEEP_INTERVAL, TACConfig
from tac.models.conversation import ConversationResponse


def get_test_config(**overrides: Any) -> dict[str, Any]:
    """Get a valid test configuration."""
    config: dict[str, Any] = {
        "account_sid": "ACtest123",
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_token",
        "conversation_configuration_id": "conv_configuration_test123",
        "phone_number": "+15551234567",
    }
    config.update(overrides)
    return config


def make_conversation(conv_id: str, status: str) -> ConversationResponse:
    return ConversationResponse(
        id=conv_id,
        accountId="ACtest123",
        configurationId="conv_configuration_test123",
        status=status,
    )


async def _closed(conv_id: str) -> ConversationResponse:
    return make_conversation(conv_id, "CLOSED")


def http_status_error(status_code: int) -> httpx.HTTPStatusError:
    response = Mock()
    response.status_code = status_code
    return httpx.HTTPStatusError(f"{status_code} error", request=Mock(), response=response)


def make_channel(**config_overrides: Any) -> SMSChannel:
    """Build an SMSChannel with a mocked Orchestrator client."""
    tac = TAC(get_test_config(**config_overrides))
    channel = SMSChannel(tac=tac)
    assert tac.conversation_orchestrator_client is not None
    tac.conversation_orchestrator_client.get_conversation = AsyncMock()  # type: ignore[method-assign]
    return channel


class TestSweepClosedConversations:
    """One sweep pass: what gets evicted and what survives."""

    @pytest.mark.asyncio
    async def test_closed_conversation_is_ended(self) -> None:
        """A CLOSED conversation is removed and fires on_conversation_ended."""
        channel = make_channel()
        ended: list[str] = []
        channel.tac.on_conversation_ended(lambda session: ended.append(session.conversation_id))

        channel._start_conversation("CH_closed")
        channel.tac.conversation_orchestrator_client.get_conversation.return_value = (
            make_conversation("CH_closed", "CLOSED")
        )

        await channel._sweep_closed_conversations()

        assert "CH_closed" not in channel._conversations
        assert ended == ["CH_closed"]

    @pytest.mark.asyncio
    async def test_missing_conversation_is_ended(self) -> None:
        """A 404 means Orchestrator no longer has it — treat as gone."""
        channel = make_channel()
        channel._start_conversation("CH_gone")
        channel.tac.conversation_orchestrator_client.get_conversation.side_effect = (
            http_status_error(404)
        )

        await channel._sweep_closed_conversations()

        assert "CH_gone" not in channel._conversations

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["ACTIVE", "INACTIVE"])
    async def test_live_conversation_is_kept(self, status: str) -> None:
        """ACTIVE and INACTIVE are both live states — INACTIVE can return to ACTIVE."""
        channel = make_channel()
        channel._start_conversation("CH_live")
        channel.tac.conversation_orchestrator_client.get_conversation.return_value = (
            make_conversation("CH_live", status)
        )

        await channel._sweep_closed_conversations()

        assert "CH_live" in channel._conversations

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [429, 500, 503])
    async def test_transient_http_error_keeps_conversation(self, status_code: int) -> None:
        """An Orchestrator blip must never tear down a live conversation."""
        channel = make_channel()
        channel._start_conversation("CH_live")
        channel.tac.conversation_orchestrator_client.get_conversation.side_effect = (
            http_status_error(status_code)
        )

        await channel._sweep_closed_conversations()

        assert "CH_live" in channel._conversations

    @pytest.mark.asyncio
    async def test_unexpected_exception_keeps_conversation(self) -> None:
        """Non-HTTP failures are inconclusive too — fail closed."""
        channel = make_channel()
        channel._start_conversation("CH_live")
        channel.tac.conversation_orchestrator_client.get_conversation.side_effect = RuntimeError(
            "boom"
        )

        await channel._sweep_closed_conversations()

        assert "CH_live" in channel._conversations

    @pytest.mark.asyncio
    async def test_sweeps_only_closed_conversations_from_a_mixed_set(self) -> None:
        """Each tracked conversation is judged independently."""
        channel = make_channel()
        for conv_id in ("CH_a", "CH_b", "CH_c"):
            channel._start_conversation(conv_id)

        statuses = {"CH_a": "ACTIVE", "CH_b": "CLOSED", "CH_c": "INACTIVE"}

        async def fake_get(conv_id: str) -> ConversationResponse:
            return make_conversation(conv_id, statuses[conv_id])

        channel.tac.conversation_orchestrator_client.get_conversation.side_effect = fake_get

        await channel._sweep_closed_conversations()

        assert set(channel._conversations) == {"CH_a", "CH_c"}

    @pytest.mark.asyncio
    async def test_no_tracked_conversations_makes_no_api_calls(self) -> None:
        """An empty dict short-circuits before touching Orchestrator."""
        channel = make_channel()

        await channel._sweep_closed_conversations()

        channel.tac.conversation_orchestrator_client.get_conversation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_conversation_removed_mid_sweep_is_not_double_ended(self) -> None:
        """If the webhook cleans up while we await, don't fire the callback twice."""
        channel = make_channel()
        ended: list[str] = []
        channel.tac.on_conversation_ended(lambda session: ended.append(session.conversation_id))
        channel._start_conversation("CH_racing")

        async def fake_get(conv_id: str) -> ConversationResponse:
            # Simulate the CLOSED webhook landing here mid-flight.
            await channel._end_conversation(conv_id)
            return make_conversation(conv_id, "CLOSED")

        channel.tac.conversation_orchestrator_client.get_conversation.side_effect = fake_get

        await channel._sweep_closed_conversations()

        assert ended == ["CH_racing"]
        assert channel._conversations == {}

    @pytest.mark.asyncio
    async def test_relay_only_mode_sweep_is_a_noop(self) -> None:
        """No Orchestrator client means nothing to reconcile against."""
        tac = TAC(get_test_config(conversation_configuration_id=None))
        channel = SMSChannel.__new__(SMSChannel)
        channel.tac = tac
        channel._conversations = {}

        await channel._sweep_closed_conversations()  # must not raise


class TestSweeperLifecycle:
    """Starting, stopping, and the enable/disable switch."""

    @pytest.mark.asyncio
    async def test_starts_on_first_conversation(self) -> None:
        channel = make_channel()
        assert channel._sweeper_task is None

        channel._start_conversation("CH_1")

        assert channel._sweeper_task is not None
        assert not channel._sweeper_task.done()
        await channel.stop_conversation_sweeper()

    @pytest.mark.asyncio
    async def test_reuses_the_running_task(self) -> None:
        channel = make_channel()
        channel._start_conversation("CH_1")
        first = channel._sweeper_task

        channel._start_conversation("CH_2")

        assert channel._sweeper_task is first
        await channel.stop_conversation_sweeper()

    @pytest.mark.asyncio
    async def test_disabled_when_interval_is_none(self) -> None:
        channel = make_channel(conversation_sweep_interval=None)

        channel._start_conversation("CH_1")

        assert channel._sweeper_task is None

    @pytest.mark.asyncio
    async def test_not_started_in_relay_only_mode(self) -> None:
        tac = TAC(get_test_config(conversation_configuration_id=None))
        # Messaging channels require Orchestrator, so drive BaseChannel via a stub.
        channel = SMSChannel.__new__(SMSChannel)
        channel.tac = tac
        channel._conversations = {}
        channel._sweeper_task = None

        channel._ensure_sweeper_started()

        assert channel._sweeper_task is None

    def test_no_running_loop_does_not_raise(self) -> None:
        """Channels are often constructed at import time, before a loop exists."""
        channel = make_channel()

        channel._start_conversation("CH_1")  # sync context: no loop

        assert channel._sweeper_task is None

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self) -> None:
        channel = make_channel()
        channel._start_conversation("CH_1")

        await channel.stop_conversation_sweeper()
        await channel.stop_conversation_sweeper()

        assert channel._sweeper_task is None

    @pytest.mark.asyncio
    async def test_stop_without_start_does_not_raise(self) -> None:
        channel = make_channel()
        await channel.stop_conversation_sweeper()

    @pytest.mark.asyncio
    async def test_loop_sweeps_repeatedly_on_the_interval(self) -> None:
        """Sleeps first, then sweeps each interval."""
        channel = make_channel(conversation_sweep_interval=0.01)
        channel._start_conversation("CH_live")
        channel.tac.conversation_orchestrator_client.get_conversation.return_value = (
            make_conversation("CH_live", "ACTIVE")
        )

        sweeps = 0
        original = channel._sweep_closed_conversations

        async def counting_sweep() -> None:
            nonlocal sweeps
            sweeps += 1
            await original()

        channel._sweep_closed_conversations = counting_sweep  # type: ignore[method-assign]
        channel._sweeper_task = asyncio.create_task(channel._sweeper_loop())

        await asyncio.sleep(0.05)
        await channel.stop_conversation_sweeper()

        assert sweeps >= 2

    @pytest.mark.asyncio
    async def test_loop_survives_a_failing_sweep(self) -> None:
        """One bad pass must not kill the loop."""
        channel = make_channel(conversation_sweep_interval=0.01)
        calls = 0

        async def flaky_sweep() -> None:
            nonlocal calls
            calls += 1
            raise RuntimeError("sweep exploded")

        channel._sweep_closed_conversations = flaky_sweep  # type: ignore[method-assign]
        channel._sweeper_task = asyncio.create_task(channel._sweeper_loop())

        await asyncio.sleep(0.05)
        still_running = not channel._sweeper_task.done()
        await channel.stop_conversation_sweeper()

        assert calls >= 2
        assert still_running


class TestSweepIntervalConfig:
    """`conversation_sweep_interval` and its env var."""

    def test_default_is_five_minutes(self) -> None:
        config = TACConfig(**get_test_config())
        assert config.conversation_sweep_interval == DEFAULT_CONVERSATION_SWEEP_INTERVAL
        assert config.conversation_sweep_interval == 300.0

    def test_none_is_accepted(self) -> None:
        config = TACConfig(**get_test_config(conversation_sweep_interval=None))
        assert config.conversation_sweep_interval is None

    @pytest.mark.parametrize("value", [0, -1, -0.5])
    def test_non_positive_is_rejected(self, value: float) -> None:
        with pytest.raises(ValueError):
            TACConfig(**get_test_config(conversation_sweep_interval=value))

    def test_env_unset_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TWILIO_CONVERSATION_SWEEP_INTERVAL", raising=False)
        assert TACConfig._sweep_interval_from_env() == DEFAULT_CONVERSATION_SWEEP_INTERVAL

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_env_blank_uses_default(self, raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TWILIO_CONVERSATION_SWEEP_INTERVAL", raw)
        assert TACConfig._sweep_interval_from_env() == DEFAULT_CONVERSATION_SWEEP_INTERVAL

    @pytest.mark.parametrize(
        "raw", ["none", "NONE", "off", "Off", "disabled", "DISABLED", "0", " 0 "]
    )
    def test_env_disabled_spellings(self, raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TWILIO_CONVERSATION_SWEEP_INTERVAL", raw)
        assert TACConfig._sweep_interval_from_env() is None

    @pytest.mark.parametrize(("raw", "expected"), [("60", 60.0), ("30.5", 30.5), (" 120 ", 120.0)])
    def test_env_numeric(self, raw: str, expected: float, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TWILIO_CONVERSATION_SWEEP_INTERVAL", raw)
        assert TACConfig._sweep_interval_from_env() == expected

    def test_env_garbage_raises_actionable_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TWILIO_CONVERSATION_SWEEP_INTERVAL", "soon")
        with pytest.raises(ValueError, match="TWILIO_CONVERSATION_SWEEP_INTERVAL"):
            TACConfig._sweep_interval_from_env()
