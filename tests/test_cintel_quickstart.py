"""Tests for the Conversation Intelligence Quickstart.

Covers:
- app.py: SCRIPT_COACH_CUSTOMER_PROMPT, handle_message_ready callback,
          OpenAI error handling, SSE broadcasts, server wiring
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

QUICKSTART_DIR = (
    Path(__file__).parent.parent / "getting_started" / "examples" / "features" / "cintel_quickstart"
)
sys.path.insert(0, str(QUICKSTART_DIR))


def _load_app_module():
    """Import app.py with environment stubs so module-level code doesn't crash."""
    env_patch = {
        "TWILIO_ACCOUNT_SID": "ACtest",
        "TWILIO_AUTH_TOKEN": "token",
        "TWILIO_API_KEY": "SKtest",
        "TWILIO_API_SECRET": "secret",
        "TWILIO_PHONE_NUMBER": "+15550000000",
        "TWILIO_CONVERSATION_CONFIGURATION_ID": "conv_configuration_test",
        "OPENAI_API_KEY": "sk-test",
        "CONVERSATION_INTELLIGENCE_CONFIGURATION_ID": "intelligence_configuration_test",
    }

    with patch.dict(os.environ, env_patch):
        with patch("tac.context.conversation.ConversationClient.get_configuration") as mock_cfg:
            mock_cfg.return_value = MagicMock(memory_store_id="mem_store_test")
            spec = importlib.util.spec_from_file_location("cintel_app", QUICKSTART_DIR / "app.py")
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# SCRIPT_COACH_CUSTOMER_PROMPT
# ---------------------------------------------------------------------------


class TestScriptCoachPrompt:
    """The AI customer prompt must describe the 4 script checkpoints and coaching behavior."""

    def setup_method(self):
        self.mod = _load_app_module()
        self.prompt = self.mod.SCRIPT_COACH_CUSTOMER_PROMPT

    def test_prompt_is_a_string(self):
        assert isinstance(self.prompt, str)
        assert len(self.prompt) > 100

    def test_prompt_contains_greeting_checkpoint(self):
        assert "GREETING" in self.prompt

    def test_prompt_contains_identity_verification_checkpoint(self):
        assert "IDENTITY VERIFICATION" in self.prompt

    def test_prompt_contains_resolution_steps_checkpoint(self):
        assert "RESOLUTION" in self.prompt

    def test_prompt_contains_closing_checkpoint(self):
        assert "CLOSING" in self.prompt or "BRAND-APPROVED" in self.prompt

    def test_prompt_describes_coaching_behavior(self):
        assert "COACHING" in self.prompt or "coach" in self.prompt.lower()

    def test_prompt_references_owl_internet(self):
        assert "Owl Internet" in self.prompt


# ---------------------------------------------------------------------------
# handle_message_ready
# ---------------------------------------------------------------------------


class TestHandleMessageReady:
    """handle_message_ready sends AI customer response and broadcasts SSE events."""

    def setup_method(self):
        self.mod = _load_app_module()
        self.mod.conversation_history.clear()

    def _make_openai_mock(self, content="AI customer response"):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = content
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        return mock_client

    def _make_context(self, conv_id="conv_test_001"):
        ctx = MagicMock()
        ctx.conversation_id = conv_id
        return ctx

    @pytest.mark.asyncio
    async def test_initialises_conversation_history_on_first_call(self):
        mock_openai = self._make_openai_mock()
        ctx = self._make_context("conv_init")

        with (
            patch.object(self.mod, "openai_client", mock_openai),
            patch.object(self.mod.voice_channel, "send_response", AsyncMock()),
        ):
            await self.mod.handle_message_ready("Hello", ctx, None)

        assert "conv_init" in self.mod.conversation_history
        assert self.mod.conversation_history["conv_init"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_appends_user_and_assistant_to_history(self):
        mock_openai = self._make_openai_mock("Customer reply")
        ctx = self._make_context("conv_history")

        with (
            patch.object(self.mod, "openai_client", mock_openai),
            patch.object(self.mod.voice_channel, "send_response", AsyncMock()),
        ):
            await self.mod.handle_message_ready("Agent says hi", ctx, None)

        history = self.mod.conversation_history["conv_history"]
        roles = [m["role"] for m in history]
        assert "user" in roles
        assert "assistant" in roles

    @pytest.mark.asyncio
    async def test_reuses_history_across_turns(self):
        mock_openai = self._make_openai_mock()
        ctx = self._make_context("conv_turns")

        with (
            patch.object(self.mod, "openai_client", mock_openai),
            patch.object(self.mod.voice_channel, "send_response", AsyncMock()),
        ):
            await self.mod.handle_message_ready("turn 1", ctx, None)
            await self.mod.handle_message_ready("turn 2", ctx, None)

        user_msgs = [m for m in self.mod.conversation_history["conv_turns"] if m["role"] == "user"]
        assert len(user_msgs) == 2

    @pytest.mark.asyncio
    async def test_sends_response_via_voice_channel(self):
        mock_openai = self._make_openai_mock("Hello customer")
        ctx = self._make_context("conv_send")
        sent = []

        async def fake_send(conv_id, text):
            sent.append((conv_id, text))

        with (
            patch.object(self.mod, "openai_client", mock_openai),
            patch.object(self.mod.voice_channel, "send_response", side_effect=fake_send),
        ):
            await self.mod.handle_message_ready("hi", ctx, None)

        assert len(sent) == 1
        assert sent[0][0] == "conv_send"
        assert sent[0][1] == "Hello customer"

    @pytest.mark.asyncio
    async def test_broadcasts_transcript_sse_events(self):
        mock_openai = self._make_openai_mock("Customer text")
        ctx = self._make_context("conv_sse")
        broadcasts = []

        with (
            patch.object(self.mod, "openai_client", mock_openai),
            patch.object(self.mod.voice_channel, "send_response", AsyncMock()),
            patch.object(
                self.mod.sse_manager, "broadcast", side_effect=lambda t, d: broadcasts.append(t)
            ),
        ):
            await self.mod.handle_message_ready("Agent text", ctx, None)

        assert "transcript-update" in broadcasts
        assert broadcasts.count("transcript-update") == 2  # agent + customer

    @pytest.mark.asyncio
    async def test_uses_gpt_5_4_mini_model(self):
        mock_openai = self._make_openai_mock()
        ctx = self._make_context("conv_model")

        with (
            patch.object(self.mod, "openai_client", mock_openai),
            patch.object(self.mod.voice_channel, "send_response", AsyncMock()),
        ):
            await self.mod.handle_message_ready("hello", ctx, None)

        call_kwargs = mock_openai.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-5.4-mini"

    @pytest.mark.asyncio
    async def test_sends_fallback_on_authentication_error(self):
        from openai import AuthenticationError

        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(
            side_effect=AuthenticationError("bad key", response=MagicMock(), body={})
        )
        ctx = self._make_context("conv_auth_err")
        sent = []

        async def fake_send(conv_id, text):
            sent.append(text)

        with (
            patch.object(self.mod, "openai_client", mock_openai),
            patch.object(self.mod.voice_channel, "send_response", side_effect=fake_send),
        ):
            await self.mod.handle_message_ready("hello", ctx, None)

        assert len(sent) == 1
        assert "trouble connecting" in sent[0]

    @pytest.mark.asyncio
    async def test_sends_fallback_on_rate_limit_error(self):
        from openai import RateLimitError

        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(
            side_effect=RateLimitError("rate limit", response=MagicMock(), body={})
        )
        ctx = self._make_context("conv_rate")
        sent = []

        async def fake_send(conv_id, text):
            sent.append(text)

        with (
            patch.object(self.mod, "openai_client", mock_openai),
            patch.object(self.mod.voice_channel, "send_response", side_effect=fake_send),
        ):
            await self.mod.handle_message_ready("hello", ctx, None)

        assert len(sent) == 1
        assert "repeat that" in sent[0]

    @pytest.mark.asyncio
    async def test_sends_fallback_on_api_connection_error(self):
        from openai import APIConnectionError

        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(
            side_effect=APIConnectionError(request=MagicMock())
        )
        ctx = self._make_context("conv_conn_err")
        sent = []

        async def fake_send(conv_id, text):
            sent.append(text)

        with (
            patch.object(self.mod, "openai_client", mock_openai),
            patch.object(self.mod.voice_channel, "send_response", side_effect=fake_send),
        ):
            await self.mod.handle_message_ready("hello", ctx, None)

        assert len(sent) == 1
        assert "technical difficulties" in sent[0]

    @pytest.mark.asyncio
    async def test_swallows_unexpected_errors(self):
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(side_effect=RuntimeError("unexpected"))
        ctx = self._make_context("conv_unexpected")

        with (
            patch.object(self.mod, "openai_client", mock_openai),
            patch.object(self.mod.voice_channel, "send_response", AsyncMock()),
        ):
            await self.mod.handle_message_ready("hello", ctx, None)  # must not raise


# ---------------------------------------------------------------------------
# handle_conversation_ended
# ---------------------------------------------------------------------------


class TestHandleConversationEnded:
    def setup_method(self):
        self.mod = _load_app_module()

    @pytest.mark.asyncio
    async def test_cleans_up_conversation_history(self):
        ctx = MagicMock()
        ctx.conversation_id = "conv_cleanup"
        self.mod.conversation_history["conv_cleanup"] = [{"role": "user", "content": "hi"}]

        with patch.object(self.mod.sse_manager, "broadcast"):
            await self.mod.handle_conversation_ended(ctx)

        assert "conv_cleanup" not in self.mod.conversation_history

    @pytest.mark.asyncio
    async def test_broadcasts_call_ended_event(self):
        ctx = MagicMock()
        ctx.conversation_id = "conv_ended"
        broadcasts = []

        with patch.object(
            self.mod.sse_manager, "broadcast", side_effect=lambda t, d: broadcasts.append(t)
        ):
            await self.mod.handle_conversation_ended(ctx)

        assert "call-ended" in broadcasts


# ---------------------------------------------------------------------------
# Server wiring
# ---------------------------------------------------------------------------


class TestServerWiring:
    def test_tac_registered_message_callback(self):
        mod = _load_app_module()
        assert mod.tac._message_ready_callback is mod.handle_message_ready

    def test_tac_registered_conversation_ended_callback(self):
        mod = _load_app_module()
        assert mod.tac._conversation_ended_callback is mod.handle_conversation_ended

    def test_voice_channel_is_initialised(self):
        mod = _load_app_module()
        assert mod.voice_channel is not None

    def test_ci_config_id_is_set(self):
        mod = _load_app_module()
        assert mod.ci_config_id == "intelligence_configuration_test"
