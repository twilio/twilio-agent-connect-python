"""Tests for the Conversation Intelligence Quickstart.

Covers:
- app.py: SCRIPT_COACH_CUSTOMER_PROMPT, handle_message_ready callback,
          OpenAI error handling, transcript polling state, server wiring
- cintel_webhook.py: parse_cintel_webhook, parse_script_adherence, parse_summary
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
    async def test_resets_summary_on_new_conversation(self):
        mock_openai = self._make_openai_mock()
        ctx = self._make_context("conv_summary_reset")
        self.mod.summary_text = "old summary from previous call"

        with (
            patch.object(self.mod, "openai_client", mock_openai),
            patch.object(self.mod.voice_channel, "send_response", AsyncMock()),
        ):
            await self.mod.handle_message_ready("hello", ctx, None)

        assert self.mod.summary_text == ""

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
    async def test_appends_agent_and_customer_turns_to_transcript(self):
        mock_openai = self._make_openai_mock("Customer text")
        ctx = self._make_context("conv_transcript")
        self.mod.transcript_turns.clear()

        with (
            patch.object(self.mod, "openai_client", mock_openai),
            patch.object(self.mod.voice_channel, "send_response", AsyncMock()),
        ):
            await self.mod.handle_message_ready("Agent text", ctx, None)

        speakers = [t["speaker"] for t in self.mod.transcript_turns]
        assert "agent" in speakers
        assert "customer" in speakers

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
    async def test_rolls_back_history_on_authentication_error(self):
        from openai import AuthenticationError

        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(
            side_effect=AuthenticationError("bad key", response=MagicMock(), body={})
        )
        ctx = self._make_context("conv_rollback_auth")

        with (
            patch.object(self.mod, "openai_client", mock_openai),
            patch.object(self.mod.voice_channel, "send_response", AsyncMock()),
        ):
            await self.mod.handle_message_ready("hello", ctx, None)

        assert len(self.mod.conversation_history.get("conv_rollback_auth", [])) == 0

    @pytest.mark.asyncio
    async def test_rolls_back_history_on_unexpected_error(self):
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(side_effect=RuntimeError("unexpected"))
        ctx = self._make_context("conv_rollback_unexpected")
        sent = []

        async def fake_send(conv_id, text):
            sent.append(text)

        with (
            patch.object(self.mod, "openai_client", mock_openai),
            patch.object(self.mod.voice_channel, "send_response", side_effect=fake_send),
        ):
            await self.mod.handle_message_ready("hello", ctx, None)  # must not raise

        assert len(self.mod.conversation_history.get("conv_rollback_unexpected", [])) == 0
        assert len(sent) == 1  # fallback response sent


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

        await self.mod.handle_conversation_ended(ctx)

        assert "conv_cleanup" not in self.mod.conversation_history

    @pytest.mark.asyncio
    async def test_sets_call_inactive_on_end(self):
        ctx = MagicMock()
        ctx.conversation_id = "conv_ended"
        self.mod.call_active = True

        await self.mod.handle_conversation_ended(ctx)

        assert self.mod.call_active is False


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


# ---------------------------------------------------------------------------
# _parse_script_hints and _register_routes (FastAPI endpoints)
# ---------------------------------------------------------------------------


class TestParseScriptHints:
    def setup_method(self):
        self.mod = _load_app_module()

    def test_extracts_example_per_category(self):
        script = (
            "Category: greeting\n"
            "- introduce_yourself: Agent introduces.\n"
            '  Example: "Hi, this is Sarah from Owl Internet."\n'
            "Category: closing\n"
            "- thank_customer: Thank them.\n"
            '  Example: "Thanks for choosing Owl Internet!"\n'
        )
        hints = self.mod._parse_script_hints(script)
        assert hints["greeting"] == "Hi, this is Sarah from Owl Internet."
        assert hints["closing"] == "Thanks for choosing Owl Internet!"

    def test_ignores_lines_without_example(self):
        script = "Category: greeting\n- introduce_yourself: Agent introduces.\n"
        hints = self.mod._parse_script_hints(script)
        assert hints == {}

    def test_only_first_example_per_category(self):
        script = 'Category: greeting\n  Example: "First example"\n  Example: "Second example"\n'
        hints = self.mod._parse_script_hints(script)
        assert hints["greeting"] == "First example"

    def test_empty_script_returns_empty_dict(self):
        assert self.mod._parse_script_hints("") == {}


class TestDashboardRoutes:
    def setup_method(self):
        import importlib.util as ilu

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
                spec = ilu.spec_from_file_location("cintel_app_routes", QUICKSTART_DIR / "app.py")
                self.mod = ilu.module_from_spec(spec)  # type: ignore[arg-type]
                spec.loader.exec_module(self.mod)  # type: ignore[union-attr]

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        test_app = FastAPI()
        self.mod._register_routes(test_app, "test_auth_token")
        self.client = TestClient(test_app, raise_server_exceptions=True)

    def test_api_transcript_returns_state(self):
        self.mod.transcript_turns.clear()
        self.mod.checkpoints.clear()
        self.mod.transcript_turns.append({"speaker": "agent", "text": "hello"})
        resp = self.client.get("/api/transcript")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["turns"]) == 1
        assert data["turns"][0]["speaker"] == "agent"

    def test_api_transcript_summary_none_when_empty(self):
        self.mod.summary_text = ""
        resp = self.client.get("/api/transcript")
        assert resp.json()["summary"] is None

    def test_api_transcript_returns_summary_when_set(self):
        self.mod.summary_text = "Call went well."
        resp = self.client.get("/api/transcript")
        assert resp.json()["summary"] == "Call went well."
        self.mod.summary_text = ""

    def test_api_reset_clears_state(self):
        self.mod.transcript_turns.append({"speaker": "agent", "text": "hi"})
        self.mod.checkpoints["greeting"] = {"category": "greeting", "completed": True}
        self.mod.summary_text = "old summary"
        self.mod.call_active = True

        resp = self.client.post("/api/reset")
        assert resp.status_code == 200
        assert len(self.mod.transcript_turns) == 0
        assert len(self.mod.checkpoints) == 0
        assert self.mod.summary_text == ""
        assert self.mod.call_active is False

    def test_api_config_returns_phone_number(self):
        with patch.dict(os.environ, {"TWILIO_PHONE_NUMBER": "+15550000000"}):
            resp = self.client.get("/api/config")
        assert resp.status_code == 200
        assert resp.json()["phone_number"] == "+15550000000"

    @pytest.mark.asyncio
    async def test_api_script_returns_empty_on_http_error(self):
        import httpx

        with patch("httpx.AsyncClient") as mock_cls:
            mc = AsyncMock()
            mc.__aenter__ = AsyncMock(return_value=mc)
            mc.__aexit__ = AsyncMock(return_value=False)
            mc.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_cls.return_value = mc

            resp = self.client.get("/api/script")
        assert resp.status_code == 200
        assert resp.json() == {}

    @pytest.mark.asyncio
    async def test_api_script_returns_hints_on_success(self):
        from unittest.mock import MagicMock

        script = 'Category: greeting\n  Example: "Hi, this is Sarah."\n'
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.raise_for_status = MagicMock()
        api_resp.json.return_value = {
            "rules": [{"operators": [{"parameters": {"script": script}}]}]
        }

        with patch("httpx.AsyncClient") as mock_cls:
            mc = AsyncMock()
            mc.__aenter__ = AsyncMock(return_value=mc)
            mc.__aexit__ = AsyncMock(return_value=False)
            mc.get = AsyncMock(return_value=api_resp)
            mock_cls.return_value = mc

            resp = self.client.get("/api/script")
        assert resp.status_code == 200
        assert resp.json().get("greeting") == "Hi, this is Sarah."


# ---------------------------------------------------------------------------
# cintel_webhook.py
# ---------------------------------------------------------------------------

_webhook_mod = importlib.import_module("cintel_webhook")
parse_cintel_webhook = _webhook_mod.parse_cintel_webhook
parse_script_adherence = _webhook_mod.parse_script_adherence
parse_summary = _webhook_mod.parse_summary
SCRIPT_ADHERENCE_OPERATOR_ID = _webhook_mod.SCRIPT_ADHERENCE_OPERATOR_ID
SUMMARY_OPERATOR_ID = _webhook_mod.SUMMARY_OPERATOR_ID


def _make_script_adherence_payload(categories: list) -> dict:
    return {
        "operatorResults": [
            {
                "operator": {"id": SCRIPT_ADHERENCE_OPERATOR_ID},
                "result": {"categories": categories},
            }
        ]
    }


def _make_summary_payload(text: str) -> dict:
    return {
        "operatorResults": [
            {
                "operator": {"id": SUMMARY_OPERATOR_ID},
                "result": {"text": text},
            }
        ]
    }


class TestParseCintelWebhook:
    def test_returns_none_for_empty_payload(self):
        result = parse_cintel_webhook({})
        assert result["script_adherence"] is None
        assert result["summary"] is None

    def test_routes_script_adherence_operator(self):
        payload = _make_script_adherence_payload([])
        result = parse_cintel_webhook(payload)
        assert result["script_adherence"] is not None
        assert result["summary"] is None

    def test_routes_summary_operator(self):
        payload = _make_summary_payload("Call went well.")
        result = parse_cintel_webhook(payload)
        assert result["summary"] is not None
        assert result["script_adherence"] is None

    def test_ignores_unknown_operator_id(self):
        payload = {"operatorResults": [{"operator": {"id": "intelligence_operator_unknown"}}]}
        result = parse_cintel_webhook(payload)
        assert result["script_adherence"] is None
        assert result["summary"] is None

    def test_handles_none_entries_in_operator_results(self):
        payload = {
            "operatorResults": [
                None,
                {"operator": {"id": SUMMARY_OPERATOR_ID}, "result": {"text": "summary"}},
            ]
        }
        # should not raise
        result = parse_cintel_webhook(payload)
        assert result["summary"] is not None


class TestParseScriptAdherence:
    def _category(self, key: str, criteria: list) -> dict:
        return {"category_key": key, "criteria": criteria}

    def _criterion(self, key: str, met: str) -> dict:
        return {"criteria_key": key, "criteria_met": met}

    def test_all_succeeded_marks_completed(self):
        op_result = {
            "result": {
                "categories": [
                    self._category("greeting", [self._criterion("introduce_yourself", "Succeeded")])
                ]
            }
        }
        checkpoints = parse_script_adherence(op_result)
        assert len(checkpoints) == 1
        assert checkpoints[0]["completed"] is True
        assert checkpoints[0]["skipped"] is False

    def test_any_failed_marks_skipped(self):
        op_result = {
            "result": {
                "categories": [
                    self._category("greeting", [self._criterion("introduce_yourself", "Failed")])
                ]
            }
        }
        checkpoints = parse_script_adherence(op_result)
        assert checkpoints[0]["skipped"] is True
        assert checkpoints[0]["completed"] is False

    def test_not_evaluated_marks_pending(self):
        op_result = {
            "result": {
                "categories": [
                    self._category(
                        "greeting", [self._criterion("introduce_yourself", "NotEvaluated")]
                    )
                ]
            }
        }
        checkpoints = parse_script_adherence(op_result)
        assert checkpoints[0]["completed"] is False
        assert checkpoints[0]["skipped"] is False
        assert checkpoints[0]["criteria"][0]["evaluated"] is False

    def test_multiple_categories_returned(self):
        op_result = {
            "result": {
                "categories": [
                    self._category("greeting", [self._criterion("c1", "Succeeded")]),
                    self._category("closing", [self._criterion("c2", "Failed")]),
                ]
            }
        }
        checkpoints = parse_script_adherence(op_result)
        assert len(checkpoints) == 2
        keys = {c["category"] for c in checkpoints}
        assert keys == {"greeting", "closing"}

    def test_empty_categories(self):
        op_result = {"result": {"categories": []}}
        assert parse_script_adherence(op_result) == []


class TestParseSummary:
    def test_extracts_summary_text(self):
        op_result = {"result": {"text": "The agent resolved the internet issue."}}
        result = parse_summary(op_result)
        assert result["summary_text"] == "The agent resolved the internet issue."

    def test_empty_text(self):
        op_result = {"result": {"text": ""}}
        result = parse_summary(op_result)
        assert result["summary_text"] == ""

    def test_missing_result_key(self):
        op_result = {}
        result = parse_summary(op_result)
        assert result["summary_text"] == ""
