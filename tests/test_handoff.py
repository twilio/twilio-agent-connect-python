"""Tests for handoff tool and HandoffPayload model."""

import base64
import importlib
import json
from typing import Any
from unittest.mock import AsyncMock
from urllib.parse import parse_qs

import httpx
import pytest

from tac import TAC
from tac.models.handoff import HandoffPayload
from tac.models.session import AuthorInfo, ConversationSession
from tac.tools.base import TACTool
from tac.tools.handoff import (
    build_handoff_payload,
    create_studio_handoff_tool,
    post_studio_handoff,
    studio_executions_url,
    studio_voice_handoff_url,
)

handoff_module = importlib.import_module("tac.tools.handoff")


def get_test_config(**overrides: Any) -> dict[str, Any]:
    """Get a valid test configuration.

    Includes ``studio_handoff_flow_sid`` by default so
    ``create_studio_handoff_tool`` can be instantiated without extra setup;
    individual tests override it (including setting it to ``None``) when
    exercising the missing-SID path.
    """
    config: dict[str, Any] = {
        "account_sid": "AC" + "a" * 32,
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_secret",
        "conversation_configuration_id": "conv_configuration_test123",
        "phone_number": "+15551234567",
        "studio_handoff_flow_sid": "FW" + "a" * 32,
    }
    config.update(overrides)
    return config


class TestHandoffPayload:
    """Test HandoffPayload model."""

    def test_creation_with_all_fields(self) -> None:
        """Test creating HandoffPayload with all fields."""
        payload = HandoffPayload(
            conversation_id="conv_123",
            memory_store_id="mem_service_abc",
            profile_id="prof_456",
            attributes={"reason": "Customer wants human", "department": "billing"},
        )
        assert payload.conversation_id == "conv_123"
        assert payload.memory_store_id == "mem_service_abc"
        assert payload.profile_id == "prof_456"
        assert payload.attributes["reason"] == "Customer wants human"
        assert payload.attributes["department"] == "billing"

    def test_creation_with_minimal_fields(self) -> None:
        """Test creating HandoffPayload with only required fields."""
        payload = HandoffPayload(
            conversation_id="conv_123",
            memory_store_id="",
            profile_id="",
        )
        assert payload.conversation_id == "conv_123"
        assert payload.memory_store_id == ""
        assert payload.profile_id == ""
        assert payload.attributes == {}

    def test_camel_case_serialization(self) -> None:
        """Test that model_dump uses camelCase aliases."""
        payload = HandoffPayload(
            conversation_id="conv_123",
            memory_store_id="mem_service_abc",
            profile_id="prof_456",
            attributes={"reason": "test"},
        )
        dumped = payload.model_dump(by_alias=True)
        assert "conversationId" in dumped
        assert "storeId" in dumped
        assert "profileId" in dumped
        assert dumped["conversationId"] == "conv_123"

    def test_json_serialization(self) -> None:
        """Test JSON serialization uses camelCase."""
        payload = HandoffPayload(
            conversation_id="conv_123",
            memory_store_id="mem_service_abc",
            profile_id="prof_456",
        )
        json_str = payload.model_dump_json(by_alias=True)
        parsed = json.loads(json_str)
        assert "conversationId" in parsed
        assert "storeId" in parsed
        assert "profileId" in parsed

    def test_populate_by_name(self) -> None:
        """Test that model accepts both Python names and aliases."""
        # Using aliases (camelCase)
        payload_alias = HandoffPayload(
            conversationId="conv_1",
            storeId="store_1",
            profileId="prof_1",
        )
        assert payload_alias.conversation_id == "conv_1"

        # Using Python names
        payload_python = HandoffPayload(
            conversation_id="conv_2",
            memory_store_id="store_2",
            profile_id="prof_2",
        )
        assert payload_python.conversation_id == "conv_2"


class TestCreateStudioHandoffTool:
    """Test create_studio_handoff_tool factory."""

    def test_returns_tac_tool(self) -> None:
        """Test that create_studio_handoff_tool returns a TACTool."""
        tac = TAC(get_test_config())
        session = ConversationSession(conversation_id="conv_123", channel="voice")

        tool = create_studio_handoff_tool(tac, session)
        assert isinstance(tool, TACTool)

    def test_tool_name_is_handoff(self) -> None:
        """Test that the tool is named 'handoff'."""
        tac = TAC(get_test_config())
        session = ConversationSession(conversation_id="conv_123", channel="voice")

        tool = create_studio_handoff_tool(tac, session)
        assert tool.name == "handoff"

    def test_schema_only_shows_reason(self) -> None:
        """Test that the LLM schema only exposes the 'reason' parameter."""
        tac = TAC(get_test_config())
        session = ConversationSession(conversation_id="conv_123", channel="voice")

        tool = create_studio_handoff_tool(tac, session)

        schema = tool.params_json_schema
        assert "reason" in schema["properties"]
        assert "reason" in schema["required"]
        # Injected params should NOT appear
        assert "tac_instance" not in schema["properties"]
        assert "session" not in schema["properties"]

    def test_openai_format(self) -> None:
        """Test OpenAI format output."""
        tac = TAC(get_test_config())
        session = ConversationSession(conversation_id="conv_123", channel="voice")

        tool = create_studio_handoff_tool(tac, session)
        openai_fmt = tool.to_openai_format()

        assert openai_fmt["type"] == "function"
        assert openai_fmt["function"]["name"] == "handoff"
        assert "reason" in openai_fmt["function"]["parameters"]["properties"]

    def test_anthropic_format(self) -> None:
        """Test Anthropic format output."""
        tac = TAC(get_test_config())
        session = ConversationSession(conversation_id="conv_123", channel="voice")

        tool = create_studio_handoff_tool(tac, session)
        anthropic_fmt = tool.to_anthropic_format()

        assert anthropic_fmt["name"] == "handoff"
        assert "reason" in anthropic_fmt["input_schema"]["properties"]


class TestHandoffExecution:
    """Test handoff tool execution."""

    @pytest.mark.asyncio
    async def test_handoff_stores_pending_payload(self) -> None:
        """Test that handoff stores payload on session for deferred delivery."""
        tac = TAC(get_test_config())
        tac.conversation_orchestrator_client.update_conversation = AsyncMock()
        tac.conversation_orchestrator_client.clear_status_callbacks = AsyncMock()
        tac.conversation_memory_client.store_id = "mem_service_abc"

        session = ConversationSession(
            conversation_id="conv_123",
            profile_id="prof_456",
            channel="voice",
        )

        tool = create_studio_handoff_tool(tac, session)
        result = await tool(reason="Customer wants human agent")

        # Verify payload is stored on session (not sent via WS)
        assert session.pending_handoff_data is not None
        assert session.pending_handoff_data.type == "end"

        handoff_data = json.loads(session.pending_handoff_data.handoff_data)
        assert handoff_data["conversationId"] == "conv_123"
        assert handoff_data["storeId"] == "mem_service_abc"
        assert handoff_data["profileId"] == "prof_456"
        assert handoff_data["attributes"]["reason"] == "Customer wants human agent"

        assert result == {"status": "handoff_initiated", "channel": "voice"}

    @pytest.mark.asyncio
    async def test_handoff_sets_conversation_inactive(self) -> None:
        """Test that handoff sets conversation status to INACTIVE."""
        tac = TAC(get_test_config())
        tac.conversation_orchestrator_client.update_conversation = AsyncMock()
        tac.conversation_orchestrator_client.clear_status_callbacks = AsyncMock()

        session = ConversationSession(conversation_id="conv_123", channel="voice")

        tool = create_studio_handoff_tool(tac, session)
        await tool(reason="Escalation needed")

        tac.conversation_orchestrator_client.update_conversation.assert_called_once_with(
            conversation_id="conv_123",
            status="INACTIVE",
        )

    @pytest.mark.asyncio
    async def test_handoff_digital_channel_posts_to_studio_flow(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that handoff on digital channels POSTs to the Studio Flow Executions URL."""
        flow_sid = "FW" + "a" * 32
        tac = TAC(
            get_test_config(
                studio_handoff_flow_sid=flow_sid,
                phone_number="+15551234567",
                api_key="SK_key",
                api_secret="tok",
            )
        )
        tac.conversation_orchestrator_client.update_conversation = AsyncMock()
        tac.conversation_orchestrator_client.clear_status_callbacks = AsyncMock()

        post_mock = AsyncMock()
        monkeypatch.setattr(handoff_module, "post_studio_handoff", post_mock)

        session = ConversationSession(conversation_id="conv_123", channel="sms")

        tool = create_studio_handoff_tool(tac, session)
        result = await tool(reason="Customer wants human")

        post_mock.assert_called_once()
        call_args, call_kwargs = post_mock.call_args
        call_payload, call_session = call_args
        assert isinstance(call_payload, HandoffPayload)
        assert call_payload.conversation_id == "conv_123"
        assert call_session is session
        assert call_kwargs == {
            "handoff_url": f"https://studio.twilio.com/v2/Flows/{flow_sid}/Executions",
            "from_address": "+15551234567",
            "api_key": "SK_key",
            "api_secret": "tok",
        }

        # Digital channels should NOT store pending handoff data
        assert session.pending_handoff_data is None
        assert result == {"status": "handoff_initiated", "channel": "sms"}

    def test_factory_raises_without_flow_sid(self) -> None:
        """Factory rejects missing studio_handoff_flow_sid — it's misconfig, not a soft fallback."""
        tac = TAC(get_test_config(studio_handoff_flow_sid=None))
        session = ConversationSession(conversation_id="conv_123", channel="sms")

        with pytest.raises(ValueError, match="studio_handoff_flow_sid"):
            create_studio_handoff_tool(tac, session)

    def test_factory_raises_in_relay_only_mode(self) -> None:
        """Factory rejects relay-only TAC — orchestrator and memory are required."""
        tac = TAC(get_test_config(conversation_configuration_id=None))
        session = ConversationSession(conversation_id="conv_123", channel="voice")

        with pytest.raises(ValueError, match="Conversation Orchestrator"):
            create_studio_handoff_tool(tac, session)

    def test_factory_raises_without_memory_store_id(self) -> None:
        """Factory rejects TAC with orchestrator but no memory store."""
        tac = TAC(get_test_config())
        tac.conversation_memory_client = None
        session = ConversationSession(conversation_id="conv_123", channel="voice")

        with pytest.raises(ValueError, match="memory store ID"):
            create_studio_handoff_tool(tac, session)

    @pytest.mark.asyncio
    async def test_handoff_digital_delivery_failure_reports_handoff_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If Studio POST raises, the tool returns handoff_failed — not a silent success."""
        tac = TAC(get_test_config())
        tac.conversation_orchestrator_client.update_conversation = AsyncMock()
        tac.conversation_orchestrator_client.clear_status_callbacks = AsyncMock()

        post_mock = AsyncMock(side_effect=httpx.HTTPError("boom"))
        monkeypatch.setattr(handoff_module, "post_studio_handoff", post_mock)

        session = ConversationSession(conversation_id="conv_123", channel="sms")

        tool = create_studio_handoff_tool(tac, session)
        result = await tool(reason="Customer wants human")

        assert result["status"] == "handoff_failed"
        assert result["channel"] == "sms"
        assert "boom" in result["error"]

    @pytest.mark.asyncio
    async def test_handoff_returns_session_channel(self) -> None:
        """Test that the returned channel matches the session channel."""
        tac = TAC(get_test_config())
        tac.conversation_orchestrator_client.update_conversation = AsyncMock()
        tac.conversation_orchestrator_client.clear_status_callbacks = AsyncMock()

        session = ConversationSession(conversation_id="conv_123", channel="voice")

        tool = create_studio_handoff_tool(tac, session)
        result = await tool(reason="test")
        assert result["channel"] == "voice"

    @pytest.mark.asyncio
    async def test_handoff_includes_store_id_from_memory_client(self) -> None:
        """Test handoff includes store_id from memory client in payload."""
        tac = TAC(get_test_config())
        tac.conversation_orchestrator_client.update_conversation = AsyncMock()
        tac.conversation_orchestrator_client.clear_status_callbacks = AsyncMock()
        tac.conversation_memory_client.store_id = "mem_service_abc"

        session = ConversationSession(
            conversation_id="conv_123",
            profile_id="prof_456",
            channel="voice",
        )

        tool = create_studio_handoff_tool(tac, session)
        await tool(reason="test")

        assert session.pending_handoff_data is not None
        handoff_data = json.loads(session.pending_handoff_data.handoff_data)
        assert handoff_data["storeId"] == "mem_service_abc"

    @pytest.mark.asyncio
    async def test_handoff_without_profile_id(self) -> None:
        """Test handoff works when profile_id is not set."""
        tac = TAC(get_test_config())
        tac.conversation_orchestrator_client.update_conversation = AsyncMock()
        tac.conversation_orchestrator_client.clear_status_callbacks = AsyncMock()

        session = ConversationSession(
            conversation_id="conv_123",
            channel="voice",
        )

        tool = create_studio_handoff_tool(tac, session)
        await tool(reason="test")

        assert session.pending_handoff_data is not None
        handoff_data = json.loads(session.pending_handoff_data.handoff_data)
        assert handoff_data["profileId"] == ""

    @pytest.mark.asyncio
    async def test_handoff_conversation_update_failure_still_succeeds(self) -> None:
        """Test that handoff succeeds even if conversation update fails."""
        tac = TAC(get_test_config())
        tac.conversation_orchestrator_client.update_conversation = AsyncMock(
            side_effect=Exception("API error")
        )
        tac.conversation_orchestrator_client.clear_status_callbacks = AsyncMock()

        session = ConversationSession(conversation_id="conv_123", channel="voice")

        tool = create_studio_handoff_tool(tac, session)
        result = await tool(reason="test")
        assert result == {"status": "handoff_initiated", "channel": "voice"}

        # Payload should still be stored on session
        assert session.pending_handoff_data is not None

    @pytest.mark.asyncio
    async def test_handoff_clears_status_callbacks(self) -> None:
        """Test that handoff clears statusCallbacks on the conversation."""
        tac = TAC(get_test_config())
        tac.conversation_orchestrator_client.update_conversation = AsyncMock()
        tac.conversation_orchestrator_client.clear_status_callbacks = AsyncMock()

        session = ConversationSession(conversation_id="conv_123", channel="voice")

        tool = create_studio_handoff_tool(tac, session)
        await tool(reason="test")

        tac.conversation_orchestrator_client.clear_status_callbacks.assert_called_once_with(
            conversation_id="conv_123",
        )

    @pytest.mark.asyncio
    async def test_handoff_order_inactive_then_clear_callbacks_then_store(self) -> None:
        """Test that handoff sets INACTIVE, clears callbacks, then stores payload."""
        tac = TAC(get_test_config())
        call_order: list[str] = []

        async def track_update(**kwargs: Any) -> None:
            call_order.append("update_conversation")

        async def track_clear(**kwargs: Any) -> None:
            call_order.append("clear_status_callbacks")

        tac.conversation_orchestrator_client.update_conversation = AsyncMock(
            side_effect=track_update
        )
        tac.conversation_orchestrator_client.clear_status_callbacks = AsyncMock(
            side_effect=track_clear
        )

        session = ConversationSession(conversation_id="conv_123", channel="voice")

        tool = create_studio_handoff_tool(tac, session)
        await tool(reason="test")

        assert call_order == ["update_conversation", "clear_status_callbacks"]
        # Payload stored after API calls
        assert session.pending_handoff_data is not None

    @pytest.mark.asyncio
    async def test_handoff_clear_callbacks_failure_still_succeeds(self) -> None:
        """Test that handoff succeeds even if clearing callbacks fails."""
        tac = TAC(get_test_config())
        tac.conversation_orchestrator_client.update_conversation = AsyncMock()
        tac.conversation_orchestrator_client.clear_status_callbacks = AsyncMock(
            side_effect=Exception("API error")
        )

        session = ConversationSession(conversation_id="conv_123", channel="voice")

        tool = create_studio_handoff_tool(tac, session)
        result = await tool(reason="test")
        assert result == {"status": "handoff_initiated", "channel": "voice"}

        # Payload should still be stored
        assert session.pending_handoff_data is not None


class TestHandoffAttributes:
    """Test static attributes on create_studio_handoff_tool."""

    @pytest.mark.asyncio
    async def test_attributes_included_in_payload(self) -> None:
        """Test that static attributes appear in the handoff payload."""
        tac = TAC(get_test_config())
        tac.conversation_orchestrator_client.update_conversation = AsyncMock()
        tac.conversation_orchestrator_client.clear_status_callbacks = AsyncMock()

        session = ConversationSession(
            conversation_id="conv_123",
            profile_id="prof_456",
            channel="voice",
        )

        tool = create_studio_handoff_tool(
            tac,
            session,
            attributes={"department": "billing", "priority": "high"},
        )
        await tool(reason="Billing dispute")

        assert session.pending_handoff_data is not None
        handoff_data = json.loads(session.pending_handoff_data.handoff_data)
        assert handoff_data["attributes"]["department"] == "billing"
        assert handoff_data["attributes"]["priority"] == "high"
        assert handoff_data["attributes"]["reason"] == "Billing dispute"

    @pytest.mark.asyncio
    async def test_no_attributes_returns_only_reason(self) -> None:
        """Test that without attributes, payload only contains reason."""
        tac = TAC(get_test_config())
        tac.conversation_orchestrator_client.update_conversation = AsyncMock()
        tac.conversation_orchestrator_client.clear_status_callbacks = AsyncMock()

        session = ConversationSession(
            conversation_id="conv_123",
            channel="voice",
        )

        tool = create_studio_handoff_tool(tac, session)
        await tool(reason="Customer request")

        assert session.pending_handoff_data is not None
        handoff_data = json.loads(session.pending_handoff_data.handoff_data)
        assert handoff_data["attributes"] == {"reason": "Customer request"}

    @pytest.mark.asyncio
    async def test_reason_overrides_attribute_reason(self) -> None:
        """Test that the LLM reason always overrides a static reason attribute."""
        tac = TAC(get_test_config())
        tac.conversation_orchestrator_client.update_conversation = AsyncMock()
        tac.conversation_orchestrator_client.clear_status_callbacks = AsyncMock()

        session = ConversationSession(
            conversation_id="conv_123",
            channel="voice",
        )

        tool = create_studio_handoff_tool(
            tac,
            session,
            attributes={"reason": "static reason", "extra": "value"},
        )
        await tool(reason="LLM reason")

        assert session.pending_handoff_data is not None
        handoff_data = json.loads(session.pending_handoff_data.handoff_data)
        assert handoff_data["attributes"]["reason"] == "LLM reason"
        assert handoff_data["attributes"]["extra"] == "value"


class TestBuildHandoffPayload:
    """Test build_handoff_payload module-level helper."""

    def test_with_memory_store_id(self) -> None:
        """Test payload includes memory_store_id passed in."""
        session = ConversationSession(
            conversation_id="conv_123",
            profile_id="prof_456",
            channel="voice",
        )
        payload = build_handoff_payload(
            session=session,
            memory_store_id="mem_service_xyz",
            attributes={"reason": "test"},
        )

        assert isinstance(payload, HandoffPayload)
        assert payload.conversation_id == "conv_123"
        assert payload.memory_store_id == "mem_service_xyz"
        assert payload.profile_id == "prof_456"
        assert payload.attributes == {"reason": "test"}

    def test_without_profile_id(self) -> None:
        """Test payload has empty profile_id when not set on session."""
        session = ConversationSession(
            conversation_id="conv_123",
            channel="voice",
        )
        payload = build_handoff_payload(
            session=session,
            memory_store_id="",
            attributes={},
        )

        assert payload.profile_id == ""


class TestPostStudioHandoff:
    """Test post_studio_handoff module-level helper (Studio Executions wire format)."""

    @staticmethod
    def _make_payload() -> HandoffPayload:
        return HandoffPayload(
            conversation_id="conv_123",
            memory_store_id="mem_abc",
            profile_id="prof_456",
            attributes={"reason": "Customer wants human", "team": "billing"},
        )

    @staticmethod
    def _install_mock_transport(
        monkeypatch: pytest.MonkeyPatch,
        handler: Any,
    ) -> None:
        transport = httpx.MockTransport(handler)
        orig_async_client = httpx.AsyncClient

        def patched(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            kwargs["transport"] = transport
            return orig_async_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", patched)

    @pytest.mark.asyncio
    async def test_posts_studio_wire_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify POST uses Studio Executions wire format with Basic auth."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["method"] = request.method
            captured["content_type"] = request.headers.get("content-type", "")
            captured["authorization"] = request.headers.get("authorization", "")
            captured["body"] = request.content.decode()
            return httpx.Response(200, json={"sid": "FN_exec"})

        self._install_mock_transport(monkeypatch, handler)

        session = ConversationSession(
            conversation_id="conv_123",
            channel="sms",
            author_info=AuthorInfo(address="+15559998888"),
        )
        payload = self._make_payload()

        await post_studio_handoff(
            payload,
            session,
            handoff_url="https://studio.twilio.com/v2/Flows/FWxxx/Executions",
            from_address="+15551234567",
            api_key="SK_test_key",
            api_secret="secret_token",
        )

        # URL + method
        assert captured["method"] == "POST"
        assert captured["url"] == "https://studio.twilio.com/v2/Flows/FWxxx/Executions"

        # Form-encoded body
        assert "application/x-www-form-urlencoded" in captured["content_type"]

        form = parse_qs(captured["body"])
        assert form["To"] == ["+15559998888"]
        assert form["From"] == ["+15551234567"]

        # Parameters is JSON with top-level HandoffData key
        params_json = form["Parameters"][0]
        params = json.loads(params_json)
        assert "HandoffData" in params
        assert params["HandoffData"]["conversationId"] == "conv_123"
        assert params["HandoffData"]["storeId"] == "mem_abc"
        assert params["HandoffData"]["profileId"] == "prof_456"
        assert params["HandoffData"]["attributes"]["reason"] == "Customer wants human"
        assert params["HandoffData"]["attributes"]["team"] == "billing"

        # Basic auth header
        auth_header = captured["authorization"]
        assert auth_header.startswith("Basic ")
        decoded = base64.b64decode(auth_header[len("Basic ") :]).decode()
        assert decoded == "SK_test_key:secret_token"

    @pytest.mark.asyncio
    async def test_missing_author_info_sends_empty_to(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When session.author_info is None, To should be empty string, no crash."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.content.decode()
            return httpx.Response(200)

        self._install_mock_transport(monkeypatch, handler)

        session = ConversationSession(conversation_id="conv_123", channel="sms")
        payload = self._make_payload()

        await post_studio_handoff(
            payload,
            session,
            handoff_url="https://studio.twilio.com/v2/Flows/FWxxx/Executions",
            from_address="+15551234567",
            api_key="SK",
            api_secret="tok",
        )

        form = parse_qs(captured["body"], keep_blank_values=True)
        assert form["To"] == [""]
        assert form["From"] == ["+15551234567"]


class TestStudioUrlHelpers:
    def test_studio_executions_url(self) -> None:
        flow_sid = "FW" + "a" * 32
        assert (
            studio_executions_url(flow_sid)
            == f"https://studio.twilio.com/v2/Flows/{flow_sid}/Executions"
        )

    def test_studio_voice_handoff_url(self) -> None:
        account_sid = "AC" + "b" * 32
        flow_sid = "FW" + "a" * 32
        assert studio_voice_handoff_url(account_sid, flow_sid) == (
            f"https://webhooks.twilio.com/v1/Accounts/{account_sid}"
            f"/Flows/{flow_sid}?Trigger=incomingCall"
        )
