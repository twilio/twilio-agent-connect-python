"""Tests for the Conversation Intelligence setup server (setup_server.py).

Covers all API endpoints:
- GET /
- POST /api/list-memory-stores
- POST /api/create-memory-store
- POST /api/verify-memory-store
- POST /api/poll-operation-status
- POST /api/create-conversation-configuration
- POST /api/create-intelligence-configuration
- POST /api/link-intelligence-configuration
- POST /api/configure-voice-webhook
"""

import importlib
import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

QUICKSTART_DIR = (
    Path(__file__).parent.parent / "getting_started" / "examples" / "features" / "cintel_quickstart"
)


def _load_server():
    spec = importlib.util.spec_from_file_location(
        "setup_server", QUICKSTART_DIR / "setup_server.py"
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def server_module():
    return _load_server()


@pytest.fixture(scope="module")
def client(server_module):
    return TestClient(server_module.app)


def _mock_http(get=None, post=None, put=None):
    """Return an async context manager mock for httpx.AsyncClient."""
    mc = AsyncMock()
    mc.__aenter__ = AsyncMock(return_value=mc)
    mc.__aexit__ = AsyncMock(return_value=False)
    if get is not None:
        mc.get = AsyncMock(return_value=get)
    if post is not None:
        mc.post = AsyncMock(return_value=post)
    if put is not None:
        mc.put = AsyncMock(return_value=put)
    return mc


def _resp(status_code, data=None, text=None):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = data or {}
    r.text = text or ""
    return r


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


class TestIndexRoute:
    def test_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_html_has_four_steps(self, client):
        body = client.get("/").text
        for step in ["Step 1", "Step 2", "Step 3", "Step 4"]:
            assert step in body


# ---------------------------------------------------------------------------
# POST /api/list-memory-stores
# ---------------------------------------------------------------------------


class TestListMemoryStores:
    def test_missing_credentials_returns_error(self, client):
        data = client.post("/api/list-memory-stores", json={"api_key": "", "api_secret": ""}).json()
        assert data["status"] == "error"

    def test_returns_store_list_on_success(self, client):
        list_resp = _resp(200, {"stores": ["mem_store_aaa"]})
        detail_resp = _resp(
            200, {"id": "mem_store_aaa", "displayName": "My Store", "status": "ACTIVE"}
        )
        mc = AsyncMock()
        mc.__aenter__ = AsyncMock(return_value=mc)
        mc.__aexit__ = AsyncMock(return_value=False)
        mc.get = AsyncMock(side_effect=[list_resp, detail_resp])

        with patch("httpx.AsyncClient", return_value=mc):
            data = client.post(
                "/api/list-memory-stores", json={"api_key": "SK", "api_secret": "s"}
            ).json()

        assert data["status"] == "success"
        assert data["stores"][0]["id"] == "mem_store_aaa"

    def test_returns_error_on_api_failure(self, client):
        mc = _mock_http(get=_resp(401, text="Unauthorized"))
        with patch("httpx.AsyncClient", return_value=mc):
            data = client.post(
                "/api/list-memory-stores", json={"api_key": "SK", "api_secret": "bad"}
            ).json()
        assert data["status"] == "error"
        assert "401" in data["message"]


# ---------------------------------------------------------------------------
# POST /api/create-memory-store
# ---------------------------------------------------------------------------


class TestCreateMemoryStore:
    def test_missing_fields_returns_error(self, client):
        data = client.post(
            "/api/create-memory-store", json={"api_key": "SK", "api_secret": "s"}
        ).json()
        assert data["status"] == "error"

    def test_returns_store_id_on_success(self, client):
        mc = _mock_http(post=_resp(201, {"id": "mem_store_new", "displayName": "My Store"}))
        with patch("httpx.AsyncClient", return_value=mc):
            data = client.post(
                "/api/create-memory-store",
                json={
                    "api_key": "SK",
                    "api_secret": "s",
                    "memory_display_name": "My Store",
                    "memory_description": "desc",
                },
            ).json()
        assert data["status"] == "success"
        assert data["memory_store_id"] == "mem_store_new"

    def test_returns_accepted_on_202(self, client):
        mc = _mock_http(
            post=_resp(
                202, {"statusUrl": "https://memory.twilio.com/v1/ControlPlane/Operations/op1"}
            )
        )
        with patch("httpx.AsyncClient", return_value=mc):
            data = client.post(
                "/api/create-memory-store",
                json={
                    "api_key": "SK",
                    "api_secret": "s",
                    "memory_display_name": "My Store",
                    "memory_description": "desc",
                },
            ).json()
        assert data["status"] == "accepted"
        assert "status_url" in data

    def test_description_too_long_returns_error(self, client):
        data = client.post(
            "/api/create-memory-store",
            json={
                "api_key": "SK",
                "api_secret": "s",
                "memory_display_name": "n",
                "memory_description": "x" * 129,
            },
        ).json()
        assert data["status"] == "error"
        assert "128" in data["message"]


# ---------------------------------------------------------------------------
# POST /api/verify-memory-store
# ---------------------------------------------------------------------------


class TestVerifyMemoryStore:
    def test_returns_success_when_active(self, client):
        mc = _mock_http(get=_resp(200, {"status": "ACTIVE"}))
        with patch("httpx.AsyncClient", return_value=mc):
            data = client.post(
                "/api/verify-memory-store",
                json={
                    "memory_store_id": "mem_store_aaa",
                    "api_key": "SK",
                    "api_secret": "s",
                },
            ).json()
        assert data["status"] == "success"

    def test_returns_pending_when_not_active(self, client):
        mc = _mock_http(get=_resp(200, {"status": "PROVISIONING"}))
        with patch("httpx.AsyncClient", return_value=mc):
            data = client.post(
                "/api/verify-memory-store",
                json={
                    "memory_store_id": "mem_store_aaa",
                    "api_key": "SK",
                    "api_secret": "s",
                },
            ).json()
        assert data["status"] == "pending"


# ---------------------------------------------------------------------------
# POST /api/poll-operation-status
# ---------------------------------------------------------------------------


class TestPollOperationStatus:
    _VALID = {
        "status_url": "https://memory.twilio.com/v1/ControlPlane/Operations/op1",
        "api_key": "SK",
        "api_secret": "s",
    }

    def test_rejects_non_https(self, client):
        data = client.post(
            "/api/poll-operation-status",
            json={
                **self._VALID,
                "status_url": "http://memory.twilio.com/v1/ControlPlane/Operations/op1",
            },
        ).json()
        assert data["status"] == "error"
        assert "https" in data["message"].lower()

    def test_rejects_non_twilio_host(self, client):
        data = client.post(
            "/api/poll-operation-status",
            json={**self._VALID, "status_url": "https://evil.com/v1/ControlPlane/Operations/op1"},
        ).json()
        assert data["status"] == "error"

    def test_returns_completed_with_result(self, client):
        mc = _mock_http(get=_resp(200, {"status": "COMPLETED", "result": {"id": "mem_store_xyz"}}))
        with patch("httpx.AsyncClient", return_value=mc):
            data = client.post("/api/poll-operation-status", json=self._VALID).json()
        assert data["status"] == "completed"
        assert data["result"]["id"] == "mem_store_xyz"

    def test_returns_pending_when_in_progress(self, client):
        mc = _mock_http(get=_resp(200, {"status": "IN_PROGRESS"}))
        with patch("httpx.AsyncClient", return_value=mc):
            data = client.post("/api/poll-operation-status", json=self._VALID).json()
        assert data["status"] == "pending"


# ---------------------------------------------------------------------------
# POST /api/create-conversation-configuration
# ---------------------------------------------------------------------------


class TestCreateConversationConfiguration:
    _BASE = {
        "api_key": "SK",
        "api_secret": "s",
        "memory_store_id": "mem_store_aaa",
        "twilio_phone": "+15550000000",
        "ngrok_domain": "test.ngrok.app",
        "configuration_display_name": "owl-support",
        "configuration_description": "Test",
    }

    def test_missing_field_returns_error(self, client):
        data = client.post(
            "/api/create-conversation-configuration", json={"api_key": "SK", "api_secret": "s"}
        ).json()
        assert data["status"] == "error"

    def test_display_name_too_long_returns_error(self, client):
        data = client.post(
            "/api/create-conversation-configuration",
            json={**self._BASE, "configuration_display_name": "x" * 33},
        ).json()
        assert data["status"] == "error"
        assert "32" in data["message"]

    def test_display_name_with_spaces_returns_error(self, client):
        data = client.post(
            "/api/create-conversation-configuration",
            json={**self._BASE, "configuration_display_name": "bad name"},
        ).json()
        assert data["status"] == "error"

    def test_returns_config_id_on_success(self, client):
        mc = _mock_http(post=_resp(201, {"id": "conv_configuration_abc"}))
        with patch("httpx.AsyncClient", return_value=mc):
            data = client.post("/api/create-conversation-configuration", json=self._BASE).json()
        assert data["status"] == "success"
        assert data["conversation_configuration_id"] == "conv_configuration_abc"

    def test_payload_includes_voice_channel(self, client):
        captured = {}

        async def cap_post(url, headers, json, **kw):
            captured["payload"] = json
            return _resp(201, {"id": "conv_configuration_abc"})

        mc = AsyncMock()
        mc.__aenter__ = AsyncMock(return_value=mc)
        mc.__aexit__ = AsyncMock(return_value=False)
        mc.post = cap_post

        with patch("httpx.AsyncClient", return_value=mc):
            client.post("/api/create-conversation-configuration", json=self._BASE)

        assert "VOICE" in captured["payload"]["channelSettings"]


# ---------------------------------------------------------------------------
# POST /api/create-intelligence-configuration
# ---------------------------------------------------------------------------


class TestCreateIntelligenceConfiguration:
    def test_missing_ngrok_returns_error(self, client):
        data = client.post(
            "/api/create-intelligence-configuration",
            json={"api_key": "SK", "api_secret": "s", "ngrok_domain": ""},
        ).json()
        assert data["status"] == "error"

    def test_sends_two_rules(self, client):
        captured = {}

        async def cap_post(url, headers, json, **kw):
            captured["payload"] = json
            return _resp(201, {"id": "intelligence_configuration_xyz"})

        mc = AsyncMock()
        mc.__aenter__ = AsyncMock(return_value=mc)
        mc.__aexit__ = AsyncMock(return_value=False)
        mc.post = cap_post

        with patch("httpx.AsyncClient", return_value=mc):
            client.post(
                "/api/create-intelligence-configuration",
                json={"api_key": "SK", "api_secret": "s", "ngrok_domain": "t.ngrok.app"},
            )

        rules = captured["payload"]["rules"]
        assert len(rules) == 2
        triggers = {r["triggers"][0]["on"] for r in rules}
        assert triggers == {"COMMUNICATION", "CONVERSATION_END"}

    def test_webhook_url_contains_ci_webhook_path(self, client):
        captured = {}

        async def cap_post(url, headers, json, **kw):
            captured["payload"] = json
            return _resp(201, {"id": "intelligence_configuration_xyz"})

        mc = AsyncMock()
        mc.__aenter__ = AsyncMock(return_value=mc)
        mc.__aexit__ = AsyncMock(return_value=False)
        mc.post = cap_post

        with patch("httpx.AsyncClient", return_value=mc):
            client.post(
                "/api/create-intelligence-configuration",
                json={"api_key": "SK", "api_secret": "s", "ngrok_domain": "myapp.ngrok.app"},
            )

        for rule in captured["payload"]["rules"]:
            assert rule["actions"][0]["url"] == "https://myapp.ngrok.app/ci-webhook"

    def test_returns_id_on_success(self, client):
        mc = _mock_http(post=_resp(201, {"id": "intelligence_configuration_abc"}))
        with patch("httpx.AsyncClient", return_value=mc):
            data = client.post(
                "/api/create-intelligence-configuration",
                json={"api_key": "SK", "api_secret": "s", "ngrok_domain": "t.ngrok.app"},
            ).json()
        assert data["status"] == "success"
        assert data["intelligence_configuration_id"] == "intelligence_configuration_abc"


# ---------------------------------------------------------------------------
# POST /api/link-intelligence-configuration
# ---------------------------------------------------------------------------


class TestLinkIntelligenceConfiguration:
    _BASE = {
        "api_key": "SK",
        "api_secret": "s",
        "conversation_configuration_id": "conv_configuration_abc",
        "intelligence_configuration_id": "intelligence_configuration_xyz",
    }

    def test_missing_fields_returns_error(self, client):
        data = client.post(
            "/api/link-intelligence-configuration", json={"api_key": "SK", "api_secret": "s"}
        ).json()
        assert data["status"] == "error"

    def test_links_when_not_already_linked(self, client):
        mc = AsyncMock()
        mc.__aenter__ = AsyncMock(return_value=mc)
        mc.__aexit__ = AsyncMock(return_value=False)
        mc.get = AsyncMock(
            return_value=_resp(200, {"displayName": "owl", "intelligenceConfigurationIds": []})
        )
        mc.put = AsyncMock(return_value=_resp(200, {}))

        with patch("httpx.AsyncClient", return_value=mc):
            data = client.post("/api/link-intelligence-configuration", json=self._BASE).json()
        assert data["status"] == "success"
        assert data["already_linked"] is False

    def test_already_linked_skips_put(self, client):
        mc = _mock_http(
            get=_resp(
                200,
                {
                    "displayName": "owl",
                    "intelligenceConfigurationIds": ["intelligence_configuration_xyz"],
                },
            )
        )
        with patch("httpx.AsyncClient", return_value=mc):
            data = client.post("/api/link-intelligence-configuration", json=self._BASE).json()
        assert data["status"] == "success"
        assert data["already_linked"] is True

    def test_get_failure_returns_error(self, client):
        mc = _mock_http(get=_resp(404, text="Not Found"))
        with patch("httpx.AsyncClient", return_value=mc):
            data = client.post("/api/link-intelligence-configuration", json=self._BASE).json()
        assert data["status"] == "error"
        assert "404" in data["message"]

    def test_put_preserves_existing_intel_ids(self, client):
        captured = {}

        mc = AsyncMock()
        mc.__aenter__ = AsyncMock(return_value=mc)
        mc.__aexit__ = AsyncMock(return_value=False)
        mc.get = AsyncMock(
            return_value=_resp(
                200,
                {
                    "displayName": "owl",
                    "intelligenceConfigurationIds": ["intelligence_configuration_existing"],
                },
            )
        )

        async def cap_put(url, headers, json, **kw):
            captured["body"] = json
            return _resp(200, {})

        mc.put = cap_put

        with patch("httpx.AsyncClient", return_value=mc):
            client.post("/api/link-intelligence-configuration", json=self._BASE)

        ids = captured["body"]["intelligenceConfigurationIds"]
        assert "intelligence_configuration_existing" in ids
        assert "intelligence_configuration_xyz" in ids


# ---------------------------------------------------------------------------
# POST /api/configure-voice-webhook
# ---------------------------------------------------------------------------


class TestConfigureVoiceWebhook:
    _BASE = {
        "account_sid": "ACtest",
        "api_key": "SK",
        "api_secret": "s",
        "twilio_phone": "+15550000000",
        "ngrok_domain": "test.ngrok.app",
    }

    def test_missing_fields_returns_error(self, client):
        data = client.post(
            "/api/configure-voice-webhook", json={"api_key": "SK", "api_secret": "s"}
        ).json()
        assert data["status"] == "error"

    def test_phone_not_found_returns_error(self, client):
        mc = _mock_http(get=_resp(200, {"incoming_phone_numbers": []}))
        with patch("httpx.AsyncClient", return_value=mc):
            data = client.post("/api/configure-voice-webhook", json=self._BASE).json()
        assert data["status"] == "error"
        assert "not found" in data["message"].lower()

    def test_returns_success_with_voice_url(self, client):
        mc = AsyncMock()
        mc.__aenter__ = AsyncMock(return_value=mc)
        mc.__aexit__ = AsyncMock(return_value=False)
        mc.get = AsyncMock(return_value=_resp(200, {"incoming_phone_numbers": [{"sid": "PN123"}]}))
        mc.post = AsyncMock(return_value=_resp(200, {}))

        with patch("httpx.AsyncClient", return_value=mc):
            data = client.post("/api/configure-voice-webhook", json=self._BASE).json()

        assert data["status"] == "success"
        assert data["voice_url"] == "https://test.ngrok.app/twiml"

    def test_voice_url_uses_twiml_path(self, client):
        mc = AsyncMock()
        mc.__aenter__ = AsyncMock(return_value=mc)
        mc.__aexit__ = AsyncMock(return_value=False)
        mc.get = AsyncMock(return_value=_resp(200, {"incoming_phone_numbers": [{"sid": "PN123"}]}))
        captured = {}

        async def cap_post(url, data, auth):
            captured["data"] = data
            return _resp(200, {})

        mc.post = cap_post

        with patch("httpx.AsyncClient", return_value=mc):
            client.post("/api/configure-voice-webhook", json=self._BASE)

        assert captured["data"]["VoiceUrl"] == "https://test.ngrok.app/twiml"
        assert captured["data"]["VoiceMethod"] == "POST"

    def test_api_failure_returns_error(self, client):
        mc = AsyncMock()
        mc.__aenter__ = AsyncMock(return_value=mc)
        mc.__aexit__ = AsyncMock(return_value=False)
        mc.get = AsyncMock(return_value=_resp(200, {"incoming_phone_numbers": [{"sid": "PN123"}]}))
        mc.post = AsyncMock(return_value=_resp(400, text="Bad Request"))

        with patch("httpx.AsyncClient", return_value=mc):
            data = client.post("/api/configure-voice-webhook", json=self._BASE).json()

        assert data["status"] == "error"
        assert "400" in data["message"]
