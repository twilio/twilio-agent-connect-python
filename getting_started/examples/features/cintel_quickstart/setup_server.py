"""
Conversation Intelligence End-to-End Setup Server

One-stop web UI to create Memory Store, Conversation Orchestrator config,
Intelligence Configuration (Script Adherence + Summary), and link them.

Usage:
    uv run getting_started/examples/features/cintel_quickstart/setup_server.py

Then open http://localhost:8081 in your browser.
"""

import base64
import logging
import os
import re

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Conversation Intelligence Setup")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

MEMORY_API_BASE = "https://memory.twilio.com/v1/ControlPlane"
CONVERSATION_API_BASE = "https://conversations.twilio.com/v2/ControlPlane"
INTELLIGENCE_BASE = "https://intelligence.twilio.com/v3"

SCRIPT_ADHERENCE_OPERATOR_ID = "intelligence_operator_01kf34tcyefpyb1t4m0nbd8rxg"
SUMMARY_OPERATOR_ID = "intelligence_operator_01kcv35pnkeysaf6z6cqtbpegn"

OWL_INTERNET_SCRIPT = """Category: greeting
- introduce_yourself: Agent identifies by first name and company name.
  Example: "Hi, this is [agent name] calling from Owl Internet."

Category: identity_verification
- verify_customer: Agent asks for account number.
  Example: "Can I get your account number to pull up your information?"

Category: resolution_steps
- acknowledge_issue: Agent actively troubleshoots the issue.
  Example: "Let me check that for you."
- offer_resolution: Agent mentions triggering a remote modem reset.
- confirms_resolution: Agent reacts positively to customer's confirmation.
  Example: "Great! I'm glad that worked."

Category: brand_approved_closing
- thank_customer: Agent thanks customer for choosing Owl Internet.
- offer_more_help: Agent explicitly asks if there's anything else they can help with.
  Example: "Is there anything else I can help you with today?\""""


def _basic_auth(api_key: str, api_secret: str) -> str:
    return "Basic " + base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()


def _auth_headers(api_key: str, api_secret: str) -> dict[str, str]:
    return {
        "Authorization": _basic_auth(api_key, api_secret),
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = os.path.join(os.path.dirname(__file__), "templates", "setup.html")
    with open(html_path) as f:
        return HTMLResponse(content=f.read())


# ---------------------------------------------------------------------------
# Prefill from .env
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Memory Store
# ---------------------------------------------------------------------------


@app.post("/api/list-memory-stores")
async def list_memory_stores(request: Request) -> dict:
    data = await request.json()
    api_key = data.get("api_key", "").strip()
    api_secret = data.get("api_secret", "").strip()

    if not api_key or not api_secret:
        return {"status": "error", "message": "api_key and api_secret are required"}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{MEMORY_API_BASE}/Stores",
                headers=_auth_headers(api_key, api_secret),
            )

            if resp.status_code != 200:
                return {
                    "status": "error",
                    "message": f"Failed to list Memory Stores: {resp.status_code} - {resp.text}",
                }

            store_ids = resp.json().get("stores", [])
            stores = []
            for store_id in store_ids[:100]:
                try:
                    detail = await client.get(
                        f"{MEMORY_API_BASE}/Stores/{store_id}",
                        headers=_auth_headers(api_key, api_secret),
                    )
                    if detail.status_code == 200:
                        d = detail.json()
                        stores.append(
                            {
                                "id": d.get("id"),
                                "displayName": d.get("displayName"),
                                "description": d.get("description"),
                                "status": d.get("status"),
                            }
                        )
                except Exception:
                    stores.append({"id": store_id, "displayName": store_id, "status": "UNKNOWN"})

        return {"status": "success", "stores": stores}

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception("Error listing Memory Stores")
        return {"status": "error", "message": str(e)}


@app.post("/api/get-memory-store")
async def get_memory_store(request: Request) -> dict:
    data = await request.json()
    memory_store_id = data.get("memory_store_id", "").strip()
    api_key = data.get("api_key", "").strip()
    api_secret = data.get("api_secret", "").strip()

    if not all([memory_store_id, api_key, api_secret]):
        return {
            "status": "error",
            "message": "memory_store_id, api_key, and api_secret are required",
        }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{MEMORY_API_BASE}/Stores/{memory_store_id}",
                headers=_auth_headers(api_key, api_secret),
            )

        if resp.status_code == 200:
            d = resp.json()
            return {
                "status": "success",
                "memory_store": {
                    "id": d.get("id"),
                    "displayName": d.get("displayName"),
                    "description": d.get("description"),
                    "status": d.get("status"),
                },
            }

        return {
            "status": "error",
            "message": f"Failed to get Memory Store: {resp.status_code} - {resp.text}",
        }

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception("Error getting Memory Store")
        return {"status": "error", "message": str(e)}


@app.post("/api/create-memory-store")
async def create_memory_store(request: Request) -> dict:
    data = await request.json()
    api_key = data.get("api_key", "").strip()
    api_secret = data.get("api_secret", "").strip()
    display_name = data.get("memory_display_name", "").strip()
    description = data.get("memory_description", "").strip()

    if not all([api_key, api_secret, display_name, description]):
        return {
            "status": "error",
            "message": "api_key, api_secret, memory_display_name, and memory_description are required",
        }

    if len(description) > 128:
        return {"status": "error", "message": "Memory description must not exceed 128 characters"}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{MEMORY_API_BASE}/Stores",
                headers=_auth_headers(api_key, api_secret),
                json={"displayName": display_name, "description": description},
            )

        if resp.status_code in (200, 201):
            result = resp.json()
            return {
                "status": "success",
                "memory_store_id": result.get("id"),
                "memory_store_name": result.get("displayName", display_name),
            }
        if resp.status_code == 202:
            result = resp.json()
            return {"status": "accepted", "status_url": result.get("statusUrl")}

        return {
            "status": "error",
            "message": f"Failed to create Memory Store: {resp.status_code} - {resp.text}",
        }

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception("Error creating Memory Store")
        return {"status": "error", "message": str(e)}


@app.post("/api/delete-memory-store")
async def delete_memory_store(request: Request) -> dict:
    data = await request.json()
    memory_store_id = data.get("memory_store_id", "").strip()
    api_key = data.get("api_key", "").strip()
    api_secret = data.get("api_secret", "").strip()

    if not all([memory_store_id, api_key, api_secret]):
        return {
            "status": "error",
            "message": "memory_store_id, api_key, and api_secret are required",
        }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.delete(
                f"{MEMORY_API_BASE}/Stores/{memory_store_id}",
                headers=_auth_headers(api_key, api_secret),
            )

        if resp.status_code in (200, 204):
            return {"status": "success", "message": "Memory Store deleted successfully"}
        if resp.status_code == 202:
            result = resp.json()
            return {"status": "accepted", "status_url": result.get("statusUrl")}

        return {
            "status": "error",
            "message": f"Failed to delete Memory Store: {resp.status_code} - {resp.text}",
        }

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception("Error deleting Memory Store")
        return {"status": "error", "message": str(e)}


@app.post("/api/verify-memory-store")
async def verify_memory_store(request: Request) -> dict:
    data = await request.json()
    memory_store_id = data.get("memory_store_id", "").strip()
    api_key = data.get("api_key", "").strip()
    api_secret = data.get("api_secret", "").strip()

    if not all([memory_store_id, api_key, api_secret]):
        return {
            "status": "error",
            "message": "memory_store_id, api_key, and api_secret are required",
        }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{MEMORY_API_BASE}/Stores/{memory_store_id}",
                headers=_auth_headers(api_key, api_secret),
            )

        if resp.status_code != 200:
            return {"status": "error", "message": f"Failed: {resp.status_code} - {resp.text}"}

        store_status = resp.json().get("status", "").upper()
        if store_status == "ACTIVE":
            return {"status": "success", "store_status": store_status}
        return {"status": "pending", "store_status": store_status}

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception("Error verifying Memory Store")
        return {"status": "error", "message": str(e)}


@app.post("/api/poll-operation-status")
async def poll_operation_status(request: Request) -> dict:
    data = await request.json()
    status_url = (data.get("status_url") or "").strip()
    api_key = (data.get("api_key") or "").strip()
    api_secret = (data.get("api_secret") or "").strip()

    if not all([status_url, api_key, api_secret]):
        return {"status": "error", "message": "status_url, api_key, and api_secret are required"}

    from posixpath import normpath
    from urllib.parse import urlparse

    try:
        parsed = urlparse(status_url)
        if parsed.scheme != "https":
            return {"status": "error", "message": "status_url must use https"}
        allowed = ["memory.twilio.com", "conversations.twilio.com"]
        if parsed.hostname not in allowed:
            return {"status": "error", "message": f"status_url host must be one of {allowed}"}
        if parsed.port not in (None, 443):
            return {"status": "error", "message": "status_url port must be 443 or omitted"}
        if ".." in status_url or "%2e" in status_url.lower():
            return {"status": "error", "message": "Path traversal detected in status_url"}
        norm = normpath(parsed.path)
        valid = ["/v1/ControlPlane/Operations/", "/v2/ControlPlane/Operations/"]
        if not any(norm.startswith(p) for p in valid):
            return {
                "status": "error",
                "message": "status_url must be a ControlPlane Operations URL",
            }
    except Exception as e:
        return {"status": "error", "message": f"Invalid status_url: {e}"}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                status_url, headers={"Authorization": _basic_auth(api_key, api_secret)}
            )

        if resp.status_code != 200:
            return {"status": "error", "message": f"Poll failed: {resp.status_code} - {resp.text}"}

        result = resp.json()
        op_status = result.get("status", "").upper()
        if op_status == "COMPLETED":
            return {
                "status": "completed",
                "result": result.get("result", {}),
                "related": result.get("related", {}),
            }
        if op_status in ("PENDING", "IN_PROGRESS", "QUEUED"):
            return {"status": "pending", "operation_status": op_status}
        if op_status == "FAILED":
            return {"status": "error", "message": f"Operation failed: {result.get('error')}"}
        return {"status": "pending", "operation_status": op_status}

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception("Error polling operation status")
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Conversation Orchestrator
# ---------------------------------------------------------------------------


@app.post("/api/detect-ngrok")
async def detect_ngrok() -> dict:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:4040/api/tunnels", timeout=2.0)
            if response.status_code == 200:
                tunnels = response.json().get("tunnels", [])
                https_tunnel = next((t for t in tunnels if t["proto"] == "https"), None)
                if https_tunnel:
                    domain = https_tunnel["public_url"].replace("https://", "")
                    return {"status": "success", "domain": domain}
    except Exception as e:
        logger.debug(f"ngrok detection failed: {e}")

    return {"status": "error", "message": "ngrok not detected. Start with: ngrok http 3340"}


@app.post("/api/list-conversation-configurations")
async def list_conversation_configurations(request: Request) -> dict:
    data = await request.json()
    api_key = data.get("api_key", "").strip()
    api_secret = data.get("api_secret", "").strip()

    if not api_key or not api_secret:
        return {"status": "error", "message": "api_key and api_secret are required"}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{CONVERSATION_API_BASE}/Configurations",
                headers=_auth_headers(api_key, api_secret),
            )

        if resp.status_code == 200:
            configs = resp.json().get("configurations", [])
            return {
                "status": "success",
                "configurations": [
                    {
                        "id": c.get("id"),
                        "displayName": c.get("displayName"),
                        "description": c.get("description"),
                        "createdAt": c.get("createdAt"),
                        "memoryStoreId": c.get("memoryStoreId"),
                    }
                    for c in configs
                ],
            }

        return {
            "status": "error",
            "message": f"Failed to list configurations: {resp.status_code} - {resp.text}",
        }

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception("Error listing configurations")
        return {"status": "error", "message": str(e)}


@app.post("/api/create-conversation-configuration")
async def create_conversation_configuration(request: Request) -> dict:
    data = await request.json()
    api_key = data.get("api_key", "").strip()
    api_secret = data.get("api_secret", "").strip()
    memory_store_id = data.get("memory_store_id", "").strip()
    twilio_phone = data.get("twilio_phone", "").strip()
    ngrok_domain = data.get("ngrok_domain", "").strip()
    display_name = data.get("configuration_display_name", "").strip()
    description = data.get("configuration_description", "").strip()

    if not all([api_key, api_secret, memory_store_id, twilio_phone, ngrok_domain, display_name]):
        return {
            "status": "error",
            "message": (
                "api_key, api_secret, memory_store_id, twilio_phone, "
                "ngrok_domain, and configuration_display_name are required"
            ),
        }

    if len(display_name) > 32:
        return {"status": "error", "message": "Display name must not exceed 32 characters"}

    if not re.match(r"^[A-Za-z0-9._~-]+$", display_name):
        return {
            "status": "error",
            "message": "Display name must be URL-safe: letters, numbers, . _ ~ - only",
        }

    if description and len(description) > 128:
        return {"status": "error", "message": "Description must not exceed 128 characters"}

    webhook_url = f"https://{ngrok_domain}/webhook"
    payload: dict = {
        "displayName": display_name,
        "conversationGroupingType": "GROUP_BY_PARTICIPANT_ADDRESSES_AND_CHANNEL_TYPE",
        "memoryStoreId": memory_store_id,
        "channelSettings": {
            "SMS": {
                "statusTimeouts": {"inactive": 2, "closed": 3},
                "captureRules": [
                    {"from": "*", "to": twilio_phone},
                    {"from": twilio_phone, "to": "*"},
                ],
            },
            "VOICE": {
                "statusTimeouts": {"inactive": 5, "closed": 30},
                "captureRules": [
                    {"from": "*", "to": twilio_phone, "metadata": {"callType": "PSTN"}}
                ],
            },
        },
        "statusCallbacks": [{"url": webhook_url, "method": "POST"}],
    }
    if description:
        payload["description"] = description

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{CONVERSATION_API_BASE}/Configurations",
                headers=_auth_headers(api_key, api_secret),
                json=payload,
            )

        if resp.status_code in (200, 201):
            result = resp.json()
            return {
                "status": "success",
                "conversation_configuration_id": result.get("id"),
            }
        if resp.status_code == 202:
            result = resp.json()
            return {"status": "accepted", "status_url": result.get("statusUrl")}

        return {
            "status": "error",
            "message": f"Failed to create Conversation Configuration: {resp.status_code} - {resp.text}",
            "payload": payload,
            "response": resp.text,
            "status_code": resp.status_code,
        }

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception("Error creating Conversation Configuration")
        return {"status": "error", "message": str(e)}


@app.post("/api/delete-conversation-configuration")
async def delete_conversation_configuration(request: Request) -> dict:
    data = await request.json()
    api_key = data.get("api_key", "").strip()
    api_secret = data.get("api_secret", "").strip()
    configuration_id = data.get("configuration_id", "").strip()

    if not all([api_key, api_secret, configuration_id]):
        return {
            "status": "error",
            "message": "api_key, api_secret, and configuration_id are required",
        }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.delete(
                f"{CONVERSATION_API_BASE}/Configurations/{configuration_id}",
                headers=_auth_headers(api_key, api_secret),
            )

        if resp.status_code in (200, 204):
            return {"status": "success", "message": f"Configuration {configuration_id} deleted"}
        if resp.status_code == 202:
            result = resp.json()
            return {"status": "accepted", "status_url": result.get("statusUrl")}

        return {
            "status": "error",
            "message": f"Failed to delete configuration: {resp.status_code} - {resp.text}",
        }

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception("Error deleting configuration")
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Intelligence Configuration
# ---------------------------------------------------------------------------


@app.post("/api/create-intelligence-configuration")
async def create_intelligence_configuration(request: Request) -> dict:
    data = await request.json()
    api_key = data.get("api_key", "").strip()
    api_secret = data.get("api_secret", "").strip()
    ngrok_domain = data.get("ngrok_domain", "").strip()

    if not all([api_key, api_secret, ngrok_domain]):
        return {"status": "error", "message": "api_key, api_secret, and ngrok_domain are required"}

    webhook_url = f"https://{ngrok_domain}/ci-webhook"
    ci_payload = {
        "displayName": "Owl Internet Support",
        "description": "Script adherence + post-call summary for voice support",
        "rules": [
            {
                "operators": [
                    {
                        "id": SCRIPT_ADHERENCE_OPERATOR_ID,
                        "parameters": {"script": OWL_INTERNET_SCRIPT},
                    }
                ],
                "triggers": [{"on": "COMMUNICATION"}],
                "actions": [{"type": "WEBHOOK", "method": "POST", "url": webhook_url}],
            },
            {
                "operators": [{"id": SUMMARY_OPERATOR_ID, "parameters": {}}],
                "triggers": [{"on": "CONVERSATION_END"}],
                "actions": [{"type": "WEBHOOK", "method": "POST", "url": webhook_url}],
            },
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{INTELLIGENCE_BASE}/ControlPlane/Configurations",
                headers=_auth_headers(api_key, api_secret),
                json=ci_payload,
            )

        if resp.status_code in (200, 201):
            result = resp.json()
            return {
                "status": "success",
                "intelligence_configuration_id": result.get("id"),
            }

        return {
            "status": "error",
            "message": f"Failed to create Intelligence Configuration: {resp.status_code} - {resp.text}",
        }

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception("Error creating Intelligence Configuration")
        return {"status": "error", "message": str(e)}


@app.post("/api/link-intelligence-configuration")
async def link_intelligence_configuration(request: Request) -> dict:
    data = await request.json()
    api_key = data.get("api_key", "").strip()
    api_secret = data.get("api_secret", "").strip()
    conv_config_id = data.get("conversation_configuration_id", "").strip()
    intel_config_id = data.get("intelligence_configuration_id", "").strip()

    if not all([api_key, api_secret, conv_config_id, intel_config_id]):
        return {
            "status": "error",
            "message": (
                "api_key, api_secret, conversation_configuration_id, "
                "and intelligence_configuration_id are required"
            ),
        }

    headers = _auth_headers(api_key, api_secret)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            get_resp = await client.get(
                f"{CONVERSATION_API_BASE}/Configurations/{conv_config_id}",
                headers=headers,
            )

            if get_resp.status_code != 200:
                return {
                    "status": "error",
                    "message": f"Could not fetch CO Configuration ({get_resp.status_code}): {get_resp.text}",
                }

            existing = get_resp.json()
            existing_ids: list[str] = existing.get("intelligenceConfigurationIds") or []

            if intel_config_id in existing_ids:
                return {"status": "success", "already_linked": True}

            put_payload: dict = {
                "displayName": existing.get("displayName", ""),
                "conversationGroupingType": existing.get("conversationGroupingType"),
                "memoryStoreId": existing.get("memoryStoreId"),
                "channelSettings": existing.get("channelSettings"),
                "intelligenceConfigurationIds": existing_ids + [intel_config_id],
            }
            if existing.get("description"):
                put_payload["description"] = existing["description"]
            if existing.get("statusCallbacks"):
                put_payload["statusCallbacks"] = existing["statusCallbacks"]
            put_resp = await client.put(
                f"{CONVERSATION_API_BASE}/Configurations/{conv_config_id}",
                headers=headers,
                json=put_payload,
            )

        if put_resp.status_code in (200, 201):
            return {"status": "success", "already_linked": False}

        if put_resp.status_code == 202:
            result = put_resp.json()
            return {
                "status": "accepted",
                "status_url": result.get("statusUrl"),
                "already_linked": False,
            }

        return {
            "status": "error",
            "message": f"Failed to link: {put_resp.status_code} - {put_resp.text}",
        }

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception("Error linking Intelligence Configuration")
        return {"status": "error", "message": str(e)}


@app.post("/api/configure-voice-webhook")
async def configure_voice_webhook(request: Request) -> dict:
    data = await request.json()
    account_sid = data.get("account_sid", "").strip()
    api_key = data.get("api_key", "").strip()
    api_secret = data.get("api_secret", "").strip()
    twilio_phone = data.get("twilio_phone", "").strip()
    ngrok_domain = data.get("ngrok_domain", "").strip()

    if not all([account_sid, api_key, api_secret, twilio_phone, ngrok_domain]):
        return {
            "status": "error",
            "message": "account_sid, api_key, api_secret, twilio_phone, and ngrok_domain are required",
        }

    voice_url = f"https://{ngrok_domain}/tac/twiml"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Look up the phone number SID
            resp = await client.get(
                f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/IncomingPhoneNumbers.json",
                params={"PhoneNumber": twilio_phone},
                auth=(api_key, api_secret),
            )

            if resp.status_code != 200:
                return {
                    "status": "error",
                    "message": f"Failed to look up phone number: {resp.status_code} - {resp.text}",
                }

            numbers = resp.json().get("incoming_phone_numbers", [])
            if not numbers:
                return {
                    "status": "error",
                    "message": f"Phone number {twilio_phone} not found in this account",
                }

            number_sid = numbers[0]["sid"]

            # Update the voice webhook
            update_resp = await client.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/IncomingPhoneNumbers/{number_sid}.json",
                data={"VoiceUrl": voice_url, "VoiceMethod": "POST"},
                auth=(api_key, api_secret),
            )

            if update_resp.status_code in (200, 201):
                return {"status": "success", "voice_url": voice_url}

            return {
                "status": "error",
                "message": f"Failed to update webhook: {update_resp.status_code} - {update_resp.text}",
            }

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception("Error configuring voice webhook")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn

    print("Starting Conversation Intelligence Setup Server...")
    print("Open http://localhost:8081 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8081)
