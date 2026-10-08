"""
TAC Quickstart Setup Server

A simple web UI to help users set up Conversation Memory and Conversation Orchestrator services
required for Twilio Agent Connect.
"""

import base64
import json
import logging
import os
import re
import sys

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="TAC Quickstart Setup")

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# API endpoints
MEMORY_API_BASE = "https://memory.twilio.com/v1/ControlPlane"


def get_basic_auth_header(api_key: str, api_secret: str) -> str:
    """Generate Basic Auth header from API key and secret."""
    credentials = f"{api_key}:{api_secret}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return f"Basic {encoded}"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Serve the setup page."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path) as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/api/create-memory-store")
async def create_memory_store(request: Request) -> dict:
    """
    Create a Memory Store in Conversation Memory.

    Expected payload:
    {
        "api_key": "SK...",
        "api_secret": "...",
        "memory_display_name": "..." (required),
        "memory_description": "..." (required)
    }
    """
    data = await request.json()

    api_key = data.get("api_key")
    api_secret = data.get("api_secret")
    memory_display_name = data.get("memory_display_name")
    memory_description = data.get("memory_description")

    if not all([api_key, api_secret, memory_display_name, memory_description]):
        return {
            "status": "error",
            "message": (
                "Missing required fields: api_key, api_secret, "
                "memory_display_name, memory_description"
            ),
        }

    # Validate memory_description length (must not exceed 128 characters)
    if len(memory_description) > 128:
        return {
            "status": "error",
            "message": "Memory description must not exceed 128 characters",
        }

    # Build payload
    payload = {
        "displayName": memory_display_name,
    }

    # Add optional description if provided
    if memory_description:
        payload["description"] = memory_description

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MEMORY_API_BASE}/Stores",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": get_basic_auth_header(api_key, api_secret),
                },
                json=payload,
                timeout=30.0,
            )

            if response.status_code in [200, 201]:
                result = response.json()
                return {
                    "status": "success",
                    "memory_store_id": result.get("id"),
                    "memory_store_name": result.get("displayName", memory_display_name),
                    "memory_store_status": result.get("status"),
                    "intelligence_service_id": result.get("intelligenceServiceId"),
                    "message": f"Memory Store created: {result.get('id')}",
                }
            elif response.status_code == 202:
                # Async operation - return status URL for polling
                result = response.json()
                logger.info("Memory Store creation accepted (async)")
                logger.info(f"  Status URL: {result.get('statusUrl')}")
                return {
                    "status": "accepted",
                    "status_url": result.get("statusUrl"),
                    "message": result.get(
                        "message", "Memory Store creation accepted for processing"
                    ),
                }
            else:
                error_text = response.text
                logger.error("Failed to create Memory Store")
                logger.error(f"  Endpoint: {MEMORY_API_BASE}/Stores")
                logger.error(f"  Payload: {json.dumps(payload, indent=2)}")
                logger.error(f"  Status: {response.status_code}")
                logger.error(f"  Response: {error_text}")
                return {
                    "status": "error",
                    "message": (
                        f"Failed to create Memory Store: {response.status_code} - {error_text}"
                    ),
                    "payload": payload,
                    "response": error_text,
                    "status_code": response.status_code,
                }

    except httpx.TimeoutException:
        logger.error("Timeout creating Memory Store")
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception(f"Error creating Memory Store: {str(e)}")
        return {"status": "error", "message": f"Error creating Memory Store: {str(e)}"}


@app.post("/api/poll-operation-status")
async def poll_operation_status(request: Request) -> dict:
    """
    Poll an operation status URL to check if async operation is complete.

    Expected payload:
    {
        "status_url": "https://memory.twilio.com/v1/ControlPlane/Operations/...",
        "api_key": "SK...",
        "api_secret": "..."
    }
    """
    data = await request.json()

    raw_status_url = data.get("status_url")
    raw_api_key = data.get("api_key")
    raw_api_secret = data.get("api_secret")

    # Validate as non-empty strings and normalize
    if not all(
        isinstance(value, str) and value.strip()
        for value in (raw_status_url, raw_api_key, raw_api_secret)
    ):
        return {
            "status": "error",
            "message": "Missing required fields: status_url, api_key, api_secret",
        }

    status_url = raw_status_url.strip()
    api_key = raw_api_key.strip()
    api_secret = raw_api_secret.strip()

    # Validate status_url to prevent SSRF and credential exfiltration
    from urllib.parse import urlparse

    try:
        parsed_url = urlparse(status_url)

        # Must be HTTPS
        if parsed_url.scheme != "https":
            return {
                "status": "error",
                "message": f"Invalid status_url scheme: {parsed_url.scheme}. Must be https.",
            }

        # Must be a Twilio domain
        allowed_hosts = ["memory.twilio.com", "conversations.twilio.com"]
        hostname = parsed_url.hostname
        if hostname not in allowed_hosts:
            return {
                "status": "error",
                "message": (
                    f"Invalid status_url host: {hostname}. Must be one of {allowed_hosts}."
                ),
            }

        # Only allow the default HTTPS port (443 or omitted)
        if parsed_url.port not in (None, 443):
            return {
                "status": "error",
                "message": f"Invalid status_url port: {parsed_url.port}. Must be 443 or omitted.",
            }

        # Must be an Operations endpoint - validate with normalized path
        # to prevent path traversal (e.g., /Operations/../Stores)
        from posixpath import normpath

        normalized_path = normpath(parsed_url.path)

        # Check for path traversal attempts
        if ".." in parsed_url.path or "%2e" in parsed_url.path.lower():
            return {
                "status": "error",
                "message": "Invalid status_url path: Path traversal detected.",
            }

        # Must match expected Operations endpoint patterns
        valid_patterns = [
            "/v1/ControlPlane/Operations/",  # Memory API
            "/v2/ControlPlane/Operations/",  # Conversation API
        ]

        if not any(normalized_path.startswith(pattern) for pattern in valid_patterns):
            return {
                "status": "error",
                "message": (
                    "Invalid status_url path: Must be a ControlPlane Operations endpoint "
                    "(e.g., /v1/ControlPlane/Operations/... or /v2/ControlPlane/Operations/...)"
                ),
            }

    except Exception as e:
        return {"status": "error", "message": f"Invalid status_url format: {str(e)}"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                status_url,
                headers={"Authorization": get_basic_auth_header(api_key, api_secret)},
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                operation_status = result.get("status", "").upper()

                if operation_status == "COMPLETED":
                    # Extract the created resource ID from result
                    return {
                        "status": "completed",
                        "operation_status": operation_status,
                        "result": result.get("result", {}),
                        "message": "Operation completed successfully",
                    }
                elif operation_status in ["PENDING", "IN_PROGRESS", "QUEUED"]:
                    return {
                        "status": "pending",
                        "operation_status": operation_status,
                        "message": f"Operation status: {operation_status}",
                    }
                elif operation_status == "FAILED":
                    return {
                        "status": "error",
                        "operation_status": operation_status,
                        "message": f"Operation failed: {result.get('error', 'Unknown error')}",
                    }
                else:
                    return {
                        "status": "pending",
                        "operation_status": operation_status,
                        "message": f"Operation status: {operation_status}",
                    }
            else:
                logger.error("Failed to poll operation status")
                logger.error(f"  Endpoint: {status_url}")
                logger.error(f"  Status: {response.status_code}")
                logger.error(f"  Response: {response.text}")
                return {
                    "status": "error",
                    "message": (
                        f"Failed to poll operation status: {response.status_code} - {response.text}"
                    ),
                    "status_code": response.status_code,
                }

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception(f"Error polling operation status: {str(e)}")
        return {"status": "error", "message": f"Error polling operation status: {str(e)}"}


@app.post("/api/get-memory-store")
async def get_memory_store(request: Request) -> dict:
    """
    Get Memory Store details by ID.

    Expected payload:
    {
        "memory_store_id": "mem_store_...",
        "api_key": "SK...",
        "api_secret": "..."
    }
    """
    data = await request.json()

    memory_store_id = data.get("memory_store_id")
    api_key = data.get("api_key")
    api_secret = data.get("api_secret")

    if not all([memory_store_id, api_key, api_secret]):
        return {
            "status": "error",
            "message": "Missing required fields: memory_store_id, api_key, api_secret",
        }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MEMORY_API_BASE}/Stores/{memory_store_id}",
                headers={"Authorization": get_basic_auth_header(api_key, api_secret)},
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "memory_store": {
                        "id": result.get("id"),
                        "displayName": result.get("displayName"),
                        "description": result.get("description"),
                        "status": result.get("status"),
                    },
                }
            else:
                endpoint = f"{MEMORY_API_BASE}/Stores/{memory_store_id}"
                logger.error("Failed to get Memory Store")
                logger.error(f"  Endpoint: {endpoint}")
                logger.error(f"  Status: {response.status_code}")
                logger.error(f"  Response: {response.text}")
                return {
                    "status": "error",
                    "message": (
                        f"Failed to get Memory Store: {response.status_code} - {response.text}"
                    ),
                    "endpoint": endpoint,
                    "response": response.text,
                    "status_code": response.status_code,
                }

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception(f"Error getting Memory Store: {str(e)}")
        return {"status": "error", "message": f"Error getting Memory Store: {str(e)}"}


@app.post("/api/list-memory-stores")
async def list_memory_stores(request: Request) -> dict:
    """
    List all Memory Stores for the account.

    Expected payload:
    {
        "api_key": "SK...",
        "api_secret": "..."
    }
    """
    data = await request.json()

    api_key = data.get("api_key")
    api_secret = data.get("api_secret")

    if not all([api_key, api_secret]):
        return {
            "status": "error",
            "message": "Missing required fields: api_key, api_secret",
        }

    try:
        async with httpx.AsyncClient() as client:
            # First, get the list of store IDs
            response = await client.get(
                f"{MEMORY_API_BASE}/Stores",
                headers={"Authorization": get_basic_auth_header(api_key, api_secret)},
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                store_ids = result.get("stores", [])

                # Fetch details for each store (in parallel for efficiency)
                stores = []
                # Limit to first 100 stores to avoid overwhelming the UI
                for store_id in store_ids[:100]:
                    try:
                        detail_response = await client.get(
                            f"{MEMORY_API_BASE}/Stores/{store_id}",
                            headers={"Authorization": get_basic_auth_header(api_key, api_secret)},
                            timeout=10.0,
                        )
                        if detail_response.status_code == 200:
                            store_data = detail_response.json()
                            stores.append(
                                {
                                    "id": store_data.get("id"),
                                    "displayName": store_data.get("displayName"),
                                    "description": store_data.get("description"),
                                    "status": store_data.get("status"),
                                }
                            )
                    except Exception as e:
                        logger.warning(f"Failed to fetch details for store {store_id}: {e}")
                        # Add store with ID only if we can't fetch details
                        stores.append(
                            {
                                "id": store_id,
                                "displayName": store_id,
                                "description": None,
                                "status": "UNKNOWN",
                            }
                        )

                return {
                    "status": "success",
                    "stores": stores,
                }
            else:
                endpoint = f"{MEMORY_API_BASE}/Stores"
                logger.error("Failed to list Memory Stores")
                logger.error(f"  Endpoint: {endpoint}")
                logger.error(f"  Status: {response.status_code}")
                logger.error(f"  Response: {response.text}")
                return {
                    "status": "error",
                    "message": (
                        f"Failed to list Memory Stores: {response.status_code} - {response.text}"
                    ),
                    "endpoint": endpoint,
                    "response": response.text,
                    "status_code": response.status_code,
                }

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception(f"Error listing Memory Stores: {str(e)}")
        return {"status": "error", "message": f"Error listing Memory Stores: {str(e)}"}


@app.post("/api/delete-memory-store")
async def delete_memory_store(request: Request) -> dict:
    """
    Delete a Memory Store.

    Expected payload:
    {
        "memory_store_id": "mem_store_...",
        "api_key": "SK...",
        "api_secret": "..."
    }

    Returns:
    - 202 with status_url for async deletion (poll with /api/poll-operation-status)
    - 200/204 for immediate deletion
    """
    data = await request.json()

    memory_store_id = data.get("memory_store_id")
    api_key = data.get("api_key")
    api_secret = data.get("api_secret")

    if not all([memory_store_id, api_key, api_secret]):
        return {
            "status": "error",
            "message": "Missing required fields: memory_store_id, api_key, api_secret",
        }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{MEMORY_API_BASE}/Stores/{memory_store_id}",
                headers={"Authorization": get_basic_auth_header(api_key, api_secret)},
                timeout=30.0,
            )

            if response.status_code in (200, 204):
                # Immediate deletion
                return {
                    "status": "success",
                    "message": "Memory Store deleted successfully",
                }
            elif response.status_code == 202:
                # Async deletion - return status URL for polling
                result = response.json()
                status_url = result.get("statusUrl")

                if not status_url:
                    logger.error("202 response missing statusUrl")
                    logger.error(f"  Response: {response.text}")
                    return {
                        "status": "error",
                        "message": "Deletion accepted but no status URL returned",
                        "response": response.text,
                    }

                return {
                    "status": "accepted",
                    "message": "Memory Store deletion request accepted",
                    "status_url": status_url,
                }
            else:
                endpoint = f"{MEMORY_API_BASE}/Stores/{memory_store_id}"
                logger.error("Failed to delete Memory Store")
                logger.error(f"  Endpoint: {endpoint}")
                logger.error(f"  Status: {response.status_code}")
                logger.error(f"  Response: {response.text}")
                return {
                    "status": "error",
                    "message": (
                        f"Failed to delete Memory Store: {response.status_code} - {response.text}"
                    ),
                    "endpoint": endpoint,
                    "response": response.text,
                    "status_code": response.status_code,
                }

    except httpx.TimeoutException:
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception(f"Error deleting Memory Store: {str(e)}")
        return {"status": "error", "message": f"Error deleting Memory Store: {str(e)}"}


@app.post("/api/verify-memory-store")
async def verify_memory_store(request: Request) -> dict:
    """
    Verify that a Memory Store is active.

    Expected payload:
    {
        "memory_store_id": "mem_store_...",
        "api_key": "SK...",
        "api_secret": "..."
    }
    """
    data = await request.json()

    memory_store_id = data.get("memory_store_id")
    api_key = data.get("api_key")
    api_secret = data.get("api_secret")

    if not all([memory_store_id, api_key, api_secret]):
        return {
            "status": "error",
            "message": "Missing required fields: memory_store_id, api_key, api_secret",
        }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MEMORY_API_BASE}/Stores/{memory_store_id}",
                headers={"Authorization": get_basic_auth_header(api_key, api_secret)},
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                store_status = result.get("status", "").upper()

                if store_status == "ACTIVE":
                    return {
                        "status": "success",
                        "store_status": store_status,
                        "message": "Memory Store is active",
                    }
                else:
                    return {
                        "status": "pending",
                        "store_status": store_status,
                        "message": f"Memory Store status: {store_status}",
                    }
            else:
                endpoint = f"{MEMORY_API_BASE}/Stores/{memory_store_id}"
                logger.error("Failed to verify Memory Store")
                logger.error(f"  Endpoint: {endpoint}")
                logger.error(f"  Status: {response.status_code}")
                logger.error(f"  Response: {response.text}")
                return {
                    "status": "error",
                    "message": (
                        f"Failed to verify Memory Store: {response.status_code} - {response.text}"
                    ),
                    "endpoint": endpoint,
                    "response": response.text,
                    "status_code": response.status_code,
                }

    except httpx.TimeoutException:
        logger.error("Timeout verifying Memory Store")
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception(f"Error verifying Memory Store: {str(e)}")
        return {"status": "error", "message": f"Error verifying Memory Store: {str(e)}"}


@app.post("/api/create-profile")
async def create_profile(request: Request) -> dict:
    """
    Create a Profile in the Memory Store.

    Expected payload:
    {
        "memory_store_id": "mem_store_...",
        "api_key": "SK...",
        "api_secret": "...",
        "email": "user@example.com",
        "phone": "+1234567890",
        "first_name": "John" (optional)
    }
    """
    data = await request.json()

    memory_store_id = data.get("memory_store_id")
    api_key = data.get("api_key")
    api_secret = data.get("api_secret")
    email = data.get("email")
    phone = data.get("phone")
    first_name = data.get("first_name")

    if not all([memory_store_id, api_key, api_secret, email, phone]):
        return {
            "status": "error",
            "message": (
                "Missing required fields: memory_store_id, api_key, api_secret, email, phone"
            ),
        }

    # Build traits object
    contact_traits: dict = {
        "email": email,
        "phone": phone,
    }
    if first_name:
        contact_traits["firstName"] = first_name

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://memory.twilio.com/v1/Stores/{memory_store_id}/Profiles",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": get_basic_auth_header(api_key, api_secret),
                },
                json={"traits": {"Contact": contact_traits}},
                timeout=30.0,
            )

            if response.status_code in [200, 201, 202]:
                result = response.json()
                return {
                    "status": "success",
                    "profile_id": result.get("id"),
                    "message": result.get("message", "Profile created successfully"),
                }
            else:
                endpoint = f"https://memory.twilio.com/v1/Stores/{memory_store_id}/Profiles"
                payload = {"traits": {"Contact": contact_traits}}
                logger.error("Failed to create Profile")
                logger.error(f"  Endpoint: {endpoint}")
                logger.error(f"  Payload: {json.dumps(payload, indent=2)}")
                logger.error(f"  Status: {response.status_code}")
                logger.error(f"  Response: {response.text}")
                return {
                    "status": "error",
                    "message": (
                        f"Failed to create Profile: {response.status_code} - {response.text}"
                    ),
                    "endpoint": endpoint,
                    "payload": payload,
                    "response": response.text,
                    "status_code": response.status_code,
                }

    except httpx.TimeoutException:
        logger.error("Timeout creating Profile")
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception(f"Error creating Profile: {str(e)}")
        return {"status": "error", "message": f"Error creating Profile: {str(e)}"}


@app.post("/api/verify-profile")
async def verify_profile(request: Request) -> dict:
    """
    Verify that a Profile was created successfully with correct traits.

    Expected payload:
    {
        "memory_store_id": "mem_store_...",
        "profile_id": "mem_profile_...",
        "api_key": "SK...",
        "api_secret": "...",
        "email": "user@example.com",
        "phone": "+1234567890",
        "first_name": "John" (optional)
    }
    """
    data = await request.json()

    memory_store_id = data.get("memory_store_id")
    profile_id = data.get("profile_id")
    api_key = data.get("api_key")
    api_secret = data.get("api_secret")
    expected_email = data.get("email")
    expected_phone = data.get("phone")
    expected_first_name = data.get("first_name")

    if not all([memory_store_id, profile_id, api_key, api_secret]):
        return {"status": "error", "message": "Missing required fields"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://memory.twilio.com/v1/Stores/{memory_store_id}/Profiles/{profile_id}",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": get_basic_auth_header(api_key, api_secret),
                },
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                traits = result.get("traits", {})
                contact = traits.get("Contact", {})

                # Verify traits match
                actual_email = contact.get("email")
                actual_phone = contact.get("phone")
                actual_first_name = contact.get("firstName")

                mismatches = []
                if expected_email and actual_email != expected_email:
                    mismatches.append(f"email (expected: {expected_email}, got: {actual_email})")
                if expected_phone and actual_phone != expected_phone:
                    mismatches.append(f"phone (expected: {expected_phone}, got: {actual_phone})")
                if expected_first_name and actual_first_name != expected_first_name:
                    mismatches.append(
                        f"firstName (expected: {expected_first_name}, got: {actual_first_name})"
                    )

                if mismatches:
                    return {
                        "status": "error",
                        "message": f"Profile traits mismatch: {', '.join(mismatches)}",
                    }

                return {
                    "status": "success",
                    "message": "Profile verified successfully",
                    "traits": contact,
                }
            elif response.status_code == 404:
                return {
                    "status": "pending",
                    "message": "Profile not found yet, still processing...",
                }
            else:
                endpoint = (
                    f"https://memory.twilio.com/v1/Stores/{memory_store_id}/Profiles/{profile_id}"
                )
                logger.error("Failed to verify Profile")
                logger.error(f"  Endpoint: {endpoint}")
                logger.error(f"  Status: {response.status_code}")
                logger.error(f"  Response: {response.text}")
                return {
                    "status": "error",
                    "message": (
                        f"Failed to verify Profile: {response.status_code} - {response.text}"
                    ),
                    "endpoint": endpoint,
                    "response": response.text,
                    "status_code": response.status_code,
                }

    except httpx.TimeoutException:
        logger.error("Timeout verifying Profile")
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception(f"Error verifying Profile: {str(e)}")
        return {"status": "error", "message": f"Error verifying Profile: {str(e)}"}


@app.post("/api/lookup-profile")
async def lookup_profile(request: Request) -> dict:
    """
    Verify that a Profile can be looked up by phone number.

    Expected payload:
    {
        "memory_store_id": "mem_store_...",
        "profile_id": "mem_profile_...",
        "api_key": "SK...",
        "api_secret": "...",
        "phone": "+1234567890"
    }
    """
    data = await request.json()

    memory_store_id = data.get("memory_store_id")
    profile_id = data.get("profile_id")
    api_key = data.get("api_key")
    api_secret = data.get("api_secret")
    phone = data.get("phone")

    if not all([memory_store_id, profile_id, api_key, api_secret, phone]):
        return {"status": "error", "message": "Missing required fields"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://memory.twilio.com/v1/Stores/{memory_store_id}/Profiles/Lookup",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": get_basic_auth_header(api_key, api_secret),
                },
                json={"idType": "phone", "value": phone},
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                profiles = result.get("profiles", [])

                if profile_id in profiles:
                    return {
                        "status": "success",
                        "message": "Profile lookup verified successfully",
                        "normalized_value": result.get("normalizedValue"),
                        "profiles": profiles,
                    }
                else:
                    return {
                        "status": "pending",
                        "message": (
                            f"Profile not yet indexed for lookup. Found profiles: {profiles}"
                        ),
                    }
            elif response.status_code == 404:
                return {
                    "status": "pending",
                    "message": "No profiles found for this phone number yet",
                }
            else:
                endpoint = f"https://memory.twilio.com/v1/Stores/{memory_store_id}/Profiles/Lookup"
                payload = {"idType": "phone", "value": phone}
                logger.error("Failed to lookup Profile")
                logger.error(f"  Endpoint: {endpoint}")
                logger.error(f"  Payload: {json.dumps(payload, indent=2)}")
                logger.error(f"  Status: {response.status_code}")
                logger.error(f"  Response: {response.text}")
                return {
                    "status": "error",
                    "message": (
                        f"Failed to lookup Profile: {response.status_code} - {response.text}"
                    ),
                    "endpoint": endpoint,
                    "payload": payload,
                    "response": response.text,
                    "status_code": response.status_code,
                }

    except httpx.TimeoutException:
        logger.error("Timeout looking up Profile")
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception(f"Error looking up Profile: {str(e)}")
        return {"status": "error", "message": f"Error looking up Profile: {str(e)}"}


# Conversation Orchestrator API endpoint
CONVERSATION_API_BASE = "https://conversations.twilio.com/v2/ControlPlane"


@app.post("/api/create-conversation-configuration")
async def create_conversation_configuration(request: Request) -> dict:
    """
    Create a Conversation Orchestrator Configuration.

    Expected payload:
    {
        "api_key": "SK...",
        "api_secret": "...",
        "memory_store_id": "mem_store_...",
        "twilio_phone": "+18001234567",
        "ngrok_domain": "your-app.ngrok.app",
        "configuration_display_name": "my-config",
        "configuration_description": "Conversation configuration description"
    }

    Required fields: api_key, api_secret, memory_store_id, twilio_phone,
    ngrok_domain, configuration_display_name, configuration_description
    """
    data = await request.json()

    api_key = data.get("api_key")
    api_secret = data.get("api_secret")
    memory_store_id = data.get("memory_store_id")
    twilio_phone = data.get("twilio_phone")
    ngrok_domain = data.get("ngrok_domain")
    # Strip values once at the beginning for efficiency
    configuration_display_name = data.get("configuration_display_name", "").strip()
    configuration_description = data.get("configuration_description", "").strip()

    if not all(
        [
            api_key,
            api_secret,
            memory_store_id,
            twilio_phone,
            ngrok_domain,
            configuration_display_name,
        ]
    ):
        return {
            "status": "error",
            "message": (
                "Missing required fields: api_key, api_secret, memory_store_id, "
                "twilio_phone, ngrok_domain, configuration_display_name"
            ),
        }

    # Use user's display name (now required)
    # displayName must be unique, URL-safe, max 32 characters
    display_name = configuration_display_name

    # Validate display name length
    if len(display_name) > 32:
        return {
            "status": "error",
            "message": "Display name must not exceed 32 characters",
            "details": f"Current length: {len(display_name)}",
        }

    # The Conversation API needs a URL-safe displayName, and the SDK's ConversationConfiguration
    # model (validated when TAC loads the configuration) disallows "." and "~". Allowing only
    # what both accept keeps a wizard-created configuration loadable by the examples.
    if not re.match(r"^[A-Za-z0-9_-]+$", display_name):
        return {
            "status": "error",
            "message": (
                "Display name may only contain letters, numbers, underscores (_), and hyphens (-)"
            ),
            "details": f"Invalid display name: {display_name}",
        }

    # Validate description length only when provided
    if configuration_description and len(configuration_description) > 128:
        return {
            "status": "error",
            "message": "Description must not exceed 128 characters",
            "details": f"Current length: {len(configuration_description)}",
        }

    # Build webhook URL
    webhook_url = f"https://{ngrok_domain}/webhook"

    # Build the request payload
    # Note: displayName is the unique identifier, description is human-readable text
    payload = {
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
            "RCS": {
                "statusTimeouts": {"inactive": 10, "closed": 15},
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

    # Add optional description if provided (no default fallback - field is omitted if not provided)
    if configuration_description:
        payload["description"] = configuration_description

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CONVERSATION_API_BASE}/Configurations",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": get_basic_auth_header(api_key, api_secret),
                },
                json=payload,
                timeout=30.0,
            )

            if response.status_code in [200, 201]:
                result = response.json()
                logger.info("Conversation Orchestrator configuration created successfully")
                logger.debug(f"  Request displayName: {display_name}")
                logger.debug(f"  Response displayName: {result.get('displayName')}")
                return {
                    "status": "success",
                    "conversation_configuration_id": result.get("id"),
                    "message": "Conversation Orchestrator configuration"
                    f" created: {result.get('id')}",
                }
            elif response.status_code == 202:
                # Async operation - return status URL for polling
                result = response.json()
                logger.info("Conversation Orchestrator configuration creation accepted (async)")
                logger.info(f"  Status URL: {result.get('statusUrl')}")
                logger.info(f"  Display name: {display_name}")
                return {
                    "status": "accepted",
                    "status_url": result.get("statusUrl"),
                    "message": "Configuration creation accepted for processing",
                }
            else:
                endpoint = f"{CONVERSATION_API_BASE}/Configurations"
                logger.error("Failed to create Conversation Orchestrator configuration")
                logger.error(f"  Endpoint: {endpoint}")
                logger.error(f"  Payload: {json.dumps(payload, indent=2)}")
                logger.error(f"  Status: {response.status_code}")
                logger.error(f"  Response: {response.text}")
                return {
                    "status": "error",
                    "message": f"Failed to create Conversation Orchestrator configuration: "
                    f"{response.status_code} - {response.text}",
                    "endpoint": endpoint,
                    "payload": payload,
                    "response": response.text,
                    "status_code": response.status_code,
                }

    except httpx.TimeoutException:
        logger.error("Timeout creating Conversation Orchestrator configuration")
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception(f"Error creating Conversation Orchestrator configuration: {str(e)}")
        msg = f"Error creating Conversation Orchestrator configuration: {e}"
        return {"status": "error", "message": msg}


@app.post("/api/list-conversation-configurations")
async def list_conversation_configurations(request: Request) -> dict:
    """
    List all Conversation Orchestrator Configurations.

    Expected payload:
    {
        "api_key": "SK...",
        "api_secret": "..."
    }
    """
    data = await request.json()

    api_key = data.get("api_key")
    api_secret = data.get("api_secret")

    if not all([api_key, api_secret]):
        return {"status": "error", "message": "Missing required fields: api_key, api_secret"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{CONVERSATION_API_BASE}/Configurations",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": get_basic_auth_header(api_key, api_secret),
                },
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                configurations = result.get("configurations", [])

                # Log what fields are actually returned by the API
                if configurations:
                    logger.debug("Sample configuration fields from API:")
                    logger.debug(f"  displayName: {configurations[0].get('displayName')}")
                    logger.debug(f"  description: {configurations[0].get('description')}")

                # Return simplified list with id, displayName, description, etc.
                simplified = [
                    {
                        "id": config.get("id"),
                        "displayName": config.get("displayName"),
                        "description": config.get("description"),
                        "createdAt": config.get("createdAt"),
                        "memoryStoreId": config.get("memoryStoreId"),
                    }
                    for config in configurations
                ]
                return {"status": "success", "configurations": simplified}
            else:
                endpoint = f"{CONVERSATION_API_BASE}/Configurations"
                logger.error("Failed to list Conversation Orchestrator configurations")
                logger.error(f"  Endpoint: {endpoint}")
                logger.error(f"  Status: {response.status_code}")
                logger.error(f"  Response: {response.text}")
                return {
                    "status": "error",
                    "message": (
                        f"Failed to list configurations: {response.status_code} - {response.text}"
                    ),
                    "response": response.text,
                    "status_code": response.status_code,
                }

    except httpx.TimeoutException:
        logger.error("Timeout listing Conversation Orchestrator configurations")
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception(f"Error listing Conversation Orchestrator configurations: {str(e)}")
        return {"status": "error", "message": f"Error listing configurations: {str(e)}"}


@app.post("/api/delete-conversation-configuration")
async def delete_conversation_configuration(request: Request) -> dict:
    """
    Delete a Conversation Orchestrator Configuration.

    Expected payload:
    {
        "api_key": "SK...",
        "api_secret": "...",
        "configuration_id": "conv_configuration_..."
    }
    """
    data = await request.json()

    api_key = data.get("api_key")
    api_secret = data.get("api_secret")
    configuration_id = data.get("configuration_id")

    if not all([api_key, api_secret, configuration_id]):
        return {
            "status": "error",
            "message": "Missing required fields: api_key, api_secret, configuration_id",
        }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{CONVERSATION_API_BASE}/Configurations/{configuration_id}",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": get_basic_auth_header(api_key, api_secret),
                },
                timeout=30.0,
            )

            if response.status_code in [200, 204]:
                logger.info(f"Deleted Conversation Orchestrator configuration: {configuration_id}")
                return {
                    "status": "success",
                    "message": f"Configuration {configuration_id} deleted successfully",
                }
            elif response.status_code == 202:
                # Async deletion - accepted
                result = response.json()
                logger.info(
                    f"Conversation Orchestrator configuration deletion accepted (async): "
                    f"{configuration_id}"
                )
                logger.info(f"  Status URL: {result.get('statusUrl')}")
                return {
                    "status": "accepted",
                    "message": f"Configuration {configuration_id} deletion accepted for processing",
                    "status_url": result.get("statusUrl"),
                }
            else:
                endpoint = f"{CONVERSATION_API_BASE}/Configurations/{configuration_id}"
                logger.error("Failed to delete Conversation Orchestrator configuration")
                logger.error(f"  Endpoint: {endpoint}")
                logger.error(f"  Status: {response.status_code}")
                logger.error(f"  Response: {response.text}")
                return {
                    "status": "error",
                    "message": (
                        f"Failed to delete configuration: {response.status_code} - {response.text}"
                    ),
                    "response": response.text,
                    "status_code": response.status_code,
                }

    except httpx.TimeoutException:
        logger.error("Timeout deleting Conversation Orchestrator configuration")
        return {"status": "error", "message": "Request timed out. Please try again."}
    except Exception as e:
        logger.exception(f"Error deleting Conversation Orchestrator configuration: {str(e)}")
        return {"status": "error", "message": f"Error deleting configuration: {str(e)}"}


if __name__ == "__main__":
    import uvicorn

    print("Starting TAC Quickstart Setup Server...")
    print("Open http://localhost:8080 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8080)
