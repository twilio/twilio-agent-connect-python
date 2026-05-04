"""
TAC Dashboard — Lightweight observation dashboard for TAC examples.

Provides two views:
- **Active Sessions** — live conversations with message viewer and profile memory hover
- **Conversation History** — closed conversations fetched from Conversation Orchestrator

Mount onto any FastAPI app with ``create_dashboard_router()``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

from tac.context.conversation import ConversationClient
from tac.context.memory import MemoryClient
from tac.models.session import ConversationSession

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_dashboard_router(
    get_sms_sessions: Callable[[], dict[str, ConversationSession]] | None = None,
    get_voice_sessions: Callable[[], dict[str, ConversationSession]] | None = None,
    get_messages: Callable[[str], list[dict[str, Any]]] | None = None,
    memory_client: MemoryClient | None = None,
    conversation_client: ConversationClient | None = None,
    ci_events_store: dict[str, list[dict[str, Any]]] | None = None,
) -> APIRouter:
    """Create a dashboard router to mount on a FastAPI app.

    Args:
        get_sms_sessions: Returns SMS channel sessions ``{conv_id: ConversationSession}``.
        get_voice_sessions: Returns Voice channel sessions.
        get_messages: Returns message list for a conversation ID.
            Each message should have ``role`` (``"user"``/``"assistant"``) and ``content`` keys.
        memory_client: Optional — enables profile memory hover popover.
        conversation_client: Optional — enables the Conversation History page.
        ci_events_store: Optional shared dict ``{conv_id: [event_dict, ...]}``.
            Populated by CI webhook middleware; read by the events API.
    """
    router = APIRouter()

    def _all_sessions() -> dict[str, ConversationSession]:
        """Merge sessions from all channel callbacks."""
        sessions: dict[str, ConversationSession] = {}
        if get_sms_sessions:
            sessions.update(get_sms_sessions())
        if get_voice_sessions:
            sessions.update(get_voice_sessions())
        return sessions

    def _resolve_channel(conv_id: str) -> str:
        if get_sms_sessions and conv_id in get_sms_sessions():
            return "sms"
        if get_voice_sessions and conv_id in get_voice_sessions():
            return "voice"
        return "chat"

    def _get_display_messages(conv_id: str) -> list[dict[str, str]]:
        """Get user/assistant messages for display."""
        if not get_messages:
            return []
        raw = get_messages(conv_id)
        messages: list[dict[str, str]] = []
        for msg in raw:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content})
        return messages

    # ---- Dashboard HTML ----

    @router.get("/dashboard")
    async def dashboard() -> HTMLResponse:
        html = (STATIC_DIR / "dashboard.html").read_text()
        return HTMLResponse(content=html)

    # ---- Active Sessions ----

    @router.get("/api/sessions")
    async def list_sessions() -> JSONResponse:
        """List ACTIVE conversations (status=ACTIVE in Conversation Orchestrator)."""
        results: list[dict[str, Any]] = []
        sessions = _all_sessions()

        # Get ACTIVE + INACTIVE conversations from Conversation Orchestrator (skip CLOSED)
        live_status: dict[str, str] | None = None
        if conversation_client:
            try:
                live_convs = await conversation_client.list_conversations(
                    status=["ACTIVE", "INACTIVE"], page_size=100
                )
                live_status = {c.id: c.status or "ACTIVE" for c in live_convs}
            except Exception as e:
                logger.warning(f"Failed to fetch conversations from Conversation Orchestrator: {e}")

        for conv_id, session in sessions.items():
            # Skip sessions that Conversation Orchestrator reports as CLOSED
            if live_status is not None and conv_id not in live_status:
                continue

            msgs = _get_display_messages(conv_id)
            last_msg = msgs[-1] if msgs else None
            results.append(
                {
                    "id": conv_id,
                    "channel": session.channel or _resolve_channel(conv_id),
                    "status": live_status.get(conv_id, "ACTIVE")
                    if live_status is not None
                    else "ACTIVE",
                    "message_count": len(msgs),
                    "last_message": last_msg,
                    "started_at": session.started_at.isoformat() if session.started_at else None,
                    "profile_id": session.profile_id,
                    "author_address": (
                        session.author_info.address if session.author_info else None
                    ),
                }
            )

        return JSONResponse(content=results)

    @router.get("/api/sessions/{conv_id}/messages")
    async def get_session_messages(conv_id: str) -> JSONResponse:
        """Get message history for an active session."""
        sessions = _all_sessions()
        session = sessions.get(conv_id)
        if not session:
            return JSONResponse(content={"error": "Session not found"}, status_code=404)

        msgs = _get_display_messages(conv_id)
        return JSONResponse(
            content={
                "id": conv_id,
                "channel": session.channel or _resolve_channel(conv_id),
                "profile_id": session.profile_id,
                "author_address": (session.author_info.address if session.author_info else None),
                "messages": msgs,
            }
        )

    # ---- Shared memory helper ----

    async def _fetch_profile_memory(profile_id: str | None) -> dict[str, Any]:
        """Fetch profile traits, observations, and summaries from Memory."""
        data: dict[str, Any] = {"traits": {}, "observations": [], "summaries": []}
        if not memory_client or not profile_id:
            return data
        try:
            profile = await memory_client.get_profile(profile_id)
            data["traits"] = profile.traits or {}
        except Exception as e:
            logger.warning(f"Failed to fetch profile {profile_id}: {e}")
        try:
            memory = await memory_client.retrieve_memory(profile_id)
            data["observations"] = [
                {
                    "content": obs.content,
                    "source": obs.source,
                    "created_at": obs.created_at,
                    "occurred_at": obs.occurred_at,
                }
                for obs in memory.observations
            ]
            data["summaries"] = [
                {
                    "content": s.content,
                    "conversation_id": s.conversation_id,
                    "created_at": s.created_at,
                }
                for s in memory.summaries
            ]
        except Exception as e:
            logger.warning(f"Failed to retrieve memory for {profile_id}: {e}")
        return data

    # ---- Agent Context (combined profile + memory + communications) ----

    @router.get("/api/sessions/{conv_id}/context")
    async def get_session_context(conv_id: str) -> JSONResponse:
        """Get agent context for a session: profile, memory, and communications."""
        sessions = _all_sessions()
        session = sessions.get(conv_id)
        if not session:
            return JSONResponse(content={"error": "Session not found"}, status_code=404)

        mem = await _fetch_profile_memory(session.profile_id)
        result: dict[str, Any] = {
            "id": conv_id,
            "profile_id": session.profile_id,
            **mem,
            "communications": [],
            "ci_events": [],
        }

        # Fetch communications from Conversation Orchestrator
        if conversation_client:
            try:
                twilio_phone = os.environ.get("TWILIO_PHONE_NUMBER", "")
                communications = await conversation_client.list_communications(
                    conversation_id=conv_id, page_size=100
                )
                for comm in communications:
                    author_address = comm.author.address if comm.author else None
                    author_channel = comm.author.channel if comm.author else None
                    is_agent = author_channel in ("SYSTEM", "API") or (
                        twilio_phone and author_address == twilio_phone
                    )
                    result["communications"].append(
                        {
                            "role": "assistant" if is_agent else "user",
                            "content": comm.content.text if comm.content else "",
                            "author_address": author_address,
                            "author_channel": author_channel,
                            "created_at": comm.created_at,
                        }
                    )
            except Exception as e:
                logger.warning(f"Context: failed to fetch communications for {conv_id}: {e}")

        # CI events
        if ci_events_store:
            result["ci_events"] = ci_events_store.get(conv_id, [])

        return JSONResponse(content=result)

    # ---- Profile Memory ----

    @router.get("/api/profiles/{profile_id}/memory")
    async def get_profile_memory(profile_id: str) -> JSONResponse:
        """Fetch profile traits and memory for hover popover."""
        if not memory_client:
            return JSONResponse(content={"error": "Memory not configured"}, status_code=404)
        mem = await _fetch_profile_memory(profile_id)
        return JSONResponse(content={"profile_id": profile_id, **mem})

    # ---- Conversation History ----

    @router.get("/api/history")
    async def list_history() -> JSONResponse:
        """List closed conversations from Conversation Orchestrator."""
        if not conversation_client:
            return JSONResponse(
                content={"error": "Conversation client not available"}, status_code=404
            )

        try:
            conversations = await conversation_client.list_conversations(
                status=["CLOSED"], page_size=50
            )
            results: list[dict[str, Any]] = []
            for conv in conversations:
                results.append(
                    {
                        "id": conv.id,
                        "name": conv.name,
                        "status": conv.status,
                        "created_at": conv.created_at,
                        "updated_at": conv.updated_at,
                        "configuration_id": conv.configuration_id,
                    }
                )
            return JSONResponse(content=results)
        except Exception as e:
            logger.error(f"Failed to list conversation history: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)

    @router.get("/api/history/{conv_id}/messages")
    async def get_history_messages(conv_id: str) -> JSONResponse:
        """Fetch communications for a closed conversation."""
        if not conversation_client:
            return JSONResponse(
                content={"error": "Conversation client not available"}, status_code=404
            )

        try:
            communications = await conversation_client.list_communications(
                conversation_id=conv_id, page_size=100
            )
            twilio_phone = os.environ.get("TWILIO_PHONE_NUMBER", "")

            messages: list[dict[str, Any]] = []
            for comm in communications:
                author_address = comm.author.address if comm.author else None
                author_channel = comm.author.channel if comm.author else None
                is_agent = author_channel in ("SYSTEM", "API") or (
                    twilio_phone and author_address == twilio_phone
                )
                messages.append(
                    {
                        "id": comm.id,
                        "role": "assistant" if is_agent else "user",
                        "content": comm.content.text if comm.content else "",
                        "author_address": author_address,
                        "author_channel": author_channel,
                        "created_at": comm.created_at,
                    }
                )
            return JSONResponse(content={"id": conv_id, "messages": messages})
        except Exception as e:
            logger.error(f"Failed to fetch history messages for {conv_id}: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)

    return router


def mount_dashboard(
    app: Any,
    tac: Any,
    channels: list[Any] | None = None,
    messages: dict[str, list[Any]] | None = None,
) -> None:
    """Mount the dashboard onto a FastAPI app in one call.

    Args:
        app: FastAPI application instance.
        tac: TAC instance. Uses ``conversation_memory_client`` and
            ``conversation_orchestrator_client`` for data fetches.
        channels: Channel instances (SMSChannel, VoiceChannel, ChatChannel, etc.).
            Sessions are read from each channel's ``_conversations`` dict.
        messages: Conversation message history ``{conv_id: [{role, content}, ...]}``.
    """
    from tac.channels.sms import SMSChannel
    from tac.channels.voice import VoiceChannel

    sms_channels = [c for c in (channels or []) if isinstance(c, SMSChannel)]
    voice_channels = [c for c in (channels or []) if isinstance(c, VoiceChannel)]
    other_channels = [c for c in (channels or []) if not isinstance(c, (SMSChannel, VoiceChannel))]

    def _get_sms_sessions() -> dict[str, ConversationSession]:
        result: dict[str, ConversationSession] = {}
        for ch in sms_channels + other_channels:
            result.update(ch._conversations)
        return result

    def _get_voice_sessions() -> dict[str, ConversationSession]:
        result: dict[str, ConversationSession] = {}
        for ch in voice_channels:
            result.update(ch._conversations)
        return result

    logger.info("Dashboard mounted at /dashboard")

    # Shared CI events store — populated by middleware, read by dashboard
    ci_events: dict[str, list[dict[str, Any]]] = {}

    router = create_dashboard_router(
        get_sms_sessions=_get_sms_sessions if (sms_channels or other_channels) else None,
        get_voice_sessions=_get_voice_sessions if voice_channels else None,
        get_messages=(
            (lambda conv_id: list(messages.get(conv_id, []))) if messages is not None else None
        ),
        memory_client=tac.conversation_memory_client,
        conversation_client=tac.conversation_orchestrator_client,
        ci_events_store=ci_events,
    )
    app.include_router(router)

    # Add CI webhook interceptor — captures events before TAC processes them
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as StarletteRequest
    from starlette.responses import Response as StarletteResponse

    class CIEventCaptureMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: StarletteRequest, call_next: Any) -> StarletteResponse:
            if request.url.path.endswith("/ci-webhook") and request.method == "POST":
                import json as _json

                body = await request.body()
                try:
                    payload = _json.loads(body)
                    conv_id = payload.get("conversationId", "")
                    if conv_id:
                        for op in payload.get("operatorResults", []):
                            op_name = (op.get("operator") or {}).get("displayName", "")
                            output_format = op.get("outputFormat", "")
                            result_data = op.get("result", {})
                            event: dict[str, Any] = {
                                "operator": op_name,
                                "output_format": output_format,
                                "created_at": op.get("dateCreated"),
                            }
                            if isinstance(result_data, dict):
                                if "label" in result_data:
                                    event["detail"] = result_data["label"]
                                elif "text" in result_data:
                                    event["detail"] = result_data["text"][:500]
                                elif "result" in result_data:
                                    event["detail"] = str(result_data["result"])[:500]
                            events_list = ci_events.setdefault(conv_id, [])
                            events_list.append(event)
                            # Cap at 200 events per conversation to prevent memory leak
                            if len(events_list) > 200:
                                ci_events[conv_id] = events_list[-200:]
                except Exception:
                    pass  # Don't block the request if capture fails
            response: StarletteResponse = await call_next(request)
            return response

    app.add_middleware(CIEventCaptureMiddleware)
