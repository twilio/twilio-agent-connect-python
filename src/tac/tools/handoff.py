"""Handoff tool for the Twilio Agent Connect."""

import json
from typing import TYPE_CHECKING, Annotated, Any

import httpx

from tac import get_logger
from tac.models.handoff import HandoffPayload, PendingHandoffData
from tac.models.session import ConversationSession
from tac.tools.base import InjectedToolArg, TACTool, function_tool
from tac.utils.redaction import mask_address

if TYPE_CHECKING:
    from tac.core.tac import TAC

logger = get_logger(__name__)


def studio_executions_url(flow_sid: str) -> str:
    """Build the Twilio Studio Flow Executions URL for a given Flow SID.

    Used for digital (messaging/chat) handoff — POST the handoff payload
    to this URL to start a Studio flow execution.
    """
    return f"https://studio.twilio.com/v2/Flows/{flow_sid}/Executions"


def studio_voice_handoff_url(account_sid: str, flow_sid: str) -> str:
    """Build the Twilio Studio Flow voice webhook URL for a given Flow SID.

    Used as the ``<Connect action=...>`` URL in TwiML for voice handoff,
    so that when ConversationRelay ends the session Twilio triggers the
    Studio flow for an incoming call.
    """
    return (
        f"https://webhooks.twilio.com/v1/Accounts/{account_sid}"
        f"/Flows/{flow_sid}?Trigger=incomingCall"
    )


def build_handoff_payload(
    session: ConversationSession,
    memory_store_id: str,
    attributes: dict[str, Any],
) -> HandoffPayload:
    """Build a HandoffPayload from session context and attributes.

    Useful for custom handoff tools that want TAC's payload shape without
    the Studio-specific delivery in ``post_studio_handoff``.

    Args:
        session: Current conversation session
        memory_store_id: Memory store ID (typically ``tac.conversation_memory_client.store_id``)
        attributes: Developer-defined attributes (including reason)

    Returns:
        HandoffPayload with conversation context and attributes
    """
    return HandoffPayload(
        conversation_id=session.conversation_id,
        memory_store_id=memory_store_id,
        profile_id=session.profile_id or "",
        attributes=attributes,
    )


async def post_studio_handoff(
    payload: HandoffPayload,
    session: ConversationSession,
    *,
    handoff_url: str,
    from_address: str,
    api_key: str,
    api_secret: str,
) -> None:
    """POST a handoff payload to a Twilio Studio Flow Executions endpoint.

    Emits the Twilio Studio Executions API wire format: form-encoded
    ``To`` / ``From`` / ``Parameters`` fields with HTTP Basic auth.
    ``Parameters`` is a JSON string keyed under ``HandoffData`` so Studio
    can reference it via ``{{flow.data.HandoffData.*}}``.

    Args:
        payload: Structured handoff payload
        session: Current conversation session (used for ``To`` address)
        handoff_url: Studio Flow Executions URL
            (``https://studio.twilio.com/v2/Flows/FWxxx/Executions``)
        from_address: Twilio phone number used as ``From``
        api_key: Twilio API Key SID (Basic auth username)
        api_secret: Twilio API Key Secret (Basic auth password)

    Raises:
        httpx.HTTPError: If the POST request fails
    """
    to_address = session.author_info.address if session.author_info else ""

    data = {
        "To": to_address or "",
        "From": from_address,
        "Parameters": json.dumps({"HandoffData": payload.model_dump(by_alias=True)}),
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            handoff_url,
            data=data,
            auth=(api_key, api_secret),
        )
        response.raise_for_status()

    logger.debug(
        "Handoff payload delivered",
        handoff_url=handoff_url,
        to=mask_address(to_address),
        from_=mask_address(from_address),
    )


async def _handoff_impl(
    reason: str,
    # Typed as Any: @function_tool resolves annotations at runtime via
    # get_type_hints(), so a forward ref to TAC (imported only under
    # TYPE_CHECKING to keep the import graph clean) fails at construction.
    # InjectedToolArg hides this parameter from the LLM schema regardless.
    tac_instance: Annotated[Any, InjectedToolArg],
    session: Annotated[ConversationSession, InjectedToolArg],
    extra_attributes: Annotated[dict[str, Any], InjectedToolArg],
) -> dict[str, Any]:
    """Internal handoff implementation wrapped by ``create_studio_handoff_tool``.

    Renamed from ``handoff`` so ``tac.tools.handoff`` unambiguously refers to
    the module, not a function (prevents a name clash if the factory ever
    re-exports ``handoff`` from ``tac.tools.__init__``).

    The injected parameters are supplied at tool-construction time and hidden
    from the LLM; only ``reason`` appears in the schema the LLM sees.

    Returns:
        ``{"status": "handoff_initiated" | "handoff_failed", "channel": ...}``.
        Delivery failures (Studio POST raising) surface as ``handoff_failed``
        with an ``error`` key so the LLM can tell the user rather than
        claiming success over a dropped connection.
    """
    attributes = {**extra_attributes, "reason": reason}

    payload = build_handoff_payload(
        session=session,
        memory_store_id=tac_instance.conversation_memory_client.store_id,
        attributes=attributes,
    )

    channel = session.channel
    co_client = tac_instance.conversation_orchestrator_client

    # 1. Set conversation to INACTIVE so no new messages are routed here.
    # Downstream (Studio/Flex) flips it back to ACTIVE on pickup and CLOSED
    # on hangup — we don't close it ourselves.
    try:
        await co_client.update_conversation(
            conversation_id=session.conversation_id,
            status="INACTIVE",
        )
    except Exception as e:
        logger.warning(
            f"Failed to set conversation INACTIVE during handoff: {e}",
            conversation_id=session.conversation_id,
        )

    # 2. Clear statusCallbacks so TAC stops receiving webhook events
    try:
        await co_client.clear_status_callbacks(
            conversation_id=session.conversation_id,
        )
    except Exception as e:
        logger.warning(
            f"Failed to clear status callbacks during handoff: {e}",
            conversation_id=session.conversation_id,
        )

    # 3. Deliver handoff payload.
    if channel == "voice":
        # Voice: store on session for deferred delivery. The voice channel
        # sends the WS "end" message after the LLM's final response so the
        # caller hears a goodbye before transfer.
        # handoffData is a JSON *string* — ConversationRelay forwards it
        # verbatim in the POST body to the action URL.
        session.pending_handoff_data = PendingHandoffData(
            handoff_data=payload.model_dump_json(by_alias=True),
        )
    else:
        # Digital channels: POST handoff payload to the Studio Flow Executions
        # endpoint. A delivery failure here means the user is stranded on a
        # conversation we've already flagged INACTIVE — report that honestly
        # so the LLM can tell the user instead of claiming success.
        config = tac_instance.config
        try:
            await post_studio_handoff(
                payload,
                session,
                handoff_url=studio_executions_url(config.studio_handoff_flow_sid),
                from_address=config.phone_number,
                api_key=config.api_key,
                api_secret=config.api_secret,
            )
        except Exception as e:
            logger.error(
                f"Failed to deliver handoff payload: {e}",
                conversation_id=session.conversation_id,
            )
            return {"status": "handoff_failed", "channel": channel, "error": str(e)}

    return {"status": "handoff_initiated", "channel": channel}


DEFAULT_HANDOFF_TOOL_NAME = "handoff"
DEFAULT_HANDOFF_TOOL_DESCRIPTION = (
    "Hand off the conversation to a human agent. "
    "Use this when the customer requests a human, or when you "
    "cannot adequately handle the request."
)


def create_studio_handoff_tool(
    tac: "TAC",
    session: ConversationSession,
    attributes: dict[str, Any] | None = None,
    *,
    name: str = DEFAULT_HANDOFF_TOOL_NAME,
    description: str = DEFAULT_HANDOFF_TOOL_DESCRIPTION,
) -> TACTool:
    """
    Create a handoff tool that delivers in the Twilio Studio Executions API shape.

    The returned tool exposes only ``handoff(reason: str)`` to the LLM.
    All other dependencies are injected at runtime.

    On digital channels, the tool POSTs to the Studio Flow Executions
    endpoint derived from ``tac.config.studio_handoff_flow_sid``
    (``https://studio.twilio.com/v2/Flows/{flow_sid}/Executions``) using
    form-encoded ``To`` / ``From`` / ``Parameters`` fields with HTTP Basic
    auth. The Studio flow can access the handoff payload via
    ``{{flow.data.HandoffData.*}}``.

    For voice channels, the payload is stored on the session and the voice
    channel automatically sends the WS ``end`` message with ``handoffData``
    after the LLM's final response is delivered.

    The tool also sets the conversation to INACTIVE and clears status callbacks
    to prevent further webhook events from being routed to TAC.

    Args:
        tac: TAC instance for building payload and posting to Studio
        session: Current conversation session
        attributes: Static attributes to include in the handoff payload
                    (e.g., ``{"department": "billing", "priority": "high"}``).
                    The LLM-provided ``reason`` is always added automatically.
        name: Tool name exposed to the LLM. Defaults to ``"handoff"``.
        description: Tool description exposed to the LLM. Customize when the
                    default's phrasing doesn't match your product vocabulary
                    or escalation policy.

    Returns:
        Configured TACTool instance for handoff

    Example:
        >>> handoff_tool = create_studio_handoff_tool(
        ...     tac,
        ...     context,
        ...     attributes={"department": "support"},
        ...     name="escalate_to_agent",
        ...     description="Escalate only for billing disputes over $100.",
        ... )

    Raises:
        ValueError: If ``tac.config.studio_handoff_flow_sid`` is unset. The
            factory is Studio-specific; a missing SID is misconfiguration,
            not a soft fallback.
    """
    if not tac.config.studio_handoff_flow_sid:
        raise ValueError(
            "create_studio_handoff_tool requires tac.config.studio_handoff_flow_sid "
            "(set TWILIO_STUDIO_HANDOFF_FLOW_SID in your environment)."
        )

    handoff_tool = function_tool(name=name, description=description)(_handoff_impl)

    return handoff_tool.configure_injection(
        tac_instance=tac,
        session=session,
        extra_attributes=attributes or {},
    )
