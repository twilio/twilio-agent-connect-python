"""Processor for Conversation Intelligence webhook events."""

import json
from typing import Any

from pydantic import ValidationError

from tac.context.memory import MemoryClient
from tac.core.config import ConversationIntelligenceConfig
from tac.core.logging import get_logger
from tac.models.intelligence import (
    OperatorProcessingResult,
    OperatorResult,
    OperatorResultEvent,
)


def _extract_profile_ids(operator_result: "OperatorResult") -> list[str]:
    """
    Extract valid profile IDs from operator result participants.

    Args:
        operator_result: The individual operator result

    Returns:
        List of valid profile IDs
    """
    profile_ids: list[str] = []

    if not operator_result.execution_details or not operator_result.execution_details.participants:
        return profile_ids

    for participant in operator_result.execution_details.participants:
        if participant.profile_id and participant.type == "CUSTOMER":
            profile_ids.append(participant.profile_id)

    return profile_ids


def _generate_content(operator_result: "OperatorResult") -> str | None:
    """
    Generate content string from the operator result based on output format.

    Args:
        operator_result: The individual operator result

    Returns:
        The content string or None if unable to extract
    """
    output_format = operator_result.output_format.upper()
    result = operator_result.result

    # Handle different result formats
    if output_format == "JSON":
        # For JSON format, extract payload
        if isinstance(result, dict):
            if "payload" in result:
                return str(result["payload"])
            # Handle nested union structure
            if "com.twilio.cai.intelligence.JSONResult" in result:
                nested = result["com.twilio.cai.intelligence.JSONResult"]
                if isinstance(nested, dict) and "payload" in nested:
                    return str(nested["payload"])
        return json.dumps(result) if result else None

    elif output_format == "CLASSIFICATION":
        if isinstance(result, dict) and "label" in result:
            return str(result["label"])
        return str(result) if result else None

    elif output_format == "EXTRACTION":
        if isinstance(result, dict) and "entities" in result:
            return json.dumps(result["entities"])
        return json.dumps(result) if result else None

    elif output_format in ("TEXT", "GENERATION"):
        if isinstance(result, dict) and "result" in result:
            return str(result["result"])
        return str(result) if result else None

    # Fallback: serialize result as JSON
    return json.dumps(result) if result else None


def _parse_summaries_content(json_content: str) -> list[str]:
    """
    Parse JSON content to extract individual summary contents.

    Supported formats:
    1. Array: {"summaries": [{"summary": "..."}, {"summary": "..."}]}
    2. Single: {"summary": "..."}
    3. Fallback: Treat raw content as single summary

    Args:
        json_content: The JSON content string

    Returns:
        List of summary content strings
    """
    try:
        payload = json.loads(json_content)

        # Try array format first
        if isinstance(payload, dict) and "summaries" in payload:
            summaries = payload["summaries"]
            if isinstance(summaries, list):
                contents = []
                for summary in summaries:
                    if isinstance(summary, dict) and summary.get("summary"):
                        contents.append(str(summary["summary"]))
                if contents:
                    return contents

        # Try single summary format
        if isinstance(payload, dict) and "summary" in payload:
            summary_content = payload["summary"]
            if summary_content:
                return [str(summary_content)]

    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: treat entire content as single summary
    return [json_content] if json_content else []


class OperatorResultProcessor:
    """Processor for Conversation Intelligence webhook events.

    This processor handles incoming CI webhook payloads, validates them,
    and creates conversation summaries in Conversation Memory based on the event type.

    Events are filtered by:
    - Configuration ID matching the provided config
    - Operator SID matching the summary operator SID in config

    Example usage:
        ```python
        from tac.context.memory import MemoryClient
        from tac.core.config import ConversationIntelligenceConfig
        from tac.intelligence import OperatorResultProcessor

        conversation_memory_client = MemoryClient(...)
        config = ConversationIntelligenceConfig(
            configuration_id="GA...",
            summary_operator_sid="LY...",
        )
        processor = OperatorResultProcessor(conversation_memory_client, config)

        result = await processor.process_event(webhook_payload)
        if result.skipped:
            print(f"Skipped: {result.skip_reason}")
        elif result.success:
            print(f"Created {result.created_count} {result.event_type}(s)")
        else:
            print(f"Error: {result.error}")
        ```
    """

    def __init__(
        self,
        conversation_memory_client: MemoryClient,
        config: ConversationIntelligenceConfig,
    ) -> None:
        """
        Initialize the CI event processor.

        Args:
            conversation_memory_client: MemoryClient instance for creating summaries
            config: ConversationIntelligenceConfig for filtering events by configuration
                ID and operator SIDs
        """
        self.conversation_memory_client = conversation_memory_client
        self.config = config
        self.logger = get_logger(__name__)

    async def process_event(self, payload: dict[str, Any]) -> OperatorProcessingResult:
        """
        Process a CI webhook payload.

        This method:
        1. Parses the payload into an OperatorResultEvent (Pydantic validates required fields)
        2. Applies filtering logic based on intelligence configuration ID and operator SIDs
        3. Iterates over operator_results array
        4. For each operator result: filters by operator SID, then generates
           content and extracts profile IDs
        5. Creates conversation summaries in Conversation Memory

        Args:
            payload: The raw webhook payload dictionary

        Returns:
            OperatorProcessingResult with status and details
        """
        # Import here to avoid circular imports at module level

        # Parse payload into OperatorResultEvent (Pydantic validates required fields)
        try:
            event = OperatorResultEvent(**payload)
        except ValidationError as e:
            self.logger.error(f"Failed to parse event payload: {e}")
            return OperatorProcessingResult(
                success=False,
                error=f"Failed to parse event payload: {e}",
            )

        # Filter by configuration ID
        if event.intelligence_configuration.id != self.config.configuration_id:
            self.logger.debug(
                f"Discarding event - configuration ID mismatch: "
                f"got {event.intelligence_configuration.id}, "
                f"expected {self.config.configuration_id}"
            )
            return OperatorProcessingResult(
                success=True,
                skipped=True,
                skip_reason=f"Configuration ID mismatch (expected {self.config.configuration_id})",
            )

        # Process each operator result in the array
        total_created = 0
        all_errors: list[str] = []
        event_types: set[str] = set()
        skipped_count = 0
        last_skip_reason: str | None = None

        for operator_result in event.operator_results:
            result = await self._process_operator_result(
                event=event,
                operator_result=operator_result,
            )

            if result.success and not result.skipped:
                total_created += result.created_count
                if result.event_type:
                    event_types.add(result.event_type)
            elif result.success and result.skipped:
                skipped_count += 1
                last_skip_reason = result.skip_reason
            elif not result.success and result.error:
                all_errors.append(result.error)

        # Aggregate results
        if total_created == 0 and all_errors:
            return OperatorProcessingResult(
                success=False,
                error="; ".join(all_errors),
            )

        # If all operators were skipped, return skipped
        if total_created == 0 and skipped_count == len(event.operator_results):
            return OperatorProcessingResult(
                success=True,
                skipped=True,
                skip_reason=last_skip_reason or "All operators skipped",
            )

        # Determine combined event type
        combined_type: str | None = None
        if len(event_types) == 1:
            combined_type = event_types.pop()
        elif len(event_types) > 1:
            combined_type = "mixed"

        return OperatorProcessingResult(
            success=True,
            event_type=combined_type,
            created_count=total_created,
        )

    async def _process_operator_result(
        self,
        event: OperatorResultEvent,
        operator_result: "OperatorResult",
    ) -> OperatorProcessingResult:
        """
        Process a single operator result from the event.

        Args:
            event: The parent webhook event (for conversation_id)
            operator_result: The individual operator result to process

        Returns:
            OperatorProcessingResult with status and count
        """
        # Filter by operator SID first so unrelated operators are skipped rather
        # than failing the whole event.
        operator_id = operator_result.operator.id if operator_result.operator else None

        if self.config.summary_operator_sid is None:
            # No summary operator configured - nothing to process
            self.logger.debug("Skipping operator - summary operator SID not configured")
            return OperatorProcessingResult(
                success=True,
                skipped=True,
                skip_reason="Summary operator SID not configured",
            )

        if operator_id != self.config.summary_operator_sid:
            # Configured summary SID doesn't match this operator - skip
            self.logger.debug(
                f"Skipping operator - SID {operator_id} doesn't match "
                f"summary ({self.config.summary_operator_sid})"
            )
            return OperatorProcessingResult(
                success=True,
                skipped=True,
                skip_reason="Operator SID mismatch",
            )

        # Generate content from result
        content = _generate_content(operator_result)
        if not content:
            self.logger.info(f"Skipping operator result {operator_result.id} with empty content")
            return OperatorProcessingResult(
                success=True,
                skipped=True,
                skip_reason="Operator result has empty content",
            )

        # Extract profile IDs from this operator result
        profile_ids = _extract_profile_ids(operator_result)
        if not profile_ids:
            self.logger.info(f"No profile IDs found in operator result {operator_result.id}")
            return OperatorProcessingResult(
                success=True,
                skipped=True,
                skip_reason="No profile IDs found in operator result execution details",
            )

        return await self._process_summary_event(
            event=event,
            operator_result=operator_result,
            content=content,
            profile_ids=profile_ids,
        )

    async def _process_summary_event(
        self,
        event: OperatorResultEvent,
        operator_result: "OperatorResult",
        content: str,
        profile_ids: list[str],
    ) -> OperatorProcessingResult:
        """
        Process a summary event by creating conversation summaries in Conversation Memory.

        Args:
            event: The parent webhook event (for conversation_id)
            operator_result: The individual operator result
            content: The generated content string
            profile_ids: List of profile IDs to create summaries for

        Returns:
            OperatorProcessingResult with status and count
        """
        # Parse summaries from content
        summary_contents = _parse_summaries_content(content)

        if not summary_contents:
            self.logger.info(f"No summaries to create from operator result {operator_result.id}")
            return OperatorProcessingResult(
                success=True,
                event_type="summary",
                skipped=True,
                skip_reason="No summary content found",
            )

        created_count = 0
        errors: list[str] = []

        # Create summaries for each profile
        for profile_id in profile_ids:
            # Build summaries payload
            summaries_payload: list[dict[str, Any]] = []
            for summary_content in summary_contents:
                summaries_payload.append(
                    {
                        "content": summary_content,
                        "conversationId": event.conversation_id,
                        "occurredAt": operator_result.date_created,
                        "source": "conversation-intelligence",
                    }
                )

            try:
                await self.conversation_memory_client.create_conversation_summaries(
                    profile_id=profile_id,
                    summaries=summaries_payload,
                )
                created_count += len(summaries_payload)
            except Exception as e:
                error_msg = f"Failed to create summaries for profile {profile_id}: {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)

        if created_count == 0 and errors:
            return OperatorProcessingResult(
                success=False,
                event_type="summary",
                error="; ".join(errors),
            )

        self.logger.info(
            f"Created {created_count} summary(ies) from operator result {operator_result.id}"
        )
        return OperatorProcessingResult(
            success=True,
            event_type="summary",
            created_count=created_count,
        )
