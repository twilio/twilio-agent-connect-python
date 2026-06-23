"""
CINTEL webhook parser for Script Adherence and Summary operators.
"""

import json
import logging

logger = logging.getLogger(__name__)


def parse_cintel_webhook(payload: dict) -> dict[str, list[dict] | dict | None]:
    """
    Parse CINTEL webhook payload for Script Adherence and Summary results.

    Returns:
        Dict with "script_adherence" (list[dict] | None) and "summary" (dict | None) keys
    """
    operator_results = payload.get("operatorResults", [])
    result: dict[str, list[dict] | dict | None] = {"script_adherence": None, "summary": None}

    for i, op_result in enumerate(operator_results):
        display_name = op_result.get("operator", {}).get("displayName", "")
        logger.debug("[CINTEL] operator_results[%d] displayName: %r", i, display_name)

        if display_name == "Script-Adherence":
            result["script_adherence"] = parse_script_adherence(op_result)
        elif display_name == "Summary":
            result["summary"] = parse_summary(op_result)
        else:
            logger.info("[CINTEL] Unrecognised operator %r — skipping", display_name)

    return result


def parse_script_adherence(op_result: dict) -> list[dict]:
    categories = op_result.get("result", {}).get("categories", [])
    logger.debug("[CINTEL] Script Adherence: %d category(ies) found", len(categories))

    checkpoints = []
    for cat in categories:
        criteria = cat.get("criteria", [])
        category_name = cat.get("category_key", "")

        criteria_values = [c.get("criteria_met") for c in criteria]
        evaluated = [v for v in criteria_values if v in ("Succeeded", "Failed")]
        all_succeeded = bool(evaluated) and all(v == "Succeeded" for v in evaluated)
        any_failed = any(v == "Failed" for v in criteria_values)

        checkpoints.append(
            {
                "category": category_name,
                "completed": all_succeeded,
                "skipped": any_failed,
                "criteria": [
                    {
                        "key": c.get("criteria_key", c.get("key", "")),
                        "met": c.get("criteria_met") == "Succeeded",
                        "evaluated": c.get("criteria_met") in ("Succeeded", "Failed"),
                    }
                    for c in criteria
                ],
            }
        )

    return checkpoints


def parse_summary(op_result: dict) -> dict:
    result_data = op_result.get("result", {})
    logger.debug("[CINTEL] Summary raw result: %s", json.dumps(result_data, indent=2))
    summary_text = result_data.get("text", "")
    logger.info("[CINTEL] Summary text: %r", summary_text)
    return {"summary_text": summary_text}
