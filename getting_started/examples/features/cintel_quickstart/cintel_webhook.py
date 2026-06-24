"""
CINTEL webhook parser for Script Adherence and Summary operators.
"""

import json
import logging

logger = logging.getLogger(__name__)


SCRIPT_ADHERENCE_OPERATOR_ID = "intelligence_operator_01kf34tcyefpyb1t4m0nbd8rxg"
SUMMARY_OPERATOR_ID = "intelligence_operator_01kcv35pnkeysaf6z6cqtbpegn"


def parse_cintel_webhook(payload: dict) -> dict[str, list[dict] | dict | None]:
    """
    Parse CINTEL webhook payload for Script Adherence and Summary results.

    Matches operators by ID since displayName is not present in the webhook payload.

    Returns:
        Dict with "script_adherence" (list[dict] | None) and "summary" (dict | None) keys
    """
    operator_results = payload.get("operatorResults", [])
    result: dict[str, list[dict] | dict | None] = {"script_adherence": None, "summary": None}

    logger.debug(f"[CINTEL] {len(operator_results)} operator result(s)")

    for op_result in operator_results:
        op_id = op_result.get("operator", {}).get("id", "")
        logger.debug(f"[CINTEL] operator id={op_id!r}")

        if op_id == SCRIPT_ADHERENCE_OPERATOR_ID:
            result["script_adherence"] = parse_script_adherence(op_result)
        elif op_id == SUMMARY_OPERATOR_ID:
            result["summary"] = parse_summary(op_result)
        else:
            logger.warning(f"[CINTEL] Unrecognised operator id {op_id!r} — skipping")

    return result


def parse_script_adherence(op_result: dict) -> list[dict]:
    categories = op_result.get("result", {}).get("categories", [])
    logger.debug(f"[CINTEL] Script Adherence: {len(categories)} category(ies) found")

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
    logger.debug(f"[CINTEL] Summary raw result: {json.dumps(result_data, indent=2)}")
    summary_text = result_data.get("text", "")
    logger.info(f"[CINTEL] Summary text: {summary_text!r}")
    return {"summary_text": summary_text}
