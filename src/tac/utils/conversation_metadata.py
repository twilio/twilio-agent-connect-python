"""Fit app metadata into Conversation Orchestrator's conversation-metadata limits."""

from __future__ import annotations

import re
from typing import Any

#: Conversation Orchestrator's documented limits for conversation metadata.
MAX_KEYS = 8
MAX_KEY_LENGTH = 128
MAX_VALUE_LENGTH = 512
_KEY_PATTERN = re.compile(r"[a-zA-Z0-9._-]+")


def fit_conversation_metadata(
    metadata: dict[str, Any] | None, *, reserved: dict[str, str]
) -> tuple[dict[str, str], list[str]]:
    """Split ``metadata`` into what CO will store and what it won't.

    ``reserved`` entries (TAC's own, like ``direction``) always come first and
    win over an app key of the same name. App entries follow in insertion
    order while they fit. An entry is skipped when its key isn't a string of
    letters, digits, ``.``, ``_`` or ``-`` up to 128 characters, its value
    isn't a string of at most 512 characters, or the 8-key limit is reached.

    Returns:
        ``(persisted, skipped_keys)``.
    """
    persisted = dict(reserved)
    skipped: list[str] = []
    for key, value in (metadata or {}).items():
        if key in reserved:
            continue
        fits = (
            isinstance(key, str)
            and len(key) <= MAX_KEY_LENGTH
            and _KEY_PATTERN.fullmatch(key) is not None
            and isinstance(value, str)
            and len(value) <= MAX_VALUE_LENGTH
            and len(persisted) < MAX_KEYS
        )
        if fits:
            persisted[key] = value
        else:
            skipped.append(str(key))
    return persisted, skipped
