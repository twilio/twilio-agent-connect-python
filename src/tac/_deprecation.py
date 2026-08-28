"""Shared helper for deprecated-alias ``__getattr__`` shims (PEP 562).

Every public re-export layer for a renamed name (e.g. ``tac``, ``tac.channels``,
``tac.channels.voice``) calls this directly from its own module ``__getattr__``
rather than delegating to another layer's ``__getattr__`` (which would forward
via a plain attribute access on that layer). Delegating adds one internal stack
frame per hop, and ``warnings.warn``'s ``stacklevel`` can only be fixed for one
specific hop count — so a chain of delegations attributes the warning to
whichever internal shim happens to be one frame up, never to the real caller.
That misattribution is not just cosmetic: Python's default warning filter
dedupes by the attributed (module, line), so once any caller anywhere in the
process triggers the warning through a given forwarding path, every other,
unrelated caller going through that same path is silently suppressed
afterward.

Calling this helper directly from each layer's own ``__getattr__`` — instead
of forwarding to a lower layer — keeps every entry point exactly two frames
from the real caller (this function, then that layer's ``__getattr__``), so
``stacklevel=3`` here is correct for all of them independently.
"""

from __future__ import annotations

import warnings
from typing import Any


def resolve_deprecated_alias(old_name: str, target: Any, version: str = "3.0") -> Any:
    """Warn that ``old_name`` is deprecated in favor of ``target``, then return it.

    Args:
        old_name: The deprecated name, as the caller referenced it.
        target: The replacement — returned as-is so the alias stays a true
            alias (``isinstance``/``==`` behave identically to using ``target``
            directly), not a subclass.
        version: The release ``old_name`` will be removed in.
    """
    warnings.warn(
        f"{old_name} is deprecated and will be removed in {version} — use "
        f"{target.__name__} instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    return target
