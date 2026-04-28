"""PII redaction utilities for log output."""

_MASK = "***"


def mask_phone(value: str | None) -> str:
    """Mask a phone number, preserving the first 2 and last 4 characters.

    Returns ``"***"`` for ``None``, empty, or short (< 7 char) inputs.
    """
    if not value:
        return _MASK
    value = value.strip()
    if not value or len(value) < 7:
        return _MASK
    return value[:2] + _MASK + value[-4:]


def mask_email(value: str | None) -> str:
    """Mask an email address, preserving the first character and full domain.

    Returns ``"***"`` for ``None``, empty, or strings without ``@``.
    """
    if not value:
        return _MASK
    value = value.strip()
    if not value:
        return _MASK
    at_index = value.find("@")
    if at_index < 1:
        return _MASK
    return value[0] + _MASK + value[at_index:]


def mask_address(value: str | None) -> str:
    """Auto-detect address type and apply the appropriate mask.

    Delegates to :func:`mask_email` if the value contains ``@``,
    otherwise to :func:`mask_phone`.
    """
    if not value or not value.strip():
        return _MASK
    if "@" in value:
        return mask_email(value)
    return mask_phone(value)
