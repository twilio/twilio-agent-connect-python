"""PII redaction utilities for log output."""

import re

_MASK = "***"

_TWIML_PARAMETER_VALUE = re.compile(
    r"(<Parameter\b[^>]*?\bvalue=)([\"'])(.*?)\2",
    re.IGNORECASE,
)


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


def redact_twiml_parameters(twiml: str | None) -> str:
    """Mask ``<Parameter value="...">`` contents, keeping the names.

    ``<Parameter>`` children carry whatever the developer put in
    ``custom_parameters`` — profile IDs, caller names. Names stay because knowing
    which parameters were sent is the point of logging the TwiML.

    Handles either quote style. TAC's own TwiML comes from the Twilio SDK, which
    always double-quotes, but this takes a plain string and shouldn't depend on
    that to stay safe.
    """
    if not twiml:
        return ""
    return _TWIML_PARAMETER_VALUE.sub(rf"\1\g<2>{_MASK}\g<2>", twiml)
