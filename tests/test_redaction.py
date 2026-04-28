"""Tests for PII redaction utilities."""

import logging

from tac.core.logging import ContextLogger, PIISafetyFilter
from tac.utils.redaction import mask_address, mask_email, mask_phone


class TestMaskPhone:
    def test_us_number(self) -> None:
        assert mask_phone("+15551234567") == "+1***4567"

    def test_international_number(self) -> None:
        assert mask_phone("+447911123456") == "+4***3456"

    def test_no_plus_prefix(self) -> None:
        assert mask_phone("5551234567") == "55***4567"

    def test_short_number(self) -> None:
        assert mask_phone("+1234") == "***"

    def test_empty(self) -> None:
        assert mask_phone("") == "***"

    def test_whitespace(self) -> None:
        assert mask_phone("   ") == "***"

    def test_seven_chars(self) -> None:
        assert mask_phone("1234567") == "12***4567"


class TestMaskEmail:
    def test_basic(self) -> None:
        assert mask_email("user@example.com") == "u***@example.com"

    def test_single_char_local(self) -> None:
        assert mask_email("a@b.com") == "a***@b.com"

    def test_long_local(self) -> None:
        assert mask_email("longuser@domain.co.uk") == "l***@domain.co.uk"

    def test_empty(self) -> None:
        assert mask_email("") == "***"

    def test_no_at_sign(self) -> None:
        assert mask_email("not-an-email") == "***"

    def test_at_start(self) -> None:
        assert mask_email("@example.com") == "***"


class TestMaskAddress:
    def test_detects_phone(self) -> None:
        assert mask_address("+15551234567") == "+1***4567"

    def test_detects_email(self) -> None:
        assert mask_address("user@example.com") == "u***@example.com"

    def test_phone_no_plus(self) -> None:
        assert mask_address("5551234567") == "55***4567"

    def test_empty(self) -> None:
        assert mask_address("") == "***"

    def test_whitespace(self) -> None:
        assert mask_address("   ") == "***"


class TestLogOutputRedaction:
    """Integration tests: verify PII is masked in actual log output."""

    def _capture_log(self, logger: ContextLogger, caplog: logging.Handler) -> None:
        """Helper to attach caplog handler to the underlying logger."""
        logger.logger.addHandler(caplog.handler)  # type: ignore[attr-defined]
        logger.logger.setLevel(logging.DEBUG)

    def test_phone_masked_in_log_message(self, caplog: object) -> None:
        raw_phone = "+15551234567"
        base = logging.getLogger("tac.test.redaction_phone")
        ctx_logger = ContextLogger(base)

        with _CaplogContext(base, caplog) as records:
            ctx_logger.debug(f"No profile found for address {mask_address(raw_phone)}")

        assert len(records) == 1
        assert raw_phone not in records[0].getMessage()
        assert "+1***4567" in records[0].getMessage()

    def test_email_masked_in_log_message(self, caplog: object) -> None:
        raw_email = "alice@example.com"
        base = logging.getLogger("tac.test.redaction_email")
        ctx_logger = ContextLogger(base)

        with _CaplogContext(base, caplog) as records:
            ctx_logger.info(f"Outbound conversation initiated to {mask_address(raw_email)}")

        assert len(records) == 1
        assert raw_email not in records[0].getMessage()
        assert "a***@example.com" in records[0].getMessage()

    def test_phone_masked_in_structured_extra(self, caplog: object) -> None:
        raw_phone = "+447911123456"
        base = logging.getLogger("tac.test.redaction_extra")
        ctx_logger = ContextLogger(base)

        with _CaplogContext(base, caplog) as records:
            ctx_logger.info("Outbound voice call placed", to=mask_phone(raw_phone))

        assert len(records) == 1
        record = records[0]
        assert raw_phone not in record.getMessage()
        assert getattr(record, "to", None) == "+4***3456"


class TestPIISafetyFilter:
    """Tests for the safety-net logging filter that catches unmask'd PII."""

    def test_scrubs_phone_in_message(self) -> None:
        base = logging.getLogger("tac.test.safety_phone_msg")
        with _FilteredCaplogContext(base) as records:
            base.warning("Caller is +15559876543, please help")

        assert len(records) == 1
        msg = records[0].getMessage()
        assert "+15559876543" not in msg
        assert "***" in msg

    def test_scrubs_email_in_message(self) -> None:
        base = logging.getLogger("tac.test.safety_email_msg")
        with _FilteredCaplogContext(base) as records:
            base.info("Contact alice@example.com for details")

        assert len(records) == 1
        msg = records[0].getMessage()
        assert "alice@example.com" not in msg
        assert "***" in msg

    def test_scrubs_phone_in_extra_field(self) -> None:
        base = logging.getLogger("tac.test.safety_phone_extra")
        with _FilteredCaplogContext(base) as records:
            base.info("Outbound call", extra={"to": "+447911123456"})

        assert len(records) == 1
        assert getattr(records[0], "to", None) == "***"

    def test_scrubs_email_in_extra_field(self) -> None:
        base = logging.getLogger("tac.test.safety_email_extra")
        with _FilteredCaplogContext(base) as records:
            base.info("Sending message", extra={"address": "bob@corp.io"})

        assert len(records) == 1
        assert getattr(records[0], "address", None) == "***"

    def test_passes_clean_messages_through(self) -> None:
        base = logging.getLogger("tac.test.safety_clean")
        with _FilteredCaplogContext(base) as records:
            base.info("Conversation started", extra={"conversation_id": "conv_abc123"})

        assert len(records) == 1
        assert records[0].getMessage() == "Conversation started"
        assert getattr(records[0], "conversation_id", None) == "conv_abc123"

    def test_scrubs_phone_in_format_args(self) -> None:
        base = logging.getLogger("tac.test.safety_phone_args")
        with _FilteredCaplogContext(base) as records:
            base.info("Calling %s now", "+15559876543")

        assert len(records) == 1
        assert "+15559876543" not in records[0].getMessage()


class _FilteredCaplogContext:
    """Context manager that captures log records through a PIISafetyFilter."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._handler = logging.Handler()
        self._filter = PIISafetyFilter()
        self.records: list[logging.LogRecord] = []
        self._handler.emit = self.records.append  # type: ignore[assignment]
        self._handler.addFilter(self._filter)

    def __enter__(self) -> list[logging.LogRecord]:
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.DEBUG)
        return self.records

    def __exit__(self, *args: object) -> None:
        self._logger.removeHandler(self._handler)


class _CaplogContext:
    """Minimal context manager that captures LogRecords from a specific logger."""

    def __init__(self, logger: logging.Logger, _caplog: object) -> None:
        self._logger = logger
        self._handler = logging.Handler()
        self.records: list[logging.LogRecord] = []
        self._handler.emit = self.records.append  # type: ignore[assignment]

    def __enter__(self) -> list[logging.LogRecord]:
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.DEBUG)
        return self.records

    def __exit__(self, *args: object) -> None:
        self._logger.removeHandler(self._handler)
