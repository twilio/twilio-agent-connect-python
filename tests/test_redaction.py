"""Tests for PII redaction utilities."""

import logging

from tac.core.logging import ConsoleFormatter, ContextLogger, JSONFormatter
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

    def test_none(self) -> None:
        assert mask_phone(None) == "***"

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

    def test_none(self) -> None:
        assert mask_email(None) == "***"

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

    def test_none(self) -> None:
        assert mask_address(None) == "***"

    def test_whitespace(self) -> None:
        assert mask_address("   ") == "***"


class TestLogOutputRedaction:
    """Integration tests: verify PII is masked in actual log output via ContextLogger."""

    def test_phone_masked_in_log_message(self) -> None:
        raw_phone = "+15551234567"
        base = logging.getLogger("tac.test.redaction_phone")
        ctx_logger = ContextLogger(base)

        with _RecordCapture(base) as records:
            ctx_logger.debug(f"No profile found for address {mask_address(raw_phone)}")

        assert len(records) == 1
        assert raw_phone not in records[0].getMessage()
        assert "+1***4567" in records[0].getMessage()

    def test_email_masked_in_log_message(self) -> None:
        raw_email = "alice@example.com"
        base = logging.getLogger("tac.test.redaction_email")
        ctx_logger = ContextLogger(base)

        with _RecordCapture(base) as records:
            ctx_logger.info(f"Outbound conversation initiated to {mask_address(raw_email)}")

        assert len(records) == 1
        assert raw_email not in records[0].getMessage()
        assert "a***@example.com" in records[0].getMessage()

    def test_phone_masked_in_structured_extra(self) -> None:
        raw_phone = "+447911123456"
        base = logging.getLogger("tac.test.redaction_extra")
        ctx_logger = ContextLogger(base)

        with _RecordCapture(base) as records:
            ctx_logger.info("Outbound voice call placed", to=mask_phone(raw_phone))

        assert len(records) == 1
        record = records[0]
        assert raw_phone not in record.getMessage()
        assert getattr(record, "to", None) == "+4***3456"


class TestFormatterSafetyNet:
    """Tests that formatters scrub PII from final output, including exception text."""

    def test_json_formatter_scrubs_phone_in_message(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            "tac.test", logging.INFO, "", 0, "Calling +15559876543 now", (), None
        )
        output = formatter.format(record)
        assert "+15559876543" not in output
        assert "***" in output

    def test_json_formatter_scrubs_email_in_message(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            "tac.test", logging.INFO, "", 0, "Contact alice@example.com", (), None
        )
        output = formatter.format(record)
        assert "alice@example.com" not in output
        assert "***" in output

    def test_json_formatter_scrubs_phone_in_extra(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord("tac.test", logging.INFO, "", 0, "Outbound call", (), None)
        record.to = "+447911123456"  # type: ignore[attr-defined]
        output = formatter.format(record)
        assert "+447911123456" not in output
        assert "***" in output

    def test_json_formatter_scrubs_exception_text(self) -> None:
        formatter = JSONFormatter()
        try:
            raise ValueError("No profile found for +15559876543")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
        record = logging.LogRecord(
            "tac.test", logging.ERROR, "", 0, "Profile lookup failed", (), exc_info
        )
        output = formatter.format(record)
        assert "+15559876543" not in output
        assert "***" in output

    def test_console_formatter_scrubs_phone_in_message(self) -> None:
        formatter = ConsoleFormatter(fmt="%(message)s")
        record = logging.LogRecord(
            "tac.test", logging.INFO, "", 0, "Calling +15559876543 now", (), None
        )
        output = formatter.format(record)
        assert "+15559876543" not in output
        assert "***" in output

    def test_console_formatter_scrubs_phone_in_extra(self) -> None:
        formatter = ConsoleFormatter(fmt="%(message)s")
        record = logging.LogRecord("tac.test", logging.INFO, "", 0, "Outbound call", (), None)
        record.to = "+447911123456"  # type: ignore[attr-defined]
        output = formatter.format(record)
        assert "+447911123456" not in output
        assert "***" in output

    def test_json_formatter_scrubs_domestic_phone(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            "tac.test", logging.INFO, "", 0, "Calling 5551234567 now", (), None
        )
        output = formatter.format(record)
        assert "5551234567" not in output
        assert "***" in output

    def test_json_formatter_scrubs_formatted_phone(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            "tac.test", logging.INFO, "", 0, "Calling (555) 123-4567 now", (), None
        )
        output = formatter.format(record)
        assert "(555) 123-4567" not in output
        assert "***" in output

    def test_json_formatter_output_remains_valid_json(self) -> None:
        import json

        formatter = JSONFormatter()
        record = logging.LogRecord(
            "tac.test", logging.INFO, "", 0, "Calling +15559876543", (), None
        )
        record.to = "+447911123456"  # type: ignore[attr-defined]
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "+15559876543" not in parsed["message"]
        assert "+447911123456" not in str(parsed.get("to", ""))

    def test_json_formatter_int_field_not_corrupted(self) -> None:
        import json

        formatter = JSONFormatter()
        record = logging.LogRecord("tac.test", logging.INFO, "", 0, "call placed", (), None)
        record.retry_count = 5551234567  # type: ignore[attr-defined]
        output = formatter.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed["retry_count"], int)

    def test_formatters_pass_clean_messages_through(self) -> None:
        for formatter in [JSONFormatter(), ConsoleFormatter(fmt="%(message)s")]:
            record = logging.LogRecord(
                "tac.test", logging.INFO, "", 0, "Conversation started", (), None
            )
            record.conversation_id = "conv_abc123"  # type: ignore[attr-defined]
            output = formatter.format(record)
            assert "Conversation started" in output
            assert "conv_abc123" in output


class _RecordCapture:
    """Context manager that captures LogRecords from a logger."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._handler = logging.Handler()
        self._prev_level: int = logging.WARNING
        self.records: list[logging.LogRecord] = []
        self._handler.emit = self.records.append  # type: ignore[assignment]

    def __enter__(self) -> list[logging.LogRecord]:
        self._prev_level = self._logger.level
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.DEBUG)
        return self.records

    def __exit__(self, *args: object) -> None:
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._prev_level)
