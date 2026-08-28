"""Tests for Voice WebSocket message models."""

import inspect
import warnings

import pytest

from tac.models.voice import (
    CustomParameters,
    InterruptMessage,
    LanguageConfig,
    PromptMessage,
    SetupMessage,
    TwiMLRequest,
    VoiceTwiMLOptionsConversationRelay,
)


class TestCustomParameters:
    """Test CustomParameters model."""

    def test_basic_creation(self) -> None:
        """Test creating CustomParameters with basic fields."""
        params = CustomParameters(conversationId="CONV123", profileId="PROFILE456")

        assert params.conversation_id == "CONV123"
        assert params.profile_id == "PROFILE456"

    def test_snake_case_fields(self) -> None:
        """Test CustomParameters accepts snake_case field names."""
        params = CustomParameters(conversation_id="CONV123", profile_id="PROFILE456")

        assert params.conversation_id == "CONV123"
        assert params.profile_id == "PROFILE456"

    def test_optional_fields(self) -> None:
        """Test CustomParameters with optional fields."""
        params = CustomParameters(conversation_id="CONV123")

        assert params.conversation_id == "CONV123"
        assert params.profile_id is None


class TestSetupMessage:
    """Test SetupMessage model."""

    def test_basic_setup_message(self) -> None:
        """Test creating a basic setup message."""
        msg = SetupMessage(
            type="setup",
            sessionId="SESSION123",
            callSid="CALL456",
            accountSid="AC789",
        )

        assert msg.type == "setup"
        assert msg.session_id == "SESSION123"
        assert msg.call_sid == "CALL456"
        assert msg.account_sid == "AC789"

    def test_setup_with_call_details(self) -> None:
        """Test setup message with full call details."""
        msg = SetupMessage(
            type="setup",
            sessionId="SESSION123",
            callSid="CALL456",
            parentCallSid="PARENT789",
            **{"from": "+15551234567", "to": "+15559876543"},
            forwardedFrom="+15551111111",
            callerName="John Doe",
            direction="inbound",
            callType="PSTN",
            callStatus="in-progress",
            accountSid="AC123",
        )

        assert msg.from_number == "+15551234567"
        assert msg.to_number == "+15559876543"
        assert msg.forwarded_from == "+15551111111"
        assert msg.caller_name == "John Doe"
        assert msg.direction == "inbound"
        assert msg.call_type == "PSTN"
        assert msg.call_status == "in-progress"

    def test_setup_type_literal(self) -> None:
        """Test setup message type must be 'setup'."""
        msg = SetupMessage(type="setup")
        assert msg.type == "setup"

    def test_setup_minimal(self) -> None:
        """Test setup message with minimal required fields."""
        msg = SetupMessage(type="setup")

        assert msg.type == "setup"
        assert msg.session_id is None
        assert msg.call_sid is None


class TestPromptMessage:
    """Test PromptMessage model."""

    def test_basic_prompt_message(self) -> None:
        """Test creating a basic prompt message."""
        msg = PromptMessage(type="prompt", voicePrompt="Hello, I need help", lang="en-US")

        assert msg.type == "prompt"
        assert msg.voice_prompt == "Hello, I need help"
        assert msg.lang == "en-US"

    def test_prompt_with_snake_case(self) -> None:
        """Test prompt message with snake_case field names."""
        msg = PromptMessage(type="prompt", voice_prompt="Hello, I need help", lang="en-US")

        assert msg.voice_prompt == "Hello, I need help"

    def test_prompt_with_last_flag(self) -> None:
        """Test prompt message with 'last' flag."""
        msg = PromptMessage(type="prompt", voicePrompt="Final message", last=True)

        assert msg.voice_prompt == "Final message"
        assert msg.last is True

    def test_prompt_type_literal(self) -> None:
        """Test prompt message type must be 'prompt'."""
        msg = PromptMessage(type="prompt")
        assert msg.type == "prompt"

    def test_prompt_minimal(self) -> None:
        """Test prompt message with minimal fields."""
        msg = PromptMessage(type="prompt")

        assert msg.type == "prompt"
        assert msg.voice_prompt is None
        assert msg.lang is None
        assert msg.last is None


class TestInterruptMessage:
    """Test InterruptMessage model."""

    def test_basic_interrupt_message(self) -> None:
        """Test creating a basic interrupt message."""
        msg = InterruptMessage(
            type="interrupt",
            utteranceUntilInterrupt="Hello, I was saying...",
            durationUntilInterruptMs=1500,
        )

        assert msg.type == "interrupt"
        assert msg.utterance_until_interrupt == "Hello, I was saying..."
        assert msg.duration_until_interrupt_ms == 1500

    def test_interrupt_with_snake_case(self) -> None:
        """Test interrupt message with snake_case field names."""
        msg = InterruptMessage(
            type="interrupt",
            utterance_until_interrupt="Hello, I was saying...",
            duration_until_interrupt_ms=1500,
        )

        assert msg.utterance_until_interrupt == "Hello, I was saying..."
        assert msg.duration_until_interrupt_ms == 1500

    def test_interrupt_type_literal(self) -> None:
        """Test interrupt message type must be 'interrupt'."""
        msg = InterruptMessage(type="interrupt")
        assert msg.type == "interrupt"

    def test_interrupt_minimal(self) -> None:
        """Test interrupt message with minimal fields."""
        msg = InterruptMessage(type="interrupt")

        assert msg.type == "interrupt"
        assert msg.utterance_until_interrupt is None
        assert msg.duration_until_interrupt_ms is None


class TestVoiceMessageAliases:
    """Test field alias behavior across all voice models."""

    def test_setup_aliases(self) -> None:
        """Test SetupMessage field aliases."""
        # Test with API-style names (camelCase)
        data = {
            "type": "setup",
            "sessionId": "SESSION123",
            "callSid": "CALL456",
            "parentCallSid": "PARENT789",
            "from": "+15551234567",
            "to": "+15559876543",
            "accountSid": "AC123",
        }
        msg = SetupMessage(**data)

        assert msg.session_id == "SESSION123"
        assert msg.call_sid == "CALL456"
        assert msg.parent_call_sid == "PARENT789"
        assert msg.from_number == "+15551234567"
        assert msg.to_number == "+15559876543"
        assert msg.account_sid == "AC123"

    def test_prompt_aliases(self) -> None:
        """Test PromptMessage field aliases."""
        # Test with API-style names
        data = {"type": "prompt", "voicePrompt": "Test message", "lang": "en-US"}
        msg = PromptMessage(**data)

        assert msg.voice_prompt == "Test message"
        assert msg.lang == "en-US"

    def test_interrupt_aliases(self) -> None:
        """Test InterruptMessage field aliases."""
        # Test with API-style names
        data = {
            "type": "interrupt",
            "utteranceUntilInterrupt": "Test utterance",
            "durationUntilInterruptMs": 2000,
        }
        msg = InterruptMessage(**data)

        assert msg.utterance_until_interrupt == "Test utterance"
        assert msg.duration_until_interrupt_ms == 2000


class TestTwiMLRequest:
    """TwiMLRequest parses Twilio webhook form fields."""

    def test_from_form_known_fields(self) -> None:
        ctx = TwiMLRequest.from_form(
            {
                "From": "+14155551234",
                "To": "+15551234567",
                "CallSid": "CA" + "1" * 32,
                "CallerCountry": "US",
                "CallerState": "CA",
                "CallerCity": "San Francisco",
                "Direction": "inbound",
            }
        )
        assert ctx.from_number == "+14155551234"
        assert ctx.to_number == "+15551234567"
        assert ctx.call_sid == "CA" + "1" * 32
        assert ctx.caller_country == "US"
        assert ctx.caller_state == "CA"
        assert ctx.caller_city == "San Francisco"
        assert ctx.direction == "inbound"
        assert ctx.extra == {}

    def test_from_form_unknown_fields_bucketed_into_extra(self) -> None:
        ctx = TwiMLRequest.from_form(
            {
                "From": "+14155551234",
                "ApiVersion": "2010-04-01",
                "ForwardedFrom": "+15559999999",
            }
        )
        assert ctx.from_number == "+14155551234"
        assert ctx.extra == {
            "ApiVersion": "2010-04-01",
            "ForwardedFrom": "+15559999999",
        }

    def test_from_form_empty(self) -> None:
        ctx = TwiMLRequest.from_form({})
        assert ctx.from_number is None
        assert ctx.extra == {}


class TestTwiMLOptionsFieldsSet:
    """Merge semantics rely on Pydantic's model_fields_set."""

    def test_unset_scalar_fields_are_none(self) -> None:
        options = VoiceTwiMLOptionsConversationRelay()
        assert options.voice is None
        assert "voice" not in options.model_fields_set

    def test_explicitly_set_fields_tracked(self) -> None:
        options = VoiceTwiMLOptionsConversationRelay(
            voice="en-US-Journey-D",
            dtmf_detection=False,
        )
        assert "voice" in options.model_fields_set
        assert "dtmf_detection" in options.model_fields_set
        # Unset fields not tracked
        assert "interruptible" not in options.model_fields_set

    def test_language_config_optional_fields(self) -> None:
        lang = LanguageConfig(code="es-MX")
        assert lang.voice is None
        assert lang.tts_provider is None
        assert lang.transcription_provider is None


class TestTwiMLOptionsDeprecatedAlias:
    """``TwiMLOptions`` is a deprecated pre-Media-Streams alias for
    ``VoiceTwiMLOptionsConversationRelay``, resolved lazily via module
    ``__getattr__`` (PEP 562) so it stays a true alias rather than a
    subclass — see ``tac.models.voice.__getattr__``.
    """

    def test_accessing_the_name_warns(self) -> None:
        import tac.models.voice as voice_module

        with pytest.warns(DeprecationWarning, match="TwiMLOptions is deprecated"):
            resolved = voice_module.TwiMLOptions

        assert resolved is VoiceTwiMLOptionsConversationRelay

    @pytest.mark.parametrize(
        "import_path",
        ["tac", "tac.models", "tac.models.voice"],
    )
    def test_warning_is_attributed_to_the_caller_not_an_internal_shim(
        self, import_path: str
    ) -> None:
        """Regression test: each re-export layer must resolve the alias
        directly rather than forwarding to another layer's __getattr__ — a
        forwarded call attributes the warning (via stacklevel) to the internal
        shim that did the forwarding, not to this line. That misattribution is
        not just a wrong file/line in the message: Python's default warning
        filter dedupes by the attributed (module, line), so once any caller
        anywhere in the process triggers the warning through a forwarding
        path, every other, unrelated caller going through that same path is
        silently suppressed afterward. See tac._deprecation.
        """
        import importlib

        module = importlib.import_module(import_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            frame = inspect.currentframe()
            assert frame is not None
            expected_lineno = frame.f_lineno + 1
            _ = module.TwiMLOptions  # this exact line must be what's reported

        assert len(caught) >= 1
        assert caught[0].filename == __file__
        assert caught[0].lineno == expected_lineno

    def test_is_a_true_alias_not_a_subclass(self) -> None:
        """isinstance/equality must behave identically to the modern name —
        this is the whole reason the alias is a module __getattr__ rather
        than a subclass with its own __init__.
        """
        with pytest.warns(DeprecationWarning):
            from tac.models.voice import TwiMLOptions

        assert TwiMLOptions is VoiceTwiMLOptionsConversationRelay

        modern = VoiceTwiMLOptionsConversationRelay(voice="en-US-Journey-D")
        deprecated = TwiMLOptions(voice="en-US-Journey-D")
        assert isinstance(modern, TwiMLOptions)
        assert isinstance(deprecated, VoiceTwiMLOptionsConversationRelay)
        assert modern == deprecated

    def test_unknown_attribute_still_raises_attribute_error(self) -> None:
        import tac.models.voice as voice_module

        with pytest.raises(AttributeError):
            voice_module.not_a_real_attribute  # noqa: B018
