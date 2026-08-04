"""Tests for interrupt callback handling."""

import asyncio

import pytest

from tac import TAC, TACConfig
from tac.models.session import ConversationSession
from tac.models.voice import InterruptMessage


def get_test_config() -> TACConfig:
    """Get test configuration."""
    return TACConfig(
        environment="prod",
        account_sid="ACtest123",
        conversation_configuration_id="conv_configuration_test123",
        auth_token="test_token_123",
        api_key="SK123",
        api_secret="test_api_token",
        phone_number="+15551234567",
    )


class TestInterruptCallbacks:
    """Tests for interrupt callback handling."""

    def test_sync_interrupt_callback(self) -> None:
        """Test that synchronous interrupt callbacks work."""
        tac = TAC(get_test_config())

        received = []

        def sync_handler(context: ConversationSession, interrupt_data: InterruptMessage) -> None:
            received.append((context.conversation_id, interrupt_data))

        tac.on_interrupt(sync_handler)

        context = ConversationSession(conversation_id="conv123", channel="VOICE")
        interrupt = InterruptMessage(
            type="interrupt",
            utteranceUntilInterrupt="Hello",
            durationUntilInterruptMs=1000,
        )

        tac.trigger_interrupt(context, interrupt)

        assert len(received) == 1
        assert received[0][0] == "conv123"

    @pytest.mark.asyncio
    async def test_async_interrupt_callback_with_task(self) -> None:
        """Test that interrupt callbacks returning Tasks are handled correctly.

        This tests the ensure_future fix - callbacks that return already-created
        Tasks should work (create_task would raise TypeError).
        """
        tac = TAC(get_test_config())

        received = []
        completion_event = asyncio.Event()

        async def async_worker(
            context: ConversationSession, interrupt_data: InterruptMessage
        ) -> None:
            received.append(context.conversation_id)
            completion_event.set()

        def callback_returning_task(
            context: ConversationSession, interrupt_data: InterruptMessage
        ) -> asyncio.Task:
            # Some wrappers might return an already-created Task
            # ensure_future handles this, create_task would raise TypeError
            return asyncio.create_task(async_worker(context, interrupt_data))

        tac.on_interrupt(callback_returning_task)

        context = ConversationSession(conversation_id="conv456", channel="VOICE")
        interrupt = InterruptMessage(
            type="interrupt",
            utteranceUntilInterrupt="Test",
            durationUntilInterruptMs=500,
        )

        # Trigger should not raise TypeError with ensure_future
        tac.trigger_interrupt(context, interrupt)

        # Wait for completion or timeout
        await asyncio.wait_for(completion_event.wait(), timeout=1.0)

        assert len(received) == 1
        assert received[0] == "conv456"

    @pytest.mark.asyncio
    async def test_async_interrupt_callback_direct_coroutine(self) -> None:
        """Test that async interrupt callbacks (coroutines) work."""
        tac = TAC(get_test_config())

        received = []
        completion_event = asyncio.Event()

        async def async_handler(
            context: ConversationSession, interrupt_data: InterruptMessage
        ) -> None:
            received.append(context.conversation_id)
            completion_event.set()

        tac.on_interrupt(async_handler)

        context = ConversationSession(conversation_id="conv789", channel="VOICE")
        interrupt = InterruptMessage(
            type="interrupt",
            utteranceUntilInterrupt="Test",
            durationUntilInterruptMs=750,
        )

        tac.trigger_interrupt(context, interrupt)

        # Wait for completion or timeout
        await asyncio.wait_for(completion_event.wait(), timeout=1.0)

        assert len(received) == 1
        assert received[0] == "conv789"

    def test_async_interrupt_callback_without_event_loop(self) -> None:
        """Test that async callback without event loop logs warning and closes coroutine."""
        import warnings

        tac = TAC(get_test_config())

        # Track if coroutine was created and whether warning was raised
        coroutine_created = False

        async def async_handler(
            context: ConversationSession, interrupt_data: InterruptMessage
        ) -> None:
            nonlocal coroutine_created
            coroutine_created = True

        tac.on_interrupt(async_handler)

        context = ConversationSession(conversation_id="conv999", channel="VOICE")
        interrupt = InterruptMessage(
            type="interrupt",
            utteranceUntilInterrupt="Test",
            durationUntilInterruptMs=100,
        )

        # Capture warnings - there should be no "coroutine was never awaited" warning
        # because we explicitly close it
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            # This runs outside an async context (no event loop)
            # Should log warning but not raise "coroutine was never awaited"
            tac.trigger_interrupt(context, interrupt)

            # Check that no RuntimeWarning about coroutine was raised
            runtime_warnings = [w for w in warning_list if issubclass(w.category, RuntimeWarning)]
            assert len(runtime_warnings) == 0, (
                f"Expected no RuntimeWarning, but got: {[str(w.message) for w in runtime_warnings]}"
            )

    def test_no_callback_registered(self) -> None:
        """Test that trigger_interrupt does nothing when no callback is registered."""
        tac = TAC(get_test_config())

        # Don't register any callback
        context = ConversationSession(conversation_id="conv_no_cb", channel="VOICE")
        interrupt = InterruptMessage(
            type="interrupt",
            utteranceUntilInterrupt="Test",
            durationUntilInterruptMs=100,
        )

        # Should not raise any exception
        tac.trigger_interrupt(context, interrupt)

    def test_sync_callback_raises_exception(self) -> None:
        """Test that exceptions in sync callbacks propagate to caller."""

        tac = TAC(get_test_config())

        def failing_handler(context: ConversationSession, interrupt_data: InterruptMessage) -> None:
            raise ValueError("Intentional sync error")

        tac.on_interrupt(failing_handler)

        context = ConversationSession(conversation_id="conv_fail", channel="VOICE")
        interrupt = InterruptMessage(
            type="interrupt",
            utteranceUntilInterrupt="Test",
            durationUntilInterruptMs=100,
        )

        # Exception should propagate
        with pytest.raises(ValueError, match="Intentional sync error"):
            tac.trigger_interrupt(context, interrupt)

    @pytest.mark.asyncio
    async def test_async_callback_raises_exception_in_background(self) -> None:
        """Test that exceptions in async callbacks don't crash the main flow."""
        tac = TAC(get_test_config())

        exception_raised = asyncio.Event()

        async def failing_async_handler(
            context: ConversationSession, interrupt_data: InterruptMessage
        ) -> None:
            exception_raised.set()
            raise RuntimeError("Intentional async error")

        tac.on_interrupt(failing_async_handler)

        context = ConversationSession(conversation_id="conv_async_fail", channel="VOICE")
        interrupt = InterruptMessage(
            type="interrupt",
            utteranceUntilInterrupt="Test",
            durationUntilInterruptMs=100,
        )

        # Should not raise - exception happens in background task
        tac.trigger_interrupt(context, interrupt)

        # Wait for the async handler to run
        await asyncio.wait_for(exception_raised.wait(), timeout=1.0)

        # Main flow should continue (no exception propagated)
        # The exception will be logged by asyncio but won't crash the trigger

    @pytest.mark.asyncio
    async def test_callback_returning_future(self) -> None:
        """Test that callbacks returning Future objects are handled correctly."""
        tac = TAC(get_test_config())

        received = []
        completion_event = asyncio.Event()

        async def async_work(
            context: ConversationSession, interrupt_data: InterruptMessage
        ) -> None:
            received.append(context.conversation_id)
            completion_event.set()

        def callback_returning_future(
            context: ConversationSession, interrupt_data: InterruptMessage
        ) -> asyncio.Future:
            # Some wrappers might return a Future instead of Task
            future = asyncio.Future()

            async def execute():
                await async_work(context, interrupt_data)
                future.set_result(None)

            asyncio.create_task(execute())
            return future

        tac.on_interrupt(callback_returning_future)

        context = ConversationSession(conversation_id="conv_future", channel="VOICE")
        interrupt = InterruptMessage(
            type="interrupt",
            utteranceUntilInterrupt="Test",
            durationUntilInterruptMs=100,
        )

        # ensure_future should handle Future objects
        tac.trigger_interrupt(context, interrupt)

        # Wait for completion
        await asyncio.wait_for(completion_event.wait(), timeout=1.0)

        assert len(received) == 1
        assert received[0] == "conv_future"

    @pytest.mark.asyncio
    async def test_wrapped_async_interrupt_callback(self) -> None:
        """Test that wrapped async callbacks (functools.partial) work correctly."""
        from functools import partial

        tac = TAC(get_test_config())

        received = []
        completion_event = asyncio.Event()

        async def async_handler_with_extra_arg(
            extra: str, context: ConversationSession, interrupt_data: InterruptMessage
        ) -> None:
            assert extra == "test_extra"
            received.append(context.conversation_id)
            completion_event.set()

        # Wrap with functools.partial
        wrapped_handler = partial(async_handler_with_extra_arg, "test_extra")
        tac.on_interrupt(wrapped_handler)

        context = ConversationSession(conversation_id="conv_wrapped", channel="VOICE")
        interrupt = InterruptMessage(
            type="interrupt",
            utteranceUntilInterrupt="Test",
            durationUntilInterruptMs=100,
        )

        tac.trigger_interrupt(context, interrupt)

        # Wait for completion
        await asyncio.wait_for(completion_event.wait(), timeout=1.0)

        assert len(received) == 1
        assert received[0] == "conv_wrapped"
