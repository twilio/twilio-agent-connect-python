"""Integration tests for the complete TAC framework."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tac import TAC, TACConfig
from tac.channels.sms import SMSChannel
from tac.context.memory import MemoryClient
from tac.core.config import TwilioMemoryConfig
from tac.models.conversation import ParticipantAddress, ParticipantResponse
from tac.models.memory import MemoryRetrievalMeta, MemoryRetrievalResponse
from tac.models.session import ConversationSession
from tac.models.tac import TACMemoryResponse


def make_sms_participants(conv_id: str = "CH123456") -> list[ParticipantResponse]:
    return [
        ParticipantResponse(
            **{  # type: ignore[arg-type]
                "id": "PA_AGENT",
                "accountId": "ACtest123",
                "conversationId": conv_id,
                "name": "+15551234567",
                "type": "AI_AGENT",
                "addresses": [
                    ParticipantAddress(channel="SMS", address="+15551234567").model_dump(
                        by_alias=True
                    )
                ],
            }
        ),
        ParticipantResponse(
            **{  # type: ignore[arg-type]
                "id": "PA_CUSTOMER",
                "accountId": "ACtest123",
                "conversationId": conv_id,
                "name": "+12345678901",
                "type": "CUSTOMER",
                "addresses": [
                    ParticipantAddress(channel="SMS", address="+12345678901").model_dump(
                        by_alias=True
                    )
                ],
            }
        ),
    ]


def create_communication_created_webhook(
    conversation_id: str,
    participant_id: str,
    message_text: str,
    timestamp: str,
    author_address: str = "+12345678901",
) -> dict[str, Any]:
    comm_id = f"comms_communication_{timestamp.replace(':', '').replace('.', '').replace('-', '')}"
    return {
        "eventType": "COMMUNICATION_CREATED",
        "timestamp": timestamp,
        "data": {
            "id": comm_id,
            "conversationId": conversation_id,
            "accountId": "ACtest123",
            "serviceId": "IStest123",
            "author": {
                "address": author_address,
                "channel": "SMS",
                "participantId": participant_id,
            },
            "content": {"type": "TEXT", "text": message_text},
            "channelId": None,
            "recipients": [
                {
                    "address": "+15551234567",
                    "channel": "SMS",
                    "participantId": "comms_participant_agent",
                    "deliveryStatus": "DELIVERED",
                }
            ],
            "createdAt": timestamp,
            "updatedAt": timestamp,
        },
    }


def create_conversation_updated_webhook(
    conversation_id: str, status: str, timestamp: str
) -> dict[str, Any]:
    return {
        "eventType": "CONVERSATION_UPDATED",
        "timestamp": timestamp,
        "data": {
            "id": conversation_id,
            "accountId": "ACtest123",
            "configurationId": "conv_configuration_test123",
            "serviceId": "IStest123",
            "status": status,
            "name": "Test Conversation",
            "createdAt": "2025-11-18T00:00:00.000Z",
            "updatedAt": timestamp,
            "configuration": {"intelligenceServiceIds": []},
        },
    }


def get_test_config(with_memory=True):
    config = {
        "account_sid": "ACtest123",
        "auth_token": "test_token_123",
        "api_key": "SK123",
        "api_secret": "test_api_token",
        "conversation_configuration_id": "conv_configuration_test123",
        "phone_number": "+15551234567",
    }
    if with_memory:
        config["memory_config"] = TwilioMemoryConfig(trait_groups=["Contact"])
    return config


def create_memory_client(tac: TAC) -> MemoryClient:
    return MemoryClient(
        store_id="MGtest123",
        api_key=tac.config.api_key,
        api_secret=tac.config.api_secret,
    )


class TestTACIntegration:
    """Integration tests for complete TAC workflow."""

    def test_configuration_validation_workflow(self):
        """Test complete workflow with configuration validation."""
        valid_configs = [
            get_test_config(),
            TACConfig(**get_test_config()),
        ]

        for config in valid_configs:
            tac = TAC(config)
            assert tac.config.auth_token == "test_token_123"

        flexible_config = get_test_config().copy()
        flexible_config["extra_field"] = "extra_value"
        tac = TAC(flexible_config)
        assert tac.config.auth_token == "test_token_123"

        for invalid_config in ["not_a_dict_or_config", 123]:
            with pytest.raises((ValueError, TypeError)):
                TAC(invalid_config)

    @pytest.mark.asyncio
    async def test_sms_channel_end_to_end_workflow(self) -> None:
        """Full inbound SMS flow: reconciliation + memory retrieval + callback."""
        tac = TAC(get_test_config())
        tac.conversation_memory_client = create_memory_client(tac)
        channel = SMSChannel(tac, config={"auto_retrieve_memory": True})

        callback_invoked = False
        received_context: ConversationSession | None = None
        received_memories: TACMemoryResponse | None = None

        def cb(
            user_message: str,
            context: ConversationSession,
            memory_response: TACMemoryResponse | None = None,
        ) -> None:
            nonlocal callback_invoked, received_context, received_memories
            callback_invoked = True
            received_context = context
            received_memories = memory_response

        tac.on_message_ready(cb)

        from tac.models.memory import ProfileLookupResponse

        tac.conversation_memory_client.lookup_profile = AsyncMock(
            return_value=ProfileLookupResponse(
                normalizedValue="+12345678901", profiles=["profile_test_123"]
            )
        )
        tac.conversation_memory_client.get_profile = AsyncMock(
            side_effect=Exception("skip profile")
        )

        empty_response = MemoryRetrievalResponse(
            observations=[],
            summaries=[],
            meta=MemoryRetrievalMeta(queryTime=0),
        )
        tac.conversation_memory_client.retrieve_memory = AsyncMock(return_value=empty_response)

        message_webhook = create_communication_created_webhook(
            "CH123456", "MB123", "Hello, I need help", "2025-11-18T00:00:02.000Z"
        )

        with patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            return_value=make_sms_participants(),
        ):
            await channel.process_webhook(message_webhook)

        tac.conversation_memory_client.retrieve_memory.assert_called_once()
        assert callback_invoked
        assert received_context is not None
        assert received_context.conversation_id == "CH123456"
        assert received_context.channel == "sms"
        assert received_context.profile_id == "profile_test_123"
        assert isinstance(received_memories, TACMemoryResponse)

    @pytest.mark.asyncio
    async def test_sms_channel_auto_initialize_conversation(self) -> None:
        """Session is auto-initialized on first inbound message."""
        tac = TAC(get_test_config())
        tac.conversation_memory_client = create_memory_client(tac)
        channel = SMSChannel(tac)

        callback_invoked = False

        def cb(
            user_message: str,
            context: ConversationSession,
            memory_response: TACMemoryResponse | None = None,
        ) -> None:
            nonlocal callback_invoked
            callback_invoked = True

        tac.on_message_ready(cb)

        empty_response = MemoryRetrievalResponse(
            observations=[],
            summaries=[],
            meta=MemoryRetrievalMeta(queryTime=0),
        )
        tac.conversation_memory_client.retrieve_memory = AsyncMock(return_value=empty_response)

        message_webhook = create_communication_created_webhook(
            "CH999999",
            "MB999",
            "First message",
            "2025-11-18T00:00:00.000Z",
            author_address="+19999999999",
        )

        with patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            return_value=make_sms_participants(conv_id="CH999999"),
        ):
            await channel.process_webhook(message_webhook)

        assert "CH999999" in channel._conversations
        assert callback_invoked

    @pytest.mark.asyncio
    async def test_sms_channel_filters_empty_messages(self) -> None:
        tac = TAC(get_test_config())
        tac.conversation_memory_client = create_memory_client(tac)
        channel = SMSChannel(tac)

        callback_invoked = False

        def cb(
            user_message: str,
            context: ConversationSession,
            memory_response: TACMemoryResponse | None = None,
        ) -> None:
            nonlocal callback_invoked
            callback_invoked = True

        tac.on_message_ready(cb)

        empty_message = create_communication_created_webhook(
            "CH111", "MB111", "", "2025-11-18T00:00:01.000Z", author_address="+11111111111"
        )

        tac.conversation_memory_client.retrieve_memory = AsyncMock()
        await channel.process_webhook(empty_message)
        tac.conversation_memory_client.retrieve_memory.assert_not_called()
        assert not callback_invoked

        whitespace_message = create_communication_created_webhook(
            "CH111",
            "MB111",
            "   \n\t   ",
            "2025-11-18T00:00:02.000Z",
            author_address="+11111111111",
        )
        await channel.process_webhook(whitespace_message)
        tac.conversation_memory_client.retrieve_memory.assert_not_called()
        assert not callback_invoked

    @pytest.mark.asyncio
    async def test_sms_channel_conversation_cleanup(self) -> None:
        """CONVERSATION_UPDATED closed event cleans up local session."""
        tac = TAC(get_test_config())
        channel = SMSChannel(tac)

        channel._conversations["CH222"] = ConversationSession(
            conversation_id="CH222", channel="sms"
        )

        await channel.process_webhook(
            create_conversation_updated_webhook("CH222", "CLOSED", "2025-11-18T00:10:00.000Z")
        )

        assert "CH222" not in channel._conversations

    @pytest.mark.asyncio
    async def test_sms_channel_multiple_concurrent_conversations(self) -> None:
        tac = TAC(get_test_config())
        tac.conversation_memory_client = create_memory_client(tac)
        channel = SMSChannel(tac)

        callback_count = 0
        conversation_ids: set[str] = set()

        def cb(
            user_message: str,
            context: ConversationSession,
            memory_response: TACMemoryResponse | None = None,
        ) -> None:
            nonlocal callback_count
            callback_count += 1
            conversation_ids.add(context.conversation_id)

        tac.on_message_ready(cb)

        empty_response = MemoryRetrievalResponse(
            observations=[],
            summaries=[],
            meta=MemoryRetrievalMeta(queryTime=0),
        )
        tac.conversation_memory_client.retrieve_memory = AsyncMock(return_value=empty_response)

        for i in range(3):
            conv_id = f"CH{i:06d}"
            with patch.object(
                tac.conversation_orchestrator_client,
                "list_participants",
                return_value=make_sms_participants(conv_id=conv_id),
            ):
                await channel.process_webhook(
                    create_communication_created_webhook(
                        conv_id,
                        f"MB{i:06d}",
                        f"Message {i}",
                        f"2025-11-18T00:01:{i:02d}.000Z",
                        author_address=f"+1{i:010d}",
                    )
                )

        assert callback_count == 3
        assert len(conversation_ids) == 3

    @pytest.mark.asyncio
    async def test_sms_channel_real_world_webhook_scenario(self) -> None:
        """Full-size realistic webhook payload processes correctly."""
        tac = TAC(get_test_config())
        tac.conversation_memory_client = create_memory_client(tac)
        channel = SMSChannel(tac)

        callback_invoked = False
        received_context: ConversationSession | None = None

        def cb(
            user_message: str,
            context: ConversationSession,
            memory_response: TACMemoryResponse | None = None,
        ) -> None:
            nonlocal callback_invoked, received_context
            callback_invoked = True
            received_context = context

        tac.on_message_ready(cb)

        empty_response = MemoryRetrievalResponse(
            observations=[],
            summaries=[],
            meta=MemoryRetrievalMeta(queryTime=0),
        )
        tac.conversation_memory_client.retrieve_memory = AsyncMock(return_value=empty_response)

        conv_id = "CHd151e6bcbe3643979a3f41f6d0da3b24"
        real_webhook = create_communication_created_webhook(
            conv_id,
            "MB723da60623f74438acee5baafbd438f0",
            "Hi, I need help resetting my password.",
            "2025-09-17T22:23:11.350Z",
            author_address="+12162622233",
        )

        participants = [
            ParticipantResponse(
                **{  # type: ignore[arg-type]
                    "id": "PA_AGENT",
                    "accountId": "ACtest123",
                    "conversationId": conv_id,
                    "name": "+15551234567",
                    "type": "AI_AGENT",
                    "addresses": [
                        ParticipantAddress(channel="SMS", address="+15551234567").model_dump(
                            by_alias=True
                        )
                    ],
                }
            ),
            ParticipantResponse(
                **{  # type: ignore[arg-type]
                    "id": "PA_CUSTOMER",
                    "accountId": "ACtest123",
                    "conversationId": conv_id,
                    "name": "+12162622233",
                    "type": "CUSTOMER",
                    "addresses": [
                        ParticipantAddress(channel="SMS", address="+12162622233").model_dump(
                            by_alias=True
                        )
                    ],
                }
            ),
        ]

        with patch.object(
            tac.conversation_orchestrator_client,
            "list_participants",
            return_value=participants,
        ):
            await channel.process_webhook(real_webhook)

        assert callback_invoked
        assert received_context is not None
        assert received_context.conversation_id == conv_id
        assert received_context.channel == "sms"

    @pytest.mark.asyncio
    async def test_sms_channel_posts_agent_on_solo_customer_inbound(self) -> None:
        """v1-bridge inbound: only customer exists → POST AI_AGENT, then deliver."""
        tac = TAC(get_test_config())
        tac.conversation_memory_client = create_memory_client(tac)
        channel = SMSChannel(tac)

        callback_invoked = False

        def cb(
            user_message: str,
            context: ConversationSession,
            memory_response: TACMemoryResponse | None = None,
        ) -> None:
            nonlocal callback_invoked
            callback_invoked = True

        tac.on_message_ready(cb)

        empty_response = MemoryRetrievalResponse(
            observations=[],
            summaries=[],
            meta=MemoryRetrievalMeta(queryTime=0),
        )
        tac.conversation_memory_client.retrieve_memory = AsyncMock(return_value=empty_response)

        only_customer = [
            ParticipantResponse(
                **{  # type: ignore[arg-type]
                    "id": "PA_CUSTOMER",
                    "accountId": "ACtest123",
                    "conversationId": "CH_SOLO",
                    "name": "+12345678901",
                    "type": "CUSTOMER",
                    "addresses": [
                        ParticipantAddress(channel="SMS", address="+12345678901").model_dump(
                            by_alias=True
                        )
                    ],
                }
            ),
        ]
        created_agent = ParticipantResponse(
            **{  # type: ignore[arg-type]
                "id": "PA_AGENT_NEW",
                "accountId": "ACtest123",
                "conversationId": "CH_SOLO",
                "name": "+15551234567",
                "type": "AI_AGENT",
                "addresses": [
                    ParticipantAddress(channel="SMS", address="+15551234567").model_dump(
                        by_alias=True
                    )
                ],
            }
        )

        message_webhook = create_communication_created_webhook(
            "CH_SOLO", "MB123", "hey help plz", "2025-11-18T00:00:00.000Z"
        )

        with (
            patch.object(
                tac.conversation_orchestrator_client,
                "list_participants",
                return_value=only_customer,
            ),
            patch.object(
                tac.conversation_orchestrator_client,
                "add_participant",
                new=AsyncMock(return_value=created_agent),
            ) as mock_add,
        ):
            await channel.process_webhook(message_webhook)

        mock_add.assert_called_once()
        assert mock_add.call_args.kwargs["participant_type"] == "AI_AGENT"
        assert callback_invoked
