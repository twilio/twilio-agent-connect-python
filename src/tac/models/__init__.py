"""Shared models for the Twilio Agent Connect."""

from tac.models.conversation import (
    ActionChannelSettings,
    ActionParticipantRef,
    ActionResponse,
    ActionTextContent,
    Communication,
    CommunicationContent,
    CommunicationParticipant,
    CommunicationRequest,
    ConversationRequest,
    ConversationResponse,
    ParticipantAddress,
    ParticipantRequest,
    ParticipantResponse,
    SendMessageActionPayload,
    SendMessageActionRequest,
)
from tac.models.handoff import HandoffPayload
from tac.models.intelligence import (
    CommunicationsRange,
    ExecutionDetails,
    IntelligenceConfiguration,
    Operator,
    OperatorResultEvent,
    Participant,
    TriggerDetails,
)
from tac.models.knowledge import Knowledge, KnowledgeBase, KnowledgeChunkResult
from tac.models.memory import (
    MemoryCommunication,
    MemoryCommunicationContent,
    MemoryMode,
    MemoryParticipant,
    MemoryRetrievalRequest,
    MemoryRetrievalResponse,
    ObservationInfo,
    ProfileResponse,
    SummaryInfo,
)
from tac.models.outbound import (
    InitiateChatConversationOptions,
    InitiateConversationResult,
    InitiateMessagingConversationOptions,
    InitiateVoiceConversationOptions,
    InitiateVoiceConversationResult,
)
from tac.models.pagination import PaginationMeta
from tac.models.session import AuthorInfo, ConversationSession
from tac.models.tac import (
    TACCommunication,
    TACCommunicationAuthor,
    TACCommunicationContent,
    TACMemoryResponse,
)
from tac.models.voice import TwiMLOptions

# Rebuild ConversationSession after importing TACMemoryResponse so Pydantic can
# resolve the forward reference used by ConversationSession. The session model
# only imports TACMemoryResponse under TYPE_CHECKING, so that name is not
# available when ConversationSession is first defined. Doing the rebuild here
# works because __init__.py imports both ConversationSession and
# TACMemoryResponse before calling model_rebuild(), satisfying the import-order
# requirement for forward-ref resolution.
ConversationSession.model_rebuild()

__all__ = [
    "ActionChannelSettings",
    "ActionParticipantRef",
    "ActionResponse",
    "ActionTextContent",
    "AuthorInfo",
    "Communication",
    "CommunicationContent",
    "CommunicationParticipant",
    "CommunicationRequest",
    "CommunicationsRange",
    "ConversationRequest",
    "ConversationResponse",
    "ConversationSession",
    "ExecutionDetails",
    "HandoffPayload",
    "InitiateChatConversationOptions",
    "InitiateConversationResult",
    "InitiateMessagingConversationOptions",
    "InitiateVoiceConversationOptions",
    "InitiateVoiceConversationResult",
    "IntelligenceConfiguration",
    "Knowledge",
    "KnowledgeBase",
    "KnowledgeChunkResult",
    "MemoryCommunication",
    "MemoryCommunicationContent",
    "MemoryMode",
    "MemoryParticipant",
    "MemoryRetrievalRequest",
    "MemoryRetrievalResponse",
    "ObservationInfo",
    "Operator",
    "OperatorResultEvent",
    "PaginationMeta",
    "Participant",
    "ParticipantAddress",
    "ParticipantRequest",
    "ParticipantResponse",
    "ProfileResponse",
    "SendMessageActionPayload",
    "SendMessageActionRequest",
    "SummaryInfo",
    "TACCommunication",
    "TACCommunicationAuthor",
    "TACCommunicationContent",
    "TACMemoryResponse",
    "TriggerDetails",
    "TwiMLOptions",
]
