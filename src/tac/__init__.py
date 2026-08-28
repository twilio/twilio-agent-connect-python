from importlib.metadata import version
from typing import TYPE_CHECKING, Any

__version__ = version("twilio-agent-connect")

# isort: off
# ``tac.core`` must load before ``tac.context.base`` to avoid a circular import
# (``tac.context.base`` -> ``tac.core.logging`` -> ``tac.core`` -> clients).
from tac.core import TAC, TACConfig, get_logger
from tac.context.base import PartnerConnector
from tac._deprecation import resolve_deprecated_alias
from tac.models.voice import VoiceTwiMLOptionsConversationRelay
from tac.utils.redaction import mask_address, mask_email, mask_phone

# isort: on

__all__ = [
    "TAC",
    "TACConfig",
    "PartnerConnector",
    "get_logger",
    "TwiMLOptions",
    "mask_address",
    "mask_email",
    "mask_phone",
    "__version__",
]

if TYPE_CHECKING:  # static type only, see tac._deprecation
    TwiMLOptions = VoiceTwiMLOptionsConversationRelay


def __getattr__(name: str) -> Any:
    # See tac._deprecation. TODO(3.0): remove.
    if name == "TwiMLOptions":
        return resolve_deprecated_alias("TwiMLOptions", VoiceTwiMLOptionsConversationRelay)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
