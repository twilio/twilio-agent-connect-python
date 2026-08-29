"""Shared scaffolding for per-provider TwiML builders."""

from __future__ import annotations

from collections.abc import Container
from typing import Any

from pydantic import BaseModel

from tac.core.config import TACConfig


class TwiMLBuilderBase:
    """Common construction and option-layering helpers for a provider's TwiML builder."""

    def __init__(self, tac_config: TACConfig, channel_config: Any) -> None:
        self.tac_config = tac_config
        self.channel_config = channel_config

    @staticmethod
    def _overlay_fields(target: BaseModel, source: BaseModel, *, skip: Container[str] = ()) -> None:
        """Apply fields explicitly set on ``source`` onto ``target``, except those in ``skip``."""
        for field in source.model_fields_set:
            if field in skip:
                continue
            setattr(target, field, getattr(source, field))

    @staticmethod
    def _missing_websocket_url_error(caller: str) -> ValueError:
        return ValueError(
            f"{caller} needs a WebSocket URL. Set TWILIO_VOICE_PUBLIC_DOMAIN "
            "(or TACConfig.voice_public_domain)."
        )

    def _default_websocket_url(self) -> str | None:
        if not self.tac_config.voice_public_domain:
            return None
        return f"wss://{self.tac_config.voice_public_domain}{self.tac_config.voice_websocket_path}"
