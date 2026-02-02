from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


def _extract_timestamp(raw: float | None) -> datetime | None:
    return datetime.fromtimestamp(raw) if raw is not None else None


class HistoryEntryRecord(BaseModel):
    run_id: str
    step: int | None = None
    timestamp: datetime | None = None
    runtime: int | None = None
    wandb_metadata: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_wandb_history(
        cls, history_entry: dict[str, Any], run_id: str
    ) -> HistoryEntryRecord:
        return cls(
            run_id=run_id,
            step=history_entry.get("_step"),
            timestamp=_extract_timestamp(history_entry.get("_timestamp")),
            runtime=history_entry.get("_runtime"),
            wandb_metadata=history_entry.get("_wandb", {}),
            metrics={k: v for k, v in history_entry.items() if not k.startswith("_")},
        )
