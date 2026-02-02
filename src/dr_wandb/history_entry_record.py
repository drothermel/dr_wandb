from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from dr_wandb.utils import extract_as_datetime

type HistoryEntry = dict[str, Any]


class HistoryEntryRecord(BaseModel):
    run_id: str
    step: int | None
    timestamp: datetime | None
    runtime: int | None
    wandb_metadata: dict[str, Any]
    metrics: dict[str, Any]

    @classmethod
    def from_wandb_history(
        cls, history_entry: HistoryEntry, run_id: str
    ) -> HistoryEntryRecord:
        return cls(
            run_id=run_id,
            step=history_entry.get("_step"),
            timestamp=extract_as_datetime(history_entry, "_timestamp"),
            runtime=history_entry.get("_runtime"),
            wandb_metadata=history_entry.get("_wandb", {}),
            metrics={k: v for k, v in history_entry.items() if not k.startswith("_")},
        )
