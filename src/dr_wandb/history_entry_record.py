from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel

type HistoryEntry = dict[str, Any]

SPECIAL_KEY_MAP = {
    "step": "_step",
    "timestamp": "_timestamp",
    "runtime": "_runtime",
    "wandb_metadata": "_wandb",
}


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
            metrics={k: v for k, v in history_entry.items() if not k.startswith("_")},
            **{k: v for k, v in history_entry.items() if k in SPECIAL_KEY_MAP},
        )
