from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from dr_wandb.constants import Base, RunId
from dr_wandb.utils import extract_as_datetime

type HistoryEntry = dict[str, Any]


class HistoryEntryRecord(Base):
    __tablename__ = "wandb_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str]
    step: Mapped[int | None]
    timestamp: Mapped[datetime | None]
    runtime: Mapped[int | None]
    wandb_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB)

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

    @classmethod
    def standard_fields(cls) -> list[str]:
        return [
            col.name
            for col in cls.__table__.columns
            if col.name not in ["wandb_metadata", "metrics"]
        ]

    def to_dict(self, include_metadata: bool = False) -> dict[str, Any]:
        return {
            **{field: getattr(self, field) for field in self.standard_fields()},
            **self.metrics,
            **({"wandb_metadata": self.wandb_metadata} if include_metadata else {}),
        }


def build_history_query(
    run_ids: list[RunId] | None = None,
) -> Select[HistoryEntryRecord]:
    query = select(HistoryEntryRecord)
    if run_ids is not None:
        query = query.where(HistoryEntryRecord.run_id.in_(run_ids))
    return query
