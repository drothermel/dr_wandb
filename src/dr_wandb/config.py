"""Export request config: what to export, how to sync, and which history rows to include."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ExportMode(StrEnum):
    METADATA = "metadata"
    HISTORY = "history"


class SyncMode(StrEnum):
    INCREMENTAL = "incremental"
    FULL_RECONCILE = "full_reconcile"


class HistoryWindow(BaseModel):
    min_step: int | None = None
    max_step: int | None = None
    max_records: int | None = None


class HistorySelection(BaseModel):
    keys: list[str] | None = None
    window: HistoryWindow | None = None


class ExportRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity: str
    project: str
    name: str
    data_root: Path = Path("./data")
    mode: ExportMode = ExportMode.METADATA
    sync_mode: SyncMode = SyncMode.INCREMENTAL
    runs_per_page: int = 500
    timeout_seconds: int = 120
    include_metadata: bool = False
    history_selection: HistorySelection | None = None
