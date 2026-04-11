from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ExportMode(StrEnum):
    METADATA = "metadata"
    HISTORY = "history"


class FetchMode(StrEnum):
    INCREMENTAL = "incremental"
    FULL_RECONCILE = "full_reconcile"


class HistoryWindow(BaseModel):
    min_step: int | None = None
    max_step: int | None = None
    max_records: int | None = None


class HistoryPolicyContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity: str
    project: str
    run_id: str
    run_name: str
    run_state: str | None
    run_updated_at: str | None
    run_last_history_step: int | None = None
    run: Any


class RunTrackingState(BaseModel):
    run_id: str
    created_at: str | None = None
    updated_at: str | None = None
    run_state: str | None = None
    last_history_step: int | None = None


class ExportState(BaseModel):
    schema_version: int = 1
    name: str
    entity: str
    project: str
    last_exported_at: str | None = None
    max_created_at: str | None = None
    runs: dict[str, RunTrackingState] = Field(default_factory=dict)


class RunSnapshot(BaseModel):
    run_id: str
    entity: str
    project: str
    exported_at: str
    raw_run: dict[str, Any] = Field(default_factory=dict)


class HistoryRow(BaseModel):
    run_id: str
    step: int | None = None
    timestamp: str | None = None
    runtime: int | float | None = None
    wandb_metadata: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


class ExportManifest(BaseModel):
    schema_version: int = 1
    name: str
    entity: str
    project: str
    mode: ExportMode
    output_format: Literal["jsonl", "parquet"]
    created_at: str
    updated_at: str
    runs_path: str
    history_path: str | None = None
    run_count: int = 0
    history_count: int = 0
    selected_history_keys: list[str] | None = None
    history_window: HistoryWindow | None = None


class ExportRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity: str
    project: str
    name: str
    data_root: Path = Path("./data")
    mode: ExportMode = ExportMode.METADATA
    output_format: Literal["jsonl", "parquet"] = "jsonl"
    fetch_mode: FetchMode = FetchMode.INCREMENTAL
    runs_per_page: int = 500
    timeout_seconds: int = 120
    include_metadata: bool = False
    history_policy: Any = None


class ExportSummary(BaseModel):
    name: str
    entity: str
    project: str
    mode: ExportMode
    fetch_mode: FetchMode
    output_dir: str
    state_path: str
    manifest_path: str
    runs_path: str
    history_path: str | None = None
    run_count: int = 0
    history_count: int = 0
    exported_at: str
