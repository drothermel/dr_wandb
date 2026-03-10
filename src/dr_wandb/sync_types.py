from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ErrorAction(StrEnum):
    SKIP = "skip"
    RETRY = "retry"
    ABORT = "abort"


class HistoryWindow(BaseModel):
    min_step: int | None = None
    max_step: int | None = None
    max_records: int | None = None


class RunDecision(BaseModel):
    status: str = "unknown"
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PatchPlan(BaseModel):
    set_config: dict[str, Any] = Field(default_factory=dict)
    add_tags: list[str] = Field(default_factory=list)
    remove_tags: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.set_config and not self.add_tags and not self.remove_tags


class RunCursor(BaseModel):
    run_id: str
    updated_at: str | None = None
    last_step: int | None = None
    history_seen: int = 0
    decision_status: str | None = None
    decision_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectSyncState(BaseModel):
    schema_version: int = 1
    entity: str = ""
    project: str = ""
    last_synced_at: str | None = None
    runs: dict[str, RunCursor] = Field(default_factory=dict)


class SyncContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity: str
    project: str
    run_id: str
    run_name: str
    run_state: str | None
    run_updated_at: str | None
    run: Any
    cursor: RunCursor | None = None


class RunEvaluation(BaseModel):
    run_id: str
    run_name: str
    run_state: str | None
    history_records: int
    decision: RunDecision
    patch_planned: bool


class PlannedPatch(BaseModel):
    entity: str
    project: str
    run_id: str
    run_name: str
    patch: PatchPlan


class ApplyResult(BaseModel):
    run_id: str
    run_name: str
    dry_run: bool
    changed: bool
    applied: bool
    changed_config_keys: list[str] = Field(default_factory=list)
    added_tags: list[str] = Field(default_factory=list)
    removed_tags: list[str] = Field(default_factory=list)


class SyncSummary(BaseModel):
    entity: str
    project: str
    state_path: str
    processed_runs: int
    planned_patches: int
    run_evaluations: list[RunEvaluation] = Field(default_factory=list)


class ExportConfig(BaseModel):
    entity: str
    project: str
    output_dir: Path
    output_format: Literal["parquet", "jsonl"] = "parquet"
    runs_per_page: int = 500
    state_path: Path | None = None
    save_every: int = 25
    incremental: bool = True
    checkpoint_every_runs: int = 25
    checkpoint_dirname: str = "_checkpoints"
    finalize_compact: bool = True
    inspection_sample_rows: int = 5
    policy_module: str = "dr_wandb.sync_policy"
    policy_class: str = "NoopPolicy"


class CheckpointRecord(BaseModel):
    checkpoint_id: int
    created_at: str
    run_rows: int
    history_rows: int
    cumulative_run_rows: int
    cumulative_history_rows: int
    run_start_index: int
    run_end_index: int
    run_ids: list[str] = Field(default_factory=list)
    run_names: list[str] = Field(default_factory=list)
    metric_keys_sample: list[str] = Field(default_factory=list)
    step_min: int | None = None
    step_max: int | None = None
    runs_file: str
    history_file: str
    record_file: str
    state_hash: str


class CheckpointManifest(BaseModel):
    schema_version: int = 1
    entity: str
    project: str
    output_format: Literal["parquet", "jsonl"] = "parquet"
    policy_module: str = ""
    policy_class: str = ""
    created_at: str
    updated_at: str
    status: Literal["in_progress", "completed", "completed_no_compact"] = "in_progress"
    checkpoint_dir: str
    runs_dir: str
    history_dir: str
    inspection_path: str
    total_run_rows: int = 0
    total_history_rows: int = 0
    last_checkpoint_id: int = 0
    checkpoints: list[CheckpointRecord] = Field(default_factory=list)


class ExportSummary(BaseModel):
    entity: str
    project: str
    state_path: str
    output_format: Literal["parquet", "jsonl"]
    runs_output_path: str
    history_output_path: str
    manifest_output_path: str
    run_count: int
    history_count: int
    exported_at: str
    checkpoint_count: int = 0
    checkpoint_manifest_path: str = ""
    finalized: bool = True
    partial_run_count: int = 0
    partial_history_count: int = 0
