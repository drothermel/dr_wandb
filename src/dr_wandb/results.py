from __future__ import annotations

from pydantic import BaseModel

from dr_wandb.config import ExportMode, HistoryWindow, SyncMode


class ExportManifest(BaseModel):
    schema_version: int = 1
    name: str
    entity: str
    project: str
    mode: ExportMode
    created_at: str
    updated_at: str
    runs_path: str
    history_path: str | None = None
    run_count: int = 0
    history_count: int = 0
    selected_history_keys: list[str] | None = None
    history_window: HistoryWindow | None = None


class ExportSummary(BaseModel):
    name: str
    entity: str
    project: str
    mode: ExportMode
    sync_mode: SyncMode
    output_dir: str
    state_path: str
    manifest_path: str
    runs_path: str
    history_path: str | None = None
    run_count: int = 0
    history_count: int = 0
    exported_at: str
