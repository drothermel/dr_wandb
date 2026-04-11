from __future__ import annotations

from pydantic import BaseModel

from dr_wandb.export.export_modes import ExportMode
from dr_wandb.export.policy import HistoryWindow


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
