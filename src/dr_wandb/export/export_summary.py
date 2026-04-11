from __future__ import annotations

from pydantic import BaseModel

from dr_wandb.export.export_modes import ExportMode, FetchMode


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
