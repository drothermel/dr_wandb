from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from dr_wandb.export.export_modes import ExportMode, FetchMode


class ExportRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity: str
    project: str
    name: str
    data_root: Path = Path("./data")
    mode: ExportMode = ExportMode.METADATA
    fetch_mode: FetchMode = FetchMode.INCREMENTAL
    runs_per_page: int = 500
    timeout_seconds: int = 120
    include_metadata: bool = False
    history_policy: Any = None
