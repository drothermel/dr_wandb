"""Define configuration models for export mode, sync mode, and request shape."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

NonBlankStr = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class ExportMode(StrEnum):
    """Enumerate the supported export payload shapes."""

    METADATA = "metadata"
    HISTORY = "history"


class SyncMode(StrEnum):
    """Enumerate whether an export reuses prior state or rebuilds from scratch."""

    INCREMENTAL = "incremental"
    FULL_RECONCILE = "full_reconcile"


class HistoryWindow(BaseModel):
    """Describe step-based and record-count bounds for history export."""

    min_step: int | None = None
    max_step: int | None = None
    max_records: int | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> HistoryWindow:
        """Reject invalid history windows after model validation."""
        if self.min_step is not None and self.min_step < 0:
            raise ValueError("min_step must be >= 0")
        if self.max_step is not None and self.max_step < 0:
            raise ValueError("max_step must be >= 0")
        if self.max_records is not None and self.max_records <= 0:
            raise ValueError("max_records must be > 0")
        if (
            self.min_step is not None
            and self.max_step is not None
            and self.min_step > self.max_step
        ):
            raise ValueError("min_step must be <= max_step")
        return self


class HistorySelection(BaseModel):
    """Describe optional key and window filters for history export."""

    keys: list[str] | None = None
    window: HistoryWindow | None = None


class ExportRequest(BaseModel):
    """Capture one export invocation and the choices that shape it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity: NonBlankStr
    project: NonBlankStr
    name: NonBlankStr
    data_root: Path = Path("./data")
    mode: ExportMode = ExportMode.METADATA
    sync_mode: SyncMode = SyncMode.INCREMENTAL
    runs_per_page: int = 500
    timeout_seconds: int = 120
    include_metadata: bool = False
    history_selection: HistorySelection | None = None
