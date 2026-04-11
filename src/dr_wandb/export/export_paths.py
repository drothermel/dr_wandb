from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, computed_field


class ExportPaths(BaseModel):
    manifest_filename: ClassVar[str] = "manifest.json"
    state_filename: ClassVar[str] = "state.json"

    name: str
    data_root: Path

    @classmethod
    def from_name_and_root(cls, name: str, data_root: Path) -> ExportPaths:
        return cls(
            name=name,
            data_root=Path(data_root),
        )

    @computed_field
    @property
    def export_dir(self) -> Path:
        return self.data_root / self.name

    @computed_field
    @property
    def manifest_path(self) -> Path:
        return self.export_dir / self.manifest_filename

    @computed_field
    @property
    def state_path(self) -> Path:
        return self.export_dir / self.state_filename

    def runs_path(self, output_format: str) -> Path:
        return self.export_dir / f"runs.{output_format}"

    def history_path(self, output_format: str) -> Path:
        return self.export_dir / f"history.{output_format}"
