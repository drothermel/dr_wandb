from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from dr_ds.atomic_io import dump_json_atomic
from pydantic import BaseModel, computed_field
import srsly

from dr_wandb.export.export_manifest import ExportManifest
from dr_wandb.export.export_state import ExportState


class ExportPaths(BaseModel):
    manifest_filename: ClassVar[str] = "manifest.json"
    state_filename: ClassVar[str] = "state.json"

    name: str
    data_root: Path

    @classmethod
    def from_name_and_root(cls, name: str, data_root: Path) -> ExportPaths:
        return cls(name=name, data_root=Path(data_root))

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

    def load_state(self, *, entity: str, project: str) -> ExportState:
        if not self.state_path.exists():
            return ExportState(name=self.name, entity=entity, project=project)
        state = ExportState.model_validate(srsly.read_json(self.state_path))
        if (
            state.name != self.name
            or state.entity != entity
            or state.project != project
        ):
            raise ValueError(
                "State identity does not match requested export: "
                f"expected ({self.name}, {entity}, {project}), "
                f"got ({state.name}, {state.entity}, {state.project})"
            )
        return state

    def save_state(self, state: ExportState) -> None:
        dump_json_atomic(self.state_path, state.model_dump(mode="python"))

    def load_manifest(self) -> ExportManifest | None:
        if not self.manifest_path.exists():
            return None
        return ExportManifest.model_validate(
            srsly.read_json(self.manifest_path)
        )

    def save_manifest(self, manifest: ExportManifest) -> None:
        dump_json_atomic(
            self.manifest_path,
            manifest.model_dump(mode="python"),
        )
