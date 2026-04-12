from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import srsly
from dr_ds.atomic_io import atomic_write_jsonl, dump_json_atomic
from pydantic import BaseModel, computed_field

from dr_wandb.config import ExportMode, ExportRequest, NonBlankStr, SyncMode
from dr_wandb.history import HistoryRow
from dr_wandb.results import ExportManifest
from dr_wandb.state import ExportState
from dr_wandb.wandb_run import RunSnapshot


class ExportStore(BaseModel):
    """Owns the on-disk layout for one named export and all of its file I/O."""

    name: NonBlankStr
    data_root: Path

    @computed_field
    @property
    def export_dir(self) -> Path:
        return self.data_root / self.name

    @property
    def manifest_path(self) -> Path:
        return self.export_dir / "manifest.json"

    @property
    def state_path(self) -> Path:
        return self.export_dir / "state.json"

    @property
    def runs_path(self) -> Path:
        return self.export_dir / "runs.jsonl"

    @property
    def history_path(self) -> Path:
        return self.export_dir / "history.jsonl"

    def load_manifest(self) -> ExportManifest | None:
        if not self.manifest_path.exists():
            return None
        return ExportManifest.model_validate(
            srsly.read_json(self.manifest_path)
        )

    def require_manifest(self) -> ExportManifest:
        manifest = self.load_manifest()
        if manifest is None:
            raise FileNotFoundError(
                f"Missing manifest for export {self.name!r} "
                f"at {self.manifest_path}"
            )
        return manifest

    def save_manifest(self, manifest: ExportManifest) -> None:
        dump_json_atomic(
            self.manifest_path, manifest.model_dump(mode="python")
        )

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

    def load_run_snapshots(self) -> list[RunSnapshot]:
        self.require_manifest()
        records = self._read_jsonl(self.runs_path)
        snapshots = [RunSnapshot.model_validate(record) for record in records]
        return sorted(
            snapshots,
            key=lambda snapshot: snapshot.sort_key,
            reverse=True,
        )

    def load_existing_snapshots(
        self,
        *,
        request: ExportRequest,
        manifest: ExportManifest | None,
    ) -> dict[str, RunSnapshot]:
        if request.sync_mode == SyncMode.FULL_RECONCILE or manifest is None:
            return {}
        records = self._read_jsonl(self.runs_path)
        return {
            snapshot.run.run_id: snapshot
            for snapshot in (
                RunSnapshot.model_validate(record) for record in records
            )
        }

    def write_run_snapshots(self, snapshots: list[RunSnapshot]) -> Path:
        path = self.runs_path
        atomic_write_jsonl(
            path,
            [snapshot.model_dump(mode="python") for snapshot in snapshots],
        )
        return path

    def iter_history_rows(self) -> Iterator[HistoryRow]:
        manifest = self.require_manifest()
        if manifest.history_path is None:
            return iter(())
        records = self._read_jsonl(self.history_path)
        return iter(HistoryRow.model_validate(row) for row in records)

    def load_existing_history_rows(
        self,
        *,
        request: ExportRequest,
        manifest: ExportManifest | None,
    ) -> list[HistoryRow]:
        if (
            request.sync_mode == SyncMode.FULL_RECONCILE
            or request.mode != ExportMode.HISTORY
            or manifest is None
            or manifest.history_path is None
        ):
            return []
        records = self._read_jsonl(self.history_path)
        return [HistoryRow.model_validate(record) for record in records]

    def write_history_rows(self, rows: list[HistoryRow]) -> Path:
        path = self.history_path
        atomic_write_jsonl(
            path, [row.model_dump(mode="python") for row in rows]
        )
        return path

    def remove_history(self) -> None:
        if self.history_path.exists():
            self.history_path.unlink()

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [
            cast(dict[str, Any], record) for record in srsly.read_jsonl(path)
        ]


def load_manifest(name: str, data_root: Path) -> ExportManifest | None:
    return ExportStore(name=name, data_root=data_root).load_manifest()


def load_run_snapshots(name: str, data_root: Path) -> list[RunSnapshot]:
    return ExportStore(name=name, data_root=data_root).load_run_snapshots()


def iter_history_rows(name: str, data_root: Path) -> Iterator[HistoryRow]:
    return ExportStore(name=name, data_root=data_root).iter_history_rows()
