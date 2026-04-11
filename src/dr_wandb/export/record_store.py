from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, cast

from dr_ds.atomic_io import atomic_write_jsonl, atomic_write_parquet_records
from dr_ds.parquet import parquet_frame_to_records
import pandas as pd
from pydantic import BaseModel
import srsly

from dr_wandb.export.export_paths import ExportPaths
from dr_wandb.export.models import ExportManifest, HistoryRow, RunSnapshot


class RecordStore(BaseModel):
    run_snapshot_json_columns: ClassVar[set[str]] = {"raw_run"}
    history_row_json_columns: ClassVar[set[str]] = {
        "wandb_metadata",
        "metrics",
        "extra",
    }

    paths: ExportPaths

    @classmethod
    def from_name_and_root(cls, name: str, data_root: Path) -> RecordStore:
        return cls(
            paths=ExportPaths.from_name_and_root(
                name=name, data_root=data_root
            )
        )

    def load_manifest(self) -> ExportManifest | None:
        return self.paths.load_manifest()

    def require_manifest(self) -> ExportManifest:
        manifest = self.load_manifest()
        assert manifest is not None, (
            f"Missing manifest for export {self.paths.name!r}"
        )
        return manifest

    def load_run_snapshots(self) -> list[RunSnapshot]:
        manifest = self.require_manifest()
        records = self.read_records(
            Path(manifest.runs_path),
            json_columns=self.run_snapshot_json_columns,
        )
        snapshots = [RunSnapshot.model_validate(record) for record in records]
        return sorted(
            snapshots,
            key=lambda snapshot: (
                str(snapshot.raw_run.get("createdAt", "")),
                snapshot.run_id,
            ),
            reverse=True,
        )

    def load_run_snapshot_dicts(self) -> list[dict[str, Any]]:
        return [
            snapshot.model_dump(mode="python")
            for snapshot in self.load_run_snapshots()
        ]

    def iter_history_rows(self) -> Iterator[HistoryRow]:
        manifest = self.require_manifest()
        if manifest.history_path is None:
            return iter(())
        rows = self.read_records(
            Path(manifest.history_path),
            json_columns=self.history_row_json_columns,
        )
        return iter(HistoryRow.model_validate(row) for row in rows)

    def read_records(
        self, path: Path, *, json_columns: set[str]
    ) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        if path.suffix == ".jsonl":
            return [
                cast(dict[str, Any], record)
                for record in srsly.read_jsonl(path)
            ]
        frame = pd.read_parquet(path)
        return parquet_frame_to_records(frame, json_columns=json_columns)

    def write_records(
        self,
        path: Path,
        records: list[dict[str, Any]],
        *,
        json_columns: set[str],
    ) -> None:
        if path.suffix == ".jsonl":
            atomic_write_jsonl(path, records)
            return

        atomic_write_parquet_records(path, records, json_columns=json_columns)

    def remove_if_exists(self, path: Path) -> None:
        if path.exists():
            path.unlink()
