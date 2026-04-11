from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dr_wandb.export.export_paths import ExportPaths
from dr_wandb.export.models import ExportManifest, HistoryRow, RunSnapshot
from dr_wandb.export.store import (
    HISTORY_ROW_JSON_COLUMNS,
    RUN_SNAPSHOT_JSON_COLUMNS,
    read_records,
)


def load_manifest(name: str, data_root: Path) -> ExportManifest:
    paths = ExportPaths.from_name_and_root(
        name=name, data_root=Path(data_root)
    )
    manifest = paths.load_manifest()
    assert manifest is not None, f"Missing manifest for export {name!r}"
    return manifest


def load_run_snapshots(name: str, data_root: Path) -> list[RunSnapshot]:
    manifest = load_manifest(name, data_root)
    records = read_records(
        Path(manifest.runs_path),
        json_columns=RUN_SNAPSHOT_JSON_COLUMNS,
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


def load_run_snapshot_dicts(
    name: str, data_root: Path
) -> list[dict[str, Any]]:
    return [
        snapshot.model_dump(mode="python")
        for snapshot in load_run_snapshots(name, data_root)
    ]


def iter_history_rows(name: str, data_root: Path) -> Iterator[HistoryRow]:
    manifest = load_manifest(name, data_root)
    if manifest.history_path is None:
        return iter(())
    rows = read_records(
        Path(manifest.history_path),
        json_columns=HISTORY_ROW_JSON_COLUMNS,
    )
    return iter(HistoryRow.model_validate(row) for row in rows)
