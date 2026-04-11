from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel
import srsly

from dr_wandb.export.models import ExportManifest, ExportState
from dr_wandb.shared.parquet import (
    parquet_frame_to_records,
    records_to_parquet_frame,
)
from dr_wandb.shared.serialization import dump_json_atomic, to_jsonable

RUN_SNAPSHOT_JSON_COLUMNS = {"raw_run"}
HISTORY_ROW_JSON_COLUMNS = {"wandb_metadata", "metrics", "extra"}


class ExportPaths(BaseModel):
    name: str
    data_root: Path
    export_dir: Path
    manifest_path: Path
    state_path: Path


def resolve_export_paths(*, name: str, data_root: Path) -> ExportPaths:
    export_dir = Path(data_root) / name
    return ExportPaths(
        name=name,
        data_root=Path(data_root),
        export_dir=export_dir,
        manifest_path=export_dir / "manifest.json",
        state_path=export_dir / "state.json",
    )


def runs_path(paths: ExportPaths, output_format: str) -> Path:
    return paths.export_dir / f"runs.{output_format}"


def history_path(paths: ExportPaths, output_format: str) -> Path:
    return paths.export_dir / f"history.{output_format}"


def load_state(paths: ExportPaths, *, entity: str, project: str) -> ExportState:
    if not paths.state_path.exists():
        return ExportState(name=paths.name, entity=entity, project=project)
    return ExportState.model_validate(srsly.read_json(paths.state_path))


def save_state(paths: ExportPaths, state: ExportState) -> None:
    dump_json_atomic(paths.state_path, state.model_dump(mode="python"))


def load_manifest(paths: ExportPaths) -> ExportManifest | None:
    if not paths.manifest_path.exists():
        return None
    return ExportManifest.model_validate(srsly.read_json(paths.manifest_path))


def save_manifest(paths: ExportPaths, manifest: ExportManifest) -> None:
    dump_json_atomic(paths.manifest_path, manifest.model_dump(mode="python"))


def read_records(path: Path, *, json_columns: set[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        return [cast(dict[str, Any], record) for record in srsly.read_jsonl(path)]
    frame = pd.read_parquet(path)
    return parquet_frame_to_records(frame, json_columns=json_columns)


def write_records(
    path: Path,
    records: list[dict[str, Any]],
    *,
    json_columns: set[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name,
            suffix=".tmp",
            delete=False,
        ) as handle:
            for record in records:
                handle.write(
                    srsly.json_dumps(to_jsonable(record), sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
            tmp_name = handle.name
        os.replace(tmp_name, path)
        return

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    frame = records_to_parquet_frame(records, json_columns=json_columns)
    frame.to_parquet(tmp_path)
    with open(tmp_path, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()
