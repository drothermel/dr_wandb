from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from dr_ds.atomic_io import atomic_write_jsonl, atomic_write_parquet_records
from dr_ds.parquet import parquet_frame_to_records
import pandas as pd
import srsly

RUN_SNAPSHOT_JSON_COLUMNS = {"raw_run"}
HISTORY_ROW_JSON_COLUMNS = {"wandb_metadata", "metrics", "extra"}


def read_records(
    path: Path, *, json_columns: set[str]
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        return [
            cast(dict[str, Any], record) for record in srsly.read_jsonl(path)
        ]
    frame = pd.read_parquet(path)
    return parquet_frame_to_records(frame, json_columns=json_columns)


def write_records(
    path: Path,
    records: list[dict[str, Any]],
    *,
    json_columns: set[str],
) -> None:
    if path.suffix == ".jsonl":
        atomic_write_jsonl(path, records)
        return

    atomic_write_parquet_records(path, records, json_columns=json_columns)


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()
