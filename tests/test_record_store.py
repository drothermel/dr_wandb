from __future__ import annotations

from pathlib import Path
from typing import Any

from dr_wandb import ExportEngine, ExportMode, ExportRequest, RecordStore
from dr_wandb.export import engine as engine_module

from tests.helpers import FakeApi, metadata_run


def test_parquet_record_store_restores_raw_run_dict(
    tmp_path: Path, monkeypatch: Any
) -> None:
    runs = [
        metadata_run(
            "run-1",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:10:00+00:00",
            state="finished",
        )
    ]
    monkeypatch.setattr(
        engine_module,
        "_build_default_api",
        lambda timeout_seconds: FakeApi(runs),
    )
    engine = ExportEngine()
    engine.export(
        ExportRequest(
            entity="ml-moe",
            project="moe",
            name="moe_runs",
            data_root=tmp_path,
            mode=ExportMode.METADATA,
            output_format="parquet",
        )
    )

    store = RecordStore.from_name_and_root("moe_runs", Path(tmp_path))
    snapshots = store.load_run_snapshots()
    assert snapshots[0].raw_run["summaryMetrics"]["loss"] == 1.23
