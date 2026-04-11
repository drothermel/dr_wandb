from __future__ import annotations

from pathlib import Path
from typing import Any

import wandb

from dr_wandb import ExportEngine, ExportMode, ExportRequest, RecordStore

from tests.helpers import FakeApi, metadata_run


def test_record_store_restores_wandb_run(
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
    monkeypatch.setattr(wandb, "Api", lambda **kwargs: FakeApi(runs))
    engine = ExportEngine()
    engine.export(
        ExportRequest(
            entity="ml-moe",
            project="moe",
            name="moe_runs",
            data_root=tmp_path,
            mode=ExportMode.METADATA,
        )
    )

    store = RecordStore.from_name_and_root("moe_runs", Path(tmp_path))
    snapshots = store.load_run_snapshots()
    assert snapshots[0].run.summary_metrics["loss"] == 1.23
