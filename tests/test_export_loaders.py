from __future__ import annotations

from pathlib import Path

from dr_wandb import ExportEngine, ExportMode, ExportRequest, load_run_snapshot_dicts

from tests.helpers import FakeApi, metadata_run


def test_parquet_loader_restores_raw_run_dict(tmp_path: Path) -> None:
    runs = [
        metadata_run(
            "run-1",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:10:00+00:00",
            state="finished",
        )
    ]
    engine = ExportEngine(api_factory=lambda timeout: FakeApi(runs))
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

    payloads = load_run_snapshot_dicts("moe_runs", Path(tmp_path))
    assert payloads[0]["raw_run"]["summaryMetrics"]["loss"] == 1.23
