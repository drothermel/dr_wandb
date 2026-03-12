from __future__ import annotations

import json
import logging
from pathlib import Path

from dr_wandb.raw_extract import (
    RawExtractConfig,
    build_raw_run_payload,
    extract_runs_raw,
    hash_raw_run_payload,
    postprocess_dedupe_runs_raw,
)


class FakeRun:
    def __init__(
        self,
        *,
        run_id: str,
        attrs: dict,
        config: dict | None = None,
        summary_metrics: dict | None = None,
        summary: dict | None = None,
        system_metrics: dict | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        history_keys: dict | None = None,
        job_type: str | None = None,
        sweep_name: str | None = None,
        user: dict | None = None,
        read_only: bool | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        heartbeat_at: str | None = None,
        storage_id: str | None = None,
        url: str | None = None,
        path: list[str] | None = None,
        display_name: str | None = None,
        name: str | None = None,
        state: str | None = None,
    ) -> None:
        self.id = run_id
        self._attrs = attrs
        self.config = config
        self.summary_metrics = summary_metrics
        self.summary = summary
        self.system_metrics = system_metrics
        self.metadata = metadata
        self.tags = tags
        self.group = group
        self.history_keys = history_keys
        self.job_type = job_type
        self.sweep_name = sweep_name
        self.user = user
        self.read_only = read_only
        self.created_at = created_at
        self.updated_at = updated_at
        self.heartbeat_at = heartbeat_at
        self.storage_id = storage_id
        self.url = url
        self.path = path
        self.display_name = display_name
        self.name = name
        self.state = state


def test_build_raw_run_payload_excludes_metadata_and_drops_summary_alias():
    run = FakeRun(
        run_id="run-1",
        attrs={
            "rawconfig": {"alpha": 1},
            "summary": {"_step": 99},
            "system_metrics": {"cpu": 1},
            "created_at": "bad-old-name",
            "unknownKey": {"x": 1},
        },
        config={"beta": 2},
        summary_metrics={"_step": 100, "accuracy": 0.5},
        system_metrics={"cpu": 2},
        metadata={"host": "example"},
        tags=["new"],
        group="MoE",
        history_keys={"lastStep": 100},
        job_type="train",
        sweep_name="sweep-a",
        user={"username": "user-a"},
        read_only=False,
        created_at="2026-03-11T00:00:00Z",
        updated_at="2026-03-11T01:00:00Z",
        heartbeat_at="2026-03-11T02:00:00Z",
        storage_id="storage-1",
        url="https://wandb.ai/entity/project/runs/run-1",
        path=["entity", "project", "run-1"],
        display_name="display-run-1",
        name="run-1-name",
        state="finished",
    )

    payload = build_raw_run_payload(run=run, include_metadata=False)

    assert payload["rawconfig"] == {"alpha": 1}
    assert payload["config"] == {"beta": 2}
    assert payload["summaryMetrics"] == {"_step": 100, "accuracy": 0.5}
    assert payload["systemMetrics"] == {"cpu": 2}
    assert payload["createdAt"] == "2026-03-11T00:00:00Z"
    assert payload["group"] == "MoE"
    assert payload["unknownKey"] == {"x": 1}
    assert "metadata" not in payload
    assert "summary" not in payload
    assert "system_metrics" not in payload
    assert "created_at" not in payload


def test_build_raw_run_payload_includes_metadata_when_requested():
    run = FakeRun(
        run_id="run-1",
        attrs={"rawconfig": {"alpha": 1}},
        metadata={"host": "example"},
    )

    payload = build_raw_run_payload(run=run, include_metadata=True)

    assert payload["metadata"] == {"host": "example"}


def test_hash_raw_run_payload_is_stable_for_equivalent_dicts():
    payload_a = {"b": 2, "a": {"z": 1, "y": [3, 2, 1]}}
    payload_b = {"a": {"y": [3, 2, 1], "z": 1}, "b": 2}

    assert hash_raw_run_payload(payload_a) == hash_raw_run_payload(payload_b)


def test_postprocess_dedupe_runs_raw_removes_exact_duplicates_and_preserves_latest_order(
    tmp_path: Path,
):
    input_path = tmp_path / "runs_raw.jsonl"
    temp_output_path = tmp_path / "runs_raw__deduping.jsonl"
    rows = [
        {
            "run_id": "run-1",
            "entity": "entity",
            "project": "project",
            "exported_at": "2026-03-11T00:00:00+00:00",
            "raw_run_hash": "hash-a",
            "raw_run": {"name": "first"},
        },
        {
            "run_id": "run-2",
            "entity": "entity",
            "project": "project",
            "exported_at": "2026-03-11T00:00:01+00:00",
            "raw_run_hash": "hash-b",
            "raw_run": {"name": "middle"},
        },
        {
            "run_id": "run-1",
            "entity": "entity",
            "project": "project",
            "exported_at": "2026-03-11T00:00:02+00:00",
            "raw_run_hash": "hash-a",
            "raw_run": {"name": "latest"},
        },
        {
            "run_id": "run-1",
            "entity": "entity",
            "project": "project",
            "exported_at": "2026-03-11T00:00:03+00:00",
            "raw_run_hash": "hash-c",
            "raw_run": {"name": "changed"},
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    temp_output_path.write_text("stale temp\n", encoding="utf-8")

    final_run_count = postprocess_dedupe_runs_raw(
        input_path=input_path,
        temp_output_path=temp_output_path,
    )

    written_rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert final_run_count == 3
    assert [row["raw_run"]["name"] for row in written_rows] == [
        "middle",
        "latest",
        "changed",
    ]
    assert not temp_output_path.exists()


def test_postprocess_dedupe_runs_raw_removes_one_malformed_trailing_line(
    tmp_path: Path,
    caplog,
):
    input_path = tmp_path / "runs_raw.jsonl"
    temp_output_path = tmp_path / "runs_raw__deduping.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_id": "run-1",
                        "entity": "entity",
                        "project": "project",
                        "exported_at": "2026-03-11T00:00:00+00:00",
                        "raw_run_hash": "hash-a",
                        "raw_run": {"name": "valid"},
                    }
                ),
                '{"broken": ',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        final_run_count = postprocess_dedupe_runs_raw(
            input_path=input_path,
            temp_output_path=temp_output_path,
        )

    lines = input_path.read_text(encoding="utf-8").splitlines()
    assert final_run_count == 1
    assert len(lines) == 1
    assert json.loads(lines[0])["run_id"] == "run-1"
    assert "Removed malformed trailing line" in caplog.text


def test_postprocess_dedupe_runs_raw_raises_for_malformed_non_trailing_line(tmp_path: Path):
    input_path = tmp_path / "runs_raw.jsonl"
    temp_output_path = tmp_path / "runs_raw__deduping.jsonl"
    original_contents = "\n".join(
        [
            '{"broken": ',
            json.dumps(
                {
                    "run_id": "run-1",
                    "entity": "entity",
                    "project": "project",
                    "exported_at": "2026-03-11T00:00:00+00:00",
                    "raw_run_hash": "hash-a",
                    "raw_run": {"name": "valid"},
                }
            ),
        ]
    )
    input_path.write_text(original_contents + "\n", encoding="utf-8")

    try:
        postprocess_dedupe_runs_raw(
            input_path=input_path,
            temp_output_path=temp_output_path,
        )
    except ValueError as exc:
        assert "Invalid JSON" in str(exc)
    else:
        raise AssertionError("Expected postprocess_dedupe_runs_raw to raise")

    assert input_path.read_text(encoding="utf-8") == original_contents + "\n"


def test_extract_runs_raw_dedupes_by_default(tmp_path: Path, monkeypatch):
    run = FakeRun(
        run_id="run-1",
        attrs={"rawconfig": {"alpha": 1}},
        config={"beta": 2},
        summary_metrics={"_step": 100},
        name="run-1-name",
    )

    class FakeApi:
        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        def runs(self, path: str, order: str, per_page: int, lazy: bool):
            assert path == "entity/project"
            assert order == "-created_at"
            assert per_page == 100
            assert lazy is False
            return [run]

    monkeypatch.setattr("dr_wandb.raw_extract.wandb.Api", FakeApi)

    config = RawExtractConfig(
        entity="entity",
        project="project",
        output_dir=tmp_path / "extract",
        runs_raw_path=tmp_path / "extract" / "runs_raw.jsonl",
        runs_raw_deduping_path=tmp_path / "extract" / "runs_raw__deduping.jsonl",
    )

    first_summary = extract_runs_raw(config)
    second_summary = extract_runs_raw(config)

    lines = (tmp_path / "extract" / "runs_raw.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert first_summary.run_count == 1
    assert second_summary.run_count == 1
    assert first_summary.final_run_count == 1
    assert second_summary.final_run_count == 1
    payload = json.loads(lines[0])
    assert payload["run_id"] == "run-1"
    assert payload["project"] == "project"
    assert payload["raw_run"]["config"] == {"beta": 2}


def test_extract_runs_raw_can_disable_postprocess_dedup(tmp_path: Path, monkeypatch):
    run = FakeRun(
        run_id="run-1",
        attrs={"rawconfig": {"alpha": 1}},
        config={"beta": 2},
        summary_metrics={"_step": 100},
        name="run-1-name",
    )

    class FakeApi:
        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        def runs(self, path: str, order: str, per_page: int, lazy: bool):
            return [run]

    monkeypatch.setattr("dr_wandb.raw_extract.wandb.Api", FakeApi)

    config = RawExtractConfig(
        entity="entity",
        project="project",
        output_dir=tmp_path / "extract",
        runs_raw_path=tmp_path / "extract" / "runs_raw.jsonl",
        runs_raw_deduping_path=tmp_path / "extract" / "runs_raw__deduping.jsonl",
        postprocess_dedup=False,
    )

    extract_runs_raw(config)
    summary = extract_runs_raw(config)

    lines = (tmp_path / "extract" / "runs_raw.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert summary.final_run_count is None


def test_extract_runs_raw_postprocesses_existing_file_when_current_fetch_is_empty(
    tmp_path: Path,
    monkeypatch,
):
    runs_raw_path = tmp_path / "extract" / "runs_raw.jsonl"
    runs_raw_path.parent.mkdir(parents=True, exist_ok=True)
    duplicate_row = {
        "run_id": "run-1",
        "entity": "entity",
        "project": "project",
        "exported_at": "2026-03-11T00:00:00+00:00",
        "raw_run_hash": "hash-a",
        "raw_run": {"name": "valid"},
    }
    runs_raw_path.write_text(
        "\n".join([json.dumps(duplicate_row), json.dumps(duplicate_row)]) + "\n",
        encoding="utf-8",
    )

    class FakeApi:
        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        def runs(self, path: str, order: str, per_page: int, lazy: bool):
            return []

    monkeypatch.setattr("dr_wandb.raw_extract.wandb.Api", FakeApi)

    config = RawExtractConfig(
        entity="entity",
        project="project",
        output_dir=tmp_path / "extract",
        runs_raw_path=runs_raw_path,
        runs_raw_deduping_path=tmp_path / "extract" / "runs_raw__deduping.jsonl",
    )

    summary = extract_runs_raw(config)

    lines = runs_raw_path.read_text(encoding="utf-8").splitlines()
    assert summary.run_count == 0
    assert summary.final_run_count == 1
    assert len(lines) == 1


def test_extract_runs_raw_raises_when_postprocess_fails_and_leaves_original_file(
    tmp_path: Path,
    monkeypatch,
):
    runs_raw_path = tmp_path / "extract" / "runs_raw.jsonl"
    runs_raw_path.parent.mkdir(parents=True, exist_ok=True)
    original_contents = "\n".join(
        [
            '{"broken": ',
            json.dumps(
                {
                    "run_id": "run-1",
                    "entity": "entity",
                    "project": "project",
                    "exported_at": "2026-03-11T00:00:00+00:00",
                    "raw_run_hash": "hash-a",
                    "raw_run": {"name": "valid"},
                }
            ),
        ]
    )
    runs_raw_path.write_text(original_contents + "\n", encoding="utf-8")

    class FakeApi:
        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        def runs(self, path: str, order: str, per_page: int, lazy: bool):
            return []

    monkeypatch.setattr("dr_wandb.raw_extract.wandb.Api", FakeApi)

    config = RawExtractConfig(
        entity="entity",
        project="project",
        output_dir=tmp_path / "extract",
        runs_raw_path=runs_raw_path,
        runs_raw_deduping_path=tmp_path / "extract" / "runs_raw__deduping.jsonl",
    )

    try:
        extract_runs_raw(config)
    except ValueError as exc:
        assert "Invalid JSON" in str(exc)
    else:
        raise AssertionError("Expected extract_runs_raw to raise")

    assert runs_raw_path.read_text(encoding="utf-8") == original_contents + "\n"


def test_extract_runs_raw_logs_total_progress_when_length_is_available(
    tmp_path: Path,
    monkeypatch,
    caplog,
):
    class FakeRuns:
        def __iter__(self):
            return iter(
                [
                    FakeRun(run_id="run-1", attrs={"rawconfig": {"alpha": 1}}, name="run-1"),
                    FakeRun(run_id="run-2", attrs={"rawconfig": {"alpha": 2}}, name="run-2"),
                ]
            )

        def __len__(self):
            return 2

    class FakeApi:
        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        def runs(self, path: str, order: str, per_page: int, lazy: bool):
            return FakeRuns()

    monkeypatch.setattr("dr_wandb.raw_extract.wandb.Api", FakeApi)

    config = RawExtractConfig(
        entity="entity",
        project="project",
        output_dir=tmp_path / "extract",
        runs_raw_path=tmp_path / "extract" / "runs_raw.jsonl",
        runs_raw_deduping_path=tmp_path / "extract" / "runs_raw__deduping.jsonl",
        runs_per_page=100,
    )

    with caplog.at_level(logging.INFO):
        extract_runs_raw(config)

    assert "Fetched 2/2 runs from entity/project: 100.0%" in caplog.text
