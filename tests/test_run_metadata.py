from __future__ import annotations

import json
from pathlib import Path

import pytest

from dr_wandb.raw_extract import RawRunRecord
from dr_wandb.run_metadata import (
    CanonicalRunMetadata,
    ParsedHistoryField,
    ParsedHistoryKeys,
    ParsedSummaryMetrics,
    UnparsedHistoryField,
    iter_raw_run_records,
    load_canonical_run_metadata,
    resolve_runs_raw_path,
)


def _write_raw_records(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_parsed_summary_metrics_promotes_top_level_and_groups_namespaced_keys():
    parsed = ParsedSummaryMetrics.from_raw(
        {
            "_runtime": 10.5,
            "_step": 100,
            "_timestamp": 123.4,
            "_wandb": {"runtime": 10.5, "foo": "bar"},
            "eval/downstream/accuracy": 0.9,
            "eval/lm/loss": 1.2,
            "plain_metric": 7,
            "_other": "value",
        }
    )

    assert parsed.runtime == 10.5
    assert parsed.step == 100
    assert parsed.timestamp == 123.4
    assert parsed.metrics == {
        "eval": {
            "downstream": {"accuracy": 0.9},
            "lm": {"loss": 1.2},
        },
        "plain_metric": 7,
    }
    assert parsed.extra == {"wandb.foo": "bar", "_other": "value"}


def test_parsed_summary_metrics_preserves_wandb_conflicts_and_collisions_in_extra():
    parsed = ParsedSummaryMetrics.from_raw(
        {
            "_runtime": 10.5,
            "_step": 100,
            "_timestamp": 123.4,
            "_wandb": {"runtime": 11.5, "step": 100},
            "eval": 1,
            "eval/downstream/accuracy": 0.9,
        }
    )

    assert parsed.runtime == 10.5
    assert parsed.step == 100
    assert parsed.metrics == {"eval": 1}
    assert parsed.extra == {
        "wandb.runtime": 11.5,
        "eval/downstream/accuracy": 0.9,
    }


def test_parsed_summary_metrics_flatten_helpers_restore_flat_metric_names():
    parsed = ParsedSummaryMetrics(
        runtime=10.5,
        step=100,
        timestamp=123.4,
        metrics={"eval": {"downstream": {"accuracy": 0.9}}, "plain_metric": 7},
        extra={"wandb.foo": "bar"},
    )

    assert parsed.flatten_metrics() == {
        "eval/downstream/accuracy": 0.9,
        "plain_metric": 7,
    }
    assert parsed.to_flat_dict(include_extra=True) == {
        "runtime": 10.5,
        "step": 100,
        "timestamp": 123.4,
        "eval/downstream/accuracy": 0.9,
        "plain_metric": 7,
        "wandb.foo": "bar",
    }


def test_parsed_history_keys_parses_common_field_shape_and_extra():
    parsed = ParsedHistoryKeys.from_raw(
        {
            "keys": {
                "_runtime": {
                    "typeCounts": [{"type": "number", "count": 1908}],
                    "monotonic": True,
                    "previousValue": 64246.5,
                }
            },
            "lastStep": 1908,
            "sets": [],
        }
    )

    assert parsed.lastStep == 1908
    assert parsed.extra == {"sets": []}
    field = parsed.fields["_runtime"]
    assert isinstance(field, ParsedHistoryField)
    assert field.key == "_runtime"
    assert field.type == "number"
    assert field.count == 1908
    assert field.monotonic is True
    assert field.previousValue == 64246.5


def test_parsed_history_keys_falls_back_for_unexpected_type_counts():
    parsed = ParsedHistoryKeys.from_raw(
        {
            "keys": {
                "metric": {
                    "typeCounts": [
                        {"type": "number", "count": 10},
                        {"type": "string", "count": 1},
                    ],
                    "previousValue": 9,
                }
            }
        }
    )

    field = parsed.fields["metric"]
    assert isinstance(field, UnparsedHistoryField)
    assert field.key == "metric"
    assert field.reason == "unexpected_type_counts"
    assert field.raw["previousValue"] == 9


def test_canonical_run_metadata_promotes_fields_and_partitions_extra():
    record = {
        "run_id": "run-1",
        "entity": "entity",
        "project": "project",
        "exported_at": "2026-03-11T10:00:00+00:00",
        "raw_run_hash": "hash-a",
        "raw_run": {
            "displayName": "pretty-name",
            "name": "fallback-name",
            "state": "finished",
            "group": "MoE",
            "createdAt": "2026-03-10T10:00:00+00:00",
            "heartbeatAt": "2026-03-10T12:00:00+00:00",
            "tags": ["tag-a", "tag-b"],
            "url": "https://wandb.ai/entity/project/runs/run-1",
            "historyLineCount": 123,
            "config": {"alpha": 1},
            "summaryMetrics": {
                "_runtime": 1.0,
                "eval/downstream/accuracy": 0.9,
            },
            "historyKeys": {
                "keys": {
                    "_runtime": {
                        "typeCounts": [{"type": "number", "count": 10}],
                        "monotonic": True,
                        "previousValue": 1.0,
                    }
                },
                "lastStep": 10,
            },
            "projectId": "project-id",
            "path": ["entity", "project", "run-1"],
        },
    }

    canonical = CanonicalRunMetadata.from_raw_run_record(
        RawRunRecord.model_validate(record)
    )

    assert canonical.name == "pretty-name"
    assert canonical.state == "finished"
    assert canonical.group == "MoE"
    assert canonical.tags == ["tag-a", "tag-b"]
    assert canonical.config == {"alpha": 1}
    assert canonical.summaryMetrics.runtime == 1.0
    assert canonical.summaryMetrics.metrics == {"eval": {"downstream": {"accuracy": 0.9}}}
    assert canonical.historyKeys.lastStep == 10
    assert isinstance(canonical.historyKeys.fields["_runtime"], ParsedHistoryField)
    assert canonical.extra == {
        "projectId": "project-id",
        "path": ["entity", "project", "run-1"],
    }


def test_resolve_runs_raw_path_uses_export_name_token(tmp_path: Path):
    path = resolve_runs_raw_path(export_name="my export", data_root=tmp_path)

    assert path == tmp_path / "my_export" / "wandb_raw_extract" / "runs_raw.jsonl"


def test_load_canonical_run_metadata_keeps_latest_record_per_run_id(tmp_path: Path):
    runs_raw_path = tmp_path / "wandb_export" / "wandb_raw_extract" / "runs_raw.jsonl"
    _write_raw_records(
        runs_raw_path,
        [
            {
                "run_id": "run-1",
                "entity": "entity",
                "project": "project",
                "exported_at": "2026-03-11T10:00:00+00:00",
                "raw_run_hash": "hash-a",
                "raw_run": {
                    "name": "older",
                    "createdAt": "2026-03-09T10:00:00+00:00",
                },
            },
            {
                "run_id": "run-1",
                "entity": "entity",
                "project": "project",
                "exported_at": "2026-03-11T11:00:00+00:00",
                "raw_run_hash": "hash-b",
                "raw_run": {
                    "displayName": "newer",
                    "createdAt": "2026-03-09T10:00:00+00:00",
                },
            },
            {
                "run_id": "run-2",
                "entity": "entity",
                "project": "project",
                "exported_at": "2026-03-11T09:00:00+00:00",
                "raw_run_hash": "hash-c",
                "raw_run": {
                    "name": "second",
                    "createdAt": "2026-03-10T10:00:00+00:00",
                },
            },
        ],
    )

    records = load_canonical_run_metadata(export_name="wandb_export", data_root=tmp_path)

    assert [record.run_id for record in records] == ["run-2", "run-1"]
    assert records[1].name == "newer"


def test_load_canonical_run_metadata_uses_hash_as_tie_breaker(tmp_path: Path):
    runs_raw_path = tmp_path / "wandb_export" / "wandb_raw_extract" / "runs_raw.jsonl"
    _write_raw_records(
        runs_raw_path,
        [
            {
                "run_id": "run-1",
                "entity": "entity",
                "project": "project",
                "exported_at": "2026-03-11T10:00:00+00:00",
                "raw_run_hash": "aaa",
                "raw_run": {"name": "older"},
            },
            {
                "run_id": "run-1",
                "entity": "entity",
                "project": "project",
                "exported_at": "2026-03-11T10:00:00+00:00",
                "raw_run_hash": "bbb",
                "raw_run": {"name": "newer"},
            },
        ],
    )

    records = load_canonical_run_metadata(export_name="wandb_export", data_root=tmp_path)

    assert len(records) == 1
    assert records[0].name == "newer"


def test_iter_raw_run_records_raises_clear_error_on_bad_json(tmp_path: Path):
    runs_raw_path = tmp_path / "wandb_export" / "wandb_raw_extract" / "runs_raw.jsonl"
    runs_raw_path.parent.mkdir(parents=True, exist_ok=True)
    runs_raw_path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        list(iter_raw_run_records(export_name="wandb_export", data_root=tmp_path))
