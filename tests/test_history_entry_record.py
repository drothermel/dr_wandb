from __future__ import annotations

from datetime import datetime

from dr_wandb.history_entry_record import HistoryEntryRecord


class TestHistoryEntryRecordFromWandbHistory:
    def test_creates_record_from_history_entry(self, mock_wandb_history_entry):
        run_id = "test_run_123"
        record = HistoryEntryRecord.from_wandb_history(mock_wandb_history_entry, run_id)

        assert record.run_id == run_id
        assert record.step == 100
        assert isinstance(record.timestamp, datetime)
        assert record.timestamp.year == 2024
        assert record.timestamp.month == 1
        assert record.timestamp.day == 15
        assert record.runtime == 1800

        assert record.wandb_metadata == {"core_version": "0.16.0"}

        expected_metrics = {
            "loss": 0.45,
            "accuracy": 0.87,
            "learning_rate": 0.001,
        }
        assert record.metrics == expected_metrics

    def test_handles_missing_optional_fields(self):
        minimal_entry = {
            "loss": 0.5,
            "accuracy": 0.8,
        }
        run_id = "test_run"

        record = HistoryEntryRecord.from_wandb_history(minimal_entry, run_id)

        assert record.run_id == run_id
        assert record.step is None
        assert record.timestamp is None
        assert record.runtime is None
        assert record.wandb_metadata == {}
        assert record.metrics == {"loss": 0.5, "accuracy": 0.8}

    def test_filters_underscore_prefixed_fields_from_metrics(self):
        entry = {
            "_step": 50,
            "_timestamp": 1705312200.0,
            "_runtime": 900,
            "_wandb": {"version": "1.0"},
            "_internal_field": "should_not_appear",
            "loss": 0.3,
            "accuracy": 0.9,
            "public_metric": 42,
        }

        record = HistoryEntryRecord.from_wandb_history(entry, "test_run")

        expected_metrics = {
            "loss": 0.3,
            "accuracy": 0.9,
            "public_metric": 42,
        }
        assert record.metrics == expected_metrics


class TestHistoryEntryRecordModelDump:
    def test_model_dump_includes_all_fields(self, mock_wandb_history_entry):
        record = HistoryEntryRecord.from_wandb_history(
            mock_wandb_history_entry, "test_run"
        )
        result = record.model_dump()

        assert result["run_id"] == "test_run"
        assert result["step"] == 100
        assert isinstance(result["timestamp"], datetime)
        assert result["runtime"] == 1800
        assert result["wandb_metadata"] == {"core_version": "0.16.0"}
        assert result["metrics"] == {
            "loss": 0.45,
            "accuracy": 0.87,
            "learning_rate": 0.001,
        }

    def test_model_dump_json_serializable(self, mock_wandb_history_entry):
        record = HistoryEntryRecord.from_wandb_history(
            mock_wandb_history_entry, "test_run"
        )
        json_str = record.model_dump_json()

        assert "test_run" in json_str
        assert "0.45" in json_str
        assert "0.87" in json_str
