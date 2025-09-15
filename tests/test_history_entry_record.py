from __future__ import annotations

from datetime import datetime

from dr_wandb.history_entry_record import HistoryEntryRecord


class TestHistoryEntryRecordFromWandbHistory:
    def test_creates_record_from_history_entry(self, mock_wandb_history_entry):
        run_id = "test_run_123"
        record = HistoryEntryRecord.from_wandb_history(mock_wandb_history_entry, run_id)

        # Test metadata fields
        assert record.run_id == run_id
        assert record.step == 100
        # Test timestamp is a datetime but don't check exact time due to timezone conversion
        assert isinstance(record.timestamp, datetime)
        assert record.timestamp.year == 2024
        assert record.timestamp.month == 1
        assert record.timestamp.day == 15
        assert record.runtime == 1800

        # Test wandb metadata
        assert record.wandb_metadata == {"core_version": "0.16.0"}

        # Test metrics (non-underscore fields)
        expected_metrics = {
            "loss": 0.45,
            "accuracy": 0.87,
            "learning_rate": 0.001,
        }
        assert record.metrics == expected_metrics

    def test_handles_missing_optional_fields(self):
        # History entry with missing optional fields
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

        # Metrics should only contain non-underscore fields
        expected_metrics = {
            "loss": 0.3,
            "accuracy": 0.9,
            "public_metric": 42,
        }
        assert record.metrics == expected_metrics

    def test_standard_fields_excludes_json_columns(self):
        fields = HistoryEntryRecord.standard_fields()

        # Should include basic fields
        assert "id" in fields
        assert "run_id" in fields
        assert "step" in fields
        assert "timestamp" in fields
        assert "runtime" in fields

        # Should exclude JSON fields
        assert "wandb_metadata" not in fields
        assert "metrics" not in fields


class TestHistoryEntryRecordToDict:
    def test_to_dict_without_metadata(self, mock_wandb_history_entry):
        record = HistoryEntryRecord.from_wandb_history(
            mock_wandb_history_entry, "test_run"
        )
        result = record.to_dict(include_metadata=False)

        # Should include standard fields
        assert "run_id" in result
        assert "step" in result
        assert "timestamp" in result
        assert "runtime" in result

        # Should include metrics
        assert "loss" in result
        assert "accuracy" in result
        assert "learning_rate" in result
        assert result["loss"] == 0.45

        # Should not include metadata
        assert "wandb_metadata" not in result

    def test_to_dict_with_metadata(self, mock_wandb_history_entry):
        record = HistoryEntryRecord.from_wandb_history(
            mock_wandb_history_entry, "test_run"
        )
        result = record.to_dict(include_metadata=True)

        # Should include standard fields
        assert "run_id" in result
        assert "step" in result

        # Should include metrics
        assert "loss" in result
        assert "accuracy" in result

        # Should include metadata
        assert "wandb_metadata" in result
        assert result["wandb_metadata"] == {"core_version": "0.16.0"}

    def test_to_dict_flattens_metrics_into_top_level(self, mock_wandb_history_entry):
        record = HistoryEntryRecord.from_wandb_history(
            mock_wandb_history_entry, "test_run"
        )
        result = record.to_dict()

        # Metrics should be flattened into top level, not nested
        assert result["loss"] == 0.45
        assert result["accuracy"] == 0.87
        assert result["learning_rate"] == 0.001

        # Should not have nested metrics dict
        assert "metrics" not in result
