from __future__ import annotations

from datetime import datetime

from dr_wandb.run_record import RunRecord


class TestRunRecordFromWandbRun:
    def test_creates_record_from_full_wandb_run(self, mock_wandb_run):
        record = RunRecord.from_wandb_run(mock_wandb_run)

        # Test core fields
        assert record.run_id == "test_run_123"
        assert record.run_name == "test_experiment"
        assert record.state == "finished"
        assert record.project == "test_project"
        assert record.entity == "test_entity"
        assert record.created_at == datetime(2024, 1, 15, 10, 30, 0)

        # Test JSON fields
        assert record.config == {"learning_rate": 0.001, "batch_size": 32, "epochs": 10}
        assert record.summary == {"final_loss": 0.25, "accuracy": 0.95, "val_loss": 0.3}
        assert record.wandb_metadata == {"notes": "test run", "tags": ["experiment"]}
        assert record.system_metrics == {"gpu_memory": 8192, "cpu_count": 4}
        assert record.system_attrs == {"framework": "pytorch", "version": "2.0"}

        # Test sweep info
        assert record.sweep_info == {
            "sweep_id": "sweep_456",
            "sweep_url": "https://wandb.ai/test_entity/test_project/sweeps/sweep_456",
        }

    def test_creates_record_from_minimal_wandb_run(self, mock_wandb_run_minimal):
        record = RunRecord.from_wandb_run(mock_wandb_run_minimal)

        # Test core fields
        assert record.run_id == "minimal_run"
        assert record.run_name == "minimal_test"
        assert record.state == "running"
        assert record.project == "test_project"
        assert record.entity == "test_entity"
        assert record.created_at is None

        # Test empty/None fields are handled gracefully
        assert record.config == {}
        assert record.summary == {}
        assert record.wandb_metadata == {}
        assert record.system_metrics == {}
        assert record.system_attrs == {}
        assert record.sweep_info == {"sweep_id": None, "sweep_url": None}

    def test_handles_none_summary_gracefully(self, mock_wandb_run):
        # Test case where summary is None/empty
        mock_wandb_run.summary = None
        record = RunRecord.from_wandb_run(mock_wandb_run)

        assert record.summary == {}

    def test_standard_fields_excludes_data_components(self):
        fields = RunRecord.standard_fields()

        # Should include basic fields
        assert "run_id" in fields
        assert "run_name" in fields
        assert "state" in fields
        assert "project" in fields
        assert "entity" in fields
        assert "created_at" in fields

        # Should exclude data component fields
        assert "config" not in fields
        assert "summary" not in fields
        assert "wandb_metadata" not in fields
        assert "system_metrics" not in fields
        assert "system_attrs" not in fields
        assert "sweep_info" not in fields


class TestRunRecordUpdate:
    def test_update_from_wandb_run(self, mock_wandb_run, mock_wandb_run_minimal):
        # Create initial record
        record = RunRecord.from_wandb_run(mock_wandb_run_minimal)
        original_id = record.run_id

        # Update with full run data
        record.update_from_wandb_run(mock_wandb_run)

        # ID should remain unchanged
        assert record.run_id == original_id

        # Other fields should be updated
        assert record.run_name == "test_experiment"
        assert record.state == "finished"
        assert record.config == {"learning_rate": 0.001, "batch_size": 32, "epochs": 10}
        assert record.summary == {"final_loss": 0.25, "accuracy": 0.95, "val_loss": 0.3}


class TestRunRecordToDict:
    def test_to_dict_with_no_include(self, mock_wandb_run):
        record = RunRecord.from_wandb_run(mock_wandb_run)
        result = record.to_dict(include=None)

        # Should only include standard fields
        assert "run_id" in result
        assert "run_name" in result
        assert "state" in result
        assert "project" in result
        assert "entity" in result
        assert "created_at" in result

        # Should not include data components
        assert "config" not in result
        assert "summary" not in result
        assert "wandb_metadata" not in result

    def test_to_dict_with_specific_includes(self, mock_wandb_run):
        record = RunRecord.from_wandb_run(mock_wandb_run)
        result = record.to_dict(include=["config", "summary"])

        # Should include standard fields
        assert "run_id" in result
        assert "run_name" in result

        # Should include requested data components
        assert "config" in result
        assert "summary" in result
        assert result["config"] == {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
        }

        # Should not include other data components
        assert "wandb_metadata" not in result
        assert "system_metrics" not in result

    def test_to_dict_with_all_include(self, mock_wandb_run):
        record = RunRecord.from_wandb_run(mock_wandb_run)
        result = record.to_dict(include="all")

        # Should include standard fields
        assert "run_id" in result
        assert "run_name" in result

        # Should include all data components
        assert "config" in result
        assert "summary" in result
        assert "wandb_metadata" in result
        assert "system_metrics" in result
        assert "system_attrs" in result
        assert "sweep_info" in result
