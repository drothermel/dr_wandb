from __future__ import annotations

from datetime import datetime

from dr_wandb.run_record import RunRecord


class TestRunRecordFromWandbRun:
    def test_creates_record_from_full_wandb_run(self, mock_wandb_run):
        record = RunRecord.from_wandb_run(mock_wandb_run)

        assert record.run_id == "test_run_123"
        assert record.run_name == "test_experiment"
        assert record.state == "finished"
        assert record.project == "test_project"
        assert record.entity == "test_entity"
        assert record.created_at == datetime(2024, 1, 15, 10, 30, 0)

        assert record.config == {"learning_rate": 0.001, "batch_size": 32, "epochs": 10}
        assert record.summary == {"final_loss": 0.25, "accuracy": 0.95, "val_loss": 0.3}
        assert record.wandb_metadata == {"notes": "test run", "tags": ["experiment"]}
        assert record.system_metrics == {"gpu_memory": 8192, "cpu_count": 4}
        assert record.system_attrs == {
            "config": {"learning_rate": 0.001, "batch_size": 32, "epochs": 10},
            "summary_metrics": {"final_loss": 0.25, "accuracy": 0.95, "val_loss": 0.3},
            "metadata": {"notes": "test run", "tags": ["experiment"]},
            "system_metrics": {"gpu_memory": 8192, "cpu_count": 4},
        }

        assert record.sweep_info == {
            "sweep_id": "sweep_456",
            "sweep_url": "https://wandb.ai/test_entity/test_project/sweeps/sweep_456",
        }

    def test_creates_record_from_minimal_wandb_run(self, mock_wandb_run_minimal):
        record = RunRecord.from_wandb_run(mock_wandb_run_minimal)

        assert record.run_id == "minimal_run"
        assert record.run_name == "minimal_test"
        assert record.state == "running"
        assert record.project == "test_project"
        assert record.entity == "test_entity"
        assert record.created_at is None

        assert record.config == {}
        assert record.summary == {}
        assert record.wandb_metadata == {}
        assert record.system_metrics == {}
        assert record.system_attrs == {}
        assert record.sweep_info == {"sweep_id": None, "sweep_url": None}

    def test_handles_none_summary_gracefully(self, mock_wandb_run):
        mock_wandb_run.summary_metrics = None
        record = RunRecord.from_wandb_run(mock_wandb_run)

        assert record.summary == {}


class TestRunRecordModelDump:
    def test_model_dump_includes_all_fields(self, mock_wandb_run):
        record = RunRecord.from_wandb_run(mock_wandb_run)
        result = record.model_dump()

        assert result["run_id"] == "test_run_123"
        assert result["run_name"] == "test_experiment"
        assert result["state"] == "finished"
        assert result["project"] == "test_project"
        assert result["entity"] == "test_entity"
        assert result["created_at"] == datetime(2024, 1, 15, 10, 30, 0)

        assert result["config"] == {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
        }
        assert result["summary"] == {
            "final_loss": 0.25,
            "accuracy": 0.95,
            "val_loss": 0.3,
        }
        assert result["wandb_metadata"] == {"notes": "test run", "tags": ["experiment"]}
        assert result["system_metrics"] == {"gpu_memory": 8192, "cpu_count": 4}
        assert result["system_attrs"] == {
            "config": {"learning_rate": 0.001, "batch_size": 32, "epochs": 10},
            "summary_metrics": {"final_loss": 0.25, "accuracy": 0.95, "val_loss": 0.3},
            "metadata": {"notes": "test run", "tags": ["experiment"]},
            "system_metrics": {"gpu_memory": 8192, "cpu_count": 4},
        }
        assert result["sweep_info"] == {
            "sweep_id": "sweep_456",
            "sweep_url": "https://wandb.ai/test_entity/test_project/sweeps/sweep_456",
        }

    def test_model_dump_json_serializable(self, mock_wandb_run):
        record = RunRecord.from_wandb_run(mock_wandb_run)
        json_str = record.model_dump_json()

        assert "test_run_123" in json_str
        assert "test_experiment" in json_str
        assert "learning_rate" in json_str
