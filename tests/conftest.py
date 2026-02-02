from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_wandb_run():
    run = Mock()
    run.id = "test_run_123"
    run.name = "test_experiment"
    run.state = "finished"
    run.project = "test_project"
    run.entity = "test_entity"
    run.created_at = datetime(2024, 1, 15, 10, 30, 0)
    run.config = {"learning_rate": 0.001, "batch_size": 32, "epochs": 10}
    run.metadata = {"notes": "test run", "tags": ["experiment"]}
    run.system_metrics = {"gpu_memory": 8192, "cpu_count": 4}
    run._attrs = {"framework": "pytorch", "version": "2.0"}
    run.sweep_id = "sweep_456"
    run.sweep_url = "https://wandb.ai/test_entity/test_project/sweeps/sweep_456"

    summary_mock = Mock()
    summary_mock._json_dict = {"final_loss": 0.25, "accuracy": 0.95, "val_loss": 0.3}
    run.summary = summary_mock

    return run


@pytest.fixture
def mock_wandb_run_minimal():
    run = Mock()
    run.id = "minimal_run"
    run.name = "minimal_test"
    run.state = "running"
    run.project = "test_project"
    run.entity = "test_entity"
    run.created_at = None
    run.config = {}
    run.metadata = None
    run.system_metrics = None
    run._attrs = {}
    run.sweep_id = None
    run.sweep_url = None

    summary_mock = Mock()
    summary_mock._json_dict = {}
    run.summary = summary_mock

    return run


@pytest.fixture
def mock_wandb_history_entry():
    return {
        "_step": 100,
        "_timestamp": 1705312200.0,
        "_runtime": 1800,
        "_wandb": {"core_version": "0.16.0"},
        "loss": 0.45,
        "accuracy": 0.87,
        "learning_rate": 0.001,
    }


@pytest.fixture
def sample_history_entries():
    return [
        {
            "_step": i,
            "_timestamp": 1705312200.0 + i * 10,
            "_runtime": i * 10,
            "_wandb": {"core_version": "0.16.0"},
            "loss": 1.0 - (i * 0.01),
            "accuracy": 0.1 + (i * 0.01),
        }
        for i in range(5)
    ]
