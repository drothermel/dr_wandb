from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine

from dr_wandb.constants import Base


@pytest.fixture
def mock_wandb_run():
    """Realistic mock WandB run object for testing data transformations."""
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

    # Mock summary with _json_dict
    summary_mock = Mock()
    summary_mock._json_dict = {"final_loss": 0.25, "accuracy": 0.95, "val_loss": 0.3}
    run.summary = summary_mock

    return run


@pytest.fixture
def mock_wandb_run_minimal():
    """Minimal mock WandB run for edge case testing."""
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

    # Mock empty summary
    summary_mock = Mock()
    summary_mock._json_dict = {}
    run.summary = summary_mock

    return run


@pytest.fixture
def mock_wandb_history_entry():
    """Sample WandB history entry for testing."""
    return {
        "_step": 100,
        "_timestamp": 1705312200.0,  # 2024-01-15 10:30:00 UTC
        "_runtime": 1800,  # 30 minutes
        "_wandb": {"core_version": "0.16.0"},
        "loss": 0.45,
        "accuracy": 0.87,
        "learning_rate": 0.001,
    }


@pytest.fixture
def sample_history_entries():
    """Multiple history entries for batch testing."""
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


@pytest.fixture
def in_memory_db():
    """SQLite in-memory database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


# project_store_with_temp_dir fixture removed - requires PostgreSQL for JSONB support
# Will be re-added in future when database integration tests are reimplemented


@pytest.fixture
def sample_run_states():
    """Sample run states for testing filtering logic."""
    return {
        "finished_run": "finished",
        "running_run": "running",
        "crashed_run": "crashed",
        "failed_run": "failed",
    }


@pytest.fixture
def sample_filter_kwargs():
    """Sample filter kwargs for query testing."""
    return {
        "project": "test_project",
        "entity": "test_entity",
        "state": "finished",
    }
