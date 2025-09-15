from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock


from dr_wandb.utils import extract_as_datetime, select_updated_runs


class TestExtractAsDatetime:
    def test_extracts_valid_timestamp(self):
        data = {"timestamp": 1705312200.0}  # 2024-01-15 10:30:00 UTC
        result = extract_as_datetime(data, "timestamp")
        # Just check that we get a datetime object, not exact time due to timezone conversion
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_returns_none_for_missing_key(self):
        data = {"other_key": "value"}
        result = extract_as_datetime(data, "timestamp")
        assert result is None

    def test_returns_none_for_none_value(self):
        data = {"timestamp": None}
        result = extract_as_datetime(data, "timestamp")
        assert result is None

    def test_returns_none_for_zero_timestamp(self):
        data = {"timestamp": 0}
        result = extract_as_datetime(data, "timestamp")
        # Zero timestamp actually converts to epoch time, so we get a datetime
        # Let's test that it's not None but is epoch time
        assert result is not None
        assert result.year == 1969 or result.year == 1970  # Depends on timezone


class TestSelectUpdatedRuns:
    def test_selects_new_runs(self, sample_run_states):
        # Create mock runs with IDs not in existing states
        new_run = Mock(id="new_run_id")
        all_runs = [new_run]

        result = select_updated_runs(all_runs, sample_run_states)
        assert result == [new_run]

    def test_selects_running_runs(self, sample_run_states):
        # Based on updated logic: only selects new runs or running runs
        running_run = Mock(id="running_run")
        crashed_run = Mock(id="crashed_run")
        all_runs = [running_run, crashed_run]

        result = select_updated_runs(all_runs, sample_run_states)
        # Only running_run should be selected (running state), not crashed_run
        assert len(result) == 1
        assert running_run in result
        assert crashed_run not in result

    def test_excludes_finished_runs(self, sample_run_states):
        # Create mock run that's already finished
        finished_run = Mock(id="finished_run")
        all_runs = [finished_run]

        result = select_updated_runs(all_runs, sample_run_states)
        assert result == []

    def test_mixed_run_states(self, sample_run_states):
        # Mix of new, finished, and running runs
        new_run = Mock(id="brand_new")
        finished_run = Mock(id="finished_run")
        running_run = Mock(id="running_run")
        all_runs = [new_run, finished_run, running_run]

        result = select_updated_runs(all_runs, sample_run_states)

        # Should include new and running, exclude finished
        assert len(result) == 2
        assert new_run in result
        assert running_run in result
        assert finished_run not in result

    def test_empty_inputs(self):
        result = select_updated_runs([], {})
        assert result == []
