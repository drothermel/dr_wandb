from __future__ import annotations


from dr_wandb.history_entry_record import build_history_query
from dr_wandb.run_record import build_run_query


class TestBuildRunQuery:
    def test_builds_basic_query_without_filters(self):
        query = build_run_query(kwargs=None)

        # Should be a valid SQLAlchemy select statement
        assert hasattr(query, "compile")

        # Should compile without errors
        compiled = query.compile(compile_kwargs={"literal_binds": True})
        assert "SELECT" in str(compiled).upper()
        assert "wandb_runs" in str(compiled)

    def test_applies_project_filter(self, sample_filter_kwargs):
        query = build_run_query(kwargs={"project": "test_project"})
        compiled = str(query.compile(compile_kwargs={"literal_binds": True}))

        assert "project" in compiled.lower()

    def test_applies_entity_filter(self, sample_filter_kwargs):
        query = build_run_query(kwargs={"entity": "test_entity"})
        compiled = str(query.compile(compile_kwargs={"literal_binds": True}))

        assert "entity" in compiled.lower()

    def test_applies_state_filter(self, sample_filter_kwargs):
        query = build_run_query(kwargs={"state": "finished"})
        compiled = str(query.compile(compile_kwargs={"literal_binds": True}))

        assert "state" in compiled.lower()

    def test_applies_run_ids_filter(self):
        run_ids = ["run1", "run2", "run3"]
        query = build_run_query(kwargs={"run_ids": run_ids})
        compiled = str(query.compile(compile_kwargs={"literal_binds": True}))

        assert "run_id" in compiled.lower()
        assert "IN" in compiled.upper()

    def test_applies_multiple_filters(self, sample_filter_kwargs):
        query = build_run_query(kwargs=sample_filter_kwargs)
        compiled = str(query.compile(compile_kwargs={"literal_binds": True}))

        # Should contain all filter criteria
        assert "project" in compiled.lower()
        assert "entity" in compiled.lower()
        assert "state" in compiled.lower()

    def test_empty_kwargs_dict(self):
        query = build_run_query(kwargs={})
        compiled = str(query.compile(compile_kwargs={"literal_binds": True}))

        # Should be basic select without WHERE clause
        assert "SELECT" in compiled.upper()
        assert "WHERE" not in compiled.upper()


class TestBuildHistoryQuery:
    def test_builds_basic_query_without_filters(self):
        query = build_history_query(run_ids=None)

        # Should be a valid SQLAlchemy select statement
        assert hasattr(query, "compile")

        # Should compile without errors
        compiled = query.compile(compile_kwargs={"literal_binds": True})
        assert "SELECT" in str(compiled).upper()
        assert "wandb_history" in str(compiled)

    def test_applies_run_ids_filter(self):
        run_ids = ["run1", "run2", "run3"]
        query = build_history_query(run_ids=run_ids)
        compiled = str(query.compile(compile_kwargs={"literal_binds": True}))

        assert "run_id" in compiled.lower()
        assert "IN" in compiled.upper()

    def test_empty_run_ids_list(self):
        query = build_history_query(run_ids=[])
        compiled = str(query.compile(compile_kwargs={"literal_binds": True}))

        # Should have empty IN clause or be optimized away
        assert "SELECT" in compiled.upper()
        assert "wandb_history" in compiled
