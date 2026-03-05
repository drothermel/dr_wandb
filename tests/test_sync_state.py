from __future__ import annotations

from pathlib import Path

from dr_wandb.sync_state import default_state_path, load_state, save_state
from dr_wandb.sync_types import ProjectSyncState, RunCursor


def test_default_state_path_is_sanitized(tmp_path: Path):
    path = default_state_path(
        "team/name",
        "proj name",
        user="alice@example.com",
        host="host.local",
        root_dir=tmp_path,
    )
    assert path.parent == tmp_path
    assert "team_name" in path.name
    assert "proj_name" in path.name
    assert "alice_example.com" in path.name


def test_load_save_state_roundtrip(tmp_path: Path):
    state_path = tmp_path / "state.json"
    state = ProjectSyncState(
        entity="ml-moe",
        project="moe",
        runs={
            "run_1": RunCursor(run_id="run_1", last_step=10, history_seen=20),
        },
    )

    save_state(state, state_path)
    loaded = load_state(state_path, entity="ml-moe", project="moe")

    assert loaded.entity == "ml-moe"
    assert loaded.project == "moe"
    assert loaded.runs["run_1"].last_step == 10
    assert loaded.runs["run_1"].history_seen == 20

    save_state(loaded, state_path)
    loaded_again = load_state(state_path, entity="ml-moe", project="moe")
    assert loaded_again.model_dump(mode="python") == loaded.model_dump(mode="python")
