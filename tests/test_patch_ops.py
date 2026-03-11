from __future__ import annotations

from dr_wandb.patch_ops import apply_run_patch
from dr_wandb.sync_types import PatchPlan


class DummyRun:
    def __init__(self) -> None:
        self.id = "run_1"
        self.name = "example"
        self.config = {"a": 1}
        self.tags = ["old"]
        self.update_calls = 0

    def update(self):
        self.update_calls += 1


def test_apply_run_patch_dry_run_does_not_write():
    run = DummyRun()
    patch = PatchPlan(set_config={"a": 2, "b": 3}, add_tags=["new"], remove_tags=["old"])

    result = apply_run_patch(run, patch, dry_run=True)

    assert result.changed is True
    assert result.applied is False
    assert run.config == {"a": 1}
    assert run.tags == ["old"]
    assert run.update_calls == 0


def test_apply_run_patch_apply_writes_changes():
    run = DummyRun()
    patch = PatchPlan(set_config={"a": 2, "b": 3}, add_tags=["new"], remove_tags=["old"])

    result = apply_run_patch(run, patch, dry_run=False)

    assert result.changed is True
    assert result.applied is True
    assert run.config["a"] == 2
    assert run.config["b"] == 3
    assert run.tags == ["new"]
    assert run.update_calls == 1
