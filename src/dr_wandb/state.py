"""Track incremental export state across runs and export invocations."""

from __future__ import annotations

from pydantic import BaseModel, Field

from dr_wandb.wandb_run import WandbRun


class RunTrackingState(BaseModel):
    """Record the incremental-export facts tracked for one run."""

    run_id: str
    created_at: str | None = None
    updated_at: str | None = None
    run_state: str | None = None
    last_history_step: int | None = None

    @classmethod
    def from_wandb_run(
        cls,
        run: WandbRun,
        *,
        last_history_step: int | None,
    ) -> RunTrackingState:
        """Build tracking state from the current stored run snapshot."""
        return cls(
            run_id=run.run_id,
            created_at=run.created_at,
            updated_at=run.updated_at,
            run_state=run.state,
            last_history_step=last_history_step,
        )


class ExportState(BaseModel):
    """Persist cross-run incremental export state for one named export."""

    schema_version: int = 1
    name: str
    entity: str
    project: str
    last_exported_at: str | None = None
    max_created_at: str | None = None
    runs: dict[str, RunTrackingState] = Field(default_factory=dict)

    def begin_run_tracking(self, wandb_run: WandbRun) -> RunTrackingState:
        """Start or refresh tracking for one run and update aggregate state."""
        prior = self.runs.get(wandb_run.run_id)
        tracking_state = RunTrackingState.from_wandb_run(
            wandb_run,
            last_history_step=(
                prior.last_history_step if prior is not None else None
            ),
        )
        self.runs[wandb_run.run_id] = tracking_state
        if wandb_run.created_at is not None and (
            self.max_created_at is None
            or wandb_run.created_at > self.max_created_at
        ):
            self.max_created_at = wandb_run.created_at
        return tracking_state
