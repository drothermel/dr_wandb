from __future__ import annotations

from pydantic import BaseModel, Field

from dr_wandb.export.wandb_run import WandbRun


class RunTrackingState(BaseModel):
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
        return cls(
            run_id=run.run_id,
            created_at=run.created_at,
            updated_at=run.updated_at,
            run_state=run.state,
            last_history_step=last_history_step,
        )


class ExportState(BaseModel):
    schema_version: int = 1
    name: str
    entity: str
    project: str
    last_exported_at: str | None = None
    max_created_at: str | None = None
    runs: dict[str, RunTrackingState] = Field(default_factory=dict)
