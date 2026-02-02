from __future__ import annotations

from datetime import datetime
from typing import Any

import wandb
from pydantic import BaseModel

from dr_wandb.constants import RunId, RunState

RUN_DATA_COMPONENTS = [
    "config",
    "summary",
    "wandb_metadata",
    "system_metrics",
    "system_attrs",
    "sweep_info",
]


class RunRecord(BaseModel):
    run_id: RunId
    run_name: str
    state: RunState
    project: str
    entity: str
    created_at: datetime | None
    config: dict[str, Any]
    summary: dict[str, Any]
    wandb_metadata: dict[str, Any]
    system_metrics: dict[str, Any]
    system_attrs: dict[str, Any]
    sweep_info: dict[str, Any]

    @classmethod
    def from_wandb_run(cls, wandb_run: wandb.apis.public.Run) -> RunRecord:
        return cls(
            run_id=wandb_run.id,
            run_name=wandb_run.name,
            state=wandb_run.state,
            project=wandb_run.project,
            entity=wandb_run.entity,
            created_at=wandb_run.created_at,
            config=dict(wandb_run.config),
            summary=dict(wandb_run.summary._json_dict) if wandb_run.summary else {},
            wandb_metadata=wandb_run.metadata or {},
            system_metrics=wandb_run.system_metrics or {},
            system_attrs=dict(wandb_run._attrs),
            sweep_info={
                "sweep_id": getattr(wandb_run, "sweep_id", None),
                "sweep_url": getattr(wandb_run, "sweep_url", None),
            },
        )
