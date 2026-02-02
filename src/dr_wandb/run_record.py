from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    import wandb

type RunId = str
type RunState = Literal["finished", "running", "crashed", "failed", "killed"]

SWEEP_INFO_KEYS = ["sweep_id", "sweep_url"]


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
        # Get summary using public API (summary_metrics)
        summary_metrics = getattr(wandb_run, "summary_metrics", None)
        summary_dict = dict(summary_metrics) if summary_metrics else {}

        # Construct system_attrs from public properties
        # Aggregate config, summary_metrics, metadata, and system_metrics into a stable dict
        system_attrs: dict[str, Any] = {}
        if wandb_run.config:
            system_attrs["config"] = dict(wandb_run.config)
        if summary_metrics:
            system_attrs["summary_metrics"] = dict(summary_metrics)
        if wandb_run.metadata:
            system_attrs["metadata"] = wandb_run.metadata
        if wandb_run.system_metrics:
            system_attrs["system_metrics"] = wandb_run.system_metrics

        return cls(
            run_id=wandb_run.id,
            run_name=wandb_run.name,
            state=wandb_run.state,
            project=wandb_run.project,
            entity=wandb_run.entity,
            created_at=wandb_run.created_at,
            config=dict(wandb_run.config),
            summary=summary_dict,
            wandb_metadata=wandb_run.metadata or {},
            system_metrics=wandb_run.system_metrics or {},
            system_attrs=system_attrs,
            sweep_info=dict(
                (key, getattr(wandb_run, key, None)) for key in SWEEP_INFO_KEYS
            ),
        )
