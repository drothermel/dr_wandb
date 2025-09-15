from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import wandb
from sqlalchemy import Select, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from dr_wandb.constants import (
    SUPPORTED_FILTER_FIELDS,
    Base,
    FilterField,
    RunId,
    RunState,
)

RUN_DATA_COMPONENTS = [
    "config",
    "summary",
    "wandb_metadata",
    "system_metrics",
    "system_attrs",
    "sweep_info",
]
type All = Literal["all"]
type RunDataComponent = Literal[
    "config",
    "summary",
    "wandb_metadata",
    "system_metrics",
    "system_attrs",
    "sweep_info",
]


class RunRecord(Base):
    __tablename__ = "wandb_runs"

    run_id: Mapped[RunId] = mapped_column(primary_key=True)
    run_name: Mapped[str]
    state: Mapped[RunState]
    project: Mapped[str]
    entity: Mapped[str]
    created_at: Mapped[datetime | None]

    config: Mapped[dict[str, Any]] = mapped_column(JSONB)
    summary: Mapped[dict[str, Any]] = mapped_column(JSONB)
    wandb_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB)
    system_metrics: Mapped[dict[str, Any]] = mapped_column(JSONB)
    system_attrs: Mapped[dict[str, Any]] = mapped_column(JSONB)
    sweep_info: Mapped[dict[str, Any]] = mapped_column(JSONB)

    @classmethod
    def standard_fields(cls) -> list[str]:
        return [
            col.name
            for col in cls.__table__.columns
            if col.name not in RUN_DATA_COMPONENTS
        ]

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
            summary=dict(wandb_run.summary._json_dict) if wandb_run.summary else {},  # noqa: SLF001
            wandb_metadata=wandb_run.metadata or {},
            system_metrics=wandb_run.system_metrics or {},
            system_attrs=dict(wandb_run._attrs),  # noqa: SLF001
            sweep_info={
                "sweep_id": getattr(wandb_run, "sweep_id", None),
                "sweep_url": getattr(wandb_run, "sweep_url", None),
            },
        )

    def update_from_wandb_run(self, wandb_run: wandb.apis.public.Run) -> None:
        updated = self.__class__.from_wandb_run(wandb_run)
        for col in self.__table__.columns:
            if col.name != "run_id":
                setattr(self, col.name, getattr(updated, col.name))

    def to_dict(
        self, include: list[RunDataComponent] | All | None = None
    ) -> dict[str, Any]:
        include = include or []
        if include == "all":
            include = RUN_DATA_COMPONENTS
        assert all(field in RUN_DATA_COMPONENTS for field in include)
        data = {k: getattr(self, k) for k in self.standard_fields()}
        for field in include:
            data[field] = getattr(self, field)
        return data


def build_run_query(kwargs: dict[FilterField, Any] | None = None) -> Select[RunRecord]:
    query = select(RunRecord)
    if kwargs is not None:
        assert all(k in SUPPORTED_FILTER_FIELDS for k in kwargs)
        assert all(v is not None for v in kwargs.values())
        if "project" in kwargs:
            query = query.where(RunRecord.project == kwargs["project"])
        if "entity" in kwargs:
            query = query.where(RunRecord.entity == kwargs["entity"])
        if "state" in kwargs:
            query = query.where(RunRecord.state == kwargs["state"])
        if "run_ids" in kwargs:
            query = query.where(RunRecord.run_id.in_(kwargs["run_ids"]))
    return query
