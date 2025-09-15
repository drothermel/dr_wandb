from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd
import wandb
from sqlalchemy import Engine, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from dr_wandb.constants import (
    DEFAULT_HISTORY_FILENAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RUNS_FILENAME,
    RUN_DATA_COMPONENTS,
    All,
    RunDataComponent,
    RunId,
    RunState,
)
from dr_wandb.utils import build_query, delete_history_for_run, extract_as_datetime


class Base(DeclarativeBase):
    pass


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
    metadata: Mapped[dict[str, Any]] = mapped_column(JSONB)
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
            summary=dict(wandb_run.summary._json_dict),  # noqa: SLF001
            metadata=wandb_run.metadata or {},
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


class HistoryRecord(Base):
    __tablename__ = "wandb_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str]
    step: Mapped[int | None]
    timestamp: Mapped[datetime | None]
    runtime: Mapped[int | None]
    wandb_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB)

    @classmethod
    def from_wandb_history(
        cls, history_entry: dict[str, Any], run_id: str
    ) -> HistoryRecord:
        return cls(
            run_id=run_id,
            step=history_entry.get("_step"),
            timestamp=extract_as_datetime(history_entry, "_timestamp"),
            runtime=history_entry.get("_runtime"),
            wandb_metadata=history_entry.get("_wandb", {}),
            metrics={k: v for k, v in history_entry.items() if not k.startswith("_")},
        )

    @classmethod
    def standard_fields(cls) -> list[str]:
        return [
            col.name
            for col in cls.__table__.columns
            if col.name not in ["wandb_metadata", "metrics"]
        ]

    def to_dict(self, include_metadata: bool = False) -> dict[str, Any]:
        return {
            **{field: getattr(self, field) for field in self.standard_fields()},
            **self.metrics,
            **({"wandb_metadata": self.wandb_metadata} if include_metadata else {}),
        }


class ProjectStore:
    def __init__(self, connection_string: str, output_dir: str | None = None) -> None:
        self.engine: Engine = create_engine(connection_string)
        self.create_tables()
        self.output_dir = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)

    def store_run(self, run: wandb.apis.public.Run) -> None:
        with Session(self.engine) as session:
            existing_run = session.get(RunRecord, run.id)
            if existing_run:
                existing_run.update_from_wandb_run(run)
            else:
                session.add(RunRecord.from_wandb_run(run))
            session.commit()

    def store_history(self, run_id: str, history: wandb.apis.public.History) -> None:
        with Session(self.engine) as session:
            delete_history_for_run(session, run_id)
            for history_entry in history:
                session.add(HistoryRecord.from_wandb_history(history_entry, run_id))
            session.commit()

    def get_runs_df(
        self,
        include: list[RunDataComponent] | All | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        with Session(self.engine) as session:
            result = session.execute(build_query("runs", kwargs=kwargs))
            return pd.DataFrame(
                [run.to_dict(include=include) for run in result.scalars().all()]
            )

    def get_history_df(
        self,
        include_metadata: bool = False,
        kwargs: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        with Session(self.engine) as session:
            result = session.execute(build_query("history", kwargs=kwargs))
        return pd.DataFrame(
            [
                {
                    "run_name": run_name,
                    "project": project_name,
                    **history.to_dict(include_metadata=include_metadata),
                }
                for history, run_name, project_name in result.all()
            ]
        )

    def get_existing_run_states(
        self, kwargs: dict[str, Any] | None = None
    ) -> dict[str, str]:
        with Session(self.engine) as session:
            result = session.execute(build_query("runs", kwargs=kwargs))
            return {run.run_id: run.state for run in result.scalars().all()}

    def export_to_parquet(
        self,
        runs_filename: str = DEFAULT_RUNS_FILENAME,
        history_filename: str = DEFAULT_HISTORY_FILENAME,
    ) -> None:
        self.output_dir.mkdir(exist_ok=True)
        logging.info(f">> Using data output directory: {self.output_dir}")
        runs_df = self.get_runs_df()
        if not runs_df.empty:
            runs_path = self.output_dir / runs_filename
            runs_df.to_parquet(runs_path, index=False)
            logging.info(f">> Wrote runs_df to {runs_path}")
        history_df = self.get_history_df()
        if not history_df.empty:
            history_path = self.output_dir / history_filename
            history_df.to_parquet(history_path, index=False)
            logging.info(f">> Wrote history_df to {history_path}")
