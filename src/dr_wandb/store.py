from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import wandb
from sqlalchemy import Engine, Select, create_engine, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from dr_wandb.constants import (
    SUPPORTED_FILTER_FIELDS,
    FilterField,
    RunId,
    RunState,
)
from dr_wandb.utils import extract_as_datetime

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data"
DEFAULT_RUNS_FILENAME = "runs_metadata"
DEFAULT_HISTORY_FILENAME = "runs_history"
RUN_DATA_COMPONENTS = [
    "config",
    "summary",
    "metadata",
    "system_metrics",
    "system_attrs",
    "sweep_info",
]
type All = Literal["all"]
type RunDataComponent = Literal[
    "config",
    "summary",
    "metadata",
    "system_metrics",
    "system_attrs",
    "sweep_info",
]
type HistoryEntry = dict[str, Any]
type History = list[HistoryEntry]


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


class HistoryEntryRecord(Base):
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
    ) -> HistoryEntryRecord:
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


def build_history_query(
    run_ids: list[RunId] | None = None,
) -> Select[HistoryEntryRecord]:
    query = select(HistoryEntryRecord)
    if run_ids is not None:
        query = query.where(HistoryEntryRecord.run_id.in_(run_ids))
    return query


def delete_history_for_runs(session: Session, run_ids: list[RunId]) -> None:
    if not run_ids:
        return
    session.execute(
        text("DELETE FROM wandb_history WHERE run_id = ANY(:run_ids)"),
        {"run_ids": run_ids},
    )


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

    def store_runs(self, runs: list[wandb.apis.public.Run]) -> None:
        with Session(self.engine) as session:
            for run in runs:
                session.add(RunRecord.from_wandb_run(run))
            session.commit()

    def store_history(self, run_id: RunId, history: History) -> None:
        with Session(self.engine) as session:
            delete_history_for_runs(session, [run_id])
            for history_entry in history:
                session.add(
                    HistoryEntryRecord.from_wandb_history(history_entry, run_id)
                )
            session.commit()

    def store_histories(
        self,
        runs: list[wandb.apis.public.Run],
        histories: list[History],
    ) -> None:
        assert len(runs) == len(histories)
        run_ids = [run.id for run in runs]
        with Session(self.engine) as session:
            delete_history_for_runs(session, run_ids)
            for run_id, history in zip(run_ids, histories, strict=False):
                for history_entry in history:
                    session.add(
                        HistoryEntryRecord.from_wandb_history(history_entry, run_id)
                    )
            session.commit()

    def get_runs_df(
        self,
        include: list[RunDataComponent] | All | None = None,
        kwargs: dict[FilterField, Any] | None = None,
    ) -> pd.DataFrame:
        with Session(self.engine) as session:
            result = session.execute(build_run_query(kwargs=kwargs))
            return pd.DataFrame(
                [run.to_dict(include=include) for run in result.scalars().all()]
            )

    def get_history_df(
        self,
        include_metadata: bool = False,
        run_ids: list[RunId] | None = None,
    ) -> pd.DataFrame:
        with Session(self.engine) as session:
            result = session.execute(build_history_query(run_ids=run_ids))
        return pd.DataFrame(
            [
                history.to_dict(include_metadata=include_metadata)
                for history in result.scalars().all()
            ]
        )

    def get_existing_run_states(
        self, kwargs: dict[FilterField, Any] | None = None
    ) -> dict[RunId, RunState]:
        with Session(self.engine) as session:
            result = session.execute(build_run_query(kwargs=kwargs))
            return {run.run_id: run.state for run in result.scalars().all()}

    def export_to_parquet(
        self,
        runs_filename: str = DEFAULT_RUNS_FILENAME,
        history_filename: str = DEFAULT_HISTORY_FILENAME,
    ) -> None:
        self.output_dir.mkdir(exist_ok=True)
        logging.info(f">> Using data output directory: {self.output_dir}")
        history_df = self.get_history_df()
        if not history_df.empty:
            history_path = self.output_dir / history_filename
            history_df.to_parquet(history_path, index=False)
            logging.info(f">> Wrote history_df to {history_path}")
        for include_type in RUN_DATA_COMPONENTS:
            runs_df = self.get_runs_df(include=include_type)
            if not runs_df.empty:
                runs_path = self.output_dir / f"{runs_filename}_{include_type}.parquet"
                runs_df.to_parquet(runs_path, index=False)
                logging.info(f">> Wrote runs_df with {include_type} to {runs_path}")
        runs_df_full = self.get_runs_df(include="all")
        if not runs_df_full.empty:
            runs_path = self.output_dir / f"{runs_filename}.parquet"
            runs_df_full.to_parquet(runs_path, index=False)
            logging.info(f">> Wrote runs_df with all parts to {runs_path}")
