from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from sqlalchemy import Engine, Select, create_engine, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data"
DEFAULT_RUNS_FILENAME = "runs_metadata.parquet"
DEFAULT_HISTORY_FILENAME = "runs_history.parquet"
SELECT_FIELDS = ["project", "entity", "state", "run_ids"]

type RunId = str
type RunState = Literal["finished", "running", "crashed", "failed", "killed"]


def build_query(
    base_query: Literal["runs", "history"],
    kwargs: dict[str, Any] | None = None,
) -> Select[Run]:
    assert base_query in ["runs", "history"]
    query = select(Run)
    if base_query == "history":
        query = select(History, Run.run_name, Run.project).join(
            Run, History.run_id == Run.run_id
        )
    if kwargs is not None:
        assert all(k in SELECT_FIELDS for k in kwargs)
        assert all(v is not None for v in kwargs.values())
        if "project" in kwargs:
            query = query.where(Run.project == kwargs["project"])
        if "entity" in kwargs:
            query = query.where(Run.entity == kwargs["entity"])
        if "state" in kwargs:
            query = query.where(Run.state == kwargs["state"])
        if "run_ids" in kwargs:
            query = query.where(Run.run_id.in_(kwargs["run_ids"]))
    return query


def delete_history_for_run(session: Session, run_id: str) -> None:
    session.execute(
        text("DELETE FROM wandb_history WHERE run_id = :run_id"),
        {"run_id": run_id},
    )


def extract_as_datetime(data: dict[str, Any], key: str) -> datetime | None:
    return datetime.fromtimestamp(data.get(key)) if data.get(key) else None


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "wandb_runs"

    run_id: Mapped[RunId] = mapped_column(primary_key=True)
    run_name: Mapped[str]
    state: Mapped[RunState]
    project: Mapped[str]
    entity: Mapped[str]
    created_at: Mapped[datetime | None]
    runtime: Mapped[int | None]
    raw_data: Mapped[dict[str, Any]] = mapped_column(JSONB)

    @classmethod
    def get_standard_fields(cls) -> list[str]:
        return [col.name for col in cls.__table__.columns if col.name != "raw_data"]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Run:
        standard_fields = cls.get_standard_fields()
        return cls(
            **{k: data.get(k) for k in standard_fields},
            raw_data={k: v for k, v in data.items() if k not in standard_fields},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **{field: getattr(self, field) for field in self.get_standard_fields()},
            **(self.raw_data if self.raw_data else {}),
        }

    def update_from_dict(self, data: dict[str, Any]) -> None:
        standard_fields = self.get_standard_fields()
        for field in standard_fields:
            if field in data:
                setattr(self, field, data[field])
        self.raw_data = {k: v for k, v in data.items() if k not in standard_fields}


class History(Base):
    __tablename__ = "wandb_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str]
    step: Mapped[int | None]
    timestamp: Mapped[datetime | None]
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB)

    @classmethod
    def get_standard_fields(cls) -> list[str]:
        return [
            col.name
            for col in cls.__table__.columns
            if col.name not in ["id", "metrics"]
        ]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> History:
        standard_fields = [*cls.get_standard_fields(), "_step", "_timestamp"]
        return cls(
            run_id=data.get("run_id"),
            step=data.get("_step"),
            timestamp=extract_as_datetime(data, "_timestamp"),
            metrics={k: v for k, v in data.items() if k not in standard_fields},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **{field: getattr(self, field) for field in self.get_standard_fields()},
            **(self.metrics if self.metrics else {}),
        }


class Store:
    def __init__(self, connection_string: str, output_dir: str | None = None) -> None:
        self.engine: Engine = create_engine(connection_string)
        self.create_tables()
        self.output_dir = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)

    def store_run(self, run_data: dict[str, Any]) -> None:
        with Session(self.engine) as session:
            existing_run = session.get(Run, run_data["run_id"])
            if existing_run:
                existing_run.update_from_dict(run_data)
            else:
                session.add(Run.from_dict(run_data))
            session.commit()

    def store_history(self, run_id: str, history_data: list[dict[str, Any]]) -> None:
        with Session(self.engine) as session:
            delete_history_for_run(session, run_id)
            for step_data in history_data:
                session.add(History.from_dict({"run_id": run_id, **step_data}))
            session.commit()

    def get_runs_df(self, kwargs: dict[str, Any] | None = None) -> pd.DataFrame:
        with Session(self.engine) as session:
            result = session.execute(build_query("runs", kwargs=kwargs))
            return pd.DataFrame([run.to_dict() for run in result.scalars().all()])

    def get_history_df(self, kwargs: dict[str, Any] | None = None) -> pd.DataFrame:
        with Session(self.engine) as session:
            result = session.execute(build_query("history", kwargs=kwargs))
            return pd.DataFrame(
                [
                    {
                        "run_name": run_name,
                        "project": project_name,
                        **history.to_dict(),
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
