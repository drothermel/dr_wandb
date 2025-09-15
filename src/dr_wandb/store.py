from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import wandb
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from dr_wandb.constants import (
    Base,
    FilterField,
    RunId,
    RunState,
)
from dr_wandb.history_entry_record import (
    HistoryEntry,
    HistoryEntryRecord,
    build_history_query,
)
from dr_wandb.run_record import (
    RUN_DATA_COMPONENTS,
    All,
    RunDataComponent,
    RunRecord,
    build_run_query,
)
from dr_wandb.utils import safe_convert_for_parquet

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data"
DEFAULT_RUNS_FILENAME = "runs_metadata"
DEFAULT_HISTORY_FILENAME = "runs_history"
type History = list[HistoryEntry]


def delete_history_for_runs(session: Session, run_ids: list[RunId]) -> None:
    if not run_ids:
        return
    session.execute(
        text("DELETE FROM wandb_history WHERE run_id = ANY(:run_ids)"),
        {"run_ids": run_ids},
    )


def save_update_run(session: Session, run: wandb.apis.public.Run) -> None:
    existing_run = session.get(RunRecord, run.id)
    if existing_run:
        existing_run.update_from_wandb_run(run)
    else:
        session.add(RunRecord.from_wandb_run(run))


def delete_add_history(session: Session, run_id: RunId, history: History) -> None:
    delete_history_for_runs(session, [run_id])
    for history_entry in history:
        session.add(HistoryEntryRecord.from_wandb_history(history_entry, run_id))


def ensure_database_exists(database_url: str) -> str:
    parsed = urlparse(database_url)
    db_name = parsed.path.lstrip("/")
    postgres_url = database_url.replace(f"/{db_name}", "/postgres")

    try:
        test_engine = create_engine(database_url)
        with test_engine.connect():
            pass
        return database_url
    except OperationalError as e:
        if "does not exist" in str(e):
            logging.info(f"Database '{db_name}' doesn't exist, creating it...")
            postgres_engine = create_engine(postgres_url)
            with postgres_engine.connect() as conn:
                conn.execute(text("COMMIT"))
                conn.execute(text(f'CREATE DATABASE "{db_name}"'))
            logging.info(f"Created database '{db_name}'")
            return database_url
        else:
            raise


class ProjectStore:
    def __init__(self, connection_string: str, output_dir: str | None = None) -> None:
        connection_string = ensure_database_exists(connection_string)
        self.engine: Engine = create_engine(connection_string)
        self.create_tables()
        self.output_dir = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)

    def store_run(self, run: wandb.apis.public.Run) -> None:
        with Session(self.engine) as session:
            save_update_run(session, run)
            session.commit()

    def store_runs(self, runs: list[wandb.apis.public.Run]) -> None:
        with Session(self.engine) as session:
            for run in runs:
                save_update_run(session, run)
            session.commit()

    def store_history(self, run_id: RunId, history: History) -> None:
        with Session(self.engine) as session:
            delete_add_history(session, run_id, history)
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

    def store_run_and_history(
        self, run: wandb.apis.public.Run, history: History
    ) -> None:
        with Session(self.engine) as session:
            delete_add_history(session, run.id, history)
            save_update_run(session, run)
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
            history_path = self.output_dir / f"{history_filename}.parquet"
            history_df = safe_convert_for_parquet(history_df)
            history_df.to_parquet(history_path, engine="pyarrow", index=False)
            logging.info(f">> Wrote history_df to {history_path}")
        for include_type in RUN_DATA_COMPONENTS:
            runs_df = self.get_runs_df(include=[include_type])
            if not runs_df.empty:
                runs_path = self.output_dir / f"{runs_filename}_{include_type}.parquet"
                runs_df = safe_convert_for_parquet(runs_df)
                runs_df.to_parquet(runs_path, engine="pyarrow", index=False)
                logging.info(f">> Wrote runs_df with {include_type} to {runs_path}")
        runs_df_full = self.get_runs_df(include="all")
        if not runs_df_full.empty:
            runs_path = self.output_dir / f"{runs_filename}.parquet"
            runs_df_full = safe_convert_for_parquet(runs_df_full)
            runs_df_full.to_parquet(runs_path, engine="pyarrow", index=False)
            logging.info(f">> Wrote runs_df with all parts to {runs_path}")
