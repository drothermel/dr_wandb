from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import typer
from pydantic import BaseModel, Field, computed_field

from dr_wandb.fetch import fetch_project_runs
from dr_wandb.utils import find_project_root, safe_convert_for_parquet

app = typer.Typer()

type OutputFormat = Literal["pkl", "parquet"]

OUTPUT_FORMAT_EXTENSION: dict[OutputFormat, str] = {
    "pkl": "pkl",
    "parquet": "parquet",
}


class ProjDownloadConfig(BaseModel):
    entity: str
    project: str
    output_dir: Path = Field(
        default_factory=lambda: (
            find_project_root(start_path=Path(__file__).parent) / "data"
        )
    )
    output_format: OutputFormat = "pkl"
    runs_only: bool = False
    runs_per_page: int = 500
    log_every: int = 20

    @computed_field
    @property
    def output_extension(self) -> str:
        return OUTPUT_FORMAT_EXTENSION[self.output_format]

    @computed_field
    @property
    def runs_output_filename(self) -> str:
        return f"{self.entity}_{self.project}_runs.{self.output_extension}"

    @computed_field
    @property
    def histories_output_filename(self) -> str:
        return f"{self.entity}_{self.project}_histories.{self.output_extension}"

    def progress_callback(
        self, run_index: int, total_runs: int | None, message: str
    ) -> None:
        if run_index % self.log_every == 0:
            if total_runs is None:
                logging.info(f">> {run_index}/?: {message}")
            else:
                logging.info(f">> {run_index}/{total_runs}: {message}")

    @computed_field
    @property
    def fetch_runs_cfg(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "project": self.project,
            "runs_per_page": self.runs_per_page,
            "progress_callback": self.progress_callback,
            "include_history": not self.runs_only,
        }


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def save_as_pickle(data: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def save_as_parquet(data: list[dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(data)
    df = safe_convert_for_parquet(df)
    df.to_parquet(path)


@app.command()
def download_project(
    entity: str,
    project: str,
    output_dir: str,
    output_format: OutputFormat = "pkl",
    runs_only: bool = False,
    runs_per_page: int = 500,
    log_every: int = 20,
) -> None:
    setup_logging()
    logging.info("\n:: Beginning Dr. Wandb Project Downloading Tool ::\n")

    cfg = ProjDownloadConfig(
        entity=entity,
        project=project,
        output_dir=Path(output_dir),
        output_format=output_format,
        runs_only=runs_only,
        runs_per_page=runs_per_page,
        log_every=log_every,
    )
    logging.info(str(cfg.model_dump_json(indent=4, exclude={"fetch_runs_cfg"})))
    logging.info("")

    runs, histories = fetch_project_runs(**cfg.fetch_runs_cfg)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    runs_file = output_path / cfg.runs_output_filename
    histories_file = output_path / cfg.histories_output_filename

    if cfg.output_format == "parquet":
        save_as_parquet(runs, runs_file)
        logging.info(f">> Saved runs to: {runs_file}")
        if not cfg.runs_only:
            # Process histories incrementally to avoid memory issues
            history_dfs = []
            for run_hist in histories:
                if run_hist:  # Only process non-empty histories
                    df = pd.DataFrame(run_hist)
                    df = safe_convert_for_parquet(df)
                    history_dfs.append(df)
            # Always create history file for consistency with pickle format
            if history_dfs:
                combined_df = pd.concat(history_dfs, ignore_index=True)
                combined_df.to_parquet(histories_file)
                logging.info(f">> Saved histories to: {histories_file}")
            else:
                # Create empty parquet file to match pickle format behavior
                empty_df = pd.DataFrame()
                empty_df.to_parquet(histories_file)
                logging.info(f">> Saved empty histories to: {histories_file}")
    else:
        save_as_pickle(runs, runs_file)
        logging.info(f">> Saved runs to: {runs_file}")
        if not cfg.runs_only:
            save_as_pickle(histories, histories_file)
            logging.info(f">> Saved histories to: {histories_file}")

    if cfg.runs_only:
        logging.info(">> Runs only mode, skipped histories")


if __name__ == "__main__":
    app()
