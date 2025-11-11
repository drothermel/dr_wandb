from typing import Any
import logging
from pydantic import BaseModel, Field, computed_field
from pathlib import Path
import typer
import pickle

from dr_wandb.fetch import fetch_project_runs

app = typer.Typer()

class ProjDownloadConfig(BaseModel):
    entity: str
    project: str
    output_dir: Path = Field(
        default_factory=lambda: (
            Path(__file__).parent.parent.parent.parent / "data"
        )
    )
    runs_only: bool = False
    runs_per_page: int = 500
    log_every: int = 20

    runs_output_filename: str = Field(
        default_factory=lambda data: (
            f"{data['entity']}_{data['project']}_runs.pkl"
        )
    )
    histories_output_filename: str = Field(
        default_factory=lambda data: (
            f"{data['entity']}_{data['project']}_histories.pkl"
        )
    )

    def progress_callback(self, run_index: int, total_runs: int, message: str)-> None:
        if run_index % self.log_every == 0:
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


@app.command()
def download_project(
    entity: str,
    project: str,
    output_dir: str,
    runs_only: bool = False,
    runs_per_page: int = 500,
    log_every: int = 20,
) -> None:
    setup_logging()
    logging.info("\n:: Beginning Dr. Wandb Project Downloading Tool ::\n")

    cfg = ProjDownloadConfig(
        entity=entity,
        project=project,
        output_dir=output_dir,
        runs_only=runs_only,
        runs_per_page=runs_per_page,
        log_every=log_every,
    )
    logging.info(str(cfg.model_dump_json(indent=4, exclude="fetch_runs_cfg")))
    logging.info("")

    runs, histories = fetch_project_runs(**cfg.fetch_runs_cfg)
    runs_filename = f"{output_dir}/{cfg.runs_output_filename}"
    histories_filename = f"{output_dir}/{cfg.histories_output_filename}"
    with open(runs_filename, 'wb') as run_file:
        pickle.dump(runs, run_file)
    logging.info(f">> Dumped runs data to: {runs_filename}")
    if not cfg.runs_only:
        with open(histories_filename, 'wb') as hist_file:
            pickle.dump(histories, hist_file)
        logging.info(f">> Dumped histories data to: {histories_filename}")
    else:
        logging.info(f">> Runs only, not dumping histories to: {histories_filename}")


if __name__ == "__main__":
    app()
