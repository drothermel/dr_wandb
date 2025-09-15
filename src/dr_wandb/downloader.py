from __future__ import annotations

import logging
from dataclasses import dataclass

import wandb

from dr_wandb.constants import ProgressCallback
from dr_wandb.store import ProjectStore
from dr_wandb.utils import default_progress_callback, select_updated_runs


@dataclass
class DownloaderStats:
    num_wandb_runs: int = 0
    num_stored_runs: int = 0
    num_new_runs: int = 0
    num_updated_runs: int = 0

    def __str__(self) -> str:
        return "\n".join(
            [
                "",
                ":: Downloader Stats ::",
                f" - # WandB runs: {self.num_wandb_runs:,}",
                f" - # Stored runs: {self.num_stored_runs:,}",
                f" - # New runs: {self.num_new_runs:,}",
                f" - # Updated runs: {self.num_updated_runs:,}",
                "",
            ]
        )


class Downloader:
    def __init__(
        self,
        store: ProjectStore,
        runs_per_page: int = 500,
    ) -> None:
        self.store = store
        self._api: wandb.Api | None = None
        self.runs_per_page = runs_per_page
        self.progress_callback: ProgressCallback = default_progress_callback

    @property
    def api(self) -> wandb.Api:
        if self._api is None:
            try:
                self._api = wandb.Api()
            except wandb.errors.UsageError as e:
                if "api_key not configured" in str(e):
                    raise RuntimeError(
                        "WandB API key not configured. "
                        "Please run 'wandb login' or set WANDB_API_KEY env var"
                    ) from e
                raise
        return self._api

    def set_progress_callback(self, progress_callback: ProgressCallback) -> None:
        self.progress_callback = progress_callback

    def get_all_runs(self, entity: str, project: str) -> list[wandb.apis.public.Run]:
        return list(self.api.runs(f"{entity}/{project}", per_page=self.runs_per_page))

    def download_runs(
        self,
        entity: str,
        project: str,
        force_refresh: bool = False,
        with_history: bool = False,
    ) -> DownloaderStats:
        wandb_runs = self.get_all_runs(entity, project)
        stored_states = self.store.get_existing_run_states(
            {"entity": entity, "project": project}
        )
        runs_to_download = (
            wandb_runs
            if force_refresh
            else select_updated_runs(wandb_runs, stored_states)
        )
        num_new_runs = len([r for r in runs_to_download if r.id not in stored_states])
        stats = DownloaderStats(
            num_wandb_runs=len(wandb_runs),
            num_stored_runs=len(stored_states),
            num_new_runs=num_new_runs,
            num_updated_runs=len(runs_to_download) - num_new_runs,
        )
        if len(runs_to_download) == 0:
            logging.info(">> No runs to download")
            return stats

        if not with_history:
            logging.info(">> Runs only mode, bulk downloading runs")
            self.store.store_runs(runs_to_download)
            return stats

        logging.info(">> Downloading runs and history data together")
        for i, run in enumerate(runs_to_download):
            self.store.store_run_and_history(run, list(run.scan_history()))
            self.progress_callback(i + 1, len(runs_to_download), run.name)
        return stats

    def download_project(
        self,
        entity: str,
        project: str,
        runs_only: bool = False,
        force_refresh: bool = False,
    ) -> DownloaderStats:
        stats = self.download_runs(
            entity, project, force_refresh, with_history=not runs_only
        )
        logging.info(">> Download completed")
        return stats

    def write_downloaded_to_parquet(self) -> None:
        logging.info(">> Beginning export to parquet")
        self.store.export_to_parquet()
