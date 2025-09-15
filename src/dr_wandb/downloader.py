from __future__ import annotations

from dataclasses import dataclass, field

import wandb

from dr_wandb.constants import ProgressCallback, RunId
from dr_wandb.store import ProjectStore
from dr_wandb.utils import default_progress_callback, select_updated_runs


@dataclass
class DownloaderStats:
    num_wandb_runs: int = 0
    num_stored_runs: int = 0
    num_new_runs: int = 0
    num_updated_runs: int = 0
    num_downloaded_histories: int = 0
    num_downloaded_history_entries: int = 0
    downloaded_rids: list[RunId] = field(default_factory=list)

    def __str__(self) -> str:
        return "\n".join(
            [
                ":: Downloader Stats ::",
                f" - # WandB runs: {self.num_wandb_runs:,}",
                f" - # Stored runs: {self.num_stored_runs:,}",
                f" - # New runs: {self.num_new_runs:,}",
                f" - # Updated runs: {self.num_updated_runs:,}",
                f" - # Histories: {self.num_downloaded_histories:,}",
                f" - # History entries: {self.num_downloaded_history_entries:,}",
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
    ) -> tuple[DownloaderStats, list[wandb.apis.public.Run]]:
        wandb_runs = self.get_all_runs(entity, project)
        stored_states = self.store.get_existing_run_states(
            {"entity": entity, "project": project}
        )
        runs_to_download = (
            wandb_runs
            if force_refresh
            else select_updated_runs(wandb_runs, stored_states)
        )
        stats = DownloaderStats(
            num_wandb_runs=len(wandb_runs),
            num_stored_runs=len(stored_states),
            num_new_runs=0,
            num_updated_runs=0,
            num_downloaded_histories=0,
            num_downloaded_history_entries=0,
            downloaded_rids=[],
        )
        if len(runs_to_download) == 0:
            return stats

        for i, run in enumerate(runs_to_download):
            self.store.store_run(run)
            self.progress_callback(i + 1, len(runs_to_download), run.name)
            if run.id not in stored_states:
                stats.num_new_runs += 1
            else:
                stats.num_updated_runs += 1
            stats.downloaded_rids.append(run.id)
        return stats, runs_to_download

    def download_histories(
        self,
        runs: list[wandb.apis.public.Run],
        stats: DownloaderStats,
    ) -> DownloaderStats:
        histories = [list(run.scan_history()) for run in runs]
        self.store.store_histories(runs, histories)
        stats.num_downloaded_histories += len(histories)
        stats.num_downloaded_history_entries += sum(
            len(history) for history in histories
        )
        return stats

    def download_project(
        self,
        entity: str,
        project: str,
        runs_only: bool = False,
        force_refresh: bool = False,
    ) -> DownloaderStats:
        stats, downloaded_runs = self.download_runs(entity, project, force_refresh)
        if not runs_only:
            stats = self.download_histories(downloaded_runs, stats)
        return stats

    def write_downloaded_to_parquet(self) -> None:
        self.store.export_to_parquet()
