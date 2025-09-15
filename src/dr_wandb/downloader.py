from __future__ import annotations

from typing import Any

import wandb

from dr_wandb.store import Store
from dr_wandb.utils import select_new_and_unfinished_runs


class Downloader:
    def __init__(
        self,
        store: Store,
        runs_per_page: int = 500,
    ) -> None:
        self.store = store
        self._api: wandb.Api | None = None
        self.runs_per_page = runs_per_page

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

    def get_runs_to_download(
        self, entity: str, project: str, force_refresh: bool = False
    ) -> list[wandb.apis.public.Run]:
        all_runs = list(
            self.api.runs(f"{entity}/{project}", per_page=self.runs_per_page)
        )
        if force_refresh:
            return all_runs
        return select_new_and_unfinished_runs(
            all_runs, self.store.get_existing_run_states(entity, project)
        )

    def download_run_data(self, run: wandb.apis.public.Run) -> dict[str, Any]:
        pass
