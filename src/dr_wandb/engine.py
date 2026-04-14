"""Coordinate W&B API reads with local export-store writes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import wandb
from dr_ds.serialization import utc_now_iso

from dr_wandb.config import ExportMode, ExportRequest, SyncMode
from dr_wandb.history import (
    HistoryRow,
    max_history_step,
    merge_history_rows,
    observed_last_history_step,
    scan_history_for_export,
)
from dr_wandb.results import ExportManifest, ExportSummary
from dr_wandb.selection import select_runs
from dr_wandb.state import ExportState
from dr_wandb.store import ExportStore
from dr_wandb.wandb_run import RunSnapshot, WandbRun

logger = logging.getLogger(__name__)


class ExportEngine:
    """Fetch runs from W&B, merge them into a named export, and persist results."""

    def __init__(self, request: ExportRequest) -> None:
        self.request = request

    def export(self) -> ExportSummary:
        """Execute one export and return a summary of the written artifacts."""
        store = ExportStore(
            name=self.request.name, data_root=self.request.data_root
        )
        existing_manifest = store.load_manifest()
        state = self._load_initial_state(store)
        snapshots = store.load_existing_snapshots(
            request=self.request, manifest=existing_manifest
        )
        existing_history_rows = store.load_existing_history_rows(
            request=self.request, manifest=existing_manifest
        )
        exported_at = utc_now_iso()
        new_history_rows: list[HistoryRow] = []

        api = wandb.Api(timeout=self.request.timeout_seconds)
        selected_runs = select_runs(api=api, request=self.request, state=state)
        logger.info(
            "Exporting %s run(s) for %s/%s into %s",
            len(selected_runs),
            self.request.entity,
            self.request.project,
            self.request.name,
        )
        last_logged_progress = 0
        for index, raw_run in enumerate(selected_runs, start=1):
            self._process_run(
                raw_run=raw_run,
                state=state,
                snapshots=snapshots,
                new_history_rows=new_history_rows,
                exported_at=exported_at,
            )
            last_logged_progress = self._log_progress(
                processed_runs=index,
                total_runs=len(selected_runs),
                last_logged_progress=last_logged_progress,
            )

        sorted_snapshots = sorted(
            snapshots.values(),
            key=lambda snapshot: snapshot.sort_key,
            reverse=True,
        )
        runs_path = store.write_run_snapshots(sorted_snapshots)

        history_rows: list[HistoryRow] = []
        history_path: Path | None = None
        if self.request.mode == ExportMode.HISTORY:
            history_rows = merge_history_rows(
                existing_history_rows=existing_history_rows,
                new_history_rows=new_history_rows,
            )
            history_path = store.write_history_rows(history_rows)
        else:
            store.remove_history()

        manifest = self._build_manifest(
            existing_manifest=existing_manifest,
            exported_at=exported_at,
            runs_path=runs_path,
            history_path=history_path,
            run_count=len(snapshots),
            history_count=len(history_rows),
        )
        store.save_manifest(manifest)

        state.last_exported_at = exported_at
        store.save_state(state)

        return ExportSummary(
            name=self.request.name,
            entity=self.request.entity,
            project=self.request.project,
            mode=self.request.mode,
            sync_mode=self.request.sync_mode,
            output_dir=str(store.export_dir),
            state_path=str(store.state_path),
            manifest_path=str(store.manifest_path),
            runs_path=str(runs_path),
            history_path=(
                str(history_path) if history_path is not None else None
            ),
            run_count=len(snapshots),
            history_count=len(history_rows),
            exported_at=exported_at,
        )

    def _log_progress(
        self,
        *,
        processed_runs: int,
        total_runs: int,
        last_logged_progress: int,
    ) -> int:
        """Log the first processed run and each additional 5% completion bucket."""
        if total_runs <= 0:
            return last_logged_progress

        percent_complete = (processed_runs * 100) // total_runs
        progress_bucket = min((percent_complete // 5) * 5, 100)
        should_log = processed_runs == 1
        if processed_runs == 1 and progress_bucket >= 5:
            last_logged_progress = progress_bucket
        elif progress_bucket >= 5 and progress_bucket > last_logged_progress:
            should_log = True
            last_logged_progress = progress_bucket

        if should_log:
            logger.info(
                "Processed %s/%s runs (%s%% complete)",
                processed_runs,
                total_runs,
                percent_complete,
            )
        return last_logged_progress

    def _load_initial_state(self, store: ExportStore) -> ExportState:
        """Load prior export state unless full reconcile explicitly resets it."""
        if self.request.sync_mode == SyncMode.FULL_RECONCILE:
            return ExportState(
                name=self.request.name,
                entity=self.request.entity,
                project=self.request.project,
            )
        return store.load_state(
            entity=self.request.entity, project=self.request.project
        )

    def _process_run(
        self,
        *,
        raw_run: Any,
        state: ExportState,
        snapshots: dict[str, RunSnapshot],
        new_history_rows: list[HistoryRow],
        exported_at: str,
    ) -> None:
        """Convert one raw run into stored snapshots and optional history rows."""
        wandb_run = WandbRun.from_wandb_run(
            raw_run,
            entity=self.request.entity,
            project=self.request.project,
            include_metadata=self.request.include_metadata,
        )
        if wandb_run.run_id == "":
            return
        snapshots[wandb_run.run_id] = RunSnapshot(
            run=wandb_run, exported_at=exported_at
        )
        tracking_state = state.begin_run_tracking(wandb_run)

        if self.request.mode != ExportMode.HISTORY:
            return
        rows = scan_history_for_export(
            request=self.request,
            wandb_run=wandb_run,
            raw_run=raw_run,
            run_last_history_step=tracking_state.last_history_step,
        )
        new_history_rows.extend(rows)
        max_step = max_history_step(rows)
        observed = observed_last_history_step(
            wandb_run=wandb_run, raw_run=raw_run
        )
        if observed is not None:
            max_step = (
                observed if max_step is None else max(max_step, observed)
            )
        tracking_state.last_history_step = max_step

    def _build_manifest(
        self,
        *,
        existing_manifest: ExportManifest | None,
        exported_at: str,
        runs_path: Path,
        history_path: Path | None,
        run_count: int,
        history_count: int,
    ) -> ExportManifest:
        """Build the manifest describing the export just written to disk."""
        selection = self.request.history_selection
        manifest_keys = (
            list(selection.keys)
            if selection is not None and selection.keys is not None
            else None
        )
        manifest_window = selection.window if selection is not None else None
        return ExportManifest(
            name=self.request.name,
            entity=self.request.entity,
            project=self.request.project,
            mode=self.request.mode,
            created_at=(
                existing_manifest.created_at
                if existing_manifest is not None
                else exported_at
            ),
            updated_at=exported_at,
            runs_path=str(runs_path),
            history_path=(
                str(history_path) if history_path is not None else None
            ),
            run_count=run_count,
            history_count=history_count,
            selected_history_keys=manifest_keys,
            history_window=manifest_window,
        )
