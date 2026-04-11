"""ExportEngine: orchestrates fetching runs from W&B and writing them to an ExportStore."""

from __future__ import annotations

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


class ExportEngine:
    """Fetch runs from W&B, merge with any prior export, and write JSONL + manifest."""

    def __init__(self, request: ExportRequest) -> None:
        self.request = request

    def export(self) -> ExportSummary:
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
        for raw_run in select_runs(api=api, request=self.request, state=state):
            self._process_run(
                raw_run=raw_run,
                state=state,
                snapshots=snapshots,
                new_history_rows=new_history_rows,
                exported_at=exported_at,
            )

        state.last_exported_at = exported_at
        store.save_state(state)

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

    def _load_initial_state(self, store: ExportStore) -> ExportState:
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
