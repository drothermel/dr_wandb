from __future__ import annotations

from pathlib import Path
from typing import Any

import wandb
from dr_ds.serialization import utc_now_iso

from dr_wandb.export.export_manifest import ExportManifest
from dr_wandb.export.export_modes import ExportMode, FetchMode
from dr_wandb.export.export_request import ExportRequest
from dr_wandb.export.export_state import ExportState, RunTrackingState
from dr_wandb.export.export_summary import ExportSummary
from dr_wandb.export.history_export import (
    max_history_step,
    merge_history_rows,
    observed_last_history_step,
    scan_history_for_export,
)
from dr_wandb.export.policy import (
    HistoryPolicyContext,
    HistoryRow,
    HistoryWindow,
)
from dr_wandb.export.record_store import RecordStore
from dr_wandb.export.run_selection import select_runs
from dr_wandb.export.run_snapshot import RunSnapshot
from dr_wandb.export.wandb_run import WandbRun


class ExportEngine:
    def __init__(self, request: ExportRequest) -> None:
        self.request = request
        self.store = RecordStore.from_name_and_root(
            name=request.name,
            data_root=request.data_root,
        )
        self.existing_manifest = self.store.load_manifest()
        if request.fetch_mode == FetchMode.FULL_RECONCILE:
            self.state = ExportState(
                name=request.name,
                entity=request.entity,
                project=request.project,
            )
        else:
            self.state = self.store.paths.load_state(
                entity=request.entity, project=request.project
            )
        self.snapshots = self.store.load_existing_snapshots(
            request=request, manifest=self.existing_manifest
        )
        self.existing_history_rows = self.store.load_existing_history_rows(
            request=request, manifest=self.existing_manifest
        )
        self.new_history_rows: list[HistoryRow] = []
        self.exported_at = utc_now_iso()
        self.first_wandb_run: WandbRun | None = None
        self.first_raw_run: Any = None
        self.runs_path: Path | None = None
        self.history_path: Path | None = None
        self.history_rows: list[HistoryRow] = []
        self.manifest: ExportManifest | None = None

    def export(self) -> ExportSummary:
        self._fetch_and_process_runs()
        self._save_state()
        self._write_records()
        self._write_manifest()
        return self._build_summary()

    def _fetch_and_process_runs(self) -> None:
        api = wandb.Api(timeout=self.request.timeout_seconds)
        for raw_run in select_runs(
            api=api, request=self.request, state=self.state
        ):
            self._process_run(raw_run)

    def _process_run(self, raw_run: Any) -> None:
        wandb_run = WandbRun.from_wandb_run(
            raw_run,
            entity=self.request.entity,
            project=self.request.project,
            include_metadata=self.request.include_metadata,
        )
        if wandb_run.run_id == "":
            return
        if self.first_wandb_run is None:
            self.first_wandb_run = wandb_run
            self.first_raw_run = raw_run

        self.snapshots[wandb_run.run_id] = RunSnapshot(
            run=wandb_run, exported_at=self.exported_at
        )
        tracking_state = self.state.begin_run_tracking(wandb_run)

        if self.request.mode == ExportMode.HISTORY:
            self._scan_run_history(
                raw_run=raw_run,
                wandb_run=wandb_run,
                tracking_state=tracking_state,
            )

    def _scan_run_history(
        self,
        *,
        raw_run: Any,
        wandb_run: WandbRun,
        tracking_state: RunTrackingState,
    ) -> None:
        ctx = HistoryPolicyContext.from_wandb_run(
            wandb_run=wandb_run,
            raw_run=raw_run,
            run_last_history_step=tracking_state.last_history_step,
        )
        rows = scan_history_for_export(request=self.request, ctx=ctx)
        self.new_history_rows.extend(rows)
        max_step = max_history_step(rows)
        observed = observed_last_history_step(
            wandb_run=wandb_run, raw_run=raw_run
        )
        if observed is not None:
            max_step = (
                observed if max_step is None else max(max_step, observed)
            )
        tracking_state.last_history_step = max_step

    def _save_state(self) -> None:
        self.state.last_exported_at = self.exported_at
        self.store.paths.save_state(self.state)

    def _write_records(self) -> None:
        sorted_snapshots = sorted(
            self.snapshots.values(),
            key=lambda snapshot: snapshot.sort_key,
            reverse=True,
        )
        self.runs_path = self.store.write_run_snapshots(sorted_snapshots)
        if self.request.mode == ExportMode.HISTORY:
            self.history_rows = merge_history_rows(
                existing_history_rows=self.existing_history_rows,
                new_history_rows=self.new_history_rows,
            )
            self.history_path = self.store.write_history_rows(
                self.history_rows
            )
        else:
            self.store.remove_history()

    def _write_manifest(self) -> None:
        manifest_keys: list[str] | None = None
        manifest_window: HistoryWindow | None = None
        if (
            self.request.history_policy is not None
            and self.first_wandb_run is not None
        ):
            ctx = HistoryPolicyContext.from_wandb_run(
                wandb_run=self.first_wandb_run,
                raw_run=self.first_raw_run,
                run_last_history_step=observed_last_history_step(
                    wandb_run=self.first_wandb_run,
                    raw_run=self.first_raw_run,
                ),
            )
            keys = self.request.history_policy.select_history_keys(ctx)
            manifest_keys = list(keys) if keys is not None else None
            manifest_window = (
                self.request.history_policy.select_history_window(ctx)
            )

        self.manifest = ExportManifest(
            name=self.request.name,
            entity=self.request.entity,
            project=self.request.project,
            mode=self.request.mode,
            created_at=(
                self.existing_manifest.created_at
                if self.existing_manifest is not None
                else self.exported_at
            ),
            updated_at=self.exported_at,
            runs_path=str(self.runs_path),
            history_path=(
                str(self.history_path)
                if self.request.mode == ExportMode.HISTORY
                else None
            ),
            run_count=len(self.snapshots),
            history_count=len(self.history_rows),
            selected_history_keys=manifest_keys,
            history_window=manifest_window,
        )
        self.store.paths.save_manifest(self.manifest)

    def _build_summary(self) -> ExportSummary:
        return ExportSummary(
            name=self.request.name,
            entity=self.request.entity,
            project=self.request.project,
            mode=self.request.mode,
            fetch_mode=self.request.fetch_mode,
            output_dir=str(self.store.paths.export_dir),
            state_path=str(self.store.paths.state_path),
            manifest_path=str(self.store.paths.manifest_path),
            runs_path=str(self.runs_path),
            history_path=(
                str(self.history_path)
                if self.request.mode == ExportMode.HISTORY
                else None
            ),
            run_count=len(self.snapshots),
            history_count=len(self.history_rows),
            exported_at=self.exported_at,
        )
