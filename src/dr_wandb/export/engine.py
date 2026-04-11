from __future__ import annotations

import wandb
from dr_ds.serialization import utc_now_iso

from dr_wandb.export.export_manifest import ExportManifest
from dr_wandb.export.export_modes import ExportMode
from dr_wandb.export.export_request import ExportRequest
from dr_wandb.export.export_summary import ExportSummary
from dr_wandb.export.history_export import (
    max_history_step,
    merge_history_rows,
    observed_last_history_step,
    scan_history_for_export,
    selected_history_keys,
    selected_history_window,
)
from dr_wandb.export.policy import HistoryPolicyContext
from dr_wandb.export.record_store import RecordStore
from dr_wandb.export.run_selection import select_runs
from dr_wandb.export.run_snapshot import RunSnapshot
from dr_wandb.export.wandb_run import WandbRun


class ExportEngine:
    def export(self, request: ExportRequest) -> ExportSummary:
        store = RecordStore.from_name_and_root(
            name=request.name,
            data_root=request.data_root,
        )
        existing = store.load_existing(request)
        api = wandb.Api(timeout=request.timeout_seconds)
        raw_runs = select_runs(api=api, request=request, state=existing.state)

        exported_at = utc_now_iso()
        new_history_rows = []
        wandb_runs: list[WandbRun] = []
        for raw_run in raw_runs:
            wandb_run = WandbRun.from_wandb_run(
                raw_run,
                entity=request.entity,
                project=request.project,
                include_metadata=request.include_metadata,
            )
            if wandb_run.run_id == "":
                continue
            wandb_runs.append(wandb_run)

            existing.snapshots[wandb_run.run_id] = RunSnapshot(
                run=wandb_run,
                exported_at=exported_at,
            )

            tracking_state = existing.state.begin_run_tracking(wandb_run)

            if request.mode == ExportMode.HISTORY:
                ctx = HistoryPolicyContext.from_wandb_run(
                    wandb_run=wandb_run,
                    raw_run=raw_run,
                    run_last_history_step=tracking_state.last_history_step,
                )
                history_rows = scan_history_for_export(
                    request=request,
                    ctx=ctx,
                )
                new_history_rows.extend(history_rows)
                max_step = max_history_step(history_rows)
                observed_last_step = observed_last_history_step(
                    wandb_run=wandb_run, raw_run=raw_run
                )
                if observed_last_step is not None:
                    max_step = (
                        observed_last_step
                        if max_step is None
                        else max(max_step, observed_last_step)
                    )
                tracking_state.last_history_step = max_step

        snapshots = sorted(
            existing.snapshots.values(),
            key=lambda snapshot: snapshot.sort_key,
            reverse=True,
        )
        if request.mode == ExportMode.HISTORY:
            history_rows = merge_history_rows(
                existing_history_rows=existing.history_rows,
                new_history_rows=new_history_rows,
            )
        else:
            history_rows = []

        existing.state.last_exported_at = exported_at
        store.paths.save_state(existing.state)

        runs_output_path = store.write_run_snapshots(snapshots=snapshots)
        if request.mode == ExportMode.HISTORY:
            history_output_path = store.write_history_rows(rows=history_rows)
        else:
            history_output_path = store.paths.history_path
            store.remove_history()

        manifest = ExportManifest(
            name=request.name,
            entity=request.entity,
            project=request.project,
            mode=request.mode,
            created_at=(
                existing.manifest.created_at
                if existing.manifest is not None
                else exported_at
            ),
            updated_at=exported_at,
            runs_path=str(runs_output_path),
            history_path=(
                str(history_output_path)
                if request.mode == ExportMode.HISTORY
                else None
            ),
            run_count=len(snapshots),
            history_count=len(history_rows),
            selected_history_keys=selected_history_keys(
                request=request,
                wandb_runs=wandb_runs,
                raw_runs=raw_runs,
            ),
            history_window=selected_history_window(
                request=request,
                wandb_runs=wandb_runs,
                raw_runs=raw_runs,
            ),
        )
        store.paths.save_manifest(manifest)
        return ExportSummary(
            name=request.name,
            entity=request.entity,
            project=request.project,
            mode=request.mode,
            fetch_mode=request.fetch_mode,
            output_dir=str(store.paths.export_dir),
            state_path=str(store.paths.state_path),
            manifest_path=str(store.paths.manifest_path),
            runs_path=str(runs_output_path),
            history_path=(
                str(history_output_path)
                if request.mode == ExportMode.HISTORY
                else None
            ),
            run_count=len(snapshots),
            history_count=len(history_rows),
            exported_at=exported_at,
        )
