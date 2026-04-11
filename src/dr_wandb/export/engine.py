from __future__ import annotations

from typing import Any

from dr_ds.serialization import utc_now_iso

from dr_wandb.export.export_manifest import ExportManifest
from dr_wandb.export.export_modes import ExportMode, FetchMode
from dr_wandb.export.export_request import ExportRequest
from dr_wandb.export.export_state import ExportState, RunTrackingState
from dr_wandb.export.export_summary import ExportSummary
from dr_wandb.export.history_export import (
    load_existing_history_rows,
    max_history_step,
    merge_history_rows,
    observed_last_history_step,
    scan_history_for_export,
    selected_history_keys,
    selected_history_window,
)
from dr_wandb.export.record_store import RecordStore
from dr_wandb.export.run_payloads import build_raw_run_payload
from dr_wandb.export.run_selection import select_runs
from dr_wandb.export.run_snapshot import RunSnapshot


def _build_default_api(timeout_seconds: int) -> Any:
    import wandb

    return wandb.Api(timeout=timeout_seconds)


class ExportEngine:
    def export(self, request: ExportRequest) -> ExportSummary:
        store = RecordStore.from_name_and_root(
            name=request.name,
            data_root=request.data_root,
        )
        paths = store.paths
        state = paths.load_state(
            entity=request.entity, project=request.project
        )
        existing_manifest = store.load_manifest()
        existing_snapshots = store.load_existing_snapshots(
            request=request,
            manifest=existing_manifest,
        )
        existing_history_rows = load_existing_history_rows(
            store=store,
            request=request,
            manifest=existing_manifest,
        )

        if request.fetch_mode == FetchMode.FULL_RECONCILE:
            state = ExportState(
                name=request.name,
                entity=request.entity,
                project=request.project,
            )
            existing_snapshots = {}
            existing_history_rows = []

        api = _build_default_api(request.timeout_seconds)
        runs = select_runs(api=api, request=request, state=state)
        exported_at = utc_now_iso()

        snapshot_by_id = dict(existing_snapshots)
        new_history_rows = []

        for run in runs:
            run_id = str(getattr(run, "id", ""))
            if run_id == "":
                continue
            tracking = state.runs.get(run_id)
            last_history_step = (
                tracking.last_history_step if tracking is not None else None
            )
            raw_run = build_raw_run_payload(
                run=run,
                include_metadata=request.include_metadata,
            )
            snapshot_by_id[run_id] = RunSnapshot(
                run_id=run_id,
                entity=request.entity,
                project=request.project,
                exported_at=exported_at,
                raw_run=raw_run,
            )

            tracking_state = RunTrackingState.from_run(
                run,
                last_history_step=last_history_step,
            )

            if request.mode == ExportMode.HISTORY:
                history_ctx = self._history_context(
                    request=request,
                    run=run,
                    run_last_history_step=last_history_step,
                )
                history_rows = scan_history_for_export(
                    request=request,
                    ctx=history_ctx,
                )
                new_history_rows.extend(history_rows)
                max_step = max_history_step(history_rows)
                observed_last_step = observed_last_history_step(run)
                if observed_last_step is not None:
                    max_step = (
                        observed_last_step
                        if max_step is None
                        else max(max_step, observed_last_step)
                    )
                tracking_state.last_history_step = max_step

            state.runs[run_id] = tracking_state
            if tracking_state.created_at is not None and (
                state.max_created_at is None
                or tracking_state.created_at > state.max_created_at
            ):
                state.max_created_at = tracking_state.created_at

        snapshots = sorted(
            snapshot_by_id.values(),
            key=lambda snapshot: snapshot.sort_key,
            reverse=True,
        )
        if request.mode == ExportMode.HISTORY:
            history_rows = merge_history_rows(
                existing_history_rows=existing_history_rows,
                new_history_rows=new_history_rows,
            )
        else:
            history_rows = []

        state.last_exported_at = exported_at
        paths.save_state(state)

        runs_output_path = store.write_run_snapshots(
            output_format=request.output_format,
            snapshots=snapshots,
        )
        if request.mode == ExportMode.HISTORY:
            history_output_path = store.write_history_rows(
                output_format=request.output_format,
                rows=history_rows,
            )
        else:
            history_output_path = paths.history_path(request.output_format)
            store.remove_history(output_format=request.output_format)
        store.remove_other_formats(output_format=request.output_format)

        manifest = ExportManifest(
            name=request.name,
            entity=request.entity,
            project=request.project,
            mode=request.mode,
            output_format=request.output_format,
            created_at=(
                existing_manifest.created_at
                if existing_manifest is not None
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
                runs=runs,
            ),
            history_window=selected_history_window(
                request=request,
                runs=runs,
            ),
        )
        paths.save_manifest(manifest)
        return ExportSummary(
            name=request.name,
            entity=request.entity,
            project=request.project,
            mode=request.mode,
            fetch_mode=request.fetch_mode,
            output_dir=str(paths.export_dir),
            state_path=str(paths.state_path),
            manifest_path=str(paths.manifest_path),
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

    def _history_context(
        self,
        *,
        request: ExportRequest,
        run: Any,
        run_last_history_step: int | None,
    ) -> Any:
        from dr_wandb.export.policy import HistoryPolicyContext

        return HistoryPolicyContext.from_run(
            entity=request.entity,
            project=request.project,
            run=run,
            run_last_history_step=run_last_history_step,
        )
