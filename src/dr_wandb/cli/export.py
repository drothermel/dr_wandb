from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dr_wandb.export_profile import (
    apply_bootstrap_archives,
    build_profile,
    resolve_bootstrap_plan,
    resolve_profile_paths,
    validate_explicit_path_mode,
)
from dr_wandb.sync_engine import SyncEngine, load_policy
from dr_wandb.sync_types import BootstrapConfig, ExportConfig, FetchMode, RefreshScope


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def export_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wandb-export")
    parser.add_argument("entity")
    parser.add_argument("project")
    parser.add_argument("output_dir", nargs="?")
    parser.add_argument("--output-format", choices=["parquet", "jsonl"], default="parquet")
    parser.add_argument(
        "--fetch-mode",
        choices=[mode.value for mode in FetchMode],
        default=FetchMode.INCREMENTAL.value,
    )
    parser.add_argument(
        "--refresh-scope",
        choices=[scope.value for scope in RefreshScope],
        default=RefreshScope.UNFINISHED_ONLY.value,
    )
    parser.add_argument("--runs-per-page", type=int, default=500)
    parser.add_argument("--state-path")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--checkpoint-every-runs", type=int, default=25)
    parser.add_argument("--no-incremental", action="store_true")
    parser.add_argument("--no-finalize-compact", action="store_true")
    parser.add_argument("--inspection-sample-rows", type=int, default=5)
    parser.add_argument("--policy-module", default="dr_wandb.sync_policy")
    parser.add_argument("--policy-class", default="NoopPolicy")
    parser.add_argument("--policy-kwargs-json")
    parser.add_argument("--output-json")
    parser.add_argument("--export-name")
    parser.add_argument("--data-root")
    parser.add_argument("--sync-root")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)
    profile = build_profile(
        export_name=args.export_name,
        data_root=args.data_root,
        sync_root=args.sync_root,
    )
    validate_explicit_path_mode(
        profile=profile,
        explicit_output_dir=args.output_dir,
        explicit_state_path=args.state_path,
        explicit_output_json=args.output_json,
    )
    profile_paths = resolve_profile_paths(profile) if profile is not None else None
    if profile_paths is None and args.output_dir is None:
        raise ValueError("output_dir is required unless profile mode is used")

    policy_kwargs = json.loads(args.policy_kwargs_json) if args.policy_kwargs_json else {}
    policy = load_policy(args.policy_module, args.policy_class, policy_kwargs=policy_kwargs)
    engine = SyncEngine(policy=policy)

    cfg = ExportConfig(
        entity=args.entity,
        project=args.project,
        output_dir=profile_paths.output_dir if profile_paths is not None else Path(args.output_dir),
        output_format=args.output_format,
        fetch_mode=FetchMode(args.fetch_mode),
        refresh_scope=RefreshScope(args.refresh_scope),
        runs_per_page=args.runs_per_page,
        state_path=(
            profile_paths.state_path
            if profile_paths is not None
            else (Path(args.state_path) if args.state_path else None)
        ),
        save_every=args.save_every,
        incremental=not args.no_incremental,
        checkpoint_every_runs=args.checkpoint_every_runs,
        finalize_compact=not args.no_finalize_compact,
        inspection_sample_rows=args.inspection_sample_rows,
        policy_module=args.policy_module,
        policy_class=args.policy_class,
        policy_kwargs=policy_kwargs,
    )
    logging.info(
        "Starting wandb export for %s/%s (format=%s, fetch_mode=%s, refresh_scope=%s, runs_per_page=%s, save_every=%s, incremental=%s, checkpoint_every_runs=%s, finalize_compact=%s)",
        cfg.entity,
        cfg.project,
        cfg.output_format,
        cfg.fetch_mode,
        cfg.refresh_scope,
        cfg.runs_per_page,
        cfg.save_every,
        cfg.incremental,
        cfg.checkpoint_every_runs,
        cfg.finalize_compact,
    )
    logging.info(
        "Using policy %s.%s with state path %s and kwargs %s",
        cfg.policy_module,
        cfg.policy_class,
        cfg.state_path,
        cfg.policy_kwargs,
    )
    summary = engine.export_project(cfg)
    payload = summary.model_dump(mode="python")

    output_path = (
        profile_paths.export_summary_path
        if profile_paths is not None
        else (Path(args.output_json) if args.output_json else None)
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        logging.info("Saved export summary to %s", output_path)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))

    return 0


def bootstrap_export_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wandb-bootstrap-export")
    parser.add_argument("entity")
    parser.add_argument("project")
    parser.add_argument("source_dir", nargs="?")
    parser.add_argument("output_dir", nargs="?")
    parser.add_argument("--output-format", choices=["parquet", "jsonl"])
    parser.add_argument("--state-path")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--no-finalize-compact", action="store_true")
    parser.add_argument("--inspection-sample-rows", type=int, default=5)
    parser.add_argument("--policy-module", default="dr_wandb.sync_policy")
    parser.add_argument("--policy-class", default="NoopPolicy")
    parser.add_argument("--policy-kwargs-json")
    parser.add_argument("--output-json")
    parser.add_argument("--export-name")
    parser.add_argument("--data-root")
    parser.add_argument("--sync-root")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)
    profile = build_profile(
        export_name=args.export_name,
        data_root=args.data_root,
        sync_root=args.sync_root,
    )
    validate_explicit_path_mode(
        profile=profile,
        explicit_output_dir=args.output_dir,
        explicit_state_path=args.state_path,
        explicit_output_json=args.output_json,
    )
    profile_paths = resolve_profile_paths(profile) if profile is not None else None
    bootstrap_plan = None
    if profile_paths is None and (args.source_dir is None or args.output_dir is None):
        raise ValueError("source_dir and output_dir are required unless profile mode is used")
    if profile_paths is not None:
        if args.overwrite_output:
            raise ValueError("--overwrite-output is not supported in profile mode")
        bootstrap_plan = resolve_bootstrap_plan(
            profile_paths=profile_paths,
            explicit_source_dir=Path(args.source_dir) if args.source_dir else None,
        )
        apply_bootstrap_archives(bootstrap_plan)

    policy_kwargs = json.loads(args.policy_kwargs_json) if args.policy_kwargs_json else {}
    policy = load_policy(args.policy_module, args.policy_class, policy_kwargs=policy_kwargs)
    engine = SyncEngine(policy=policy)

    cfg = BootstrapConfig(
        entity=args.entity,
        project=args.project,
        source_dir=(
            bootstrap_plan.source_dir
            if bootstrap_plan is not None
            else Path(args.source_dir)
        ),
        output_dir=(
            bootstrap_plan.output_dir
            if bootstrap_plan is not None
            else Path(args.output_dir)
        ),
        output_format=args.output_format,
        state_path=(
            bootstrap_plan.state_path
            if bootstrap_plan is not None
            else (Path(args.state_path) if args.state_path else None)
        ),
        overwrite_output=args.overwrite_output,
        finalize_compact=not args.no_finalize_compact,
        inspection_sample_rows=args.inspection_sample_rows,
        policy_module=args.policy_module,
        policy_class=args.policy_class,
        policy_kwargs=policy_kwargs,
    )
    logging.info(
        "Starting bootstrap export for %s/%s (source_dir=%s, output_dir=%s, output_format=%s, overwrite_output=%s, finalize_compact=%s)",
        cfg.entity,
        cfg.project,
        cfg.source_dir,
        cfg.output_dir,
        cfg.output_format,
        cfg.overwrite_output,
        cfg.finalize_compact,
    )
    logging.info(
        "Using policy %s.%s with state path %s and kwargs %s",
        cfg.policy_module,
        cfg.policy_class,
        cfg.state_path,
        cfg.policy_kwargs,
    )
    summary = engine.bootstrap_export(cfg)
    payload = summary.model_dump(mode="python")

    output_path = (
        bootstrap_plan.bootstrap_summary_path
        if bootstrap_plan is not None
        else (Path(args.output_json) if args.output_json else None)
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        logging.info("Saved bootstrap summary to %s", output_path)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(export_main())
