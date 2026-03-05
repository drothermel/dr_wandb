from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dr_wandb.sync_engine import (
    SyncEngine,
    load_policy,
    read_patch_jsonl,
    write_patch_jsonl,
)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def sync_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wandb-sync")
    parser.add_argument("entity")
    parser.add_argument("project")
    parser.add_argument("--runs-per-page", type=int, default=500)
    parser.add_argument("--state-path")
    parser.add_argument("--policy-module", default="dr_wandb.sync_policy")
    parser.add_argument("--policy-class", default="NoopPolicy")
    parser.add_argument("--output-json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)
    policy = load_policy(args.policy_module, args.policy_class)
    engine = SyncEngine(policy=policy)

    summary = engine.sync_project(
        entity=args.entity,
        project=args.project,
        state_path=Path(args.state_path) if args.state_path else None,
        runs_per_page=args.runs_per_page,
    )

    payload = summary.model_dump(mode="python")
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        logging.info("Saved sync summary to %s", output_path)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))

    return 0


def plan_patches_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wandb-plan-patches")
    parser.add_argument("entity")
    parser.add_argument("project")
    parser.add_argument("output_jsonl")
    parser.add_argument("--runs-per-page", type=int, default=500)
    parser.add_argument("--state-path")
    parser.add_argument("--policy-module", default="dr_wandb.sync_policy")
    parser.add_argument("--policy-class", default="NoopPolicy")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)
    policy = load_policy(args.policy_module, args.policy_class)
    engine = SyncEngine(policy=policy)

    plans = engine.plan_patches(
        entity=args.entity,
        project=args.project,
        state_path=Path(args.state_path) if args.state_path else None,
        runs_per_page=args.runs_per_page,
    )

    output_path = Path(args.output_jsonl)
    write_patch_jsonl(plans, output_path)
    logging.info("Planned %s patches", len(plans))
    logging.info("Saved patch plan to %s", output_path)

    return 0


def apply_patches_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wandb-apply-patches")
    parser.add_argument("patches_jsonl")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)
    plans = read_patch_jsonl(Path(args.patches_jsonl))
    engine = SyncEngine()
    results = engine.apply_patch_plans(plans, dry_run=not args.apply)
    payload = [result.model_dump(mode="python") for result in results]

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        logging.info("Saved apply results to %s", output_path)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(sync_main())
