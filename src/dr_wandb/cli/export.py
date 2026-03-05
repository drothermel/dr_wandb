from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dr_wandb.sync_engine import SyncEngine, load_policy
from dr_wandb.sync_types import ExportConfig


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
    parser.add_argument("output_dir")
    parser.add_argument("--output-format", choices=["parquet", "jsonl"], default="parquet")
    parser.add_argument("--runs-per-page", type=int, default=500)
    parser.add_argument("--state-path")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--policy-module", default="dr_wandb.sync_policy")
    parser.add_argument("--policy-class", default="NoopPolicy")
    parser.add_argument("--output-json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)
    policy = load_policy(args.policy_module, args.policy_class)
    engine = SyncEngine(policy=policy)

    cfg = ExportConfig(
        entity=args.entity,
        project=args.project,
        output_dir=Path(args.output_dir),
        output_format=args.output_format,
        runs_per_page=args.runs_per_page,
        state_path=Path(args.state_path) if args.state_path else None,
        save_every=args.save_every,
        policy_module=args.policy_module,
        policy_class=args.policy_class,
    )
    summary = engine.export_project(cfg)
    payload = summary.model_dump(mode="python")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        logging.info("Saved export summary to %s", output_path)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(export_main())
