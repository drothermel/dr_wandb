from __future__ import annotations

from typing import Any

from dr_wandb.sync_types import ApplyResult, PatchPlan


def _current_tags(run: Any) -> list[str]:
    tags = getattr(run, "tags", None)
    if tags is None:
        return []
    return list(tags)


def apply_run_patch(run: Any, patch: PatchPlan, *, dry_run: bool = True) -> ApplyResult:
    existing_config = dict(getattr(run, "config", {}) or {})
    changed_keys = [
        key for key, value in patch.set_config.items() if existing_config.get(key) != value
    ]

    current_tags = _current_tags(run)
    next_tags = set(current_tags)
    added_tags = [tag for tag in patch.add_tags if tag not in next_tags]
    for tag in patch.add_tags:
        next_tags.add(tag)

    removed_tags = [tag for tag in patch.remove_tags if tag in next_tags]
    for tag in patch.remove_tags:
        next_tags.discard(tag)

    changed = bool(changed_keys or added_tags or removed_tags)
    applied = False

    if changed and not dry_run:
        run.config = dict(getattr(run, "config", None) or {})
        for key in changed_keys:
            run.config[key] = patch.set_config[key]
        run.tags = sorted(next_tags)
        run.update()
        applied = True

    return ApplyResult(
        run_id=getattr(run, "id", ""),
        run_name=getattr(run, "name", ""),
        dry_run=dry_run,
        changed=changed,
        applied=applied,
        changed_config_keys=changed_keys,
        added_tags=added_tags,
        removed_tags=removed_tags,
    )
