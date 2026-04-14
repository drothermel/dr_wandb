# Changelog

## 3.0.2

- Added CLI progress logging so `wandb-export` reports the first processed
  run and then each additional 5% completion bucket during exports.
- Added `DR_WANDB_LOG_LEVEL` so CLI log verbosity can be overridden for batch
  runs and CI without changing code.

## 3.0.1

- Updated `dr-wandb` to depend on `dr-ds==0.1.4`.
- Added regression coverage for W&B run attributes like `user` that may
  arrive as non-JSON-serializable SDK objects.
- Verified export/store round-trips now normalize those objects into
  JSON-safe data instead of failing during snapshot writes.
- Added a first pass of public API docstrings plus repo-level documentation
  and verification guidance for the shared library/CLI surface.
