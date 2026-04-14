# CLAUDE.md

## Documentation Style Guide

`dr-wandb` is a shared library and CLI surface. Treat docstrings as part of
the public contract, especially where behavior is incremental, stateful, or
opinionated.

### What to document

- Add docstrings to public modules, public functions, public classes, and
  exported helpers that downstream repos are expected to call.
- Prefer docstrings over inline comments. Add inline comments only for
  invariants or non-obvious implementation details that callers would not
  learn from the public API docs.
- Keep private helpers lightly documented unless their behavior is subtle
  enough that a short docstring materially improves maintenance.

### Docstring content

- Start with a short imperative summary sentence.
- Describe behavior, not implementation history.
- Call out state transitions, incremental-vs-full behavior, ordering
  guarantees, deduplication semantics, filesystem layout, and failure modes
  when they matter.
- Be explicit about lossy or opinionated normalization, especially when
  translating raw W&B SDK objects into stored JSON-safe data.

### Style rules

- Be concise. Most function docstrings should be 1-5 lines.
- Prefer plain English over section-heavy API reference formatting.
- Do not restate obvious type information unless runtime behavior is more
  specific than the signature.

## Verification Workflow

### Local iteration

Run the smallest targeted test command that covers the code you changed.

### Before commit

Run the full repo checks:

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
```

### Before publish

Make sure the package still builds cleanly:

```bash
uv build
```
