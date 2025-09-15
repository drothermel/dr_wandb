# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Development
- `us` runs `uv sync --group all` - Install all dependencies including dev, test, and test-ml groups
- `lint` runs `uv run ruff check --fix .` - Lint code with ruff and apply autofixes where possible
- `ft` runs `uv run ruff format .` - Format code with ruff  
- `mp` runs `uv run mypy src/` - Type check with mypy
- `pt` runs `uv run pytest` - Run tests with pytest (supports parallel execution with xdist)

### Configuration Philosophy
- Strict type checking with mypy
- Comprehensive linting with ruff (88 char line length, extensive rule set)
- Google-style docstrings
- ML-friendly settings (higher complexity thresholds, performance-focused rules)
- Uses assertions over exceptions for validation (performance requirement)
- Excludes notebooks and outputs directories from linting

### Testing Setup
- pytest with coverage, parallel execution (xdist), timeout protection
- Separate test groups: dev, test, test-ml for different environments
- Custom markers for slow, serial, and integration tests
