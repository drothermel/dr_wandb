# CLAUDE.md

This file provides project level guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚨 READ FIRST - DESIGN PHILOSOPHY
**MANDATORY:** Before starting ANY work, read `docs/processes/design_philosophy.md` to understand the core principles and methodology that guide this project. All code changes must align with these principles.

- **No Backward Compatibility**: This is a research library - breaking changes are acceptable for better design
- **Fail Fast, Fail Loudly**: Use assertions, avoid defensive programming that hides bugs
- **No Exception Handling**: Never use try-catch blocks - let errors surface immediately
- **Assertions Over Exceptions**: Use `assert condition, "message"` instead of `raise ValueError()`
- **Minimize Friction**: Every design choice should reduce friction between idea and visualization
- **Embrace Change, Demand Consistency**: When making changes, update ALL affected parts

Remember: The goal is code that *disappears* into the background, allowing researchers to focus on their work.

## Essential Commands
- `us` runs `uv sync` - Install all dependencies including dev, test, and test-ml groups
- `lint` runs `uv run ruff check --fix .` - Lint code with ruff and apply autofixes where possible
- `ft` runs `uv run ruff format .` - Format code with ruff  
- `uv run pytest` - Run tests with pytest (supports parallel execution with xdist)
- `lint_fix` - Run ruff format and then check with --fix

**IMPORTANT**: Do NOT run tests, linting, type checking, or formatting unless explicitly requested by the user. Focus on the requested changes only.

## 🎯 CODE STYLE REQUIREMENTS

### Zero Comments Policy
- **NEVER add ANY comments** - no docstrings, no inline comments, no block comments
- Code must be self-documenting through clear naming and structure
- Remove ALL existing comments when editing files (docstrings, # comments, etc.)

### Comprehensive Typing
- **ALL function signatures** must have complete type hints for parameters and return values
- Use `from typing import Any, Optional` etc. as needed
- Prefer `list`, `dict` etc over `List` and `Dict`
- Add `from __future__ import annotations` and use modern type hints
- Import types like `import pandas as pd` when using `pd.DataFrame` in hints
- If a circular import exists, use `TYPE_CHECKING` to gate
- All `__init__` methods must have `-> None` return type
- All class methods need proper `self` typing context
- Use specific types over `Any` when possible (e.g., `pd.DataFrame` not `Any`)
- Create custom types for clarity: `type GroupKey = Tuple[Tuple[str, Any], ...]`
- Example pattern:
  ```python
  def method_name(self, param: str, optional_param: Optional[int] = None) -> Dict[str, Any]:
  ```

### File Structure
- **ALL imports at the very top** - no imports anywhere else in the file
- Type aliases near top after imports
- Magic values should NEVER be hardcoded throughout, all constants be semantically named at the top of the module
- No module-level docstrings - remove entirely
- Class definitions without docstrings
- Methods without docstrings but with full type hints

### Replace Comments with Structure
- **Instead of comments** → Extract succinctly named helper functions
- **Instead of complex types** → Create descriptive type aliases
- Examples:
  ```python
  # BAD: Complex code with comments
  def process_data(self, data):
      # Convert categorical columns to numeric for ML processing
      processed = data.copy()
      # ... complex logic ...
      
  # GOOD: Self-documenting through function names and types
  type CategoricalColumns = List[str]
  type NumericData = pd.DataFrame
  
  def process_data(self, data: pd.DataFrame) -> NumericData:
      return self._convert_categorical_to_numeric(data)
      
  def _convert_categorical_to_numeric(self, data: pd.DataFrame) -> NumericData:
      # Clear, focused function that explains itself
  ```

### Fail Fast and Loud: Asserts Not Try-Except
- **Always aim to check assumptions with asserts**
- Avoid nested try-except blocks
- Instead, identify assumptions and assert them at the top of the function

## 🛠️ DEVELOPMENT WORKFLOW

### When Editing Files
1. **Read design philosophy first** - understand the core method principles
2. **Strip ALL comments** - docstrings, inline comments, everything
3. **Add comprehensive type hints** - every parameter, every return value
4. **Extract helper functions** - instead of complex inline logic with comments
5. **Import required typing modules** - add to imports as needed
6. **Test functionality** - ensure no behavioral changes from refactoring

### Code Quality Gates
- **Use type hints** on all functions
- **ALL imports at file top** - never mid-file, never in functions, never anywhere else
- **Use assertions, not exceptions** - single line `assert condition, "message"` instead of try-catch or raising exceptions
- **Never use try-catch blocks** - let errors bubble up; use assertions for validation
- **Show full modified functions**, not just diffs
- **Prefer explicit code** over clever code
- **Follow "Leave No Trace"** - remove all legacy patterns when making changes

### Git Shortcuts
| Shortcut | Command | Use |
|----------|---------|-----|
| `gst` | `git status` | Check state |
| `gd_agent` | `git --no-pager diff` | See changes |
| `glo` | `git log --oneline -10` | Recent commits |
| `ga .` | `git add .` | Stage files |
| `gc -m "msg"` | `git commit -m "msg"` | Commit |

### 📋 COMMIT STRATEGY
- **Small, semantic commits**: 20-30 lines per commit with clear purpose
- **Single line messages**: Succinct and clear, imperative mood
- **Quality gates**: Run linting/formatting before commits only when explicitly requested
- **Incremental building**: Each commit should be reviewable and complete
