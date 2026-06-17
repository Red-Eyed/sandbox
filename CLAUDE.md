# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A **sandbox** — a scratch repo for quick experiments and one-off tries, not a single
coherent application. Each experiment lives on its own **orphan branch** with no shared
history, so `main` stays essentially empty (just `.gitignore` and `LICENSE`).

Implications when working here:

- **Treat each branch as its own project.** Don't assume code, dependencies, or conventions
  carry over from `main` or from another branch — they almost certainly don't. Read the
  branch you're actually on before reasoning about it.
- **Don't try to unify or refactor across branches.** Orphan branches are deliberately
  independent; there is no cross-branch architecture to preserve.
- **`main` is intentionally bare.** An empty-looking checkout is expected, not a mistake to
  "fix" by adding scaffolding.

## Starting a new experiment

Create an orphan branch so the experiment carries no history from `main`:

```sh
git checkout --orphan <experiment-name>
git rm -rf .          # clear the index inherited from the previous branch
```

Then scaffold whatever that experiment needs (e.g. `uv init` for Python — see below).

## Conventions

- The `.gitignore` is the standard Python template, so experiments default to Python. Use
  **`uv`** for all Python work (`uv run`, `uv pip install`, `uv sync`) — never bare
  `pip`/`python`. Lint/format/type-check with `uv run ruff check`, `uv run ruff format`, and
  `uv run pyrefly check` before committing.
- There is no repo-wide build/test command — each branch defines its own. Check that
  branch's `pyproject.toml` / `README` / scripts for how to build, run, and test it.
