# AGENTS.md

## CLI Usage
- `python cli.py load <scenario>` — copies scenario into `sandbox/` (required first step)
- `python cli.py check` — runs TDD verification + healing loop
- `python cli.py request "feature" --context file.md` — feature request with optional spec

## Sandbox & State
- All work happens in `sandbox/` (never edit scenarios/ directly)
- Always run `load` before `check` or `request`

## Coverage Gate (test_verifier_node)
- Enforces ≥90% branch coverage (`tools.py:77 COVERAGE_THRESHOLD`)
- If below threshold and tests pass, auto-writes edge-case tests (nulls, negatives, boundaries) up to 2 attempts
- Loops back to verifier until satisfied or max attempts reached

## Self-Healing & Escalation
- On test failure: captures traceback, proposes fix, retries (max 3 attempts)
- After 3 failures → escalator node (prompts for manual hint)

## Human-in-the-Loop
- Always pauses on `request` mode
- Pauses on plans containing: "new file", "database", "dependency"
- Input: `y` / `n` / free-text feedback

## Ecosystem & Tooling
- Auto-detects Python (pytest) vs Node (jest) via `package.json` vs `requirements.txt`
- Missing packages are auto-installed (`pip install` / `npm install`) inside `run_tests*`
- `.env` must contain `GEMINI_API_KEY`

## File Discovery
- `list_project_files` ignores: `node_modules`, `dist`, `build`, `__pycache__`, `.venv`, `.autonomie_backup`

## Verification
- No root-level lint, typecheck, or test scripts exist
- All test/coverage execution and healing is internal to the LangGraph
