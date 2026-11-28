@AGENTS.md

## Claude Code specifics
- You run on Jedd's Mac inside the repo, so read outputs from `results/` and logs directly instead of asking Jedd to paste them.
- Run `pytest tests/` yourself after code changes and report the pass count from the actual output.
- `scripts/evaluate_test.py` is blocked in `.claude/settings.json` until CP7. Don't work around the block. Ask Jedd to lift it.
