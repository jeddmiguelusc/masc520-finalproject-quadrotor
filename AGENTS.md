# AGENTS.md — MASC 520 final project (NeuroBEM quadrotor dynamics)

Shared instructions for every AI agent working in this repo (Claude Code, Claude desktop app, Antigravity, Cursor, Codex, ...).
Owner: Jedd. Solo project. AI use is allowed and is cited in the report.

## Start of every session
1. Read `docs/ai/project-plan.md` (plan of record) and the top entry of `docs/ai/progress-log.md`.
2. In 2–3 lines, tell Jedd which checkpoint we're on, what the last session finished, and what's left today.
3. Run `git log --oneline -10` and `git status`. If there are commits newer than the top log entry, another agent worked here without logging it, so say so before starting.

## Wrap-up (when Jedd says "wrap up", or a checkpoint gate passes)
- Add a new dated entry at the top of `docs/ai/progress-log.md`, using the entry format at the top of that file:
  checkpoint; done; files created/changed; key numbers with their source file in `results/`; decisions + rejected alternatives; open issues; next step.
- Never rewrite old entries. Put corrections in a new entry.
- If scope, dates or cuts changed, update `docs/ai/project-plan.md`.
- Name the tool you are in the entry title, e.g. `## 2026-11-28 (Claude Code) — CP7: ...`.
- Tell Jedd which plan-doc checkboxes to tick. The living checklist is a claude.ai artifact, and only Claude on claude.ai can edit it.

## Hard rules
- Every reported number must come from a file in `results/`, never from chat or memory. Quote the file path.
- Validation only. The test set is evaluated exactly once, at CP7 (`scripts/evaluate_test.py`), and only after Jedd says go. Never load the test part of any split otherwise.
- Model inputs: vel, ang vel, gravity-in-body, motor speeds, dmot, vbat. Never acc or ang acc.
- Equal tuning budget across models. Log git hash + config for every run (`results/runs.csv`).
- Hardware: train on the Mac with MPS. Never train the TCN (M3) on CPU (~90× slower).
- On MPS, move tensors to CPU before converting to float64 (`.cpu().double()`, never `.double().cpu()`).
- Do not commit the dataset (no license). The repo only has the download script.
- `docs/decisions.md` is written by Jedd. Don't edit it. At each decision-log step, give Jedd a factual summary he can use to write it himself.
- Don't overwrite a committed script whose results already exist in `results/`. Add a new script under its own name.
- Commit with a clear message after each unit of work so other agents can see it.

## Key facts
- Data: NeuroBEM, 247 segment CSVs → 95 flights (94 used; the 164 Hz flight is excluded). Splits in `data/splits.json` (A, A_r1–A_r3, B_s). Details: see the plan's "Data facts".
- Metric: paper-definition RMSE (per component, Table II), flight-bootstrap 95% CIs. `runs.csv` stores vector norms = √3 × paper F.
- Tests: `pytest tests/` (99 passing as of Nov 27).

## Layout
- `src/data/` loader, frames, targets, splits, dataset · `src/models/` P0b physics, MLP, TCN, transformer, ensemble · `src/eval/` metrics, baselines, noise · `src/sim/` rollout integrators
- `scripts/` training queues, eval, rollout eval, fine-tune · `configs/` YAML per model · `results/` all numbers + figures · `tests/`
- `docs/ai/` plan + progress log (shared with all agents) · `docs/decisions.md` (Jedd's)

## Deadlines
Results freeze Sat Nov 28, 6 pm (tag v0.9-results) · Poster Mon Nov 30, 11:59 pm (target 8 pm) · 10-min talk TBC (assume Dec 1–2) · IEEE report Sun Dec 6.

## Reply format
End every reply with a "Steps for Jedd" section listing exactly what he needs to do (files, where they go, commands to run).
