# AGENTS.md

## How to build paper
- Export figures: `python3 scripts/gen_figs.py paper/figs`
- Export tables: `python3 scripts/gen_tables.py paper/tables`
- Build PDF (from repo root): `bash scripts/build_paper.sh` or `make paper`

## Guardrails
- When changing experiment code, prefer running minimal validations (e.g., `pytest -q` or a smoke script if available) before updating paper assets.
- Use `\label{}` + `\cref{}` for cross-references in LaTeX.
- Cite only entries from `paper/ai-philo-refs.bib`.

