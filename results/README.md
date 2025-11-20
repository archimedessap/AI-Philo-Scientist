# Paper-ready Evaluation Assets

This folder tracks only the lightweight evaluation summaries that are cited in the paper. Raw run folders under `evaluation_results/` or `theory_visuals_multi/` stay locally to avoid bloating the repo. A one-page quicklook for reviewers is in `results/overview.md` (with pipeline SVG and top-ranked theories).

## Contents

- `prior_eval_paper/run_20251007_prior_eval/final_evaluation_summary.json`
- `prior_eval_paper/run_20251008_prior_eval_multi/final_evaluation_summary.json`

Both files are canonicalized copies of the corresponding `final_evaluation_summary.json` artifacts captured under `evaluation_results/`. The entries are sorted alphabetically by theory name to guarantee stable diffs.

## Regeneration

1. Re-run the evaluation pipelines (e.g., `run_prior_theories_evaluation.py`).
2. Copy the resulting `final_evaluation_summary.json` into a new timestamped directory under `results/prior_eval_paper/`.
3. （可选）运行 `python3 scripts/gen_tables.py tmp_tables` 在本地生成最新的表格或统计摘要；本分支不再维护 LaTeX 或论文目录，仅保留核心运行代码与关键评估结果。

This keeps the paper artifacts reproducible without forcing every large intermediate result into version control.
