#!/usr/bin/env bash
set -euo pipefail

# Build the paper PDF, then sanitize log paths and summarize warnings.

cd "$(dirname "$0")"

# Generate figures and tables from repository scripts (if available)
if command -v python3 >/dev/null 2>&1; then
  (
    cd .. && python3 scripts/gen_figs.py paper/figs || echo "[warn] gen_figs failed (missing deps?)"
  )
  (
    cd .. && python3 scripts/gen_tables.py paper/tables || echo "[warn] gen_tables failed (missing deps?)"
  )
  (
    cd .. && python3 scripts/gen_registry_tables.py paper/tables || echo "[warn] gen_registry_tables failed"
  )
else
  echo "[warn] python3 not found; skipping figure/table generation"
fi

# Compile with latexmk (uses local .latexmkrc if present)
latexmk -pdf -interaction=nonstopmode -file-line-error ai-philo-crossAI-arxiv.tex

# Normalize absolute paths in aux files to relative paths
./clean_log_paths.sh

# Summarize warnings into a file for quick review
./summarize_warnings.sh

echo "Build complete. See ai-philo-crossAI-arxiv.pdf and warnings_summary.txt."
