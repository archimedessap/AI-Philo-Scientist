#!/usr/bin/env bash
set -euo pipefail

# Extract key LaTeX warnings/errors into a concise summary file.

cd "$(dirname "$0")"

LOG="ai-philo-crossAI-arxiv.log"
OUT="warnings_summary.txt"

if [[ ! -f "$LOG" ]]; then
  echo "No log file: $LOG. Build first (./build.sh)." >&2
  exit 0
fi

{
  echo "# LaTeX Warnings Summary"
  date
  echo
  # Core patterns: errors, LaTeX/Package/pdfTeX warnings, box issues, undefineds
  grep -nEi '(^!|LaTeX Warning|Package [^ ]+ Warning|pdfTeX warning|Overfull \\hbox|Underfull \\hbox|Undefined|Missing|not found|No file|Rerun to get)' "$LOG" || true
} > "$OUT"

echo "Wrote $OUT"

