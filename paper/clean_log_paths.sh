#!/usr/bin/env bash
set -euo pipefail

# Replace absolute paths pointing to this paper directory with 'paper/' in
# LaTeX-generated files to avoid environment-specific path leakage.

cd "$(dirname "$0")"

BASE_DIR="$(pwd)"

sanitize() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  local tmp="$f.tmp.$$"
  # Map absolute path of this folder to a stable relative prefix
  sed "s|${BASE_DIR}/|paper/|g" "$f" > "$tmp" && mv "$tmp" "$f"
}

sanitize ai-philo-crossAI-arxiv.log
sanitize ai-philo-crossAI-arxiv.fls
sanitize ai-philo-crossAI-arxiv.fdb_latexmk

echo "Sanitized absolute paths in .log/.fls/.fdb_latexmk to 'paper/'."

