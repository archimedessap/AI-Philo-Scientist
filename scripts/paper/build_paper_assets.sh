#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="paper_assets"
TOP_N=${TOP_N:-20}
CASE_N=${CASE_N:-3}

echo "[1/3] 汇总注册库与最近运行 (Top-$TOP_N)"
python scripts/paper/aggregate_results.py \
  --registry_index global_theory_registry/theory_index.json \
  --manifests_glob 'output_*/*/run_manifest.json' \
  --output_dir "$OUT_DIR" \
  --top_n "$TOP_N"

echo "[2/3] 生成案例卡片 (Top-$CASE_N)"
python scripts/paper/generate_case_cards.py \
  --registry_index global_theory_registry/theory_index.json \
  --output_dir "$OUT_DIR/case_cards" \
  --top_n "$CASE_N"

echo "[3/3] 完成。输出目录: $OUT_DIR"

