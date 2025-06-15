#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_refinement_loop.py
======================

批量运行理论深度优化 (MVP)。
步骤：
1. 读取 final_evaluation_summary.json
2. 选 Top-N 理论（借助 candidate_selector）
3. 对每个理论调用 depth_optimizer.optimize_once

用法示例：
python run_refinement_loop.py \
  --summary_file data/.../final_evaluation_summary.json \
  --theories_root data/.../eval_ready_theories \
  --output_root data/.../4_refinement \
  --top_n 5
"""

from __future__ import annotations
import argparse
import os
import json
from pathlib import Path
from typing import List, Dict

# 动态加入项目根目录，方便包导入
import sys as _sys, os as _os
_PROJECT_ROOT = _os.path.abspath(_os.path.dirname(__file__))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

from tools.refinement.candidate_selector import load_summary, select_top_n, attach_json_path
from tools.refinement.depth_optimizer import optimize_once

# 默认最大迭代次数（与 depth_optimizer 保持一致）
DEFAULT_MAX_ITERS = 3

def run_batch_refinement(summary_file: str, theories_root: str, output_root: str,
                         top_n: int, min_improve: float, eval_mode: str,
                         max_iters: int,
                         judge_src: str,
                         judge_name: str,
                         dialog_src: str,
                         dialog_name: str):
    Path(output_root).mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_file)
    selected = select_top_n(summary, top_n)
    attach_json_path(selected, theories_root)

    # 保存选取列表
    with open(Path(output_root) / "selected_candidates.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 已选择 Top-{top_n} 理论，开始逐个优化...\n")

    for idx, entry in enumerate(selected, 1):
        theory_name = entry["theory_name"]
        print("="*80)
        print(f"({idx}/{len(selected)}) 优化 {theory_name}")
        print("="*80)
        optimize_once(
            entry,
            output_dir=output_root+"/depth_runs",
            min_improve=min_improve,
            eval_mode=eval_mode,
            max_iters=max_iters,
            judge_model_source=judge_src,
            judge_model_name=judge_name,
            dialog_model_source=dialog_src,
            dialog_model_name=dialog_name,
        )

    print("\n🎉 批量优化完成，结果位于:", output_root)


def main():
    parser = argparse.ArgumentParser(description="批量理论深度优化 (MVP)")
    parser.add_argument("--summary_file", required=True, help="final_evaluation_summary.json 路径")
    parser.add_argument("--theories_root", required=True, help="eval_ready_theories 目录")
    parser.add_argument("--output_root", required=True, help="输出根目录（会创建 selected_candidates.json 和 depth_runs/）")
    parser.add_argument("--top_n", type=int, default=5, help="筛选前 N")
    parser.add_argument("--min_improve", type=float, default=0.05, help="视为改进的最小分数增量")
    parser.add_argument("--eval_mode", choices=["quick", "real"], default="quick", help="评估模式：quick / real")
    parser.add_argument("--max_iters", type=int, default=DEFAULT_MAX_ITERS, help="最大迭代轮数")
    parser.add_argument("--judge_model_source", default="deepseek", help="评审 LLM 来源")
    parser.add_argument("--judge_model_name", default="deepseek-reasoner", help="评审 LLM 名称")
    parser.add_argument("--dialog_model_source", default="deepseek", help="对话 LLM 来源")
    parser.add_argument("--dialog_model_name", default="deepseek-reasoner", help="对话 LLM 名称")

    args = parser.parse_args()

    run_batch_refinement(
        args.summary_file,
        args.theories_root,
        args.output_root,
        args.top_n,
        args.min_improve,
        args.eval_mode,
        args.max_iters,
        args.judge_model_source,
        args.judge_model_name,
        args.dialog_model_source,
        args.dialog_model_name,
    )

if __name__ == "__main__":
    main() 