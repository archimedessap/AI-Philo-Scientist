#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_m3_auto_refinement.py
=========================
一次性完成 Top-N 筛选 + 深度优化循环 + 结果排行榜汇总。

使用示例：
python run_m3_auto_refinement.py \
  --summary_file data/demo/final_evaluation_summary.json \
  --theories_root data/demo/eval_ready_theories \
  --output_dir data/m3_runs \
  --top_n 5 \
  --eval_mode real \
  --max_iters 2
"""

from __future__ import annotations
import os
import sys
import argparse
import time
import json
from pathlib import Path
import subprocess

# -----------------------------------------------------------------------------
# 第 0 步: 工具函数
# -----------------------------------------------------------------------------

def run_cmd(cmd: list[str]):
    print("\n$ " + " ".join(cmd))
    return subprocess.call(cmd)

# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="M3 一键深度优化循环")
    parser.add_argument("--summary_file", required=True, help="final_evaluation_summary.json 路径")
    parser.add_argument("--theories_root", required=True, help="eval_ready_theories 目录")
    parser.add_argument("--output_dir", default="data/m3_runs", help="输出根目录")
    parser.add_argument("--top_n", type=int, default=5, help="筛选前 N")
    parser.add_argument("--eval_mode", choices=["quick", "real"], default="quick", help="评估模式")
    parser.add_argument("--max_iters", type=int, default=3, help="深度优化最大迭代次数")
    parser.add_argument("--min_improve", type=float, default=0.05, help="视为有效提升的最小 Δ 分")
    parser.add_argument("--judge_model_source", default="deepseek", help="评审 LLM 来源")
    parser.add_argument("--judge_model_name", default="deepseek-reasoner", help="评审 LLM 名称")
    parser.add_argument("--dialog_model_source", default="deepseek", help="对话 LLM 来源")
    parser.add_argument("--dialog_model_name", default="deepseek-reasoner", help="对话 LLM 名称")

    args = parser.parse_args()

    # 创建唯一输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_dir) / f"run_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 运行输出目录: {run_root}")

    # ---------------------------------------------------------------------
    # 步骤 1: 执行批量深度优化
    # ---------------------------------------------------------------------
    depth_output_dir = run_root / "depth_output"
    depth_output_dir.mkdir(exist_ok=True)

    cmd_refine = [
        "python", "run_refinement_loop.py",
        "--summary_file", args.summary_file,
        "--theories_root", args.theories_root,
        "--output_root", str(depth_output_dir),
        "--top_n", str(args.top_n),
        "--min_improve", str(args.min_improve),
        "--eval_mode", args.eval_mode,
        "--max_iters", str(args.max_iters),
        "--judge_model_source", args.judge_model_source,
        "--judge_model_name", args.judge_model_name,
        "--dialog_model_source", args.dialog_model_source,
        "--dialog_model_name", args.dialog_model_name,
    ]

    if run_cmd(cmd_refine) != 0:
        print("[FATAL] 深度优化阶段失败。")
        sys.exit(1)

    # ---------------------------------------------------------------------
    # 步骤 2: 生成排行榜 (根据 depth_runs/*/summary.json)
    # ---------------------------------------------------------------------
    summaries_dir = depth_output_dir / "depth_runs"
    leaderboard = []
    for summary_file in summaries_dir.rglob("summary.json"):
        with open(summary_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            leaderboard.append({
                "theory_name": data["theory_name"],
                "final_score": data["final_score"],
                "iterations": data["iterations"],
                "summary_path": str(summary_file)
            })

    leaderboard.sort(key=lambda x: x["final_score"], reverse=True)
    lb_path = run_root / "leaderboard.json"
    with open(lb_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=2)

    # 同时输出 markdown 方便查看
    md_lines = ["| Rank | Theory | Score | Iters |", "| ---- | ------ | ----- | ----- |"]
    for idx, item in enumerate(leaderboard, 1):
        md_lines.append(f"| {idx} | {item['theory_name']} | {item['final_score']:.3f} | {item['iterations']} |")
    with open(run_root / "leaderboard.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("\n🏆 排行榜已生成:", lb_path)
    print("运行全部完成！🎉")


if __name__ == "__main__":
    main() 