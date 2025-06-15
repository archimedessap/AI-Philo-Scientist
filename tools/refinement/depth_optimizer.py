#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
depth_optimizer.py - 单理论深度优化循环 (MVP)
================================================

读取由 candidate_selector 选出的理论条目，对其进行 1 轮：
    评估 → (占位)对话优化 → 再评估
并保存评估结果与改进版本。
"""

from __future__ import annotations
import os
import json
import argparse
from typing import Dict
from pathlib import Path

# --- 处理从脚本路径直接执行时的包导入问题 ---
import sys as _sys, os as _os
_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

from tools.refinement.eval_proxy import evaluate
from tools.refinement.dialogue_optimizer import improve as dialogue_improve


DEFAULT_MIN_IMPROVE = 0.05  # 5pp
DEFAULT_MAX_ITERS = 3


def optimize_once(
    theory_entry: Dict,
    output_dir: str,
    min_improve: float = DEFAULT_MIN_IMPROVE,
    eval_mode: str = "quick",
    max_iters: int = DEFAULT_MAX_ITERS,
    judge_model_source: str = "deepseek",
    judge_model_name: str = "deepseek-reasoner",
    dialog_model_source: str = "deepseek",
    dialog_model_name: str = "deepseek-reasoner",
):
    """对单个理论执行多轮优化循环并保存日志"""

    theory_name = theory_entry["theory_name"]
    origin_path = theory_entry["json_path"]

    work_dir = Path(output_dir) / theory_name.replace(" ", "_")
    work_dir.mkdir(parents=True, exist_ok=True)

    current_path = origin_path
    history = []

    for it in range(max_iters):
        iter_dir = work_dir / f"iter_{it}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # 评估
        eval_res = evaluate(
            current_path,
            mode=eval_mode,
            model_source=judge_model_source,
            model_name=judge_model_name,
        )
        with open(iter_dir / "eval.json", "w", encoding="utf-8") as f:
            json.dump(eval_res, f, ensure_ascii=False, indent=2)

        print(f"[EVAL] {theory_name} iter={it} role_score={eval_res['role_score']:.3f}")

        # 如果不是第一轮，计算提升
        if it > 0:
            prev_score = history[-1]["role_score"]
            delta = eval_res["role_score"] - prev_score
            if delta < min_improve:
                print(f"[STOP] Δ={delta:.3f} < min_improve ({min_improve}), 提前结束迭代")
                break

        # 准备下一轮
        if it == max_iters - 1:
            break  # 达到最大迭代次数

        hints = {
            "role_score": eval_res["role_score"],
            "details": eval_res.get("details", {})
        }
        next_iter_dir = work_dir / f"iter_{it+1}"
        next_iter_dir.mkdir(exist_ok=True)

        new_path = dialogue_improve(
            current_path,
            hints,
            output_dir=str(next_iter_dir),
            model_source=dialog_model_source,
            model_name=dialog_model_name,
        )

        history.append(eval_res)
        current_path = new_path

    # 保存汇总
    summary = {
        "theory_name": theory_name,
        "iterations": len(history)+1,
        "scores": [h["role_score"] for h in history] + [eval_res["role_score"]],
        "final_score": eval_res["role_score"],
    }
    with open(work_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] {theory_name} 优化完成，共 {summary['iterations']} 轮，最终分 {summary['final_score']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="单理论深度优化 (MVP)")
    parser.add_argument("--candidate_entry", required=True, help="由 candidate_selector 生成的单条 JSON 字符串或文件路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--min_improve", type=float, default=DEFAULT_MIN_IMPROVE, help="视为改进所需的最小分数增量")
    parser.add_argument("--eval_mode", choices=["quick", "real"], default="quick", help="评估模式：quick / real")
    parser.add_argument("--max_iters", type=int, default=DEFAULT_MAX_ITERS, help="最大迭代轮数")
    parser.add_argument("--judge_model_source", default="deepseek", help="评审 LLM 来源")
    parser.add_argument("--judge_model_name", default="deepseek-reasoner", help="评审 LLM 名称")
    parser.add_argument("--dialog_model_source", default="deepseek", help="对话 LLM 来源")
    parser.add_argument("--dialog_model_name", default="deepseek-reasoner", help="对话 LLM 名称")

    args = parser.parse_args()

    # 支持传文件或直接传 json 字符串方便测试
    if os.path.exists(args.candidate_entry):
        with open(args.candidate_entry, "r", encoding="utf-8") as f:
            entry = json.load(f)
    else:
        entry = json.loads(args.candidate_entry)

    optimize_once(
        entry,
        args.output_dir,
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