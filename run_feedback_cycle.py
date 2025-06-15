#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_feedback_cycle.py
====================
一次性执行"理论生成 → 评估 → 深度对话优化"多代循环。

流程 (每代)：
1. 调用 run_full_cycle.py 生成新理论并评估，得到 main_run_dir
2. 调用 run_m3_auto_refinement.py 对 Top-N 理论做深度优化
3. 将优化后理论收集到 next_theory_pool/，作为下一代合成的输入

默认仅把"最佳迭代版本"引回；若无 JSON 通过 Schema 校验，则保留原版本。
"""

from __future__ import annotations
import subprocess
import sys
import argparse
from pathlib import Path
import json
import shutil
import glob
import time

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------

def run_cmd(cmd: list[str]):
    """简易同步执行，出错即退出。"""
    print("\n$ " + " ".join(cmd))
    res = subprocess.call(cmd)
    if res != 0:
        print(f"[FATAL] 命令失败: {' '.join(cmd)}")
        sys.exit(res)


def find_unique(glob_pattern: str) -> Path:
    """确保只找到一个文件/目录"""
    matches = glob.glob(glob_pattern)
    if not matches:
        raise FileNotFoundError(f"未找到匹配: {glob_pattern}")
    if len(matches) > 1:
        raise RuntimeError(f"匹配到多个路径: {matches}")
    return Path(matches[0])


def collect_improved_theories(depth_runs_dir: Path, dest_dir: Path):
    """遍历 depth_runs，每条理论挑选最终迭代的 improved_*.json 复制到 dest_dir"""
    from tools.refinement.schema_validator import is_valid_theory_json

    dest_dir.mkdir(parents=True, exist_ok=True)

    for theory_dir in depth_runs_dir.iterdir():
        if not theory_dir.is_dir():
            continue
        # 找到最后一轮 iter_*
        iter_dirs = sorted([p for p in theory_dir.iterdir() if p.is_dir() and p.name.startswith("iter_")],
                           key=lambda p: int(p.name.split("_")[1]))
        if not iter_dirs:
            continue
        last_iter = iter_dirs[-1]
        # 优先取 improved_*.json
        improved_files = list(last_iter.glob("improved_*.json"))
        source_json = None
        if improved_files:
            source_json = improved_files[0]
        else:
            # fallback to origin in iter_0 or work_dir root
            origin = theory_dir / "iter_0" / source_json if False else None  # placeholder
        if source_json is None:
            continue
        # 校验
        with open(source_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not is_valid_theory_json(data):
            continue
        shutil.copy(source_json, dest_dir / source_json.name)

# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("多代反馈循环控制脚本")
    parser.add_argument("--generations", type=int, default=2, help="循环代数")
    parser.add_argument("--initial_theories_dir", type=str, default="data/theories_v2.1", help="初始理论池目录")
    parser.add_argument("--output_root", type=str, default="data/feedback_runs", help="总输出根目录")

    # 透传常用参数
    parser.add_argument("--synthesis_model_source", default="google")
    parser.add_argument("--synthesis_model_name",   default="gemini-1.5-pro-latest")
    parser.add_argument("--evaluation_model_source", default="deepseek")
    parser.add_argument("--evaluation_model_name",   default="deepseek-reasoner")
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--max_iters", type=int, default=2)
    parser.add_argument("--min_improve", type=float, default=0.03)

    args = parser.parse_args()

    base_root = Path(args.output_root)
    base_root.mkdir(parents=True, exist_ok=True)

    theories_dir_for_next = Path(args.initial_theories_dir).resolve()

    for gen in range(1, args.generations + 1):
        print("\n" + "="*80)
        print(f"🌀 Generation {gen}")
        print("="*80)
        gen_dir = base_root / f"generation_{gen}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Phase 1: run_full_cycle
        # ------------------------------------------------------------------
        full_cycle_dir = gen_dir / "full_cycle"
        cmd_full = [
            "python", "run_full_cycle.py",
            "--existing_theories_dir", str(theories_dir_for_next),
            "--base_output_dir", str(full_cycle_dir),
            "--max_pairs_to_analyze", "5",
            "--variants_per_contradiction", "2",
            "--synthesis_model_source", args.synthesis_model_source,
            "--synthesis_model_name", args.synthesis_model_name,
            "--evaluation_model_source", args.evaluation_model_source,
            "--evaluation_model_name", args.evaluation_model_name
        ]
        run_cmd(cmd_full)

        # 找到 run_* 目录
        run_subdir = find_unique(str(full_cycle_dir / "run_*/"))

        # ------------------------------------------------------------------
        # Phase 2: run_m3_auto_refinement
        # ------------------------------------------------------------------
        summary_file = find_unique(str(run_subdir / "2_evaluation_output" / "*/final_evaluation_summary.json"))
        theories_root = find_unique(str(run_subdir / "1_synthesis_output" / "*/eval_ready_theories"))

        refinement_dir = gen_dir / "refinement"
        cmd_refine = [
            "python", "run_m3_auto_refinement.py",
            "--summary_file", str(summary_file),
            "--theories_root", str(theories_root),
            "--output_dir", str(refinement_dir),
            "--top_n", str(args.top_n),
            "--eval_mode", "real",
            "--max_iters", str(args.max_iters),
            "--min_improve", str(args.min_improve),
            "--judge_model_source", args.evaluation_model_source,
            "--judge_model_name", args.evaluation_model_name,
            "--dialog_model_source", args.evaluation_model_source,
            "--dialog_model_name", args.evaluation_model_name
        ]
        run_cmd(cmd_refine)

        depth_runs_dir = refinement_dir / next(iter((refinement_dir).iterdir())) / "depth_output/depth_runs"
        if not depth_runs_dir.exists():
            depth_runs_dir = refinement_dir / "depth_output/depth_runs"
        improved_pool = gen_dir / "improved_theories_pool"
        collect_improved_theories(depth_runs_dir, improved_pool)

        # 下代理论池：使用改写版本 + 原理论
        theories_dir_for_next = improved_pool
        print(f"[GENERATION {gen}] 将 {len(list(improved_pool.glob('*.json')))} 个改写理论作为下一轮输入。")
        time.sleep(1)  # 简要停顿，便于查看日志

    print("\n🎉 多代循环完成，所有结果位于:", base_root)


if __name__ == "__main__":
    main() 