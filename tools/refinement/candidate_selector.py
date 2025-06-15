#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Candidate Selector
==================
读取 `final_evaluation_summary.json`，按成功率(或综合得分)排序，挑选 Top-N 理论。
该脚本属于"深度优化 MVP"流程的第一步。

用法示例：
-------------
python tools/refinement/candidate_selector.py \
    --summary_file data/full_cycle_runs/run_20250610_211016/3_final_evaluation_output/final_evaluation_summary.json \
    --top_n 5 \
    --theories_root data/full_cycle_runs/run_20250610_211016/1_synthesis_output/synthesis_20250610_211016/eval_ready_theories \
    --output_file selected_candidates.json
"""

import argparse
import json
import os
import re
from typing import List, Dict

DEFAULT_TOP_N = 5


def slugify(text: str) -> str:
    """将理论名称转为安全文件名，例如 "Quantum Gravity" -> "Quantum_Gravity""" 
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")
    return slug[:120]  # 避免文件名过长


def load_summary(summary_path: str) -> List[Dict]:
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("summary_file 应是包含多个理论结果的 JSON 列表。")
    return data


def select_top_n(summary: List[Dict], top_n: int) -> List[Dict]:
    # 首选 success_rate，其次 combined_score 或 overall_score
    def score_key(item):
        # 尝试多个字段
        for key in ("success_rate", "overall_score", "combined_score"):
            if key in item and item[key] is not None:
                return item[key]
        return 0.0

    sorted_items = sorted(summary, key=score_key, reverse=True)
    return sorted_items[:top_n]


def _build_slug_mapping(theories_root: str) -> Dict[str, str]:
    """扫描目录，建立 slug -> 实际文件路径 映射"""
    mapping = {}
    for fname in os.listdir(theories_root):
        if not fname.lower().endswith('.json'):
            continue
        slug = slugify(os.path.splitext(fname)[0])
        mapping[slug.lower()] = os.path.join(theories_root, fname)
    return mapping


def attach_json_path(items: List[Dict], theories_root: str):
    """为每个 theory 附加 json_path 字段，尽量匹配真实文件名"""
    slug_map = _build_slug_mapping(theories_root)

    for it in items:
        theory_name = it["theory_name"]
        slug = slugify(theory_name).lower()
        if slug in slug_map:
            it["json_path"] = slug_map[slug]
        else:
            # 退而求其次：把理论名转小写直接搜索包含关系
            matched = None
            for key, path in slug_map.items():
                if slug in key:
                    matched = path
                    break
            if matched:
                it["json_path"] = matched
            else:
                it["json_path"] = os.path.join(theories_root, f"{slug}.json")
                it["_warning"] = "未找到精确匹配文件，路径可能不存在"


def main():
    parser = argparse.ArgumentParser(description="Select Top-N theories from evaluation summary")
    parser.add_argument("--summary_file", type=str, required=True, help="Path to final_evaluation_summary.json")
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N, help="Number of top theories to select")
    parser.add_argument("--theories_root", type=str, default=None, help="Root directory containing theory JSON files")
    parser.add_argument("--output_file", type=str, default=None, help="Where to save selected list (JSON). Defaults to stdout only.")

    args = parser.parse_args()

    summary = load_summary(args.summary_file)
    selected = select_top_n(summary, args.top_n)

    if args.theories_root:
        attach_json_path(selected, args.theories_root)

    # 打印到终端
    print(json.dumps(selected, ensure_ascii=False, indent=2))

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(selected, f, ensure_ascii=False, indent=2)
        print(f"已保存 Top-{args.top_n} 理论列表到: {args.output_file}")


if __name__ == "__main__":
    main() 