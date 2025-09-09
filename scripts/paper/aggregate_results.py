#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aggregate_results.py
---------------------
基于仓库资产，生成论文所需的榜单与代际曲线数据（CSV）与可视化（PNG）。

输入：
- 注册库索引：global_theory_registry/theory_index.json
- 运行清单：扫描 output_*/run_*/run_manifest.json（或其它 output_*/*/run_manifest.json）

输出（默认写入 paper_assets/）：
- leaderboard_topN.csv               # 全局理论空间榜单（Top-N 按综合分）
- generation_curves_latest.csv       # 最近一次运行的代际曲线
- fig_leaderboard_topN.png           # 榜单图
- fig_generation_curves_latest.png   # 代际曲线图
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from pathlib import Path
from statistics import mean
from typing import Dict, List, Any, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="汇总注册库与运行清单，生成论文数据与图表")
    p.add_argument(
        "--registry_index",
        type=str,
        default="global_theory_registry/theory_index.json",
        help="注册库索引 JSON 路径",
    )
    p.add_argument(
        "--manifests_glob",
        type=str,
        default="output_*/*/run_manifest.json",
        help="运行清单文件通配路径（glob）",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="paper_assets",
        help="输出目录（将写入 CSV/PNG）",
    )
    p.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="榜单 Top-N 数量",
    )
    p.add_argument(
        "--no_fig",
        action="store_true",
        help="仅输出 CSV，不生成图表",
    )
    return p.parse_args()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str | Path, rows: List[List[Any]]) -> None:
    import csv

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def find_latest_manifest(manifest_paths: List[str]) -> Optional[str]:
    """按 run_id 中的时间戳或文件 mtime 选择最新的 manifest。"""
    if not manifest_paths:
        return None

    def parse_run_ts(p: str) -> Optional[datetime]:
        # 期望路径含有 run_YYYYMMDD_HHMMSS
        m = re.search(r"run_(\d{8}_\d{6})", p)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
        except Exception:
            return None

    candidates: List[Tuple[datetime, str]] = []
    for p in manifest_paths:
        ts = parse_run_ts(p)
        if ts is None:
            # 回退：用文件修改时间
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(p))
            except Exception:
                continue
            candidates.append((mtime, p))
        else:
            candidates.append((ts, p))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def build_leaderboard(registry_index_path: str, top_n: int) -> List[List[Any]]:
    idx = load_json(registry_index_path)
    theories = idx.get("theories", {})

    rows: List[Dict[str, Any]] = []
    for tid, meta in theories.items():
        rows.append(
            {
                "theory_id": tid,
                "theory_name": meta.get("theory_name", "未知理论"),
                "source_type": meta.get("source_type", "unknown"),
                "run_id": meta.get("run_id", ""),
                "generation": meta.get("generation", ""),
                "composite_score": meta.get("composite_score", None),
                "success_rate": meta.get("success_rate", None),
                "theory_file": meta.get("theory_file", ""),
                "evaluation_file": meta.get("evaluation_file", ""),
                "registered_at": meta.get("registered_at", ""),
            }
        )

    # 按综合分排序
    rows.sort(key=lambda r: (r["composite_score"] is None, -(r["composite_score"] or 0.0)))
    top = rows[:top_n]

    header = [
        "rank",
        "theory_id",
        "theory_name",
        "source_type",
        "run_id",
        "generation",
        "composite_score",
        "success_rate",
        "theory_file",
        "evaluation_file",
        "registered_at",
    ]
    csv_rows: List[List[Any]] = [header]
    for i, r in enumerate(top, start=1):
        csv_rows.append(
            [
                i,
                r["theory_id"],
                r["theory_name"],
                r["source_type"],
                r["run_id"],
                r["generation"],
                r["composite_score"],
                r["success_rate"],
                r["theory_file"],
                r["evaluation_file"],
                r["registered_at"],
            ]
        )

    return csv_rows


def build_generation_curves(manifest_path: str) -> List[List[Any]]:
    m = load_json(manifest_path)
    run_id = m.get("run_id", Path(manifest_path).parent.name)
    theories = m.get("theories", {})
    generations_meta = m.get("generations", {})

    # 收集每代的分数
    per_gen_scores: Dict[str, List[float]] = {}
    for _, tinfo in theories.items():
        gen = str(tinfo.get("generation", "0"))
        score = tinfo.get("score", None)
        if score is None:
            continue
        per_gen_scores.setdefault(gen, []).append(score)

    gens_sorted = sorted(per_gen_scores.keys(), key=lambda g: int(g))

    header = [
        "run_id",
        "generation",
        "total_theories",
        "promoted_count",
        "best_score",
        "avg_score",
    ]
    rows: List[List[Any]] = [header]

    for g in gens_sorted:
        scores = per_gen_scores[g]
        best = max(scores) if scores else None
        avg = mean(scores) if scores else None
        prom = generations_meta.get(g, {}).get("promoted_count", None)
        rows.append([run_id, int(g), len(scores), prom, best, avg])

    return rows


def plot_leaderboard(csv_rows: List[List[Any]], out_path: str) -> None:
    import matplotlib.pyplot as plt

    header, data = csv_rows[0], csv_rows[1:]
    names = [str(r[2]) for r in data]
    scores = [float(r[6]) if r[6] is not None else 0.0 for r in data]

    plt.figure(figsize=(10, max(4, 0.4 * len(names))))
    y_pos = list(range(len(names)))
    plt.barh(y_pos, scores, color="#4C78A8")
    plt.yticks(y_pos, names, fontsize=8)
    plt.xlabel("Composite Score")
    plt.title("Global Theory Space Top-N Leaderboard")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_generation_curves(csv_rows: List[List[Any]], out_path: str) -> None:
    import matplotlib.pyplot as plt

    header, data = csv_rows[0], csv_rows[1:]
    gens = [int(r[1]) for r in data]
    best = [float(r[4]) if r[4] is not None else None for r in data]
    avg = [float(r[5]) if r[5] is not None else None for r in data]
    prom = [r[3] if isinstance(r[3], (int, float)) else None for r in data]

    plt.figure(figsize=(8, 4))
    plt.plot(gens, best, marker="o", label="Best Score", color="#F58518")
    plt.plot(gens, avg, marker="s", label="Avg Score", color="#54A24B")
    if any(p is not None for p in prom):
        try:
            plt.twinx()
            plt.plot(gens, [p if p is not None else 0 for p in prom],
                     marker="^", linestyle=":", color="#9C755F", label="Promoted")
        except Exception:
            pass
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.title("Latest Run: Generation Curves")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    # 1) 榜单（全局注册库）
    leaderboard_rows = build_leaderboard(args.registry_index, args.top_n)
    leaderboard_csv = str(Path(args.output_dir) / "leaderboard_topN.csv")
    write_csv(leaderboard_csv, leaderboard_rows)

    # 2) 选择最近运行清单
    manifests = glob(args.manifests_glob)
    latest_manifest = find_latest_manifest(manifests)
    gen_csv_path = None
    if latest_manifest:
        gen_rows = build_generation_curves(latest_manifest)
        gen_csv_path = str(Path(args.output_dir) / "generation_curves_latest.csv")
        write_csv(gen_csv_path, gen_rows)
    else:
        print("[WARN] 未找到运行清单，跳过代际曲线生成", file=sys.stderr)

    if not args.no_fig:
        # 图：榜单
        try:
            plot_leaderboard(leaderboard_rows, str(Path(args.output_dir) / "fig_leaderboard_topN.png"))
        except Exception as e:
            print(f"[WARN] 无法绘制榜单图: {e}")

        # 图：最近运行代际曲线
        if gen_csv_path is not None:
            try:
                gen_rows = [r for r in load_csv(gen_csv_path)]  # noqa
            except Exception:
                gen_rows = None
            try:
                rows_to_plot = gen_rows if gen_rows else []
                if not rows_to_plot:
                    rows_to_plot = build_generation_curves(latest_manifest)
                plot_generation_curves(rows_to_plot, str(Path(args.output_dir) / "fig_generation_curves_latest.png"))
            except Exception as e:
                print(f"[WARN] 无法绘制代际曲线图: {e}")


def load_csv(path: str | Path) -> List[List[str]]:
    import csv

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return [row for row in reader]


if __name__ == "__main__":
    main()

