#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_case_cards.py
----------------------
从 Global Theory Registry 中选取 Top-N 理论，生成论文“案例卡片” Markdown。

输出：paper_assets/case_cards/*.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成论文案例卡片（Markdown）")
    p.add_argument(
        "--registry_index",
        type=str,
        default="global_theory_registry/theory_index.json",
        help="注册库索引 JSON 路径",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="paper_assets/case_cards",
        help="案例卡片输出目录",
    )
    p.add_argument(
        "--top_n",
        type=int,
        default=3,
        help="导出前 N 个案例",
    )
    return p.parse_args()


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9\-_.]+", "_", name)
    return name[:80] if len(name) > 80 else name


def safe_get(d: Dict[str, Any], keys: List[str], default: Any = "") -> Any:
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def format_list(items: Any) -> str:
    if isinstance(items, list):
        return "\n".join([f"- {str(x)}" for x in items])
    if isinstance(items, str):
        return items
    return ""


def build_card_md(rank: int, meta: Dict[str, Any], theory: Dict[str, Any]) -> str:
    # 元信息
    header_name = theory.get("name") or meta.get("theory_name") or "未命名理论"
    t_id = meta.get("theory_id", "")
    source_type = meta.get("source_type", "")
    run_id = meta.get("run_id", "")
    generation = meta.get("generation", "")
    composite_score = meta.get("composite_score", "")
    success_rate = meta.get("success_rate", "")

    # 主要内容（容错提取）
    description = safe_get(
        theory,
        ["description", "detailed_description", "summary", "overview"],
        "",
    )
    core_assumptions = safe_get(theory, ["core_assumptions", "core_principles"], [])
    math = safe_get(
        theory,
        ["mathematical_formalism", "mathematical_formulation", "math_formalism"],
        "",
    )
    predictions = safe_get(theory, ["empirical_predictions", "predictions"], [])
    philosophy = safe_get(theory, ["philosophy", "philosophical_stance"], {})

    # 渲染 Markdown
    lines: List[str] = []
    lines.append(f"# {header_name}")
    lines.append("")
    lines.append("**案例排名**: Top {}".format(rank))
    lines.append("")
    lines.append("**元信息**:")
    lines.append(f"- ID: `{t_id}`")
    lines.append(f"- 来源: `{source_type}`  |  Run: `{run_id}`  |  代: `{generation}`")
    lines.append(f"- 综合分: `{composite_score}`  |  实验成功率: `{success_rate}`")
    lines.append("")
    if description:
        lines.append("**概述**:")
        lines.append(str(description))
        lines.append("")
    if core_assumptions:
        lines.append("**核心假设**:")
        lines.append(format_list(core_assumptions))
        lines.append("")
    if math:
        lines.append("**数学形式**:")
        lines.append("```")
        lines.append(str(math))
        lines.append("```")
        lines.append("")
    if predictions:
        lines.append("**经验预测**:")
        lines.append(format_list(predictions))
        lines.append("")
    if isinstance(philosophy, dict) and philosophy:
        lines.append("**哲学立场**:")
        for k, v in philosophy.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    idx = load_json(args.registry_index)
    theories_meta = idx.get("theories", {})

    # 排序选 Top-N
    metas = list(theories_meta.values())
    metas.sort(key=lambda m: (m.get("composite_score") is None, -(m.get("composite_score") or 0.0)))
    metas = metas[: args.top_n]

    for i, meta in enumerate(metas, start=1):
        tpath = meta.get("theory_file")
        if not tpath or not Path(tpath).exists():
            # 路径可能是相对路径，尝试从根相对
            alt = Path.cwd() / tpath if tpath else None
            if not (alt and alt.exists()):
                print(f"[WARN] 无法找到理论文件: {tpath}")
                theory = {"name": meta.get("theory_name", "未知理论")}
            else:
                theory = load_json(alt)
        else:
            theory = load_json(tpath)

        name_for_file = sanitize_filename(theory.get("name") or meta.get("theory_name") or f"case_{i}")
        out_path = Path(args.output_dir) / f"{i:02d}_{name_for_file}.md"
        out_md = build_card_md(i, meta, theory)
        out_path.write_text(out_md, encoding="utf-8")
        print(f"[OK] 写入案例卡片: {out_path}")


if __name__ == "__main__":
    main()

