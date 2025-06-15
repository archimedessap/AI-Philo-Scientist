#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dialogue_optimizer.py - 占位版
==============================

MVP 阶段：返回原始理论 JSON 路径，不做任何修改。
后续版本将接入真实 LLM 对话，根据评估提示改写理论。
"""

import shutil
import os
import json
from typing import Dict


def _build_prompt(original_json: Dict, hints: Dict) -> str:
    """根据原理论与评审详情构造对话提示"""
    # 取出评审详情
    details = hints.get("details", {})
    # 对角色按得分升序排序，低分在前
    ordered = sorted(details.items(), key=lambda kv: kv[1].get("score", 0))
    rationale_text = []
    for role_id, info in ordered:
        role_name = role_id.capitalize()
        rationale = info.get("rationale", "")
        score = info.get("score", 0)
        rationale_text.append(f"{role_name} 打分 {score}/10, 原因: {rationale}")
    rationale_block = "\n".join(rationale_text) if rationale_text else "暂无评审详情"

    prompt = f"""
你是一位资深量子理论家，需要改进下面这份量子诠释理论。
改进目标：
1. 重点解决评审者指出的低分/弱点。
2. 力争提升整体平均分至少 0.03。
3. 生成完整、符合 Schema v2.1 的 JSON 文件，字段保持原有结构，可修改内容包括 summary / philosophy / formalism / parameters 等。

--- 原理论 JSON (请勿修改字段名，只修改内容) ---
{json.dumps(original_json, ensure_ascii=False, indent=2)}

--- 多角色评审详情 (供参考) ---
{rationale_block}

请输出**唯一**的 JSON（无 Markdown 代码块包装）。
"""
    return prompt


def improve(
    theory_path: str,
    hints: Dict,
    output_dir: str,
    model_source: str = "deepseek",
    model_name: str = "deepseek-reasoner",
) -> str:
    """调用 LLM 根据评审 hints 生成改进版理论 JSON。

    若解析失败，则复制原文件并添加 _warning 字段。
    """

    os.makedirs(output_dir, exist_ok=True)

    # 加载原理论
    with open(theory_path, "r", encoding="utf-8") as f:
        orig_json = json.load(f)

    prompt = _build_prompt(orig_json, hints)

    # 调用 LLM
    from theory_generation.llm_interface import LLMInterface

    llm = LLMInterface(model_source=model_source, model_name=model_name)
    response = llm.query([{"role": "user", "content": prompt}], temperature=0.7)

    new_json = llm.extract_json(response)

    # --- 校验改写结果 ---
    from tools.refinement.schema_validator import is_valid_theory_json
    if new_json and not is_valid_theory_json(new_json):
        print("[WARN] LLM 生成的 JSON 不符合 Schema，回退到原理论")
        new_json = None

    # 关闭客户端，避免资源泄漏
    try:
        llm.close()
    except Exception:
        pass

    base_name = os.path.basename(theory_path)
    new_path = os.path.join(output_dir, f"improved_{base_name}")

    if not new_json:
        # 解析失败，复制原文件
        shutil.copyfile(theory_path, new_path)
        return new_path

    # 保存新 JSON
    with open(new_path, "w", encoding="utf-8") as f:
        json.dump(new_json, f, ensure_ascii=False, indent=2)

    return new_path 