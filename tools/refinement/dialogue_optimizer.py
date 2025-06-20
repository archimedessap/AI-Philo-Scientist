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
import random
from typing import Dict


def _get_refinement_strategy(iteration: int, theory_name: str) -> str:
    """根据迭代次数和理论名称选择不同的精炼策略"""
    strategies = [
        "深化哲学基础",
        "强化数学形式化", 
        "改进实验预测",
        "完善物理解释",
        "增强逻辑一致性"
    ]
    
    # 使用理论名称和迭代次数作为种子，确保可重复但多样化
    seed = hash(theory_name + str(iteration)) % len(strategies)
    return strategies[seed]


def _build_prompt(original_json: Dict, hints: Dict, iteration: int = 0) -> str:
    """根据原理论与评审详情构造对话提示"""
    # 取出评审详情
    details = hints.get("details", {})
    current_score = hints.get("role_score", 0.0)
    
    # 对角色按得分升序排序，低分在前
    ordered = sorted(details.items(), key=lambda kv: kv[1].get("score", 0))
    
    # 构建具体的改进建议
    improvement_strategies = []
    weak_areas = []
    
    for role_id, info in ordered[:3]:  # 关注最弱的3个角色
        role_name = role_id.capitalize()
        rationale = info.get("rationale", "")
        score = info.get("score", 0)
        
        if score < 6:  # 低分角色
            weak_areas.append(f"{role_name}: {rationale}")
            
            # 根据角色类型提供具体建议
            if "experimental" in role_id.lower():
                improvement_strategies.append("加强实验预测的数学精确性和可验证性")
            elif "philosopher" in role_id.lower():
                improvement_strategies.append("深化哲学基础，明确本体论立场")
            elif "mathematician" in role_id.lower():
                improvement_strategies.append("完善数学形式化表述，增加严格性")
            elif "physicist" in role_id.lower():
                improvement_strategies.append("强化与已知物理现象的联系")
    
    # 构建改进策略文本
    strategies_text = "\n".join([f"- {strategy}" for strategy in improvement_strategies])
    weak_areas_text = "\n".join([f"- {area}" for area in weak_areas])
    
    # 获取本轮的重点策略
    theory_name = original_json.get('name', 'Unknown Theory')
    focus_strategy = _get_refinement_strategy(iteration, theory_name)
    
    prompt = f"""
你是一位顶尖的量子理论专家，需要对下面的量子诠释理论进行**深度改进**。

当前理论得分：{current_score:.3f}/1.0
本轮改进重点：**{focus_strategy}**

## 主要弱点分析：
{weak_areas_text}

## 改进策略：
{strategies_text}

## 改进要求：
1. **针对性改进**：重点解决上述弱点，特别关注"{focus_strategy}"
2. **保持一致性**：确保改进后的理论内部逻辑一致
3. **增强独特性**：突出理论的创新点和独特优势
4. **完善表述**：使用更精确、更专业的语言
5. **扩展深度**：在薄弱环节增加更多细节和论证

## 具体改进方向：
- 如果哲学基础薄弱：明确本体论立场，完善测量理论
- 如果数学表述不足：增加严格的数学形式化
- 如果实验预测模糊：提供具体的、可验证的预测
- 如果物理解释不清：强化与量子现象的联系

## 原理论JSON：
{json.dumps(original_json, ensure_ascii=False, indent=2)}

请输出改进后的完整JSON（无Markdown包装，保持原有字段结构）：
"""
    return prompt


def improve(
    theory_path: str,
    hints: Dict,
    output_dir: str,
    model_source: str = "deepseek",
    model_name: str = "deepseek-reasoner",
    iteration: int = 0,
) -> str:
    """调用 LLM 根据评审 hints 生成改进版理论 JSON。

    若解析失败，则复制原文件并添加 _warning 字段。
    """

    os.makedirs(output_dir, exist_ok=True)

    # 加载原理论
    with open(theory_path, "r", encoding="utf-8") as f:
        orig_json = json.load(f)

    prompt = _build_prompt(orig_json, hints, iteration)

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