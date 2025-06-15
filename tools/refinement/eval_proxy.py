#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_proxy.py - 快速评估代理 (MVP 占位版)
=========================================

本模块为"深度优化 MVP"提供一个轻量级的评估接口，
先不真正调用实验/角色评估脚本，而是生成**可重复的伪随机分数**，
以便把优化链路跑通。

后续可以在 `fast_eval()` 内接入 `run_role_evaluation.py` 或其他真实评估逻辑。
"""

from __future__ import annotations
import os
import json
import hashlib
import asyncio
from typing import Dict

# No longer using subprocess; real evaluation will call TheoryEvaluator directly


def _deterministic_score(theory_name: str) -> float:
    """根据理论名称生成 0.5~1.0 之间的可重复分数"""
    # 使用 SHA256 哈希保证跨平台一致
    h = hashlib.sha256(theory_name.encode("utf-8")).hexdigest()
    # 取前 8 位转为 int
    n = int(h[:8], 16)
    # 映射到 0.5 - 1.0
    return 0.5 + (n % 5000) / 10000.0  # 0.5-0.9999


def fast_eval(theory_path: str, quick: bool = True) -> Dict:
    """快速评估理论 (MVP)。

    Args:
        theory_path: 理论 JSON 文件路径
        quick: 是否采用快速评估（占位）。占位版仅支持 True。

    Returns:
        dict, 形如 {"theory_name": ..., "role_score": ...}
    """
    if not os.path.exists(theory_path):
        raise FileNotFoundError(f"理论文件不存在: {theory_path}")

    with open(theory_path, "r", encoding="utf-8") as f:
        theory_data = json.load(f)

    # 理论 JSON 至少需要有 "name" 字段或文件名推断
    theory_name = theory_data.get("name")
    if not theory_name:
        theory_name = os.path.splitext(os.path.basename(theory_path))[0]

    score = _deterministic_score(theory_name)

    return {
        "theory_name": theory_name,
        "role_score": score
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="快速评估代理 (占位版)")
    parser.add_argument("theory_json", help="理论 JSON 文件路径")
    args = parser.parse_args()
    result = fast_eval(args.theory_json)
    print(json.dumps(result, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# real evaluation using run_role_evaluation (role scores)
# ---------------------------------------------------------------------------


async def _async_real_eval(theory_path: str,
                           model_source: str = "deepseek",
                           model_name: str = "deepseek-reasoner") -> Dict:
    """对单个理论执行多角色评估，返回平均 role_score (0-1)。"""

    # 延迟导入，避免无 API key 环境报错
    from theory_validation.agent_validation.theory_evaluator import TheoryEvaluator
    from theory_generation.llm_interface import LLMInterface

    # 加载理论 JSON
    with open(theory_path, "r", encoding="utf-8") as f:
        theory_json = json.load(f)

    theory_name = theory_json.get("name") or os.path.splitext(os.path.basename(theory_path))[0]

    # 构造适配的 theory dict，参考 demo.auto_role_evaluation
    adapted_theory = {
        'name': theory_name,
        'core_principles': theory_json.get('summary', ''),
        'detailed_description': f"哲学立场: {theory_json.get('philosophy', {}).get('ontology', '')}\n测量解释: {theory_json.get('philosophy', {}).get('measurement', '')}",
        'quantum_phenomena_explanation': {
            'wave_function_collapse': theory_json.get('philosophy', {}).get('measurement', ''),
            'measurement_problem': theory_json.get('philosophy', {}).get('measurement', ''),
            'non_locality': theory_json.get('philosophy', {}).get('ontology', ''),
        },
        'philosophical_stance': theory_json.get('philosophy', {}),
        'mathematical_formulation': theory_json.get('formalism', {}),
        'parameters': theory_json.get('parameters', {}),
        'semantics': theory_json.get('semantics', {})
    }

    llm = LLMInterface(model_source=model_source, model_name=model_name)
    evaluator = TheoryEvaluator(llm)

    # 逐角色评估
    role_scores = []
    details = {}
    for role_id, role_info in evaluator.evaluation_roles.items():
        eval_res = await evaluator._evaluate_as_role(adapted_theory, role_id, role_info)

        score = eval_res.get('score', 0)
        rationale = (
            eval_res.get('detailed_comments')
            or eval_res.get('improvement_suggestions')
            or eval_res.get('strengths')
            or ""
        )
        # 若 strengths / weaknesses 数组，合并为字符串
        if isinstance(rationale, list):
            rationale = "\n".join(rationale)

        details[role_id] = {
            "score": score,
            "rationale": rationale
        }
        role_scores.append(score)

    avg_score = sum(role_scores) / len(role_scores) if role_scores else 0

    # 关闭客户端，避免事件循环关闭警告
    try:
        await llm.aclose()
    except Exception:
        pass

    return {
        "theory_name": theory_name,
        "role_score": avg_score / 10.0,  # 归一化
        "details": details
    }


def real_eval(theory_path: str,
              model_source: str = "deepseek",
              model_name: str = "deepseek-reasoner") -> Dict:
    """同步封装，方便在非 async 环境调用。"""
    return asyncio.run(_async_real_eval(theory_path, model_source, model_name))


# ---------------------------------------------------------------------------
# unified evaluate helper
# ---------------------------------------------------------------------------


def evaluate(theory_path: str, mode: str = "quick",
             model_source: str = "deepseek",
             model_name: str = "deepseek-reasoner") -> Dict:
    """统一入口
    mode = "quick" 使用 deterministic fast_eval
    mode = "real"  使用 LLM 角色评估
    """
    mode = mode.lower()
    if mode == "real":
        return real_eval(theory_path, model_source, model_name)
    return fast_eval(theory_path)


# ---------------------------------------------------------------------------
# new evaluate_real function
# ---------------------------------------------------------------------------


def evaluate_real(theory_path: str,
                  model_source: str = "deepseek",
                  model_name: str = "deepseek-reasoner") -> Dict:
    """
    Returns:
        dict, 形如 {"role_score": ..., "details": {...}}
    """
    result = real_eval(theory_path, model_source, model_name)
    details = {}
    for role_id, eval_res in result["details"].items():
        details[role_id] = {
            "score": eval_res["score"],
            "rationale": eval_res["rationale"]
        }
    return {
        "role_score": result["role_score"],
        "details": details
    } 