#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
矛盾分析器

同时支持：
1. 基于现有理论文件（schema v2.1）检测矛盾；
2. 基于短卡集合生成结构化矛盾表。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class ContradictionAnalyzer:
    """多用途矛盾分析工具：既能分析理论 Pair，也能处理短卡集合。"""

    def __init__(
        self,
        llm_interface,
        schema_path: str = "schemas/contradiction.schema.json",
    ) -> None:
        self.llm = llm_interface
        self.schema_path = Path(schema_path)
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Contradiction schema not found: {self.schema_path}")
        self.schema = json.loads(self.schema_path.read_text(encoding="utf-8"))

        # direct 模式所需的理论缓存
        self.theories: Dict[str, Dict[str, Any]] = {}
        self.contradictions: List[Dict[str, Any]] = []

        # 比较维度，用于旧版 direct 工作流的提示构建
        self.key_dimensions = [
            "wave_function_reality",
            "measurement_process",
            "observer_role",
            "determinism",
            "non_locality",
            "mathematical_formalism",
            "ontological_status",
            "quantum_classical_boundary",
        ]

    # ------------------------------------------------------------------
    # direct 模式：加载理论与分析矛盾
    # ------------------------------------------------------------------
    def load_theories(self, theories_dir: str, schema_version: Optional[str] = None) -> None:
        """从目录加载理论 JSON 文件，便于后续 pairwise 矛盾分析。"""
        if not os.path.exists(theories_dir):
            print(f"[ERROR] 目录不存在: {theories_dir}")
            return

        theory_files = [f for f in os.listdir(theories_dir) if f.endswith(".json")]
        loaded_count = 0

        for file_name in theory_files:
            file_path = os.path.join(theories_dir, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                print(f"[ERROR] 加载理论文件 {file_name} 失败: {exc}")
                continue

            def _register_theory(theory_payload: Dict[str, Any]):
                theory_name = theory_payload.get("theory_name") or theory_payload.get("name")
                if not theory_name:
                    return

                if schema_version:
                    version = theory_payload.get("metadata", {}).get("schema_version")
                    if version != schema_version:
                        print(
                            f"[INFO] 跳过 {file_name} 中的 {theory_name}: schema版本不匹配 "
                            f"(需要 {schema_version}, 实际 {version})"
                        )
                        return

                self.theories[theory_name] = theory_payload
                print(f"[INFO] 已加载理论: {theory_name}")

            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict):
                        _register_theory(entry)
                        loaded_count += 1
            elif isinstance(data, dict):
                _register_theory(data)
                loaded_count += 1

        print(f"[INFO] 共加载 {len(self.theories)} 个理论")

    async def find_contradictions(self, theory1_name: str, theory2_name: str) -> Dict[str, Any]:
        """比较两个理论，输出矛盾点分析。"""
        theory1 = self.theories.get(theory1_name)
        theory2 = self.theories.get(theory2_name)

        missing: List[str] = []
        if not theory1:
            missing.append(theory1_name)
        if not theory2:
            missing.append(theory2_name)
        if missing:
            msg = f"未找到理论: {', '.join(missing)}"
            print(f"[ERROR] {msg}")
            return {"error": msg}

        prompt = self._build_analysis_prompt(theory1, theory2)
        response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        try:
            raw_result = self.llm.extract_json(response)
        except Exception as exc:
            print(f"[ERROR] 解析矛盾分析响应失败: {exc}")
            return {"error": str(exc), "theory1": theory1_name, "theory2": theory2_name}

        analysis_result = self._normalize_analysis_result(raw_result, theory1_name, theory2_name)
        if not analysis_result:
            print("[ERROR] 矛盾分析响应无法解析为有效JSON")
            return {"error": "无法解析响应", "raw_response": response}

        self.contradictions.append(analysis_result)
        count = len(analysis_result.get("contradictions", []))
        print(f"[INFO] 找到 {count} 个矛盾点: {theory1_name} vs {theory2_name}")
        return analysis_result

    def _build_analysis_prompt(self, theory1: Dict[str, Any], theory2: Dict[str, Any]) -> str:
        """构建 direct 模式提示词。"""
        theory1_name = theory1.get("theory_name") or theory1.get("name") or "理论1"
        theory2_name = theory2.get("theory_name") or theory2.get("name") or "理论2"
        theory1_desc = theory1.get("core_principles") or theory1.get("description", "")
        theory2_desc = theory2.get("core_principles") or theory2.get("description", "")

        prompt = f"""
你是一位量子诠释专家，请分析下列两种理论之间的关键矛盾。

理论A: {theory1_name}
核心内容: {theory1_desc}

理论B: {theory2_name}
核心内容: {theory2_desc}

重点关注（但不限于）以下维度：
1. 波函数本体论地位
2. 测量问题处理
3. 决定论与随机性
4. 观察者角色
5. 非局域性解释

请以JSON格式返回矛盾列表，每个矛盾包含：
- dimension
- theory1_position
- theory2_position
- core_tension
- philosophical_implications
- importance_score (1-10)
        """
        return prompt

    def _normalize_analysis_result(
        self,
        raw_result: Any,
        theory1_name: str,
        theory2_name: str,
    ) -> Optional[Dict[str, Any]]:
        """将 LLM 返回的矛盾结果归一化为带有 contradictions 列表的结构。"""
        if raw_result is None:
            return None

        if isinstance(raw_result, list):
            normalized: Dict[str, Any] = {
                "theory1": theory1_name,
                "theory2": theory2_name,
                "contradictions": raw_result,
            }
            return normalized

        if isinstance(raw_result, dict):
            normalized = dict(raw_result)
            normalized.setdefault("theory1", theory1_name)
            normalized.setdefault("theory2", theory2_name)

            contradictions = normalized.get("contradictions")
            if isinstance(contradictions, dict):
                contradictions = [contradictions]
            elif contradictions is None:
                if self._looks_like_single_contradiction(normalized):
                    contr_entry = {k: normalized.pop(k) for k in list(normalized.keys()) if k in self._contradiction_keys()}
                    contradictions = [contr_entry]
                else:
                    contradictions = []
            elif not isinstance(contradictions, list):
                contradictions = [contradictions]

            normalized["contradictions"] = contradictions
            return normalized

        # 未知类型时返回空结构，避免后续崩溃
        return {
            "theory1": theory1_name,
            "theory2": theory2_name,
            "contradictions": [],
        }

    def _looks_like_single_contradiction(self, data: Dict[str, Any]) -> bool:
        """判断字典是否类似单条矛盾记录。"""
        keys = set(data.keys())
        return bool(keys & self._contradiction_keys())

    @staticmethod
    def _contradiction_keys() -> set:
        """矛盾记录常见字段集合。"""
        return {
            "dimension",
            "theory1_position",
            "theory2_position",
            "core_tension",
            "philosophical_implications",
            "importance_score",
            "issue",
            "summary",
        }

    # ------------------------------------------------------------------
    # 短卡模式：生成结构化矛盾表
    # ------------------------------------------------------------------
    async def build_table(
        self,
        cards: Iterable[Dict[str, Any]],
        task_hint: str = "",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """基于短卡集合生成矛盾表。"""
        cards_list = list(cards)
        if len(cards_list) < 2:
            raise ValueError("At least two cards are required to analyze contradictions.")

        messages = self._build_messages(cards_list, task_hint)
        result = await self.llm.query_structured_json(
            messages=messages,
            schema=self.schema,
            schema_name="contradiction_table",
            temperature=temperature,
        )

        if isinstance(result, list):
            result = {"contradictions": result}
        elif isinstance(result, dict) and "items" in result and "contradictions" not in result:
            result = {"contradictions": result.get("items", [])}
        elif isinstance(result, dict) and {"A", "B", "issue", "one_line"}.issubset(result.keys()):
            result = {"contradictions": [result]}

        if not result or "contradictions" not in result:
            print(f"[ERROR] 结构化矛盾输出格式异常: {result}")
            raise ValueError("Structured contradiction result missing required field 'contradictions'.")
        return result

    def _build_messages(self, cards: List[Dict[str, Any]], task_hint: str) -> List[Dict[str, str]]:
        system_prompt = (
            "You are a quantum foundations researcher. Analyse the supplied interpretation cards. "
            "Compare their commitments and produce concise contradiction items. "
            "Use the provided JSON schema. Prefer deep, substantive disagreements over terminology. "
            "Enumerate multiple contradictions (aim for five or more) spanning distinct issue categories whenever the cards allow; "
            "do not stop after a single conflict."
        )

        if task_hint:
            system_prompt += f" Task focus: {task_hint.strip()}"

        cards_text = "\n\n".join(self._format_card(card, index) for index, card in enumerate(cards, start=1))
        user_prompt = (
            "Interpretation cards:\n"
            f"{cards_text}\n\n"
            "Generate a contradiction table that captures the diverse conflicts among these cards. "
            "Cover every major incompatibility cluster (collapse, ontology, determinism, information, locality, etc.) when supported by the cards. "
            "Produce at least two contradictions and prefer five or more. "
            "For each contradiction: use card ids or names in fields 'A' and 'B'; pick the `issue` enum that fits best. "
            "The `one_line` text must be under 25 ASCII characters and explain the tension plainly."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _format_card(self, card: Dict[str, Any], index: int) -> str:
        predictions = card.get("predictions", [])
        predictions_text = "; ".join(predictions) if predictions else "No distinct empirical deviations."
        key_claims = "; ".join(card.get("key_claims", []))
        return (
            f"[{index}] id={card.get('id')} | name={card.get('name')}\n"
            f"one_line: {card.get('one_line')}\n"
            f"math_relation: type={card.get('math_relation_to_SQM', {}).get('type')} | "
            f"math_change={card.get('math_relation_to_SQM', {}).get('math_change')} | "
            f"eq: {card.get('math_relation_to_SQM', {}).get('equations_summary')}\n"
            f"claims: {key_claims}\n"
            f"born_rule: {card.get('born_rule')} | measurement: {card.get('measurement_update')} | locality: {card.get('locality_note')}\n"
            f"predictions: {predictions_text}\n"
            f"tags: {', '.join(card.get('tags', []))}"
        )


__all__ = ["ContradictionAnalyzer"]
