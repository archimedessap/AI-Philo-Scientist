#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
network_synthesis_adapter.py - 矛盾网络合成方法适配器（CNS-Lite）

以最小工程编排利用 LLM 代理：
1) 枚举/抽取 pairwise 矛盾 → 2) 构建矛盾网络 → 3) 规划概念放松 →
4) 统一合成新理论 → 5) 输出评估就绪候选。

提供 dry-run 回退：当外部 API 不可用时，使用确定性模板产出简化产物，保证可运行与可集成。
"""

from __future__ import annotations

import os
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List

try:
    from .base_adapter import TheoryGenerationMethod, GenerationResult
    from ..llm_interface import LLMInterface
    from ..direct_synthesis.contradiction_analyzer import ContradictionAnalyzer
    from ..direct_synthesis.hypothesis_generator import HypothesisGenerator
    from ..network.contradiction_graph_builder import ContradictionGraphBuilder
    from ..network.relaxation_planner import RelaxationPlanner
except ImportError:
    import sys
    parent = Path(__file__).parent.parent
    sys.path.insert(0, str(parent))
    from methods.base_adapter import TheoryGenerationMethod, GenerationResult
    from llm_interface import LLMInterface
    from direct_synthesis.contradiction_analyzer import ContradictionAnalyzer
    from direct_synthesis.hypothesis_generator import HypothesisGenerator
    from network.contradiction_graph_builder import ContradictionGraphBuilder
    from network.relaxation_planner import RelaxationPlanner


class NetworkSynthesisAdapter(TheoryGenerationMethod):
    def __init__(self,
                 theories_dir: str,
                 output_dir: str,
                 max_pairs: int = 15,
                 variants_per_contradiction: int = 1,  # not used; keep signature compatible
                 model_source: str = "google",
                 model_name: str = "gemini-2.5-flash",
                 relaxation_budget: int = 6,
                 diversity_level: float = 0.6,
                 num_seeds: int = 1,
                 dry_run: bool = False,
                 **kwargs):
        super().__init__(
            theories_dir=theories_dir,
            output_dir=output_dir,
            max_pairs=max_pairs,
            variants_per_contradiction=variants_per_contradiction,
            model_source=model_source,
            model_name=model_name,
            **kwargs
        )
        self.relaxation_budget = relaxation_budget
        self.diversity_level = diversity_level
        self.dry_run = dry_run
        self.num_seeds = max(1, int(num_seeds))
        self.synthesis_dir = self.output_dir / f"network_synthesis_{time.strftime('%Y%m%d_%H%M%S')}"
        self.synthesis_dir.mkdir(parents=True, exist_ok=True)

    def generate(self) -> Dict[str, Any]:
        try:
            result = asyncio.run(self._async_generate())
            return self._ensure_output_format(result)
        except Exception as e:
            self._log_error(f"网络合成出错: {e}")
            return GenerationResult(
                success=False,
                error_message=str(e),
                output_dir=str(self.synthesis_dir)
            ).to_dict()

    async def _async_generate(self) -> Dict[str, Any]:
        self._log_info("开始矛盾网络合成 (CNS-Lite)")

        llm = None
        if not self.dry_run:
            try:
                llm = LLMInterface(self.model_source, self.model_name, request_interval=1.0)
            except Exception as e:
                self._log_warning(f"LLM初始化失败，切换 dry-run: {e}")
                self.dry_run = True

        # 1) 加载理论 & 选择配对
        analyzer = ContradictionAnalyzer(llm) if not self.dry_run else ContradictionAnalyzer(llm_interface=None)
        analyzer.load_theories(str(self.theories_dir), schema_version=None)
        theory_names = list(analyzer.theories.keys())
        if len(theory_names) < 2:
            raise ValueError("至少需要2个理论")

        from itertools import combinations
        pairs = list(combinations(theory_names, 2))[: self.max_pairs]
        self._log_info(f"选取 {len(pairs)} 对理论用于矛盾分析")

        # 2) pairwise 矛盾
        analyses: List[Dict[str, Any]] = []
        if self.dry_run:
            # 生成最小的占位矛盾以支持后续流程
            for (a, b) in pairs:
                analyses.append({
                    "theory1": a, "theory2": b,
                    "contradictions": [
                        {"dimension": "wave_function_reality", "theory1_position": "ontic", "theory2_position": "epistemic", "core_tension": "status of Ψ", "importance_score": 7},
                        {"dimension": "measurement_process", "theory1_position": "collapse", "theory2_position": "unitary/decoherence", "core_tension": "nature of outcome definiteness", "importance_score": 6}
                    ],
                    "summary": f"Synthetic tensions between {a} and {b}."
                })
        else:
            for (a, b) in pairs:
                self._log_info(f"分析矛盾: {a} vs {b}")
                res = await analyzer.find_contradictions(a, b)
                if "error" in res:
                    self._log_warning(f"跳过失败对 {a},{b}: {res['error']}")
                    continue
                analyses.append(res)

        (self.synthesis_dir / "pairwise_contradictions.json").write_text(
            json.dumps(analyses, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 3) 矛盾网络
        graph_builder = ContradictionGraphBuilder(llm_interface=llm)
        graph = graph_builder.build_graph(analyses)
        (self.synthesis_dir / "network.json").write_text(
            json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 4) 概念放松
        planner = RelaxationPlanner(llm_interface=llm)
        plan = planner.plan(graph, budget=self.relaxation_budget)
        (self.synthesis_dir / "relaxation_plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 5) 统一合成
        eval_ready = self.synthesis_dir / "eval_ready_theories"
        eval_ready.mkdir(exist_ok=True)

        theories_out: List[Dict[str, Any]] = []
        parents = sorted(list({p for an in analyses for p in [an.get('theory1'), an.get('theory2')] if p}))

        if self.dry_run or llm is None:
            # 生成 num_seeds 个占位理论
            for k in range(self.num_seeds):
                theory = {
                    "name": f"Network-Integrated Interpretation (CNS-Lite) - {time.strftime('%m%d_%H%M')}-S{k+1}",
                    "metadata": {
                        "uid": f"THEORY-{time.strftime('%Y%m%d-%H%M%S')}-S{k+1}",
                        "schema_version": "2.1",
                        "author": "AI Physicist",
                        "lineage": {"method": "Network Synthesis from Contradiction", "parents": parents},
                    },
                    "mathematical_relation_to_sqm": "Interpretation",
                    "summary": "A relational-perspectival, layered-ontology synthesis under minimal relaxations.",
                    "core_principles": {
                        "ontological_commitments": "Layered potentiality + perspectival actuality.",
                        "epistemological_stances": "Agent-relative definiteness with intersubjective consistency.",
                        "key_postulates": [
                            "Unitary background; context-driven actualization.",
                            "No-signalling via holistic relational potential."
                        ]
                    },
                    "formalism": {
                        "mathematical_objects": "Hilbert spaces with context indices",
                        "governing_equations": ["i\\hbar \\partial_t |\\Psi\rangle = \\hat{H} |\\Psi\rangle"],
                        "comparison_with_sqm": {"agreements": "standard", "modifications": "n/a", "extensions": "context indices"}
                    },
                    "predictions_and_verifiability": {
                        "reproduces_sqm_predictions": "standard limits",
                        "deviations_from_sqm": [
                            {"prediction_name": "Context anomalies", "description": "weak deviations in extreme contexts", "mathematical_derivation": "heuristic", "experimental_setup": "macroscopic entanglement"}
                        ],
                        "unanswered_questions": "thresholds and agent criteria"
                    }
                }
                theories_out.append(theory)
        else:
            hg = HypothesisGenerator(llm)
            for k in range(self.num_seeds):
                # 轻微改变生成参数以增加多样性
                try:
                    theory = await hg.generate_from_contradictions_list(
                        analyses, plan,
                        generation_params={
                            "creativity_level": 0.55 + 0.1 * (k % 2),
                            "mathematical_rigor": 0.7,
                            "philosophical_depth": 0.7,
                            "emphasis_on_testability": 0.65
                        },
                        max_items=8 - (k % 2)
                    )
                    if "error" in theory:
                        self._log_warning(f"第{k+1}个统一合成失败: {theory['error']}")
                        continue
                    # 保证名称唯一
                    # 轻量去重并附加种子标识
                    if theory.get("name"):
                        theory["name"] = f"{theory['name']} (Seed {k+1})"
                    theories_out.append(theory)
                except Exception as e:
                    self._log_warning(f"第{k+1}个统一合成异常: {e}")

        if not theories_out:
            return GenerationResult(
                success=False,
                error_message="no theories generated",
                output_dir=str(self.synthesis_dir)
            ).to_dict()

        # --- 强制补齐最小 Schema，确保注册与评估稳定 ---
        def _ensure_minimal_schema(obj: Dict[str, Any], idx: int) -> Dict[str, Any]:
            now_tag = time.strftime('%m%d_%H%M')
            uid_tag = time.strftime('%Y%m%d-%H%M%S')
            # name
            if not obj.get('name') and obj.get('theory_name'):
                obj['name'] = obj['theory_name']
            if not obj.get('name'):
                obj['name'] = f"Network Unified Theory - {now_tag}-S{idx}"
            # metadata
            md = obj.setdefault('metadata', {})
            md.setdefault('uid', f"THEORY-{uid_tag}-S{idx}")
            md.setdefault('schema_version', '2.1')
            md.setdefault('author', 'AI Physicist')
            lineage = md.setdefault('lineage', {})
            lineage.setdefault('method', 'Network Synthesis from Contradiction')
            lineage.setdefault('parents', parents)
            # relation
            obj.setdefault('mathematical_relation_to_sqm', 'Interpretation')
            return obj

        # 保存理论并准备评估
        for theory in theories_out:
            theory = _ensure_minimal_schema(theory, theories_out.index(theory) + 1)
            safe_name = theory.get("name", "network_theory").replace(" ", "_").replace("/", "_").lower()
            eval_file = eval_ready / f"{safe_name}.json"
            eval_file.write_text(json.dumps(theory, ensure_ascii=False, indent=2), encoding="utf-8")
            self._log_info(f"网络合成完成: {eval_file}")

        return GenerationResult(
            success=True,
            theories=theories_out,
            metadata={
                "synthesis_dir": str(self.synthesis_dir),
                "eval_theories_dir": str(eval_ready),
                "method": "network_synthesis",
            },
            output_dir=str(self.synthesis_dir)
        ).to_dict()
