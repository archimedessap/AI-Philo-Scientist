#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
理论一致性验证器

验证理论的内部一致性和逻辑严谨性，
检查理论内部不同部分之间是否存在逻辑矛盾。
"""

import os
import json
from typing import Dict, Any

class ConsistencyValidator:
    """验证理论的内部一致性和逻辑严谨性"""
    
    def __init__(self, llm_interface):
        """
        初始化一致性验证器
        
        Args:
            llm_interface: LLM接口实例
        """
        self.llm = llm_interface
    
    async def validate(self, theory: Dict) -> Dict:
        """
        验证理论的一致性
        
        Args:
            theory: 理论数据
            
        Returns:
            Dict: 验证结果
        """
        theory_name = theory.get("name", theory.get("theory_name", "未命名理论"))
        print(f"[INFO] 验证理论的内部一致性: {theory_name}")
        
        # 构建提示
        prompt = self._build_consistency_prompt(theory)
        
        # 调用LLM进行一致性验证
        response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2  # 低温度，保证评估的确定性
        )
        
        # 解析响应
        result = self.llm.extract_json(response)
        
        if not result:
            print(f"[WARN] 无法解析LLM响应为有效JSON，将使用默认结果")
            # 创建默认结果
            result = {
                "overall_score": 5.0,
                "dimension_scores": {
                    "logical_consistency": 5.0,
                    "conceptual_coherence": 5.0,
                    "principle_alignment": 5.0
                },
                "analysis": "无法解析验证结果",
                "inconsistencies": [],
                "recommendations": []
            }
        
        return result
    
    def _build_consistency_prompt(self, theory: Dict) -> str:
        """
        构建一致性验证提示
        
        Args:
            theory: 理论数据
            
        Returns:
            str: 验证提示
        """
        # 提取理论关键部分进行验证
        name = theory.get("name", theory.get("theory_name", "未命名理论"))
        principles = theory.get("core_principles", [])
        if isinstance(principles, list):
            principles_text = "\n".join([f"- {p}" for p in principles])
        else:
            principles_text = principles
        
        # 提取数学形式化
        math_formalism = theory.get("mathematical_formalism", {})
        if isinstance(math_formalism, dict):
            equations = math_formalism.get("key_equations", [])
            math_text = "\n".join([
                f"方程: {eq.get('name', '未命名')}\n{eq.get('equation', '')}\n{eq.get('description', '')}"
                for eq in equations
            ])
        else:
            math_text = str(math_formalism)
        
        # 提取量子现象解释
        phenomena = theory.get("quantum_phenomena_explanations", {})
        if isinstance(phenomena, dict):
            phenomena_text = "\n".join([f"- {k}: {v}" for k, v in phenomena.items()])
        else:
            phenomena_text = str(phenomena)
        
        # 构建完整提示
        prompt = f"""
作为量子物理学的专家评审，请评估以下量子诠释理论的内部一致性和逻辑严谨性。

理论名称: {name}

核心原则:
{principles_text}

数学形式化:
{math_text}

量子现象解释:
{phenomena_text}

请严格分析该理论的内部一致性，特别关注:
1. 逻辑一致性: 核心原则之间是否存在逻辑矛盾
2. 概念连贯性: 概念定义是否清晰且一致使用
3. 数学严谨性: 数学形式是否自洽，与物理概念是否对应
4. 原则与应用一致性: 核心原则与具体量子现象解释是否一致

请返回JSON格式的分析结果:
{{
  "overall_score": 一致性总体评分(0-10),
  "dimension_scores": {{
    "logical_consistency": 逻辑一致性评分(0-10),
    "conceptual_coherence": 概念连贯性评分(0-10),
    "mathematical_rigor": 数学严谨性评分(0-10),
    "principle_alignment": 原则与应用一致性评分(0-10)
  }},
  "analysis": "总体一致性分析",
  "inconsistencies": [
    {{
      "type": "不一致类型",
      "description": "详细描述",
      "severity": 严重程度(1-10)
    }}
  ],
  "recommendations": [
    "改进建议1",
    "改进建议2"
  ]
}}

请确保你的评估是严格、客观和深入的，找出任何可能的内部矛盾或概念不清晰之处。
"""
        return prompt
