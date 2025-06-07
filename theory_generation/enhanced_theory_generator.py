#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强版量子理论生成器

基于优秀理论标准的七大标准设计：
1. 概念清晰性 - 明确本体论、操作性定义
2. 解释完整性 - 统一解释量子现象  
3. 数学一致性 - 严格数学表述、预测准确
4. 实验可区分性 - 新的可测量预测
5. 渐进保守性 - 包容现有成功、适当极限还原
6. 直觉可理解性 - 提供物理直觉
7. 哲学连贯性 - 清晰因果关系、时空观念

保持与现有JSON格式的完全兼容性。
"""

import os
import json
import time
from typing import List, Dict, Any, Optional

class EnhancedTheoryGenerator:
    """基于七大标准的增强理论生成器"""
    
    def __init__(self, llm_interface):
        """
        初始化理论生成器
        
        Args:
            llm_interface: LLM接口实例
        """
        self.llm = llm_interface
        self.generated_theories = []
        
        # 好理论的评价标准权重
        self.theory_standards = {
            "conceptual_clarity": 0.15,      # 概念清晰性
            "explanatory_completeness": 0.20, # 解释完整性
            "mathematical_consistency": 0.15,  # 数学一致性
            "experimental_distinguishability": 0.20, # 实验可区分性
            "progressive_conservatism": 0.10,  # 渐进保守性
            "intuitive_comprehensibility": 0.10, # 直觉可理解性
            "philosophical_coherence": 0.10   # 哲学连贯性
        }
    
    async def generate_high_quality_theory(self, theory1: str, theory2: str, 
                                         contradictions: List[Dict], 
                                         generation_config: Optional[Dict] = None) -> Dict:
        """
        基于好理论标准生成高质量理论
        
        Args:
            theory1: 第一个理论名称
            theory2: 第二个理论名称
            contradictions: 矛盾点列表
            generation_config: 生成配置
            
        Returns:
            Dict: 生成的高质量理论（兼容现有格式）
        """
        if generation_config is None:
            generation_config = {
                "focus_on_standards": ["explanatory_completeness", "experimental_distinguishability"],
                "mathematical_depth": "moderate",  # "basic", "moderate", "advanced"
                "novelty_level": "incremental",    # "incremental", "moderate", "revolutionary"
                "experimental_orientation": True
            }
        
        print(f"[INFO] 生成基于 {theory1} vs {theory2} 矛盾的高质量理论...")
        
        # 构建基于七大标准的提示
        prompt = self._build_standards_based_prompt(theory1, theory2, contradictions, generation_config)
        
        # 设置生成温度
        temperature = self._get_optimal_temperature(generation_config)
        
        # 调用LLM生成理论
        response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        
        # 解析并验证结果
        try:
            theory = self.llm.extract_json(response)
            
            if not theory:
                return {"error": "无法解析理论响应", "raw_response": response}
            
            # 确保格式兼容性
            theory = self._ensure_format_compatibility(theory)
            
            # 质量验证
            quality_report = self._assess_theory_quality(theory)
            theory["quality_assessment"] = quality_report
            
            # 添加生成元数据
            theory["generation_metadata"] = {
                "source_theories": [theory1, theory2],
                "contradictions_addressed": [c.get("dimension", "") for c in contradictions[:3]],
                "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "generation_config": generation_config,
                "generation_method": "enhanced_standards_based"
            }
            
            self.generated_theories.append(theory)
            
            print(f"[INFO] 成功生成高质量理论: {theory.get('name', '未命名理论')}")
            print(f"[INFO] 质量评分: {quality_report.get('overall_score', 0):.2f}/10")
            
            return theory
            
        except Exception as e:
            print(f"[ERROR] 理论生成失败: {str(e)}")
            return {"error": str(e), "theories": [theory1, theory2]}
    
    def _build_standards_based_prompt(self, theory1: str, theory2: str, 
                                    contradictions: List[Dict], config: Dict) -> str:
        """
        基于七大标准构建生成提示
        """
        # 处理矛盾点
        top_contradictions = sorted(
            contradictions, 
            key=lambda x: x.get("importance_score", 5),
            reverse=True
        )[:3]
        
        contradictions_text = ""
        for i, c in enumerate(top_contradictions, 1):
            dimension = c.get("dimension", f"矛盾点 {i}")
            theory1_pos = c.get("theory1_position", "未明确")
            theory2_pos = c.get("theory2_position", "未明确")
            core = c.get("core_tension", "")
            
            contradictions_text += f"""
**矛盾点 {i}: {dimension}**
- {theory1} 的立场: {theory1_pos}
- {theory2} 的立场: {theory2_pos}
- 核心张力: {core}
"""
        
        # 根据配置调整要求重点
        focused_standards = config.get("focus_on_standards", [])
        standards_requirements = self._get_standards_requirements(focused_standards, config)
        
        prompt = f"""
# 高质量量子理论生成任务

你是一位量子物理学理论家，需要基于现有理论矛盾创造一个新的量子力学诠释理论。你的理论必须符合优秀科学理论的核心标准。

## 基础理论对比
**理论对比**: {theory1} vs {theory2}

{contradictions_text}

## 优秀理论的核心标准

{standards_requirements}

## 理论输出格式

请以JSON格式输出你的理论，**严格遵循以下结构**以确保与现有系统兼容：

```json
{{
  "name": "<理论名称 - 简洁但体现核心创新>",
  "summary": "<100字以内的理论摘要，突出主要创新点>",
  
  "philosophy": {{
    "ontology": "<详细描述：根据这个理论，什么是真实存在的？>",
    "measurement": "<详细描述：测量在这个理论中意味着什么？观察者的角色？>"
  }},
  
  "parameters": {{
    "<符号1>": {{
      "value": <典型数值>,
      "unit": "<物理单位>",
      "role": "<这个参数在理论中的详细物理作用>"
    }},
    "<符号2>": {{
      "value": <典型数值>,
      "unit": "<物理单位>",
      "role": "<这个参数在理论中的详细物理作用>"
    }}
    // 包含2-3个有意义的新参数，如果理论不需要新参数则留空 {{}}
  }},
  
  "formalism": {{
    "hamiltonian": "<完整的哈密顿量，LaTeX格式>",
    "state_equation": "<状态演化方程，LaTeX格式>",
    "measurement_rule": "<测量规则的详细描述>"
  }},
  
  "semantics": {{
    "<重要符号/概念>": "<这个符号/概念在理论中的深层物理意义>",
    "<重要符号/概念>": "<这个符号/概念在理论中的深层物理意义>",
    "overall_picture": "<理论的整体物理图像，如何理解量子世界>"
  }}
}}
```

## 关键要求

1. **概念清晰**: 每个概念都要有明确的物理意义，避免纯数学符号游戏
2. **解释力强**: 能够自然地解释测量问题、量子纠缠、干涉现象等
3. **数学严谨**: 方程必须数学上自洽，参数值要合理
4. **实验相关**: 提供可以与现有理论区分的具体预测
5. **连续性**: 在适当极限下回到标准量子力学
6. **可理解**: 提供直观的物理图像，不只是抽象数学
7. **逻辑一致**: 避免因果悖论，保持逻辑连贯性

请创建一个真正推进我们对量子世界理解的理论，而不仅仅是现有理论的重新包装。
"""
        return prompt
    
    def _get_standards_requirements(self, focused_standards: List[str], config: Dict) -> str:
        """
        根据配置生成标准要求描述
        """
        all_standards = {
            "conceptual_clarity": "**概念清晰性**: 明确的本体论承诺，每个概念都有操作性定义",
            "explanatory_completeness": "**解释完整性**: 统一解释所有量子现象，不需要针对不同现象的专门假设",
            "mathematical_consistency": "**数学一致性**: 严格的数学表述，保持量子力学的成功预测",
            "experimental_distinguishability": "**实验可区分性**: 提供新的可测量预测，明确与其他理论的区别",
            "progressive_conservatism": "**渐进保守性**: 包容现有成功，在适当极限下还原到经典物理",
            "intuitive_comprehensibility": "**直觉可理解性**: 提供物理直觉和可视化图像",
            "philosophical_coherence": "**哲学连贯性**: 清晰的因果关系，与时空观念协调"
        }
        
        if focused_standards:
            # 突出重点标准
            requirements = "### 重点关注标准:\n"
            for standard in focused_standards:
                if standard in all_standards:
                    requirements += f"- {all_standards[standard]}\n"
            
            requirements += "\n### 其他重要标准:\n"
            for standard, desc in all_standards.items():
                if standard not in focused_standards:
                    requirements += f"- {desc}\n"
        else:
            # 全部标准
            requirements = "### 核心标准:\n"
            for desc in all_standards.values():
                requirements += f"- {desc}\n"
        
        return requirements
    
    def _get_optimal_temperature(self, config: Dict) -> float:
        """
        根据配置确定最优生成温度
        """
        novelty = config.get("novelty_level", "incremental")
        math_depth = config.get("mathematical_depth", "moderate")
        
        base_temp = 0.6
        
        # 根据新颖性调整
        if novelty == "revolutionary":
            base_temp += 0.3
        elif novelty == "moderate":
            base_temp += 0.15
        # incremental 保持基础温度
        
        # 根据数学深度调整
        if math_depth == "advanced":
            base_temp -= 0.1  # 更保守，确保数学正确性
        elif math_depth == "basic":
            base_temp += 0.1  # 更灵活
        
        return min(max(base_temp, 0.3), 0.9)  # 限制在合理范围内
    
    def _ensure_format_compatibility(self, theory: Dict) -> Dict:
        """
        确保理论格式与现有系统兼容
        """
        # 确保必需字段存在
        required_fields = ["name", "parameters", "formalism"]
        for field in required_fields:
            if field not in theory:
                if field == "parameters":
                    theory[field] = {}
                elif field == "formalism":
                    theory[field] = {
                        "hamiltonian": "H_QM",
                        "state_equation": "i\\hbar \\partial_t \\psi = H \\psi",
                        "measurement_rule": "Born rule"
                    }
                else:
                    theory[field] = f"未指定的{field}"
        
        # 确保formalism字段的结构
        if "formalism" in theory:
            formalism = theory["formalism"]
            if "hamiltonian" not in formalism:
                formalism["hamiltonian"] = "H_QM"
            if "state_equation" not in formalism:
                formalism["state_equation"] = "i\\hbar \\partial_t \\psi = H \\psi"
            if "measurement_rule" not in formalism:
                formalism["measurement_rule"] = "Born rule"
        
        return theory
    
    def _assess_theory_quality(self, theory: Dict) -> Dict:
        """
        评估理论质量
        """
        scores = {}
        
        # 概念清晰性评分
        clarity_score = 0
        if "philosophy" in theory and "ontology" in theory["philosophy"]:
            clarity_score += 5
        if "semantics" in theory and len(theory["semantics"]) > 2:
            clarity_score += 5
        scores["conceptual_clarity"] = min(clarity_score, 10)
        
        # 数学一致性评分
        math_score = 0
        if "formalism" in theory:
            formalism = theory["formalism"]
            if "hamiltonian" in formalism and len(formalism["hamiltonian"]) > 5:
                math_score += 4
            if "state_equation" in formalism and len(formalism["state_equation"]) > 10:
                math_score += 3
            if "measurement_rule" in formalism:
                math_score += 3
        scores["mathematical_consistency"] = min(math_score, 10)
        
        # 参数新颖性评分
        param_score = 0
        if "parameters" in theory:
            params = theory["parameters"]
            param_score = min(len(params) * 3, 10)  # 每个参数3分，最多10分
        scores["parameter_novelty"] = param_score
        
        # 解释性评分
        explanation_score = 0
        if "summary" in theory and len(theory["summary"]) > 50:
            explanation_score += 5
        if "philosophy" in theory and "measurement" in theory["philosophy"]:
            explanation_score += 5
        scores["explanatory_power"] = min(explanation_score, 10)
        
        # 计算总分
        overall_score = sum(scores.values()) / len(scores)
        
        return {
            "detailed_scores": scores,
            "overall_score": overall_score,
            "assessment_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    async def generate_theory_family(self, theory1: str, theory2: str,
                                   contradictions: List[Dict], 
                                   family_size: int = 3) -> List[Dict]:
        """
        生成理论家族（多个相关变体）
        """
        theories = []
        
        # 定义不同的生成配置
        configs = [
            {
                "focus_on_standards": ["explanatory_completeness", "experimental_distinguishability"],
                "mathematical_depth": "moderate",
                "novelty_level": "incremental",
                "experimental_orientation": True
            },
            {
                "focus_on_standards": ["mathematical_consistency", "conceptual_clarity"],
                "mathematical_depth": "advanced",
                "novelty_level": "moderate", 
                "experimental_orientation": False
            },
            {
                "focus_on_standards": ["intuitive_comprehensibility", "philosophical_coherence"],
                "mathematical_depth": "moderate",
                "novelty_level": "moderate",
                "experimental_orientation": True
            }
        ]
        
        for i in range(family_size):
            config = configs[i % len(configs)]
            
            print(f"[INFO] 生成理论家族成员 {i+1}/{family_size}")
            
            theory = await self.generate_high_quality_theory(
                theory1, theory2, contradictions, config
            )
            
            if "error" not in theory:
                # 添加家族标识
                if "name" in theory and f"(Variant {i+1})" not in theory["name"]:
                    theory["name"] = f"{theory['name']} (Variant {i+1})"
                theory["family_member"] = i + 1
                
                theories.append(theory)
        
        return theories 