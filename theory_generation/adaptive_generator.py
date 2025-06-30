#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应理论生成器

结合创新层次框架和精确反馈机制的智能理论生成器。
能够根据目标创新层次和历史反馈动态调整生成策略。
"""

import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from theory_generation.innovation_framework import InnovationFramework, InnovationLevel

class AdaptiveTheoryGenerator:
    """自适应量子理论生成器"""
    
    def __init__(self, llm_interface, temperature=0.7, performance_monitor=None):
        self.llm = llm_interface
        self.temperature = temperature
        self.innovation_framework = InnovationFramework()
        self.generation_history = []
        self.feedback_memory = {}
        self.performance_monitor = performance_monitor
        
    async def generate_targeted_theory(self,
                                     contradiction: Dict,
                                     target_innovation_level: InnovationLevel,
                                     feedback_context: Optional[Dict] = None) -> Dict:
        """
        根据目标创新层次生成理论
        
        Args:
            contradiction: 矛盾分析结果
            target_innovation_level: 目标创新层次
            feedback_context: 反馈上下文（来自之前的评估）
            
        Returns:
            Dict: 生成的理论
        """
        print(f"[INFO] 目标创新层次: {target_innovation_level.value}")
        
        # 获取创新目标配置
        innovation_targets = self.innovation_framework.generate_innovation_targets(target_innovation_level)
        
        # 构建增强提示
        prompt = self._build_adaptive_prompt(
            contradiction, 
            innovation_targets, 
            feedback_context
        )
        
        # 根据创新层次调整生成参数
        generation_params = self._get_adaptive_parameters(target_innovation_level, feedback_context)
        
        # 生成理论
        if self.performance_monitor:
            self.performance_monitor.record_api_call()
        
        response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=generation_params["temperature"]
        )
        
        # 解析并验证
        theory = self.llm.extract_json(response)
        
        if not theory:
            print("[ERROR] 无法解析LLM响应为JSON")
            return {"error": "无法解析响应", "raw_response": response}
        
        # 添加时间戳和创新标记
        theory = self._post_process_theory(theory, target_innovation_level)
        
        # 验证创新层次是否达标
        actual_level, innovation_scores = self.innovation_framework.assess_theory_innovation_level(theory)
        
        print(f"[INFO] 实际创新层次: {actual_level.value}")
        print(f"[INFO] 创新评分: {innovation_scores}")
        
        # 记录生成历史
        self._record_generation(theory, target_innovation_level, actual_level, innovation_scores)
        
        return theory
    
    def _build_adaptive_prompt(self, 
                             contradiction: Dict, 
                             innovation_targets: Dict,
                             feedback_context: Optional[Dict] = None) -> str:
        """构建自适应生成提示"""
        
        theory1 = contradiction.get("theory1", "理论1")
        theory2 = contradiction.get("theory2", "理论2")
        contradictions = contradiction.get("contradictions", [])
        
        # 构建矛盾描述
        contradictions_text = ""
        for i, cont in enumerate(contradictions, 1):
            cont_name = cont.get("contradiction", f"矛盾{i}")
            theory1_pos = cont.get("theory1_position", "")
            theory2_pos = cont.get("theory2_position", "")
            
            contradictions_text += f"""
### 矛盾 {i}: {cont_name}
- **{theory1}的立场**: {theory1_pos}
- **{theory2}的立场**: {theory2_pos}
"""
        
        # 构建创新指导
        target_level = innovation_targets["target_level"]
        mathematical_targets = "\n".join([f"- {t}" for t in innovation_targets["mathematical_targets"]])
        conceptual_targets = "\n".join([f"- {t}" for t in innovation_targets["conceptual_targets"]])
        experimental_targets = "\n".join([f"- {t}" for t in innovation_targets["experimental_targets"]])
        philosophical_targets = "\n".join([f"- {t}" for t in innovation_targets["philosophical_targets"]])
        
        guidance = innovation_targets["generation_guidance"]
        
        # 构建反馈改进指导
        feedback_guidance = ""
        if feedback_context:
            feedback_guidance = self._build_feedback_guidance(feedback_context)
        
        prompt = f"""
# ADVANCED QUANTUM THEORY GENERATION TASK

## MISSION
You are tasked with creating a **{target_level}** level quantum theory that resolves the contradictions between **{theory1}** and **{theory2}**. This theory must meet specific innovation criteria while maintaining scientific rigor.

## CONTRADICTION ANALYSIS
{contradictions_text}

## INNOVATION REQUIREMENTS

### Target Innovation Level: **{target_level.upper()}**

### Mathematical Innovation Targets:
{mathematical_targets}

### Conceptual Innovation Targets:
{conceptual_targets}

### Experimental Innovation Targets:
{experimental_targets}

### Philosophical Innovation Targets:
{philosophical_targets}

## GENERATION GUIDANCE

### Mathematical Focus:
{guidance["mathematical_focus"]}

### Conceptual Focus:
{guidance["conceptual_focus"]}

### Experimental Focus:
{guidance["experimental_focus"]}

### Philosophical Focus:
{guidance["philosophical_focus"]}

{feedback_guidance}

## CRITICAL INNOVATION CRITERIA

Your theory will be evaluated on these dimensions (target scores for {target_level}):
- **Mathematical Novelty**: {innovation_targets["success_criteria"]["mathematical_novelty"]:.1f}/1.0
- **Conceptual Breakthrough**: {innovation_targets["success_criteria"]["conceptual_breakthrough"]:.1f}/1.0
- **Experimental Distinguishability**: {innovation_targets["success_criteria"]["experimental_distinguishability"]:.1f}/1.0
- **Philosophical Depth**: {innovation_targets["success_criteria"]["philosophical_depth"]:.1f}/1.0
- **Paradigm Shift Potential**: {innovation_targets["success_criteria"]["paradigm_shift_potential"]:.1f}/1.0

## OUTPUT REQUIREMENTS

Your output **MUST** be a single, valid JSON object that strictly adheres to the "Quantum Theory Schema v2.1":

```json
{{
  "name": "string (A name that reflects the innovation level and key concepts)",
  "metadata": {{
    "uid": "string (Unique identifier)",
    "schema_version": "2.1",
    "author": "AI Physicist",
    "tags": ["array of strings reflecting innovation level and key concepts"],
    "lineage": {{
      "method": "Adaptive Generation (Innovation Level: {target_level})",
      "parents": ["{theory1}", "{theory2}"],
      "inspiration": "string (Explain how theory achieves {target_level} innovation)"
    }},
    "innovation_targets": {{
      "target_level": "{target_level}",
      "mathematical_innovation": {innovation_targets["success_criteria"]["mathematical_novelty"]:.1f},
      "conceptual_breakthrough": {innovation_targets["success_criteria"]["conceptual_breakthrough"]:.1f},
      "experimental_distinguishability": {innovation_targets["success_criteria"]["experimental_distinguishability"]:.1f}
    }}
  }},
  "mathematical_relation_to_sqm": "string (Choose based on innovation level: Interpretation/Modification/Extension)",
  "summary": "string (One paragraph emphasizing innovative aspects)",
  "core_principles": {{
    "ontological_commitments": "string (What fundamentally exists - be specific about innovations)",
    "epistemological_stances": "string (How knowledge is acquired - emphasize new perspectives)",
    "key_postulates": ["array of innovative postulates specific to {target_level} level"]
  }},
  "formalism": {{
    "mathematical_objects": "string (List mathematical objects - emphasize novel ones for {target_level})",
    "governing_equations": ["array of equations in LaTeX - show mathematical innovation"],
    "comparison_with_sqm": {{
      "agreements": "string (What is retained from SQM)",
      "modifications": "string (What is changed - be specific about innovation level)",
      "extensions": "string (What is added beyond SQM)"
    }}
  }},
  "predictions_and_verifiability": {{
    "reproduces_sqm_predictions": "string (How theory reduces to SQM)",
    "deviations_from_sqm": [
      {{
        "prediction_name": "string (Novel prediction reflecting {target_level} innovation)",
        "description": "string (Detailed description of the new prediction)",
        "mathematical_derivation": "string (Show how innovation leads to this prediction)",
        "experimental_setup": "string (Specific, feasible experimental design)"
      }}
    ],
    "unanswered_questions": "string (New research directions opened by this innovation)"
  }}
}}
```

**MANDATORY REQUIREMENTS:**
1. **Innovation Level Compliance**: The theory MUST demonstrably achieve the {target_level} innovation level
2. **Mathematical Precision**: All equations must be syntactically correct LaTeX
3. **Experimental Testability**: Include at least one specific, feasible experimental prediction
4. **Conceptual Coherence**: All innovations must be internally consistent and well-motivated
5. **Scientific Grounding**: Maintain connection to established physics while pushing boundaries

**WARNING**: Theories that fail to meet the {target_level} innovation criteria will be automatically rejected. Push the boundaries appropriately for this level!
"""
        
        return prompt
    
    def _build_feedback_guidance(self, feedback_context: Dict) -> str:
        """根据反馈构建改进指导"""
        guidance = "\n## FEEDBACK-BASED IMPROVEMENTS\n"
        
        if "evaluation_results" in feedback_context:
            eval_results = feedback_context["evaluation_results"]
            
            # 分析评估分数和评语
            if "role_score" in eval_results:
                score = eval_results["role_score"]
                if score < 0.6:
                    guidance += "**CRITICAL**: Previous theory scored poorly. Focus on fundamental improvements.\n"
                elif score < 0.8:
                    guidance += "**MODERATE**: Previous theory needs significant enhancements.\n"
                else:
                    guidance += "**GOOD**: Build on previous strengths while addressing weaknesses.\n"
            
            # 分析具体角色反馈
            if "details" in eval_results:
                for role, details in eval_results["details"].items():
                    if "rationale" in details:
                        rationale = details["rationale"]
                        score = details.get("score", 0)
                        
                        if score < 6:
                            guidance += f"\n**{role.capitalize()} Critical Issues**:\n{rationale[:200]}...\n"
                            guidance += f"**Action Required**: Address these {role} concerns directly.\n"
        
        if "innovation_gaps" in feedback_context:
            gaps = feedback_context["innovation_gaps"]
            guidance += f"\n**Innovation Gaps to Address**:\n"
            for gap in gaps:
                guidance += f"- {gap}\n"
        
        return guidance
    
    def _get_adaptive_parameters(self, 
                               target_level: InnovationLevel, 
                               feedback_context: Optional[Dict] = None) -> Dict:
        """根据创新层次和反馈获取自适应参数"""
        
        # 使用初始化时传入的温度作为基准
        base_temp = self.temperature
        
        # 创新层次对温度的微调
        level_adjustments = {
            InnovationLevel.INTERPRETATION: -0.1,
            InnovationLevel.PARAMETER_EXTENSION: 0.0,
            InnovationLevel.EQUATION_MODIFICATION: 0.1,
            InnovationLevel.FRAMEWORK_EXTENSION: 0.2,
            InnovationLevel.PARADIGM_REVOLUTION: 0.25
        }
        
        adjusted_temp = base_temp + level_adjustments.get(target_level, 0.0)
        
        # 根据反馈再次微调
        if feedback_context:
            if "low_innovation_score" in feedback_context:
                adjusted_temp = min(adjusted_temp + 0.1, 0.99)
            elif "high_innovation_score" in feedback_context:
                adjusted_temp = max(adjusted_temp - 0.1, 0.1)
        
        return {"temperature": min(max(adjusted_temp, 0.1), 0.99)}
    
    def _post_process_theory(self, theory: Dict, target_level: InnovationLevel) -> Dict:
        """后处理生成的理论"""
        
        # 添加时间戳
        if "name" in theory:
            timestamp = time.strftime("%m%d_%H%M", time.localtime())
            if not any(f"-{ts}" in theory["name"] for ts in [timestamp[:4], timestamp[5:]]):
                theory["name"] = f"{theory['name']}-{timestamp}"
        
        # 确保metadata存在并完整
        if "metadata" not in theory:
            theory["metadata"] = {}
        
        theory["metadata"]["generation_info"] = {
            "source": "adaptive_generation",
            "target_innovation_level": target_level.value,
            "generation_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "adaptive_parameters": self._get_adaptive_parameters(target_level)
        }
        
        return theory
    
    def _record_generation(self, 
                          theory: Dict, 
                          target_level: InnovationLevel,
                          actual_level: InnovationLevel,
                          innovation_scores: Dict) -> None:
        """记录生成历史"""
        
        record = {
            "timestamp": time.time(),
            "theory_name": theory.get("name", "Unknown"),
            "target_innovation_level": target_level.value,
            "actual_innovation_level": actual_level.value,
            "innovation_scores": innovation_scores,
            "level_match": target_level == actual_level,
            "overall_innovation_score": sum(innovation_scores.values()) / len(innovation_scores)
        }
        
        self.generation_history.append(record)
        
    async def generate_progressive_series(self, 
                                        contradiction: Dict,
                                        max_level: InnovationLevel = InnovationLevel.FRAMEWORK_EXTENSION) -> List[Dict]:
        """
        生成渐进式理论系列，从低创新层次到高创新层次
        
        Args:
            contradiction: 矛盾分析
            max_level: 最高创新层次
            
        Returns:
            List[Dict]: 渐进式理论系列
        """
        
        # 定义渐进序列
        progression = [
            InnovationLevel.INTERPRETATION,
            InnovationLevel.PARAMETER_EXTENSION, 
            InnovationLevel.EQUATION_MODIFICATION,
            InnovationLevel.FRAMEWORK_EXTENSION,
            InnovationLevel.PARADIGM_REVOLUTION
        ]
        
        # 找到目标层次的索引
        max_index = progression.index(max_level)
        target_levels = progression[:max_index + 1]
        
        theories = []
        feedback_context = None
        
        for i, level in enumerate(target_levels):
            print(f"[INFO] 生成渐进系列 {i+1}/{len(target_levels)}: {level.value}")
            
            theory = await self.generate_targeted_theory(
                contradiction, 
                level, 
                feedback_context
            )
            
            if "error" not in theory:
                theories.append(theory)
                
                # 为下一层次提供反馈上下文
                if i < len(target_levels) - 1:
                    feedback_context = {
                        "previous_theory": theory,
                        "previous_level": level,
                        "target_next_level": target_levels[i + 1]
                    }
        
        return theories
    
    def analyze_generation_performance(self) -> Dict:
        """分析生成性能"""
        if not self.generation_history:
            return {"message": "没有生成历史数据"}
        
        total_generations = len(self.generation_history)
        level_matches = sum(1 for record in self.generation_history if record["level_match"])
        match_rate = level_matches / total_generations
        
        avg_innovation_score = sum(record["overall_innovation_score"] for record in self.generation_history) / total_generations
        
        level_distribution = {}
        for record in self.generation_history:
            level = record["actual_innovation_level"]
            level_distribution[level] = level_distribution.get(level, 0) + 1
        
        return {
            "total_generations": total_generations,
            "level_match_rate": match_rate,
            "average_innovation_score": avg_innovation_score,
            "level_distribution": level_distribution,
            "latest_records": self.generation_history[-5:]  # 最近5次记录
        } 