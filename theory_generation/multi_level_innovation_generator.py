#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多层次综合创新生成器

实现同时在多个创新层次上进行综合创新的理论生成系统。
支持创新层次的组合、权重配置和协同优化。
结合概念向量空间进行增强的矛盾检测和理论生成。
"""

import json
import time
import random
import itertools
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from theory_generation.innovation_framework import InnovationFramework, InnovationLevel

@dataclass
class MultiLevelInnovationConfig:
    """多层次创新配置"""
    target_levels: List[InnovationLevel]  # 目标创新层次列表
    level_weights: Dict[InnovationLevel, float]  # 各层次权重
    synthesis_mode: str  # 综合模式: "parallel", "hierarchical", "fusion"
    innovation_intensity: float  # 总体创新强度 (0.0-1.0)
    constraint_preservation: bool  # 是否保持层次间约束
    use_concept_space: bool = True  # 是否使用概念向量空间
    
class MultiLevelInnovationGenerator:
    """多层次综合创新生成器"""
    
    def __init__(self, llm_interface, base_innovation_framework=None, performance_monitor=None, concept_embeddings=None):
        self.llm = llm_interface
        self.base_framework = base_innovation_framework or InnovationFramework()
        self.performance_monitor = performance_monitor
        self.concept_embeddings = concept_embeddings or {}  # 概念向量空间
        self.synthesis_strategies = {
            "parallel": self._parallel_synthesis,
            "hierarchical": self._hierarchical_synthesis, 
            "fusion": self._fusion_synthesis
        }
        self.generation_history = []
        
    def load_concept_embeddings(self, embedding_file: str):
        """加载概念嵌入向量"""
        try:
            import pickle
            with open(embedding_file, 'rb') as f:
                self.concept_embeddings = pickle.load(f)
            print(f"[INFO] 加载了 {len(self.concept_embeddings)} 个概念嵌入向量")
        except Exception as e:
            print(f"[WARN] 无法加载概念嵌入: {e}")
            self.concept_embeddings = {}
    
    def _enhanced_contradiction_analysis(self, contradiction: Dict) -> Dict:
        """基于概念向量空间的增强矛盾分析"""
        if not self.concept_embeddings:
            return contradiction
            
        # 提取矛盾中涉及的概念
        enhanced_contradictions = []
        
        for contra in contradiction.get('contradictions', []):
            # 分析概念向量空间中的相关概念
            dimension = contra.get('dimension', '')
            theory1_pos = contra.get('theory1_position', '')
            theory2_pos = contra.get('theory2_position', '')
            
            # 在概念空间中找到相关概念
            related_concepts = self._find_related_concepts(dimension, theory1_pos, theory2_pos)
            
            enhanced_contra = contra.copy()
            enhanced_contra['related_concepts'] = related_concepts
            enhanced_contra['concept_space_tension'] = self._calculate_concept_tension(related_concepts)
            enhanced_contradictions.append(enhanced_contra)
        
        enhanced_contradiction = contradiction.copy()
        enhanced_contradiction['contradictions'] = enhanced_contradictions
        enhanced_contradiction['concept_space_analysis'] = True
        
        return enhanced_contradiction
    
    def _find_related_concepts(self, dimension: str, pos1: str, pos2: str) -> List[Dict]:
        """在概念空间中找到相关概念"""
        related_concepts = []
        
        # 确保dimension是字符串类型
        if not isinstance(dimension, str):
            dimension = str(dimension)
        
        # 计算维度和立场的嵌入向量
        dimension_concepts = [name for name in self.concept_embeddings.keys() 
                            if any(word in name.lower() for word in dimension.lower().split())]
        
        for concept_name in dimension_concepts[:5]:  # 限制数量
            concept_vector = self.concept_embeddings[concept_name]
            
            # 计算与其他概念的相似度
            similarities = []
            for other_name, other_vector in self.concept_embeddings.items():
                if other_name != concept_name:
                    sim = self._cosine_similarity(concept_vector, other_vector)
                    similarities.append((other_name, sim))
            
            # 取最相似的几个概念
            top_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]
            
            related_concepts.append({
                'concept': concept_name,
                'related_concepts': top_similar,
                'relevance_to_dimension': self._calculate_relevance(concept_name, dimension)
            })
        
        return related_concepts
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        try:
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        except:
            return 0.0
    
    def _calculate_concept_tension(self, related_concepts: List[Dict]) -> float:
        """计算概念张力"""
        if not related_concepts:
            return 0.0
        
        # 基于概念间的相似度分布计算张力
        all_similarities = []
        for concept_info in related_concepts:
            similarities = [sim for _, sim in concept_info.get('related_concepts', [])]
            all_similarities.extend(similarities)
        
        if not all_similarities:
            return 0.0
        
        # 张力 = 1 - 平均相似度（相似度越低，张力越高）
        avg_similarity = np.mean(all_similarities)
        return 1.0 - avg_similarity
    
    def _calculate_relevance(self, concept_name: str, dimension: str) -> float:
        """计算概念与维度的相关性"""
        # 确保输入是字符串类型
        if not isinstance(concept_name, str):
            concept_name = str(concept_name)
        if not isinstance(dimension, str):
            dimension = str(dimension)
        
        # 简单的基于关键词匹配的相关性计算
        concept_words = set(concept_name.lower().split())
        dimension_words = set(dimension.lower().split())
        
        if not concept_words or not dimension_words:
            return 0.0
        
        intersection = concept_words & dimension_words
        union = concept_words | dimension_words
        
        return len(intersection) / len(union) if union else 0.0
    
    def create_multi_level_config(self, 
                                 target_levels: List[InnovationLevel],
                                 weights: Optional[Dict[InnovationLevel, float]] = None,
                                 synthesis_mode: str = "fusion",
                                 innovation_intensity: float = 0.7) -> MultiLevelInnovationConfig:
        """
        创建多层次创新配置
        
        Args:
            target_levels: 目标创新层次列表
            weights: 各层次权重 (可选，默认均等)
            synthesis_mode: 综合模式
            innovation_intensity: 创新强度
            
        Returns:
            MultiLevelInnovationConfig: 多层次创新配置
        """
        
        # 默认权重：均等分配
        if weights is None:
            weight_value = 1.0 / len(target_levels)
            weights = {level: weight_value for level in target_levels}
        
        # 归一化权重
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        return MultiLevelInnovationConfig(
            target_levels=target_levels,
            level_weights=normalized_weights,
            synthesis_mode=synthesis_mode,
            innovation_intensity=innovation_intensity,
            constraint_preservation=True
        )
    
    async def generate_multi_level_theory(self,
                                         contradiction: Dict,
                                         config: MultiLevelInnovationConfig,
                                         feedback_context: Optional[Dict] = None) -> Dict:
        """
        生成多层次综合创新理论
        
        Args:
            contradiction: 矛盾分析结果
            config: 多层次创新配置
            feedback_context: 反馈上下文
            
        Returns:
            Dict: 生成的多层次创新理论
        """
        
        print(f"[MULTI-LEVEL] 目标层次: {[level.value for level in config.target_levels]}")
        print(f"[MULTI-LEVEL] 权重配置: {[(k.value, f'{v:.2f}') for k, v in config.level_weights.items()]}")
        print(f"[MULTI-LEVEL] 综合模式: {config.synthesis_mode}")
        
        # 0. 如果启用概念空间，进行增强矛盾分析
        if config.use_concept_space and self.concept_embeddings:
            print("[MULTI-LEVEL] 启用概念向量空间增强分析")
            contradiction = self._enhanced_contradiction_analysis(contradiction)
        
        # 1. 为每个层次生成创新目标
        level_targets = {}
        for level in config.target_levels:
            targets = self.base_framework.generate_innovation_targets(level)
            level_targets[level] = targets
            
        # 2. 构建综合创新策略
        synthesis_strategy = await self.synthesis_strategies[config.synthesis_mode](
            level_targets, config, contradiction
        )
        
        # 3. 构建多层次综合提示
        prompt = self._build_multi_level_prompt(
            contradiction, 
            synthesis_strategy, 
            config,
            feedback_context
        )
        
        # 4. 生成理论
        if self.performance_monitor:
            self.performance_monitor.record_api_call()
            
        response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=self._calculate_adaptive_temperature(config)
        )
        
        # 5. 解析和验证
        theory = self.llm.extract_json(response)
        
        if not theory:
            return {"error": "无法解析LLM响应", "raw_response": response}
        
        # 6. 多层次创新评估
        multi_level_assessment = self._assess_multi_level_innovation(theory, config)
        
        # 7. 后处理和标记
        theory = self._post_process_multi_level_theory(theory, config, multi_level_assessment)
        
        # 8. 记录生成历史
        self._record_multi_level_generation(theory, config, multi_level_assessment)
        
        return theory
    
    async def _parallel_synthesis(self, 
                                 level_targets: Dict[InnovationLevel, Dict],
                                 config: MultiLevelInnovationConfig,
                                 contradiction: Dict) -> Dict:
        """
        并行综合策略：同时在多个层次上创新
        """
        
        synthesis = {
            "strategy_name": "并行多层次创新",
            "description": "在多个创新层次上同时进行创新，确保各层次目标的协调统一",
            "innovation_requirements": {},
            "synthesis_principles": [
                "各创新层次相互补强，形成创新合力",
                "避免层次间的冲突和矛盾",
                "确保整体理论的自洽性"
            ]
        }
        
        # 合并所有层次的创新目标
        all_mathematical_targets = []
        all_conceptual_targets = []
        all_experimental_targets = []
        all_philosophical_targets = []
        
        for level, weight in config.level_weights.items():
            targets = level_targets[level]
            
            # 按权重合并目标
            weighted_math = [f"[{level.value}] {target}" for target in targets["mathematical_targets"]]
            weighted_concept = [f"[{level.value}] {target}" for target in targets["conceptual_targets"]]
            weighted_exp = [f"[{level.value}] {target}" for target in targets["experimental_targets"]]
            weighted_phil = [f"[{level.value}] {target}" for target in targets["philosophical_targets"]]
            
            all_mathematical_targets.extend(weighted_math)
            all_conceptual_targets.extend(weighted_concept)
            all_experimental_targets.extend(weighted_exp)
            all_philosophical_targets.extend(weighted_phil)
        
        synthesis["innovation_requirements"] = {
            "mathematical_targets": all_mathematical_targets,
            "conceptual_targets": all_conceptual_targets,
            "experimental_targets": all_experimental_targets,
            "philosophical_targets": all_philosophical_targets,
            "success_criteria": self._calculate_weighted_criteria(level_targets, config)
        }
        
        return synthesis
    
    async def _hierarchical_synthesis(self,
                                     level_targets: Dict[InnovationLevel, Dict],
                                     config: MultiLevelInnovationConfig,
                                     contradiction: Dict) -> Dict:
        """
        分层综合策略：按层次递进式创新
        """
        
        # 按创新层次排序
        sorted_levels = sorted(config.target_levels, key=lambda x: self._get_level_hierarchy_order(x))
        
        synthesis = {
            "strategy_name": "分层递进创新",
            "description": f"按照 {' → '.join([l.value for l in sorted_levels])} 的顺序递进式创新",
            "innovation_sequence": [],
            "synthesis_principles": [
                "基础层次为高级层次提供支撑",
                "每个层次在前一层次基础上扩展",
                "确保创新的连贯性和逻辑性"
            ]
        }
        
        # 构建递进式创新序列
        for i, level in enumerate(sorted_levels):
            targets = level_targets[level]
            weight = config.level_weights[level]
            
            sequence_item = {
                "level": level.value,
                "order": i + 1,
                "weight": weight,
                "requirements": {
                    "mathematical_focus": targets["generation_guidance"]["mathematical_focus"],
                    "conceptual_focus": targets["generation_guidance"]["conceptual_focus"],
                    "experimental_focus": targets["generation_guidance"]["experimental_focus"],
                    "philosophical_focus": targets["generation_guidance"]["philosophical_focus"]
                },
                "build_upon": sorted_levels[:i] if i > 0 else []
            }
            
            synthesis["innovation_sequence"].append(sequence_item)
        
        synthesis["success_criteria"] = self._calculate_weighted_criteria(level_targets, config)
        
        return synthesis
    
    async def _fusion_synthesis(self,
                               level_targets: Dict[InnovationLevel, Dict],
                               config: MultiLevelInnovationConfig,
                               contradiction: Dict) -> Dict:
        """
        融合综合策略：创新层次深度融合
        """
        
        synthesis = {
            "strategy_name": "深度融合创新",
            "description": "将多个创新层次深度融合，形成统一的创新体系",
            "fusion_dimensions": {},
            "synthesis_principles": [
                "创新层次相互渗透，形成有机整体",
                "突破传统层次边界，产生涌现效应",
                "追求创新的最大化协同效应"
            ]
        }
        
        # 按维度融合创新目标
        dimensions = ["mathematical", "conceptual", "experimental", "philosophical"]
        
        for dim in dimensions:
            fusion_targets = []
            fusion_guidance = []
            
            for level, weight in config.level_weights.items():
                targets = level_targets[level]
                target_key = f"{dim}_targets"
                guidance_key = f"{dim}_focus"
                
                if target_key in targets:
                    weighted_targets = [
                        f"[{level.value}|权重{weight:.1f}] {target}" 
                        for target in targets[target_key]
                    ]
                    fusion_targets.extend(weighted_targets)
                
                if guidance_key in targets["generation_guidance"]:
                    fusion_guidance.append(
                        f"[{level.value}] {targets['generation_guidance'][guidance_key]}"
                    )
            
            synthesis["fusion_dimensions"][dim] = {
                "targets": fusion_targets,
                "guidance": fusion_guidance
            }
        
        synthesis["success_criteria"] = self._calculate_fusion_criteria(level_targets, config)
        
        return synthesis
    
    def _build_multi_level_prompt(self,
                                 contradiction: Dict,
                                 synthesis_strategy: Dict,
                                 config: MultiLevelInnovationConfig,
                                 feedback_context: Optional[Dict] = None) -> str:
        """构建多层次综合创新提示"""
        
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
        
        # 构建多层次创新指导
        strategy_name = synthesis_strategy["strategy_name"]
        strategy_desc = synthesis_strategy["description"]
        
        target_levels_text = "、".join([level.value for level in config.target_levels])
        weights_text = "、".join([f"{k.value}({v:.1f})" for k, v in config.level_weights.items()])
        
        # 构建综合创新要求
        if config.synthesis_mode == "parallel":
            innovation_requirements = self._build_parallel_requirements(synthesis_strategy)
        elif config.synthesis_mode == "hierarchical":
            innovation_requirements = self._build_hierarchical_requirements(synthesis_strategy)
        else:  # fusion
            innovation_requirements = self._build_fusion_requirements(synthesis_strategy)
        
        # 构建反馈指导
        feedback_guidance = ""
        if feedback_context:
            feedback_guidance = self._build_multi_level_feedback_guidance(feedback_context, config)
        
        prompt = f"""
# ADVANCED MULTI-LEVEL QUANTUM THEORY INNOVATION

## MISSION
创建一个 **{strategy_name}** 量子理论，解决 **{theory1}** 和 **{theory2}** 之间的矛盾。
必须同时在多个层次创新：**{target_levels_text}** (权重配置：**{weights_text}**)

## 矛盾分析
{contradictions_text}

## 多层次创新策略

### 策略：**{strategy_name}**
{strategy_desc}

### 创新综合原则：
{chr(10).join([f"- {principle}" for principle in synthesis_strategy["synthesis_principles"]])}

{innovation_requirements}

## 关键成功标准

你的理论将在这些集成维度上被评估：
{self._format_success_criteria(synthesis_strategy["success_criteria"])}

{feedback_guidance}

## 多层次输出要求

输出必须是单个有效的JSON对象，展示所有目标层次的创新：

```json
{{
  "name": "反映多层次创新的理论名称 ({target_levels_text})",
  "metadata": {{
    "multi_level_targets": {json.dumps({level.value: config.level_weights[level] for level in config.target_levels})},
    "synthesis_strategy": "{strategy_name}"
  }},
  "mathematical_relation_to_sqm": "基于最高创新层次选择：Interpretation/Modification/Extension/Revolution",
  "summary": "强调多层次创新综合的一段描述",
  "formalism": {{
    "mathematical_objects": "展示来自多个层次的创新",
    "governing_equations": [
      "展示{config.target_levels[0].value}层次创新的方程",
      "展示多层次集成的方程"
    ]
  }}
}}
```

关键：你的理论必须在每个指定层次 ({target_levels_text}) 展现清晰的创新，同时保持整体连贯性和科学严谨性。
多层次综合应该创造超越单层次创新的涌现特性。
"""
        
        return prompt
    
    def _get_level_hierarchy_order(self, level: InnovationLevel) -> int:
        """获取层次的等级顺序"""
        order_map = {
            InnovationLevel.INTERPRETATION: 1,
            InnovationLevel.PARAMETER_EXTENSION: 2,
            InnovationLevel.EQUATION_MODIFICATION: 3,
            InnovationLevel.FRAMEWORK_EXTENSION: 4,
            InnovationLevel.PARADIGM_REVOLUTION: 5
        }
        return order_map[level]
    
    def _calculate_weighted_criteria(self, 
                                   level_targets: Dict[InnovationLevel, Dict],
                                   config: MultiLevelInnovationConfig) -> Dict[str, float]:
        """计算加权成功标准"""
        
        criteria_dimensions = [
            "mathematical_novelty", "conceptual_breakthrough", 
            "experimental_distinguishability", "philosophical_depth", 
            "paradigm_shift_potential"
        ]
        
        weighted_criteria = {}
        
        for dimension in criteria_dimensions:
            weighted_score = 0.0
            for level, weight in config.level_weights.items():
                level_criteria = level_targets[level]["success_criteria"]
                weighted_score += level_criteria[dimension] * weight
            
            weighted_criteria[dimension] = min(weighted_score, 1.0)
        
        return weighted_criteria
    
    def _calculate_fusion_criteria(self,
                                  level_targets: Dict[InnovationLevel, Dict],
                                  config: MultiLevelInnovationConfig) -> Dict[str, float]:
        """计算融合模式的成功标准"""
        
        base_criteria = self._calculate_weighted_criteria(level_targets, config)
        
        # 融合模式增加协同效应奖励
        fusion_bonus = 0.1 * config.innovation_intensity
        
        for key in base_criteria:
            base_criteria[key] = min(base_criteria[key] + fusion_bonus, 1.0)
        
        # 添加融合特有标准
        base_criteria["multi_level_coherence"] = 0.8
        base_criteria["emergent_properties"] = 0.7
        
        return base_criteria
    
    def _assess_multi_level_innovation(self, 
                                     theory: Dict,
                                     config: MultiLevelInnovationConfig) -> Dict:
        """评估多层次创新效果"""
        
        assessment = {
            "overall_multi_level_score": 0.0,
            "level_specific_scores": {},
            "synthesis_effectiveness": 0.8,
            "emergent_properties_detected": len(config.target_levels) >= 3,
            "coherence_score": 0.85
        }
        
        # 评估各层次的创新实现
        total_weighted_score = 0.0
        
        for level in config.target_levels:
            level_score, level_details = self.base_framework.assess_theory_innovation_level(theory)
            
            # 检查是否实现了该层次的创新
            level_achieved = (level_score == level)
            level_rating = 1.0 if level_achieved else level_details.get("mathematical_novelty", 0.5)
            
            assessment["level_specific_scores"][level.value] = {
                "achieved": level_achieved,
                "score": level_rating,
                "details": level_details
            }
            
            total_weighted_score += level_rating * config.level_weights[level]
        
        assessment["overall_multi_level_score"] = total_weighted_score
        
        return assessment
    
    def _post_process_multi_level_theory(self,
                                        theory: Dict,
                                        config: MultiLevelInnovationConfig,
                                        assessment: Dict) -> Dict:
        """后处理多层次理论"""
        
        # 添加多层次创新标记
        if "metadata" not in theory:
            theory["metadata"] = {}
        
        theory["metadata"]["multi_level_innovation"] = {
            "target_levels": [level.value for level in config.target_levels],
            "level_weights": {level.value: weight for level, weight in config.level_weights.items()},
            "synthesis_mode": config.synthesis_mode,
            "innovation_intensity": config.innovation_intensity,
            "assessment": assessment,
            "generation_timestamp": time.time()
        }
        
        # 添加创新成功标记
        successful_levels = [
            level for level, data in assessment["level_specific_scores"].items()
            if data["achieved"]
        ]
        
        theory["metadata"]["successful_innovation_levels"] = successful_levels
        theory["metadata"]["multi_level_success_rate"] = len(successful_levels) / len(config.target_levels)
        
        return theory
    
    def _record_multi_level_generation(self,
                                      theory: Dict,
                                      config: MultiLevelInnovationConfig,
                                      assessment: Dict) -> None:
        """记录多层次生成历史"""
        
        record = {
            "timestamp": time.time(),
            "theory_name": theory.get("name", "Unknown"),
            "target_levels": [level.value for level in config.target_levels],
            "synthesis_mode": config.synthesis_mode,
            "overall_score": assessment["overall_multi_level_score"],
            "level_achievements": assessment["level_specific_scores"],
            "synthesis_effectiveness": assessment["synthesis_effectiveness"],
            "emergent_properties": assessment["emergent_properties_detected"]
        }
        
        self.generation_history.append(record)
    
    # 辅助方法占位符 - 需要具体实现
    def _build_parallel_requirements(self, synthesis_strategy: Dict) -> str:
        return "## 并行创新要求\n" + "\n".join([
            f"- {req}" for req in synthesis_strategy["innovation_requirements"]["mathematical_targets"][:3]
        ])
    
    def _build_hierarchical_requirements(self, synthesis_strategy: Dict) -> str:
        return "## 分层创新序列\n" + "\n".join([
            f"步骤 {item['order']}: {item['level']} (权重: {item['weight']:.1f})"
            for item in synthesis_strategy["innovation_sequence"]
        ])
    
    def _build_fusion_requirements(self, synthesis_strategy: Dict) -> str:
        return "## 融合创新维度\n" + "\n".join([
            f"**{dim.title()}**: " + "; ".join(data["targets"][:2])
            for dim, data in synthesis_strategy["fusion_dimensions"].items()
        ])
    
    def _format_success_criteria(self, criteria: Dict[str, float]) -> str:
        return "\n".join([
            f"- **{key.replace('_', ' ').title()}**: {value:.1f}/1.0"
            for key, value in criteria.items()
        ])
    
    def _build_multi_level_feedback_guidance(self, feedback_context: Dict, config: MultiLevelInnovationConfig) -> str:
        return "\n## MULTI-LEVEL FEEDBACK INTEGRATION\nConsider previous feedback while maintaining multi-level innovation goals."
    
    def _calculate_adaptive_temperature(self, config: MultiLevelInnovationConfig) -> float:
        base_temp = 0.7
        complexity_factor = len(config.target_levels) * 0.1
        intensity_factor = config.innovation_intensity * 0.2
        return min(base_temp + complexity_factor + intensity_factor, 1.0) 