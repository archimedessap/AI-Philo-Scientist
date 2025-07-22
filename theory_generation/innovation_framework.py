#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创新框架 - 分层次的理论创新目标设定与评估

这个模块实现了一个多层次的创新评估和目标设定框架，
用于指导和评估量子理论的创新程度。
"""

import json
import time
from typing import Dict, List, Any, Tuple
from enum import Enum

class InnovationLevel(Enum):
    """创新层次枚举"""
    INTERPRETATION = "interpretation"        # 纯诠释性（无数学改动）
    PARAMETER_EXTENSION = "parameter_ext"    # 参数扩展（增加新参数）
    EQUATION_MODIFICATION = "equation_mod"   # 方程修改（改变现有方程）
    FRAMEWORK_EXTENSION = "framework_ext"    # 框架扩展（新数学结构）
    PARADIGM_REVOLUTION = "paradigm_rev"     # 范式革命（全新物理图像）

class InnovationFramework:
    """理论创新框架"""
    
    def __init__(self):
        self.innovation_criteria = {
            InnovationLevel.INTERPRETATION: {
                "mathematical_novelty": 0.0,
                "conceptual_breakthrough": 0.3,
                "experimental_distinguishability": 0.2,
                "philosophical_depth": 0.8,
                "paradigm_shift_potential": 0.1
            },
            InnovationLevel.PARAMETER_EXTENSION: {
                "mathematical_novelty": 0.4,
                "conceptual_breakthrough": 0.5,
                "experimental_distinguishability": 0.6,
                "philosophical_depth": 0.5,
                "paradigm_shift_potential": 0.3
            },
            InnovationLevel.EQUATION_MODIFICATION: {
                "mathematical_novelty": 0.7,
                "conceptual_breakthrough": 0.6,
                "experimental_distinguishability": 0.8,
                "philosophical_depth": 0.6,
                "paradigm_shift_potential": 0.5
            },
            InnovationLevel.FRAMEWORK_EXTENSION: {
                "mathematical_novelty": 0.8,
                "conceptual_breakthrough": 0.8,
                "experimental_distinguishability": 0.9,
                "philosophical_depth": 0.7,
                "paradigm_shift_potential": 0.7
            },
            InnovationLevel.PARADIGM_REVOLUTION: {
                "mathematical_novelty": 0.9,
                "conceptual_breakthrough": 0.9,
                "experimental_distinguishability": 0.95,
                "philosophical_depth": 0.9,
                "paradigm_shift_potential": 0.9
            }
        }
    
    def assess_theory_innovation_level(self, theory: Dict[str, Any]) -> Tuple[InnovationLevel, Dict[str, float]]:
        """
        评估理论的创新层次
        
        Args:
            theory: 理论数据
            
        Returns:
            Tuple[InnovationLevel, Dict]: (创新层次, 详细评分)
        """
        scores = self._calculate_innovation_scores(theory)
        
        # 使用更精确的层次判断逻辑，基于特征而非简单平均分
        level = self._determine_innovation_level_by_features(theory, scores)
        
        return level, scores
    
    def _determine_innovation_level_by_features(self, theory: Dict[str, Any], scores: Dict[str, float]) -> InnovationLevel:
        """基于特征而非简单平均分确定创新层次"""
        
        # 检查数学关系类型 - 这是最直接的指标
        math_relation = theory.get("mathematical_relation_to_sqm", "")
        if not isinstance(math_relation, str):
            math_relation = str(math_relation)
        math_relation = math_relation.lower()
        
        # 检查是否有新参数
        formalism = theory.get("formalism", {})
        math_objects = formalism.get("mathematical_objects", "")
        if not isinstance(math_objects, str):
            math_objects = str(math_objects)
        math_objects = math_objects.lower()
        
        # 安全地提取方程文本 - 处理两种数据结构
        equations_text = ""
        
        # 1. 检查新格式：governing_equations (数组)
        governing_equations = formalism.get("governing_equations", [])
        if isinstance(governing_equations, list):
            string_equations = []
            for eq in governing_equations:
                if isinstance(eq, str):
                    string_equations.append(eq)
                elif isinstance(eq, dict):
                    string_equations.extend([str(v) for v in eq.values() if isinstance(v, str)])
            if string_equations:
                equations_text += " ".join(string_equations) + " "
        
        # 2. 检查旧格式：equations (字典)
        equations = formalism.get("equations", {})
        if isinstance(equations, dict):
            equation_values = [str(v) for v in equations.values() if isinstance(v, str)]
            if equation_values:
                equations_text += " ".join(equation_values) + " "
        elif isinstance(equations, str):
            equations_text += equations + " "
            
        equations = equations_text.strip().lower()
        
        # 特征检测
        has_new_parameters = any(keyword in math_objects or keyword in equations for keyword in [
            "parameter", "coefficient", "constant", "β", "κ", "α", "γ", "λ", "new"
        ])
        
        has_equation_modification = any(keyword in equations for keyword in [
            "modified", "non-linear", "stochastic", "additional term", "correction"
        ])
        
        has_framework_extension = any(keyword in math_objects for keyword in [
            "extended hilbert", "new space", "additional dimension", "tensor network"
        ])
        
        # 检查数学新颖性分数
        math_score = scores.get("mathematical_novelty", 0)
        
        # 层次判断逻辑
        if "extension" in math_relation and has_framework_extension and math_score > 0.6:
            return InnovationLevel.FRAMEWORK_EXTENSION
        elif "modification" in math_relation and has_equation_modification and math_score > 0.5:
            return InnovationLevel.EQUATION_MODIFICATION
        elif ("extension" in math_relation or "modification" in math_relation) and has_new_parameters and math_score > 0.4:
            return InnovationLevel.PARAMETER_EXTENSION
        elif math_score > 0.3 or scores.get("experimental_distinguishability", 0) > 0.4:
            return InnovationLevel.PARAMETER_EXTENSION  # 有实验区分性也算参数扩展
        else:
            return InnovationLevel.INTERPRETATION
    
    def _calculate_innovation_scores(self, theory: Dict[str, Any]) -> Dict[str, float]:
        """计算各维度的创新评分"""
        scores = {}
        
        # 1. 数学新颖性
        scores["mathematical_novelty"] = self._assess_mathematical_novelty(theory)
        
        # 2. 概念突破性
        scores["conceptual_breakthrough"] = self._assess_conceptual_breakthrough(theory)
        
        # 3. 实验可区分性
        scores["experimental_distinguishability"] = self._assess_experimental_distinguishability(theory)
        
        # 4. 哲学深度
        scores["philosophical_depth"] = self._assess_philosophical_depth(theory)
        
        # 5. 范式转移潜力
        scores["paradigm_shift_potential"] = self._assess_paradigm_shift_potential(theory)
        
        return scores
    
    def _assess_mathematical_novelty(self, theory: Dict[str, Any]) -> float:
        """评估数学新颖性"""
        score = 0.0
        
        # 检查数学关系类型
        math_relation = theory.get("mathematical_relation_to_sqm", "")
        if not isinstance(math_relation, str):
            math_relation = str(math_relation)
        math_relation = math_relation.lower()
        if "modification" in math_relation:
            score += 0.4
        elif "extension" in math_relation:
            score += 0.6
        
        # 检查新数学对象
        formalism = theory.get("formalism", {})
        math_objects = formalism.get("mathematical_objects", "")
        if not isinstance(math_objects, str):
            math_objects = str(math_objects)
        math_objects = math_objects.lower()
        
        novel_math_indicators = [
            "new space", "novel algebra", "additional dimension",
            "extended hilbert", "modified metric", "new operator",
            "tensor network", "category theory", "non-commutative"
        ]
        
        for indicator in novel_math_indicators:
            if indicator in math_objects:
                score += 0.1
        
        # 检查方程创新 - 同时处理两种数据结构
        equations_text = ""
        
        # 1. 检查新格式：governing_equations (数组)
        governing_equations = formalism.get("governing_equations", [])
        if isinstance(governing_equations, list):
            # 处理字符串列表
            string_equations = []
            for eq in governing_equations:
                if isinstance(eq, str):
                    string_equations.append(eq)
                elif isinstance(eq, dict):
                    # 如果是字典，提取所有文本值
                    string_equations.extend([str(v) for v in eq.values() if isinstance(v, str)])
            if string_equations:
                # 确保所有元素都是字符串再调用lower()
                safe_equations = [str(eq) for eq in string_equations]
                equations_text += " ".join(safe_equations).lower() + " "
        
        # 2. 检查旧格式：equations (字典)
        equations = formalism.get("equations", {})
        if isinstance(equations, dict):
            # 处理字典格式（如consistent_histories中的结构）
            equation_values = [str(v) for v in equations.values()]
            if equation_values:
                equations_text += " ".join(equation_values).lower() + " "
        elif isinstance(equations, str):
            # 处理单个字符串
            equations_text += equations.lower() + " "
        elif isinstance(equations, list):
            # 处理列表格式
            equation_values = [str(eq) for eq in equations]
            if equation_values:
                equations_text += " ".join(equation_values).lower() + " "
            
        equations_text = equations_text.strip()
        if equations_text:
            equation_innovations = [
                "non-linear", "stochastic", "fractional", "discrete",
                "emergent", "holographic", "topological"
            ]
            
            for innovation in equation_innovations:
                if innovation in equations_text:
                    score += 0.15
        
        return min(score, 1.0)
    
    def _assess_conceptual_breakthrough(self, theory: Dict[str, Any]) -> float:
        """评估概念突破性"""
        score = 0.0
        
        # 检查核心原则的创新性
        core_principles = theory.get("core_principles", {})
        ontology = core_principles.get("ontological_commitments", "")
        if not isinstance(ontology, str):
            ontology = str(ontology)
        ontology = ontology.lower()
        
        postulates = core_principles.get("key_postulates", [])
        # 确保postulates是字符串列表
        safe_postulates = [str(p) for p in postulates] if postulates else []
        
        breakthrough_concepts = [
            "emergence", "holism", "information", "computation",
            "consciousness", "spacetime", "causal set", "digital physics",
            "quantum gravity", "many minds", "modal realism"
        ]
        
        all_text = f"{ontology} {' '.join(safe_postulates)}".lower()
        
        for concept in breakthrough_concepts:
            if concept in all_text:
                score += 0.15
        
        # 检查是否提出新的基本实体
        if "new" in ontology and ("entity" in ontology or "object" in ontology):
            score += 0.2
        
        return min(score, 1.0)
    
    def _assess_experimental_distinguishability(self, theory: Dict[str, Any]) -> float:
        """评估实验可区分性"""
        score = 0.0
        
        predictions = theory.get("predictions_and_verifiability", {})
        deviations = predictions.get("deviations_from_sqm", [])
        
        if isinstance(deviations, list):
            # 每个独特预测增加分数
            score += len(deviations) * 0.3
            
            # 检查预测的具体性
            for deviation in deviations:
                if isinstance(deviation, dict):
                    exp_setup = deviation.get("experimental_setup", "")
                    if not isinstance(exp_setup, str):
                        exp_setup = str(exp_setup)
                    exp_setup = exp_setup.lower()
                    if any(keyword in exp_setup for keyword in [
                        "specific", "precision", "measurement", "detector",
                        "interferometer", "collider", "telescope"
                    ]):
                        score += 0.2
        
        return min(score, 1.0)
    
    def _assess_philosophical_depth(self, theory: Dict[str, Any]) -> float:
        """评估哲学深度"""
        score = 0.0
        
        core_principles = theory.get("core_principles", {})
        epistemology = core_principles.get("epistemological_stances", "")
        if not isinstance(epistemology, str):
            epistemology = str(epistemology)
        epistemology = epistemology.lower()
        
        philosophical_depth_indicators = [
            "reality", "knowledge", "measurement", "observer",
            "causality", "locality", "determinism", "realism",
            "anti-realism", "instrumentalism", "phenomenology"
        ]
        
        for indicator in philosophical_depth_indicators:
            if indicator in epistemology:
                score += 0.1
        
        # 检查是否处理经典哲学问题
        classical_problems = [
            "mind-body", "free will", "identity", "time",
            "space", "consciousness", "interpretation"
        ]
        
        summary = theory.get("summary", "")
        if not isinstance(summary, str):
            summary = str(summary)
        summary = summary.lower()
        for problem in classical_problems:
            if problem in summary:
                score += 0.15
        
        return min(score, 1.0)
    
    def _assess_paradigm_shift_potential(self, theory: Dict[str, Any]) -> float:
        """评估范式转移潜力"""
        score = 0.0
        
        # 检查是否挑战基本假设
        summary = theory.get("summary", "")
        if not isinstance(summary, str):
            summary = str(summary)
        summary = summary.lower()
        paradigm_shift_indicators = [
            "fundamental", "revolutionary", "paradigm", "breakthrough",
            "transform", "redefine", "challenge", "overthrow"
        ]
        
        for indicator in paradigm_shift_indicators:
            if indicator in summary:
                score += 0.2
        
        # 检查跨学科整合
        interdisciplinary_indicators = [
            "cognitive science", "computer science", "philosophy",
            "neuroscience", "cosmology", "mathematics", "biology"
        ]
        
        all_content = json.dumps(theory).lower()
        for field in interdisciplinary_indicators:
            if field in all_content:
                score += 0.1
        
        return min(score, 1.0)
    
    def generate_innovation_targets(self, desired_level: InnovationLevel) -> Dict[str, Any]:
        """
        根据期望的创新层次生成创新目标
        
        Args:
            desired_level: 期望的创新层次
            
        Returns:
            Dict: 创新目标配置
        """
        criteria = self.innovation_criteria[desired_level]
        
        targets = {
            "target_level": desired_level.value,
            "mathematical_targets": self._generate_mathematical_targets(desired_level),
            "conceptual_targets": self._generate_conceptual_targets(desired_level),
            "experimental_targets": self._generate_experimental_targets(desired_level),
            "philosophical_targets": self._generate_philosophical_targets(desired_level),
            "success_criteria": criteria,
            "generation_guidance": self._generate_guidance_prompts(desired_level)
        }
        
        return targets
    
    def _generate_mathematical_targets(self, level: InnovationLevel) -> List[str]:
        """生成数学创新目标"""
        targets = {
            InnovationLevel.INTERPRETATION: [
                "保持标准量子力学数学形式",
                "专注于数学对象的物理解释"
            ],
            InnovationLevel.PARAMETER_EXTENSION: [
                "引入1-2个新的物理参数",
                "保持原有方程结构，增加新项"
            ],
            InnovationLevel.EQUATION_MODIFICATION: [
                "修改薛定谔方程或其他核心方程",
                "引入非线性或随机项",
                "改变演化动力学"
            ],
            InnovationLevel.FRAMEWORK_EXTENSION: [
                "扩展希尔伯特空间结构",
                "引入新的数学对象（算子、度量等）",
                "建立新的数学关系"
            ],
            InnovationLevel.PARADIGM_REVOLUTION: [
                "提出全新的数学框架",
                "超越传统量子力学结构",
                "建立革命性的数学基础"
            ]
        }
        
        return targets[level]
    
    def _generate_conceptual_targets(self, level: InnovationLevel) -> List[str]:
        """生成概念创新目标"""
        targets = {
            InnovationLevel.INTERPRETATION: [
                "提供新的物理图像和直觉",
                "澄清测量和坍缩概念"
            ],
            InnovationLevel.PARAMETER_EXTENSION: [
                "引入新的物理概念或机制",
                "扩展现有理论的概念框架"
            ],
            InnovationLevel.EQUATION_MODIFICATION: [
                "重新定义基本物理过程",
                "提出新的因果关系模型"
            ],
            InnovationLevel.FRAMEWORK_EXTENSION: [
                "建立新的本体论框架",
                "提出革新的物理原理"
            ],
            InnovationLevel.PARADIGM_REVOLUTION: [
                "完全重新构想物理现实",
                "提出跨学科的统一理论",
                "挑战基础物理假设"
            ]
        }
        
        return targets[level]
    
    def _generate_experimental_targets(self, level: InnovationLevel) -> List[str]:
        """生成实验目标"""
        targets = {
            InnovationLevel.INTERPRETATION: [
                "与现有实验完全一致",
                "提供新的实验解释角度"
            ],
            InnovationLevel.PARAMETER_EXTENSION: [
                "预测可测量的新参数效应",
                "提出参数测定实验"
            ],
            InnovationLevel.EQUATION_MODIFICATION: [
                "预测偏离标准QM的现象",
                "设计关键判决性实验"
            ],
            InnovationLevel.FRAMEWORK_EXTENSION: [
                "预测全新的物理现象",
                "提出前沿技术实验方案"
            ],
            InnovationLevel.PARADIGM_REVOLUTION: [
                "预测范式转移的实验特征",
                "设计概念验证实验",
                "提出技术突破方向"
            ]
        }
        
        return targets[level]
    
    def _generate_philosophical_targets(self, level: InnovationLevel) -> List[str]:
        """生成哲学目标"""
        targets = {
            InnovationLevel.INTERPRETATION: [
                "澄清本体论和认识论立场",
                "解决经典哲学问题"
            ],
            InnovationLevel.PARAMETER_EXTENSION: [
                "扩展物理实在的理解",
                "深化因果关系概念"
            ],
            InnovationLevel.EQUATION_MODIFICATION: [
                "重新审视决定论与随机性",
                "探索新的时空概念"
            ],
            InnovationLevel.FRAMEWORK_EXTENSION: [
                "建立新的形而上学框架",
                "整合科学与哲学视角"
            ],
            InnovationLevel.PARADIGM_REVOLUTION: [
                "彻底重新构想现实本质",
                "提出后量子哲学框架",
                "整合意识、信息、物质"
            ]
        }
        
        return targets[level]
    
    def _generate_guidance_prompts(self, level: InnovationLevel) -> Dict[str, str]:
        """生成创新指导提示"""
        guidance = {
            InnovationLevel.INTERPRETATION: {
                "mathematical_focus": "保持数学严谨性，专注于解释",
                "conceptual_focus": "提供清晰的物理图像和直觉理解",
                "experimental_focus": "解释现有实验，提供新视角",
                "philosophical_focus": "深入探讨认识论和本体论问题"
            },
            InnovationLevel.PARAMETER_EXTENSION: {
                "mathematical_focus": "审慎引入新参数，保持理论自洽",
                "conceptual_focus": "为新参数提供物理动机和解释",
                "experimental_focus": "设计参数测量和验证实验",
                "philosophical_focus": "探讨新参数的本体论地位"
            },
            InnovationLevel.EQUATION_MODIFICATION: {
                "mathematical_focus": "确保数学修改的物理合理性",
                "conceptual_focus": "为方程修改提供深刻的物理洞察",
                "experimental_focus": "预测并设计偏离验证实验",
                "philosophical_focus": "重新审视因果性和决定论"
            },
            InnovationLevel.FRAMEWORK_EXTENSION: {
                "mathematical_focus": "建立自洽的扩展数学框架",
                "conceptual_focus": "提出革新的物理原理和概念",
                "experimental_focus": "预测新现象并设计验证方案",
                "philosophical_focus": "建立新的本体论和认识论基础"
            },
            InnovationLevel.PARADIGM_REVOLUTION: {
                "mathematical_focus": "创造全新的数学语言和结构",
                "conceptual_focus": "重新定义物理现实的基本概念",
                "experimental_focus": "设计范式验证的关键实验",
                "philosophical_focus": "整合跨学科视角，挑战基础假设"
            }
        }
        
        return guidance[level] 