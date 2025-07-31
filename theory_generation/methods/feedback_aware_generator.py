#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feedback_aware_generator.py - 基于反馈的理论生成器
===============================================

将评估反馈直接集成到理论生成过程中
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path

from .unified_generator_adapter import UnifiedSpaceBasedGenerator
from utils.logging_config import get_logger


class FeedbackAwareGenerator(UnifiedSpaceBasedGenerator):
    """基于评估反馈的理论生成器"""
    
    def __init__(self, feedback_data: Optional[Dict] = None, **kwargs):
        """
        初始化反馈感知生成器
        
        Args:
            feedback_data: 包含评估反馈的字典
            **kwargs: 传递给父类的参数
        """
        super().__init__(**kwargs)
        self.feedback_data = feedback_data or {}
        self.logger = get_logger('feedback_aware_generator')
        
    def _analyze_feedback_patterns(self) -> Dict[str, List[str]]:
        """分析反馈中的模式"""
        patterns = {
            'mathematical_gaps': [],
            'experimental_needs': [],
            'conceptual_issues': [],
            'successful_aspects': []
        }
        
        # 数学相关的关键词
        math_keywords = ['数学', '公式', '方程', 'mathematical', 'equation', 'formalism']
        # 实验相关的关键词
        exp_keywords = ['实验', '预测', '验证', 'experiment', 'prediction', 'testable']
        # 概念相关的关键词
        concept_keywords = ['定义', '概念', '清晰', 'definition', 'concept', 'clarity']
        
        # 分析每个理论的反馈
        for theory_name, feedback in self.feedback_data.items():
            if 'evaluations' in feedback:
                for role, eval_data in feedback['evaluations'].items():
                    # 分析弱点
                    if 'weaknesses' in eval_data:
                        for weakness in eval_data['weaknesses']:
                            weakness_lower = weakness.lower()
                            if any(kw in weakness_lower for kw in math_keywords):
                                patterns['mathematical_gaps'].append(weakness)
                            elif any(kw in weakness_lower for kw in exp_keywords):
                                patterns['experimental_needs'].append(weakness)
                            elif any(kw in weakness_lower for kw in concept_keywords):
                                patterns['conceptual_issues'].append(weakness)
                    
                    # 收集成功的方面
                    if 'strengths' in eval_data:
                        patterns['successful_aspects'].extend(eval_data['strengths'])
        
        return patterns
    
    def _adjust_concept_weights_based_on_feedback(self, patterns: Dict[str, List[str]]):
        """根据反馈调整概念权重"""
        # 如果有数学缺陷，增加数学相关概念的权重
        if patterns['mathematical_gaps']:
            self._log_info("检测到数学形式化不足，增强数学概念权重")
            # 增加数学相关概念的重要性
            for concept_name in self.concept_space:
                if any(term in concept_name.lower() for term in ['equation', 'formula', 'mathematical', 'operator']):
                    if concept_name in self.concept_importance:
                        self.concept_importance[concept_name] *= 1.5
                    else:
                        self.concept_importance[concept_name] = 1.5
        
        # 如果缺乏实验预测，增加实验相关概念
        if patterns['experimental_needs']:
            self._log_info("检测到实验预测不足，增强实验相关概念")
            # 可以从文献中寻找实验相关的概念
            experimental_concepts = ['measurement', 'observation', 'detector', 'interference', 'correlation']
            for concept_name in self.concept_space:
                if any(term in concept_name.lower() for term in experimental_concepts):
                    if concept_name in self.concept_importance:
                        self.concept_importance[concept_name] *= 1.3
                    else:
                        self.concept_importance[concept_name] = 1.3
    
    def _generate_feedback_informed_prompt(self, gap_info: Dict, patterns: Dict) -> str:
        """生成包含反馈信息的提示"""
        base_prompt = super()._generate_theory_from_gap(gap_info)
        
        # 添加反馈相关的指导
        feedback_guidance = "\n\n基于先前理论的评估反馈，请特别注意以下方面：\n"
        
        if patterns['mathematical_gaps']:
            feedback_guidance += "\n数学形式化要求：\n"
            for gap in patterns['mathematical_gaps'][:3]:
                feedback_guidance += f"- {gap}\n"
            feedback_guidance += "请确保包含明确的数学方程和形式化描述。\n"
        
        if patterns['experimental_needs']:
            feedback_guidance += "\n实验预测要求：\n"
            for need in patterns['experimental_needs'][:3]:
                feedback_guidance += f"- {need}\n"
            feedback_guidance += "请提供具体的、可量化的实验预测。\n"
        
        if patterns['conceptual_issues']:
            feedback_guidance += "\n概念澄清要求：\n"
            for issue in patterns['conceptual_issues'][:3]:
                feedback_guidance += f"- {issue}\n"
            feedback_guidance += "请确保所有核心概念都有清晰的定义。\n"
        
        if patterns['successful_aspects']:
            feedback_guidance += "\n成功的方面（请保持）：\n"
            for aspect in patterns['successful_aspects'][:3]:
                feedback_guidance += f"- {aspect}\n"
        
        return base_prompt + feedback_guidance
    
    async def _generate_from_space(self) -> List[Dict]:
        """重写：基于反馈的理论生成"""
        self._log_info("🎯 基于反馈生成新理论...")
        
        # 分析反馈模式
        if self.feedback_data:
            patterns = self._analyze_feedback_patterns()
            self._log_info(f"识别到 {len(patterns['mathematical_gaps'])} 个数学缺陷，"
                          f"{len(patterns['experimental_needs'])} 个实验需求")
            
            # 调整概念权重
            self._adjust_concept_weights_based_on_feedback(patterns)
        else:
            patterns = {'mathematical_gaps': [], 'experimental_needs': [], 
                       'conceptual_issues': [], 'successful_aspects': []}
        
        theories = []
        
        # 为每个概念空白生成理论
        for i, gap in enumerate(self.conceptual_gaps[:self.num_theories_to_generate]):
            self._log_info(f"为概念空白区域 {i+1} 生成理论...")
            
            try:
                # 生成包含反馈的提示
                prompt = self._generate_feedback_informed_prompt(gap, patterns)
                
                # 调用LLM生成
                response = await self.llm_interface.generate(prompt)
                
                # 解析响应
                theory = self._parse_theory_response(response)
                
                if theory:
                    # 添加反馈相关的元数据
                    theory['generation_metadata'] = {
                        'feedback_informed': True,
                        'addressed_gaps': {
                            'mathematical': len(patterns['mathematical_gaps']) > 0,
                            'experimental': len(patterns['experimental_needs']) > 0,
                            'conceptual': len(patterns['conceptual_issues']) > 0
                        }
                    }
                    theories.append(theory)
                    self._log_info(f"✅ 成功生成反馈感知理论: {theory.get('name', '未命名')}")
                
            except Exception as e:
                self._log_error(f"生成理论失败: {e}")
                continue
        
        return theories
    
    @classmethod
    def from_evaluation_results(cls, evaluation_dir: str, **kwargs):
        """从评估结果目录创建反馈感知生成器"""
        feedback_data = {}
        
        # 读取评估结果
        eval_path = Path(evaluation_dir)
        if eval_path.exists():
            for eval_file in eval_path.glob("*_role_evaluation.json"):
                try:
                    with open(eval_file, 'r', encoding='utf-8') as f:
                        eval_data = json.load(f)
                        theory_name = eval_file.stem.replace('_role_evaluation', '')
                        feedback_data[theory_name] = eval_data
                except Exception as e:
                    print(f"读取评估文件失败 {eval_file}: {e}")
        
        return cls(feedback_data=feedback_data, **kwargs)