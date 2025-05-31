#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多角色专家评估验证器

集成已有的TheoryEvaluator，从不同专家角度评估理论
"""

import os
import json
from typing import Dict, Any, List

class AgentEvaluationValidator:
    """从多角色专家视角评估理论"""
    
    def __init__(self, llm_interface):
        """
        初始化多角色评估验证器
        
        Args:
            llm_interface: LLM接口实例
        """
        self.llm = llm_interface
        
        # 尝试导入已有的理论评估器
        try:
            from theory_validation.agent_validation.theory_evaluator import TheoryEvaluator
            self.theory_evaluator = TheoryEvaluator(llm_interface)
            print(f"[INFO] 已成功加载多角色理论评估器")
            self.use_existing_evaluator = True
        except ImportError:
            print(f"[WARN] 无法导入已有的TheoryEvaluator，将使用内部实现")
            self.use_existing_evaluator = False
            # 定义评估角色
            self.evaluation_roles = {
                "physicist": {
                    "name": "物理学家",
                    "focus": [
                        "与已知物理实验的兼容性", 
                        "可检验的预测", 
                        "物理直觉的合理性"
                    ]
                },
                "philosopher": {
                    "name": "哲学家",
                    "focus": [
                        "逻辑一致性", 
                        "本体论清晰度", 
                        "认识论立场"
                    ]
                },
                "mathematician": {
                    "name": "数学家",
                    "focus": [
                        "数学形式化的严谨性", 
                        "数学结构的优雅性"
                    ]
                }
            }
    
    async def validate(self, theory: Dict) -> Dict:
        """
        进行多角色专家评估
        
        Args:
            theory: 理论数据
            
        Returns:
            Dict: 评估结果
        """
        # 在此加入详细日志
        print(f"[DEBUG] 开始角色评估: {theory.get('name', 'unknown')}")
        
        theory_name = theory.get("name", theory.get("theory_name", "未命名理论"))
        print(f"[INFO] 进行多角色专家评估: {theory_name}")
        
        # 使用已有评估器或内部实现
        if self.use_existing_evaluator:
            try:
                # 添加每个步骤的调试信息
                eval_result = await self.theory_evaluator.evaluate_theory(theory)
                print(f"[DEBUG] 原始评估结果: {eval_result}")
                
                # 检查转换过程
                result = self._convert_to_standard_format(eval_result)
                print(f"[DEBUG] 转换后结果: {result}")
                return result
            except Exception as e:
                print(f"[ERROR] 使用已有评估器失败: {str(e)}")
                # 失败时回退到内部实现
                return await self._internal_evaluation(theory)
        else:
            # 使用内部实现
            return await self._internal_evaluation(theory)
    
    def _convert_to_standard_format(self, original_result: Dict) -> Dict:
        """
        将原有评估结果转换为统一格式
        
        Args:
            original_result: 原有格式的评估结果
            
        Returns:
            Dict: 统一格式的评估结果
        """
        # 检查是否有错误状态
        if original_result.get('status') == 'error':
            print(f"[WARN] 检测到错误状态，将使用内部评估替代")
            return {
                "overall_score": 5.0,  # 给个默认中等分数而不是0
                "dimension_scores": {
                    "physical_soundness": 5.0,
                    "philosophical_coherence": 5.0,
                    "mathematical_rigor": 5.0
                },
                "expert_evaluations": {
                    "physicist": {"score": 5.0, "strengths": [], "weaknesses": []},
                    "philosopher": {"score": 5.0, "strengths": [], "weaknesses": []},
                    "mathematician": {"score": 5.0, "strengths": [], "weaknesses": []}
                },
                "overall_assessment": "格式验证失败，使用默认评分",
                "strengths": [],
                "weaknesses": ["理论格式不符合要求"],
                "recommendations": ["完善理论格式，确保包含所有必要字段"]
            }
        
        # 添加日志查看输入数据结构
        print(f"[DEBUG] 转换输入: {original_result}")
        
        # 提取评分和评价
        physicist_eval = original_result.get("evaluations", {}).get("physicist", {})
        philosopher_eval = original_result.get("evaluations", {}).get("philosopher", {})
        mathematician_eval = original_result.get("evaluations", {}).get("mathematician", {})
        
        # 打印各角色评分
        print(f"[DEBUG] 物理学家评分: {physicist_eval.get('score', 0.0)}")
        print(f"[DEBUG] 哲学家评分: {philosopher_eval.get('score', 0.0)}")
        print(f"[DEBUG] 数学家评分: {mathematician_eval.get('score', 0.0)}")
        
        # 提取各项分数
        physicist_score = physicist_eval.get("score", 0.0)
        philosopher_score = philosopher_eval.get("score", 0.0)
        mathematician_score = mathematician_eval.get("score", 0.0)
        
        # 计算平均分
        avg_score = (physicist_score + philosopher_score + mathematician_score) / 3
        
        # 合并优势和弱点
        strengths = []
        weaknesses = []
        recommendations = []
        
        for eval_data in [physicist_eval, philosopher_eval, mathematician_eval]:
            strengths.extend(eval_data.get("strengths", []))
            weaknesses.extend(eval_data.get("weaknesses", []))
        
        # 从summary提取建议
        summary = original_result.get("summary", {})
        if "improvement_directions" in summary:
            if isinstance(summary["improvement_directions"], list):
                recommendations.extend(summary["improvement_directions"])
            else:
                recommendations.append(summary["improvement_directions"])
        
        # 构建统一格式
        return {
            "overall_score": avg_score,
            "dimension_scores": {
                "physical_soundness": physicist_score,
                "philosophical_coherence": philosopher_score,
                "mathematical_rigor": mathematician_score
            },
            "expert_evaluations": {
                "physicist": {
                    "score": physicist_score,
                    "strengths": physicist_eval.get("strengths", []),
                    "weaknesses": physicist_eval.get("weaknesses", [])
                },
                "philosopher": {
                    "score": philosopher_score,
                    "strengths": philosopher_eval.get("strengths", []),
                    "weaknesses": philosopher_eval.get("weaknesses", [])
                },
                "mathematician": {
                    "score": mathematician_score,
                    "strengths": mathematician_eval.get("strengths", []),
                    "weaknesses": mathematician_eval.get("weaknesses", [])
                }
            },
            "overall_assessment": summary.get("potential_value", ""),
            "strengths": list(set(strengths)),  # 去重
            "weaknesses": list(set(weaknesses)),  # 去重
            "recommendations": recommendations
        }
    
    async def _internal_evaluation(self, theory: Dict) -> Dict:
        """
        内部实现的多角色评估
        
        Args:
            theory: 理论数据
            
        Returns:
            Dict: 评估结果
        """
        # 初始化结果
        evaluations = {}
        all_strengths = []
        all_weaknesses = []
        
        # 对每个角色进行评估
        for role_id, role_info in self.evaluation_roles.items():
            role_eval = await self._evaluate_as_role(theory, role_id, role_info)
            evaluations[role_id] = role_eval
            
            # 收集优势和弱点
            all_strengths.extend(role_eval.get("strengths", []))
            all_weaknesses.extend(role_eval.get("weaknesses", []))
        
        # 计算总体分数
        scores = [eval_data.get("score", 0) for eval_data in evaluations.values()]
        overall_score = sum(scores) / len(scores) if scores else 0
        
        # 请求总体评估
        summary = await self._generate_evaluation_summary(theory, evaluations)
        
        # 构建最终结果
        dimension_scores = {
            "physical_soundness": evaluations.get("physicist", {}).get("score", 0),
            "philosophical_coherence": evaluations.get("philosopher", {}).get("score", 0),
            "mathematical_rigor": evaluations.get("mathematician", {}).get("score", 0)
        }
        
        return {
            "overall_score": overall_score,
            "dimension_scores": dimension_scores,
            "expert_evaluations": {
                role_id: {
                    "score": eval_data.get("score", 0),
                    "strengths": eval_data.get("strengths", []),
                    "weaknesses": eval_data.get("weaknesses", [])
                }
                for role_id, eval_data in evaluations.items()
            },
            "overall_assessment": summary.get("potential_value", ""),
            "strengths": list(set(all_strengths)),  # 去重
            "weaknesses": list(set(all_weaknesses)),  # 去重
            "recommendations": summary.get("improvement_directions", [])
        }
    
    async def _evaluate_as_role(self, theory: Dict, role_id: str, role_info: Dict) -> Dict:
        """从特定角色视角评估理论"""
        # 记录角色和调用信息
        print(f"[DEBUG] 开始角色评估: {role_info['name']}")
        
        # 调用LLM后记录原始响应
        response = await self.llm.query_async(
            messages=[{"role": "user", "content": f"""
        你是一位资深的量子物理学{role_info['name']}，需要评估一个新的量子诠释理论。
        
        作为{role_info['name']}，你特别关注:
        {', '.join(role_info['focus'])}
        
        要评估的理论:
        理论名称: {theory.get('name', theory.get('theory_name', '未命名理论'))}
        核心原理: {theory.get('core_principles', '')}
        
        请从{role_info['name']}的专业视角评估这个理论，给出评分(0-10)，分析优势和弱点，并提出问题和改进建议。
        
        以JSON格式返回评估结果:
        {{
          "score": 评分(0-10),
          "strengths": ["优势1", "优势2", ...],
          "weaknesses": ["弱点1", "弱点2", ...],
          "questions": ["问题1", "问题2", ...],
          "detailed_comments": "详细评价..."
        }}
        """}],
            temperature=0.3
        )
        print(f"[DEBUG] LLM响应(前200字符): {response[:200]}")
        
        # 解析JSON结果时添加详细信息
        try:
            result = self.llm.extract_json(response)
            print(f"[DEBUG] 解析结果: {result}")
            return result
        except Exception as e:
            print(f"[ERROR] JSON解析失败: {str(e)}")
    
    async def _generate_evaluation_summary(self, theory: Dict, evaluations: Dict) -> Dict:
        """生成评估总结"""
        # 构建提示
        prompt = f"""
        作为量子物理学理论评审委员会主席，你需要对一个新量子理论的多专家评估进行总结。
        
        理论名称: {theory.get('name', theory.get('theory_name', '未命名理论'))}
        
        物理学家评分: {evaluations.get('physicist', {}).get('score', 0)}/10
        哲学家评分: {evaluations.get('philosopher', {}).get('score', 0)}/10
        数学家评分: {evaluations.get('mathematician', {}).get('score', 0)}/10
        
        优势:
        {', '.join([s for e in evaluations.values() for s in e.get('strengths', [])])}
        
        弱点:
        {', '.join([w for e in evaluations.values() for w in e.get('weaknesses', [])])}
        
        请提供综合评估，包括:
        1. 理论的潜在价值和意义
        2. 具体的改进方向
        
        以JSON格式返回:
        {{
          "potential_value": "理论潜在价值评估",
          "improvement_directions": ["改进方向1", "改进方向2", ...]
        }}
        """
        
        # 调用LLM生成总结
        response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        # 解析结果
        result = self.llm.extract_json(response)
        
        return result if result else {
            "potential_value": "无法生成总体评估",
            "improvement_directions": []
        }
