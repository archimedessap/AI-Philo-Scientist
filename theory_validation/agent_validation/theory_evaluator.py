#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
量子理论评估器

从物理学家、哲学家和数学家多个角度评估生成的新量子诠释理论。
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional

def safe_get_nested(obj, path, subpath=None, default=''):
    """安全获取嵌套字段，无论是字典还是字符串"""
    value = obj.get(path, default) if isinstance(obj, dict) else default
    
    # 如果没有子路径，或者值不是字典，直接返回值
    if subpath is None or not isinstance(value, dict):
        return value
        
    # 如果值是字典且有子路径，继续获取
    return value.get(subpath, default)

class TheoryEvaluator:
    """多角色量子理论评估器"""
    
    def __init__(self, llm_interface):
        """
        初始化理论评估器
        
        Args:
            llm_interface: LLM接口实例
        """
        self.llm = llm_interface
        self.evaluation_results = []  # 评估结果
        self.experiments = []  # 添加空实验列表
        
        # 安全导入验证器
        try:
            from theory_experiment.experimetal_validation.schema_validator import SchemaValidator
            self.schema_validator = SchemaValidator()
        except ImportError:
            # 如果导入失败，创建一个简单的替代验证器
            class SimpleValidator:
                def validate_theory(self, theory):
                    return True  # 总是返回有效
            self.schema_validator = SimpleValidator()
            print("[WARN] 无法导入SchemaValidator，使用简单验证器替代")
        
        # 定义评估角色及其关注点
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
                    "认识论立场", 
                    "与哲学传统的关系"
                ]
            },
            "mathematician": {
                "name": "数学家",
                "focus": [
                    "数学形式化的严谨性", 
                    "数学结构的优雅性",
                    "与现有数学框架的兼容性"
                ]
            }
        }
    
    async def evaluate_theory(self, theory_json, predictor_module=None):
        """评估理论"""
        # 初始化基本结果
        base_result = {
            'theory_name': theory_json.get('name', '未命名理论'),
            'theory_id': theory_json.get('id', 'AUTO_' + str(hash(theory_json.get('name', '')))[0:8]),
            'evaluations': {},  # 确保初始化评估字段
            'avg_chi2': 0,
            'conflicts': [],
            'detailed_results': []
        }
        
        # 进行格式验证但不阻止评估流程
        valid_format = self.schema_validator.validate_theory(theory_json)
        if not valid_format:
            print(f"[INFO] 理论缺少id字段，已自动生成临时ID")
            base_result['status'] = 'info'  # 改为info而非warning
        
        # 如果没有实验数据，直接返回基本结果
        if not self.experiments:
            base_result['status'] = 'warning'
            base_result['message'] = '没有实验数据可供评估'
            return base_result
        
        # 直接进行角色评估
        for role_id, role_info in self.evaluation_roles.items():
            print(f"[INFO] 开始{role_info['name']}评估")
            eval_result = await self._evaluate_as_role(theory_json, role_id, role_info)
            base_result['evaluations'][role_id] = eval_result
            print(f"[INFO] {role_info['name']}评估完成，得分: {eval_result.get('score', '未知')}")
        
        # 生成评估总结
        if len(base_result['evaluations']) > 0:
            summary = await self._generate_evaluation_summary(theory_json, base_result['evaluations'])
            base_result['summary'] = summary
        
        return base_result
    
    async def _evaluate_as_role(self, theory: Dict, role_id: str, role_info: Dict) -> Dict:
        """
        从特定角色视角评估理论
        
        Args:
            theory: 要评估的理论
            role_id: 角色ID
            role_info: 角色信息
            
        Returns:
            Dict: 角色评估结果
        """
        role_name = role_info["name"]
        focus_points = role_info["focus"]
        
        focus_text = "\n".join([f"- {point}" for point in focus_points])
        
        # 构建提示
        prompt = f"""
        你是一位资深的量子物理学{role_name}，需要评估一个新提出的量子诠释理论。
        
        作为{role_name}，你特别关注:
        {focus_text}
        
        要评估的理论:
        理论名称: {theory.get('name', '未命名理论')}
        核心原理: {theory.get('core_principles', '')}
        详细描述: {theory.get('detailed_description', '')}
        量子现象解释:
        - 波函数坍缩: {safe_get_nested(theory, 'quantum_phenomena_explanation', 'wave_function_collapse', '')}
        - 测量问题: {safe_get_nested(theory, 'quantum_phenomena_explanation', 'measurement_problem', '')}
        - 非局域性: {safe_get_nested(theory, 'quantum_phenomena_explanation', 'non_locality', '')}
        哲学立场: {theory.get('philosophical_stance', '')}
        数学表述: {theory.get('mathematical_formulation', '')}
        
        请从{role_name}的视角评估这个理论，考虑上述关注点，以JSON格式返回评估结果:
        {{
          "strengths": [
            "理论优势1",
            "理论优势2"
          ],
          "weaknesses": [
            "理论弱点1",
            "理论弱点2"
          ],
          "questions": [
            "有待解决的问题1",
            "有待解决的问题2"
          ],
          "score": 评分(0-10),
          "detailed_comments": "详细评价...",
          "improvement_suggestions": "改进建议..."
        }}
        
        请基于理论的科学和哲学价值进行客观评估，给出合理的评分和具体的分析。
        """
        
        # 调用LLM进行评估
        try:
            response = await self.llm.query_async(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3  # 使用较低的温度以获得确定性结果
            )
            
            # 解析结果
            evaluation = self.llm.extract_json(response)
            if evaluation:
                evaluation["role"] = role_name
                return evaluation
            
            return {
                "role": role_name,
                "error": "评估结果解析失败",
                "score": 0
            }
        except Exception as e:
            print(f"[ERROR] {role_name}评估失败: {str(e)}")
            return {
                "role": role_name,
                "error": str(e),
                "score": 0
            }
    
    async def _generate_evaluation_summary(self, theory: Dict, evaluations: Dict) -> Dict:
        """
        生成评估总结
        
        Args:
            theory: 被评估的理论
            evaluations: 所有角色的评估结果
            
        Returns:
            Dict: 评估总结
        """
        # 提取各角色评估的要点
        evaluation_summary = ""
        overall_score = evaluations.get("overall_score", 0)
        
        for role_id, eval_data in evaluations.get("evaluations", {}).items():
            role_name = eval_data.get("role", role_id)
            score = eval_data.get("score", 0)
            strengths = "\n".join([f"- {s}" for s in eval_data.get("strengths", [])])
            weaknesses = "\n".join([f"- {w}" for w in eval_data.get("weaknesses", [])])
            
            evaluation_summary += f"""
            {role_name}评分: {score}/10
            优势:
            {strengths}
            
            弱点:
            {weaknesses}
            
            """
        
        # 构建提示
        prompt = f"""
        作为量子物理学理论评审委员会主席，你需要对一个新提出的量子诠释理论做出总体评价。
        
        理论名称: {theory.get('name', '未命名理论')}
        核心原理: {theory.get('core_principles', '')}
        
        各专家评价摘要:
        {evaluation_summary}
        
        总体评分: {overall_score:.2f}/10
        
        请提供总体评价，包括:
        1. 这个理论的潜在价值
        2. 是否推荐进一步发展这个理论
        3. 建议的改进方向
        
        以JSON格式返回:
        {{
          "potential_value": "对这个理论潜在价值的评价...",
          "recommendation": "推荐意见...",
          "improvement_directions": "改进方向..."
        }}
        
        请根据专家评价和总体评分给出平衡、客观的建议。
        """
        
        # 调用LLM生成总结
        try:
            response = await self.llm.query_async(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            # 解析结果
            summary = self.llm.extract_json(response)
            return summary if summary else {
                "potential_value": "无法生成评价",
                "recommendation": "无法提供推荐意见",
                "improvement_directions": []
            }
        except Exception as e:
            print(f"[ERROR] 生成评估总结失败: {str(e)}")
            return {
                "potential_value": f"评估过程出错: {str(e)}",
                "recommendation": "无法提供推荐意见",
                "improvement_directions": []
            }
    
    async def evaluate_theories(self, theories: List[Dict]) -> List[Dict]:
        """
        评估多个理论
        
        Args:
            theories: 理论列表
            
        Returns:
            List[Dict]: 评估结果列表
        """
        results = []
        for i, theory in enumerate(theories):
            print(f"[INFO] 评估理论 {i+1}/{len(theories)}")
            result = await self.evaluate_theory(theory)
            results.append(result)
        
        self.evaluation_results = results
        return results
    
    def save_evaluation_results(self, output_path: str) -> None:
        """
        保存评估结果
        
        Args:
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_results, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 评估结果已保存到: {output_path}")
        except Exception as e:
            print(f"[ERROR] 保存评估结果失败: {str(e)}") 