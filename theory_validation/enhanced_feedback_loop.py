#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强反馈闭环系统

精确分析评估结果，生成针对性改进指导，并验证改进效果。
实现从评估到改进的完整闭环控制。
"""

import json
import time
import random
import statistics
from typing import Dict, List, Any, Optional, Tuple
from theory_generation.innovation_framework import InnovationFramework, InnovationLevel
from theory_generation.adaptive_generator import AdaptiveTheoryGenerator

class EnhancedFeedbackLoop:
    """增强的评估-改进反馈闭环"""
    
    def __init__(self, llm_interface, evaluator=None):
        self.llm = llm_interface
        self.evaluator = evaluator
        self.innovation_framework = InnovationFramework()
        self.adaptive_generator = AdaptiveTheoryGenerator(llm_interface)
        self.feedback_history = []
        
    async def run_improvement_cycle(self, 
                                  theory: Dict, 
                                  evaluation_results: Dict,
                                  max_iterations: int = 3,
                                  target_score_threshold: float = 0.8) -> Dict:
        """
        运行完整的改进循环
        
        Args:
            theory: 初始理论
            evaluation_results: 评估结果
            max_iterations: 最大改进迭代次数
            target_score_threshold: 目标分数阈值
            
        Returns:
            Dict: 改进循环结果
        """
        
        cycle_results = {
            "initial_theory": theory,
            "initial_evaluation": evaluation_results,
            "iterations": [],
            "final_theory": theory,
            "final_evaluation": evaluation_results,
            "improvement_achieved": False,
            "cycle_summary": {}
        }
        
        current_theory = theory
        current_evaluation = evaluation_results
        
        for iteration in range(max_iterations):
            print(f"\n[INFO] 改进循环 第{iteration + 1}轮")
            
            # 1. 分析当前评估结果
            feedback_analysis = self.analyze_evaluation_feedback(current_evaluation)
            
            # 2. 生成改进策略
            improvement_strategy = self.generate_improvement_strategy(
                current_theory, 
                feedback_analysis
            )
            
            # 3. 执行理论改进
            improved_theory = await self.improve_theory(
                current_theory, 
                improvement_strategy
            )
            
            if "error" in improved_theory:
                print(f"[WARNING] 第{iteration + 1}轮改进失败: {improved_theory['error']}")
                break
            
            # 4. 重新评估改进后的理论
            if self.evaluator:
                new_evaluation = await self.evaluator.evaluate_theory(improved_theory)
            else:
                # 如果没有评估器，使用模拟评估
                new_evaluation = self._simulate_evaluation(improved_theory, current_evaluation)
            
            # 5. 分析改进效果
            improvement_analysis = self.analyze_improvement(
                current_evaluation, 
                new_evaluation
            )
            
            # 6. 记录迭代结果
            iteration_result = {
                "iteration": iteration + 1,
                "feedback_analysis": feedback_analysis,
                "improvement_strategy": improvement_strategy,
                "improved_theory": improved_theory,
                "new_evaluation": new_evaluation,
                "improvement_analysis": improvement_analysis,
                "score_improvement": improvement_analysis.get("score_improvement", 0)
            }
            
            cycle_results["iterations"].append(iteration_result)
            
            # 7. 更新当前状态
            current_theory = improved_theory
            current_evaluation = new_evaluation
            
            # 8. 检查是否达到目标
            current_score = self._extract_overall_score(new_evaluation)
            if current_score >= target_score_threshold:
                print(f"[SUCCESS] 达到目标分数阈值 {target_score_threshold}")
                cycle_results["improvement_achieved"] = True
                break
            
            # 9. 检查是否有显著改进
            score_improvement = improvement_analysis.get("score_improvement", 0)
            if score_improvement < 0.03:  # 降低阈值，3%以下才认为是小改进
                print(f"[INFO] 改进幅度较小 ({score_improvement:.3f})，提前结束循环")
                break
            elif score_improvement > 0.1:  # 10%以上的改进继续优化
                print(f"[INFO] 显著改进 ({score_improvement:.3f})，继续优化")
            else:
                print(f"[INFO] 中等改进 ({score_improvement:.3f})，继续尝试")
        
        # 10. 生成循环总结
        cycle_results["final_theory"] = current_theory
        cycle_results["final_evaluation"] = current_evaluation
        cycle_results["cycle_summary"] = self._generate_cycle_summary(cycle_results)
        
        return cycle_results
    
    def analyze_evaluation_feedback(self, evaluation_results: Dict) -> Dict:
        """
        分析评估反馈，提取关键问题和改进点
        
        Args:
            evaluation_results: 评估结果
            
        Returns:
            Dict: 反馈分析结果
        """
        
        analysis = {
            "overall_assessment": {},
            "role_specific_issues": {},
            "innovation_gaps": [],
            "mathematical_issues": [],
            "experimental_issues": [],
            "philosophical_issues": [],
            "priority_improvements": []
        }
        
        # 1. 总体评估分析
        overall_score = self._extract_overall_score(evaluation_results)
        analysis["overall_assessment"] = {
            "score": overall_score,
            "performance_level": self._classify_performance(overall_score),
            "improvement_urgency": "high" if overall_score < 0.6 else "medium" if overall_score < 0.8 else "low"
        }
        
        # 2. 角色特定问题分析
        if "details" in evaluation_results:
            details_data = evaluation_results["details"]
            if isinstance(details_data, dict):
                for role, details in details_data.items():
                    # 处理details可能是字典或数值的情况
                    if isinstance(details, dict):
                        role_score = details.get("score", 0)
                        rationale = details.get("rationale", "")
                    else:
                        # 如果details是数值，直接使用
                        role_score = float(details) if details is not None else 0
                        rationale = ""
                    
                    issues = self._extract_role_issues(role, role_score, rationale)
                    analysis["role_specific_issues"][role] = issues
                    
                    # 根据角色类型分类问题
                    if role.lower() in ["mathematician", "mathematical_physicist"]:
                        analysis["mathematical_issues"].extend(issues["specific_problems"])
                    elif role.lower() in ["experimentalist", "experimental_physicist"]:
                        analysis["experimental_issues"].extend(issues["specific_problems"])
                    elif role.lower() in ["philosopher", "philosophy_of_science"]:
                        analysis["philosophical_issues"].extend(issues["specific_problems"])
        
        # 3. 创新差距分析
        analysis["innovation_gaps"] = self._identify_innovation_gaps(evaluation_results)
        
        # 4. 优先级排序
        analysis["priority_improvements"] = self._prioritize_improvements(analysis)
        
        return analysis
    
    def generate_improvement_strategy(self, 
                                    theory: Dict, 
                                    feedback_analysis: Dict) -> Dict:
        """
        根据反馈分析生成改进策略
        
        Args:
            theory: 当前理论
            feedback_analysis: 反馈分析结果
            
        Returns:
            Dict: 改进策略
        """
        
        strategy = {
            "improvement_targets": [],
            "mathematical_adjustments": [],
            "conceptual_enhancements": [],
            "experimental_enhancements": [],
            "philosophical_clarifications": [],
            "prompt_modifications": {},
            "generation_parameters": {}
        }
        
        # 1. 基于优先级设定改进目标
        for improvement in feedback_analysis["priority_improvements"]:
            strategy["improvement_targets"].append({
                "target": improvement["issue"],
                "priority": improvement["priority"],
                "approach": self._determine_improvement_approach(improvement)
            })
        
        # 2. 数学方面的调整
        if feedback_analysis["mathematical_issues"]:
            strategy["mathematical_adjustments"] = [
                self._design_mathematical_fix(issue) 
                for issue in feedback_analysis["mathematical_issues"]
            ]
        
        # 3. 概念增强
        innovation_gaps = feedback_analysis["innovation_gaps"]
        if innovation_gaps:
            strategy["conceptual_enhancements"] = [
                self._design_conceptual_enhancement(gap)
                for gap in innovation_gaps
            ]
        
        # 4. 实验方面增强
        if feedback_analysis["experimental_issues"]:
            strategy["experimental_enhancements"] = [
                self._design_experimental_enhancement(issue)
                for issue in feedback_analysis["experimental_issues"]
            ]
        
        # 5. 哲学澄清
        if feedback_analysis["philosophical_issues"]:
            strategy["philosophical_clarifications"] = [
                self._design_philosophical_clarification(issue)
                for issue in feedback_analysis["philosophical_issues"]
            ]
        
        # 6. 生成参数调整
        performance_level = feedback_analysis["overall_assessment"]["performance_level"]
        strategy["generation_parameters"] = self._adjust_generation_parameters(performance_level)
        
        return strategy
    
    async def improve_theory(self, theory: Dict, improvement_strategy: Dict) -> Dict:
        """
        根据改进策略改进理论
        
        Args:
            theory: 原理论
            improvement_strategy: 改进策略
            
        Returns:
            Dict: 改进后的理论
        """
        
        # 构建改进提示
        improvement_prompt = self._build_improvement_prompt(theory, improvement_strategy)
        
        # 获取生成参数
        gen_params = improvement_strategy.get("generation_parameters", {"temperature": 0.7})
        
        print("[INFO] 正在基于反馈改进理论...")
        
        # 调用LLM进行改进
        response = await self.llm.query_async(
            messages=[{"role": "user", "content": improvement_prompt}],
            temperature=gen_params.get("temperature", 0.7)
        )
        
        # 解析改进结果
        improved_theory = self.llm.extract_json(response)
        
        if not improved_theory:
            return {"error": "无法解析改进结果", "raw_response": response}
        
        # 添加改进元信息
        if "metadata" not in improved_theory:
            improved_theory["metadata"] = {}
        
        improved_theory["metadata"]["improvement_info"] = {
            "improved_from": theory.get("name", "Unknown"),
            "improvement_strategy": improvement_strategy["improvement_targets"],
            "improvement_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
        return improved_theory
    
    def analyze_improvement(self, 
                          old_evaluation: Dict, 
                          new_evaluation: Dict) -> Dict:
        """
        分析改进效果
        
        Args:
            old_evaluation: 改进前的评估
            new_evaluation: 改进后的评估
            
        Returns:
            Dict: 改进分析结果
        """
        
        old_score = self._extract_overall_score(old_evaluation)
        new_score = self._extract_overall_score(new_evaluation)
        score_improvement = new_score - old_score
        
        analysis = {
            "score_improvement": score_improvement,
            "improvement_percentage": (score_improvement / max(old_score, 0.1)) * 100,
            "improvement_significance": self._classify_improvement_significance(score_improvement),
            "role_specific_changes": {},
            "improvement_areas": [],
            "regression_areas": []
        }
        
        # 分析各角色评分变化
        if "details" in old_evaluation and "details" in new_evaluation:
            for role in old_evaluation["details"]:
                if role in new_evaluation["details"]:
                    # 处理details可能是字典或数值的情况
                    old_details = old_evaluation["details"][role]
                    new_details = new_evaluation["details"][role]
                    
                    old_role_score = old_details.get("score", 0) if isinstance(old_details, dict) else float(old_details)
                    new_role_score = new_details.get("score", 0) if isinstance(new_details, dict) else float(new_details)
                    role_change = new_role_score - old_role_score
                    
                    analysis["role_specific_changes"][role] = {
                        "old_score": old_role_score,
                        "new_score": new_role_score,
                        "change": role_change,
                        "change_type": "improvement" if role_change > 0 else "regression" if role_change < 0 else "unchanged"
                    }
                    
                    if role_change > 0.5:
                        analysis["improvement_areas"].append(role)
                    elif role_change < -0.5:
                        analysis["regression_areas"].append(role)
        
        return analysis
    
    def _extract_overall_score(self, evaluation_results: Dict) -> float:
        """提取总体评分"""
        if "role_score" in evaluation_results:
            return float(evaluation_results["role_score"])
        elif "overall_score" in evaluation_results:
            return float(evaluation_results["overall_score"])
        elif "details" in evaluation_results:
            # 计算各角色平均分
            scores = [details.get("score", 0) for details in evaluation_results["details"].values()]
            return statistics.mean(scores) / 10.0 if scores else 0.0
        else:
            return 0.0
    
    def _classify_performance(self, score: float) -> str:
        """分类性能水平"""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "good"
        elif score >= 0.6:
            return "moderate"
        elif score >= 0.4:
            return "poor"
        else:
            return "very_poor"
    
    def _extract_role_issues(self, role: str, score: float, rationale: str) -> Dict:
        """提取角色特定问题"""
        issues = {
            "score": score,
            "performance_level": "low" if score < 6 else "medium" if score < 8 else "high",
            "specific_problems": [],
            "improvement_suggestions": []
        }
        
        # 基于rationale提取具体问题
        rationale_lower = rationale.lower()
        
        # 数学问题检测
        math_indicators = ["equation", "formula", "mathematical", "calculation", "derivation", "inconsistent"]
        for indicator in math_indicators:
            if indicator in rationale_lower:
                issues["specific_problems"].append(f"Mathematical issue: {indicator}")
        
        # 实验问题检测
        exp_indicators = ["experiment", "testable", "prediction", "observable", "measurement"]
        for indicator in exp_indicators:
            if indicator in rationale_lower:
                issues["specific_problems"].append(f"Experimental issue: {indicator}")
        
        # 概念问题检测
        concept_indicators = ["unclear", "vague", "contradictory", "inconsistent", "confusing"]
        for indicator in concept_indicators:
            if indicator in rationale_lower:
                issues["specific_problems"].append(f"Conceptual issue: {indicator}")
        
        return issues
    
    def _identify_innovation_gaps(self, evaluation_results: Dict) -> List[str]:
        """识别创新差距"""
        gaps = []
        
        # 基于评估反馈识别创新不足
        if "details" in evaluation_results:
            details_data = evaluation_results["details"]
            if isinstance(details_data, dict):
                for role, details in details_data.items():
                    # 处理details可能是字典或数值的情况
                    if isinstance(details, dict):
                        rationale = details.get("rationale", "").lower()
                    else:
                        # 如果details是数值，无法分析rationale，跳过
                        continue
                    
                    if "not innovative" in rationale or "lack of novelty" in rationale:
                        gaps.append("General innovation deficit")
                    
                    if "mathematical" in rationale and ("same" in rationale or "standard" in rationale):
                        gaps.append("Mathematical innovation deficit")
                    
                    if "experimental" in rationale and ("no new" in rationale or "same as" in rationale):
                        gaps.append("Experimental innovation deficit")
                    
                    if "conceptual" in rationale and ("familiar" in rationale or "known" in rationale):
                        gaps.append("Conceptual innovation deficit")
        
        return list(set(gaps))  # 去重
    
    def _prioritize_improvements(self, analysis: Dict) -> List[Dict]:
        """优先级排序改进点"""
        improvements = []
        
        # 基于角色评分和问题严重性设定优先级
        for role, issues in analysis["role_specific_issues"].items():
            if issues["performance_level"] == "low":
                for problem in issues["specific_problems"]:
                    improvements.append({
                        "issue": problem,
                        "role": role,
                        "priority": "high",
                        "urgency": "immediate"
                    })
            elif issues["performance_level"] == "medium":
                for problem in issues["specific_problems"]:
                    improvements.append({
                        "issue": problem,
                        "role": role,
                        "priority": "medium",
                        "urgency": "moderate"
                    })
        
        # 创新差距优先级
        for gap in analysis["innovation_gaps"]:
            improvements.append({
                "issue": gap,
                "role": "general",
                "priority": "high",
                "urgency": "strategic"
            })
        
        # 按优先级排序
        priority_order = {"high": 3, "medium": 2, "low": 1}
        improvements.sort(key=lambda x: priority_order.get(x["priority"], 0), reverse=True)
        
        return improvements
    
    def _build_improvement_prompt(self, theory: Dict, improvement_strategy: Dict) -> str:
        """构建改进提示"""
        
        theory_name = theory.get("name", "Unknown Theory")
        improvement_targets = improvement_strategy["improvement_targets"]
        
        targets_text = "\n".join([
            f"- **{target['target']}** (优先级: {target['priority']})"
            for target in improvement_targets
        ])
        
        prompt = f"""
# THEORY IMPROVEMENT TASK

## CURRENT THEORY
**Theory Name**: {theory_name}

## IMPROVEMENT TARGETS
Based on detailed evaluation feedback, please improve the theory by addressing these specific issues:

{targets_text}

## SPECIFIC IMPROVEMENTS REQUIRED

### Mathematical Enhancements:
"""
        
        # 添加具体改进指导
        for adjustment in improvement_strategy.get("mathematical_adjustments", []):
            prompt += f"- {adjustment}\n"
        
        prompt += "\n### Conceptual Enhancements:\n"
        for enhancement in improvement_strategy.get("conceptual_enhancements", []):
            prompt += f"- {enhancement}\n"
        
        prompt += "\n### Experimental Enhancements:\n"
        for enhancement in improvement_strategy.get("experimental_enhancements", []):
            prompt += f"- {enhancement}\n"
        
        prompt += "\n### Philosophical Clarifications:\n"
        for clarification in improvement_strategy.get("philosophical_clarifications", []):
            prompt += f"- {clarification}\n"
        
        prompt += f"""

## ORIGINAL THEORY SUMMARY
**Name**: {theory.get('name', 'Unknown')}
**Core Philosophy**: {theory.get('philosophy', {}).get('ontology', {}).get('fundamental_entities', 'Not specified')}
**Key Innovation**: {theory.get('philosophy', {}).get('innovation_claim', 'Not specified')}
**Mathematical Framework**: {theory.get('formalism', {}).get('mathematical_objects', 'Not specified')}

## OUTPUT REQUIREMENTS
Provide the IMPROVED theory as a single, valid JSON object using the "Quantum Theory Schema v2.1" structure. 

**CRITICAL REQUIREMENTS**:
1. **Address ALL identified issues** directly and substantially
2. **Maintain core theoretical strengths** while fixing weaknesses  
3. **Output ONLY the JSON object** - no additional text or explanations
4. **Use proper JSON formatting** with valid syntax
5. **Keep the same general structure** as the original theory but with improvements

**JSON Schema Structure Required**:
```json
{{
  "name": "string (improved theory name)",
  "metadata": {{ ... }},
  "philosophy": {{ ... }},
  "formalism": {{ ... }},
  "predictions_and_verifiability": {{ ... }}
}}
```

**CRITICAL**: Make substantial improvements that directly address the evaluation feedback. Half-measures will not be accepted.
"""
        
        return prompt
    
    def _simulate_evaluation(self, theory: Dict, baseline_evaluation: Dict) -> Dict:
        """模拟评估（当没有真实评估器时）"""
        # 这是一个简化的模拟，实际使用时应该用真实的评估器
        simulated = baseline_evaluation.copy()
        
        # 模拟小幅提升
        if "role_score" in simulated:
            current_score = simulated["role_score"]
            improvement = random.uniform(0.05, 0.15)  # 5-15%的改进
            simulated["role_score"] = min(current_score + improvement, 1.0)
        
        return simulated
    
    def _generate_cycle_summary(self, cycle_results: Dict) -> Dict:
        """生成循环总结"""
        initial_score = self._extract_overall_score(cycle_results["initial_evaluation"])
        final_score = self._extract_overall_score(cycle_results["final_evaluation"])
        total_improvement = final_score - initial_score
        
        return {
            "total_iterations": len(cycle_results["iterations"]),
            "initial_score": initial_score,
            "final_score": final_score,
            "total_improvement": total_improvement,
            "improvement_percentage": (total_improvement / max(initial_score, 0.1)) * 100,
            "improvement_achieved": cycle_results["improvement_achieved"],
            "key_improvements": [
                iteration["improvement_analysis"]["improvement_areas"]
                for iteration in cycle_results["iterations"]
                if iteration["improvement_analysis"]["improvement_areas"]
            ],
            "persistent_issues": self._identify_persistent_issues(cycle_results)
        }
    
    def _classify_improvement_significance(self, score_improvement: float) -> str:
        """分类改进显著性"""
        if score_improvement >= 0.2:
            return "major"
        elif score_improvement >= 0.1:
            return "significant" 
        elif score_improvement >= 0.05:
            return "moderate"
        elif score_improvement > 0:
            return "minor"
        elif score_improvement == 0:
            return "none"
        else:
            return "regression"
    
    def _determine_improvement_approach(self, improvement: Dict) -> str:
        """确定改进方法"""
        issue = improvement["issue"].lower()
        
        if "mathematical" in issue:
            return "mathematical_refinement"
        elif "experimental" in issue:
            return "experimental_enhancement"
        elif "conceptual" in issue:
            return "conceptual_clarification"
        elif "innovation" in issue:
            return "innovation_boost"
        else:
            return "general_improvement"
    
    def _design_mathematical_fix(self, issue: str) -> str:
        """设计数学修复方案"""
        if "equation" in issue.lower():
            return "Review and correct mathematical equations for physical consistency"
        elif "formula" in issue.lower():
            return "Ensure all formulas are properly derived and dimensionally correct"
        elif "inconsistent" in issue.lower():
            return "Resolve mathematical inconsistencies in the formalism"
        else:
            return "Enhance mathematical rigor and precision"
    
    def _design_conceptual_enhancement(self, gap: str) -> str:
        """设计概念增强方案"""
        if "innovation deficit" in gap.lower():
            return "Introduce genuinely novel conceptual elements"
        elif "mathematical innovation" in gap.lower():
            return "Develop new mathematical structures or modify existing ones"
        elif "experimental innovation" in gap.lower():
            return "Propose unique experimental predictions and tests"
        else:
            return "Strengthen conceptual foundations and novelty"
    
    def _design_experimental_enhancement(self, issue: str) -> str:
        """设计实验增强方案"""
        if "testable" in issue.lower():
            return "Develop more specific and feasible experimental tests"
        elif "prediction" in issue.lower():
            return "Formulate clear, quantitative experimental predictions"
        else:
            return "Improve experimental accessibility and verifiability"
    
    def _design_philosophical_clarification(self, issue: str) -> str:
        """设计哲学澄清方案"""
        if "unclear" in issue.lower():
            return "Clarify philosophical assumptions and implications"
        elif "contradictory" in issue.lower():
            return "Resolve philosophical contradictions and tensions"
        else:
            return "Deepen philosophical analysis and coherence"
    
    def _adjust_generation_parameters(self, performance_level: str) -> Dict:
        """调整生成参数"""
        if performance_level == "very_poor":
            return {"temperature": 0.9, "focus": "fundamental_reconstruction"}
        elif performance_level == "poor":
            return {"temperature": 0.8, "focus": "significant_improvements"}
        elif performance_level == "moderate":
            return {"temperature": 0.7, "focus": "targeted_enhancements"}
        else:
            return {"temperature": 0.6, "focus": "fine_tuning"}
    
    def _identify_persistent_issues(self, cycle_results: Dict) -> List[str]:
        """识别持续存在的问题"""
        persistent = []
        
        if len(cycle_results["iterations"]) > 1:
            # 比较不同迭代中的问题
            first_iter = cycle_results["iterations"][0]
            last_iter = cycle_results["iterations"][-1]
            
            first_issues = set()
            last_issues = set()
            
            if "feedback_analysis" in first_iter:
                for improvement in first_iter["feedback_analysis"].get("priority_improvements", []):
                    first_issues.add(improvement["issue"])
            
            if "feedback_analysis" in last_iter:
                for improvement in last_iter["feedback_analysis"].get("priority_improvements", []):
                    last_issues.add(improvement["issue"])
            
            persistent = list(first_issues.intersection(last_issues))
        
        return persistent 