#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学形式分类器

用于判断量子理论是否使用与经典量子力学相同的数学公式
"""
import re
import json
from typing import Dict, Any, Tuple, List

class MathematicalClassifier:
    """量子理论数学形式分类器"""
    
    def __init__(self):
        # 标准量子力学的关键数学元素
        self.standard_qm_equations = [
            r"i\\?ħ\\?\s*\\?partial_t\\?\s*\\?psi\\?\s*=\\?\s*H\\?\s*\\?psi",  # 薛定谔方程
            r"i\\?\\hbar\\?\s*\\?frac\\{\\?partial\\?\\}\\{\\?partial\\s*t\\?\\}\\?\\|\\?\\psi\\?\\rangle\\?\s*=\\?\s*H\\?\\|\\?\\psi\\?\\rangle",
            r"\\|\\?\\psi\\?\\rangle\\?\s*=\\?\s*\\sum\\?\s*c_i\\?\\|\\?\\phi_i\\?\\rangle",  # 态叠加
            r"P\\?\(\\?a_n\\?\)\\?\s*=\\?\s*\\|\\?\\langle\\?\\phi_n\\?\\|\\?\\psi\\?\\rangle\\?\\|\\?\\^\\?2",  # 玻恩规则
        ]
        
        # 修改的数学元素关键词
        self.modification_indicators = [
            "non-linear",
            "stochastic",
            "collapse",
            "localization",
            "guidance equation",
            "pilot wave",
            "hidden variable",
            "additional dynamics",
            "modified hamiltonian",
            "new parameter",
            "spontaneous",
            "objective reduction"
        ]
        
        # 纯诠释性关键词（不改变数学）
        self.interpretation_indicators = [
            "interpretation",
            "complementarity",
            "wave function collapse",
            "measurement problem",
            "observer",
            "classical apparatus",
            "epistemic",
            "ontic"
        ]
    
    def classify_theory_mathematics(self, theory: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        分类理论的数学形式
        
        Args:
            theory: 理论数据字典
            
        Returns:
            Tuple[str, Dict]: (分类结果, 详细分析)
            分类结果: "standard_qm", "modified_qm", "extended_qm"
        """
        analysis = {
            "mathematical_relation": theory.get("mathematical_relation_to_sqm", ""),
            "uses_standard_equations": False,
            "has_modifications": False,
            "modification_evidence": [],
            "standard_evidence": [],
            "confidence": 0.0
        }
        
        # 1. 检查明确的数学关系声明
        math_relation = theory.get("mathematical_relation_to_sqm", {})
        if isinstance(math_relation, dict):
            relation_type = math_relation.get("type", "").lower()
        elif isinstance(math_relation, str):
            relation_type = math_relation.lower()
        else:
            relation_type = ""
        
        # 2. 分析形式主义部分
        formalism = theory.get("formalism", {})
        equations = formalism.get("equations", {})
        
        # 检查是否使用标准薛定谔方程
        state_evolution = equations.get("state_evolution", "")
        additional_dynamics = equations.get("additional_dynamics", "")
        
        # 3. 检查标准方程
        if self._contains_standard_schrodinger(state_evolution):
            analysis["uses_standard_equations"] = True
            analysis["standard_evidence"].append("Standard Schrödinger equation found")
        
        # 4. 检查修改证据 - 但排除纯诠释性的"collapse"提及
        modification_evidence = self._find_modification_evidence(theory)
        # 过滤掉纯诠释性的证据
        filtered_evidence = self._filter_interpretation_evidence(modification_evidence, theory)
        analysis["modification_evidence"] = filtered_evidence
        analysis["has_modifications"] = len(filtered_evidence) > 0
        
        # 5. 最终分类 - 更严格的判断
        if relation_type == "interpretation":
            # 如果明确声明是诠释，且没有真正的数学修改，则为标准QM
            if not analysis["has_modifications"]:
                classification = "standard_qm"
                analysis["confidence"] = 0.9
            else:
                classification = "modified_qm"
                analysis["confidence"] = 0.7
        elif relation_type == "modification" or self._has_genuine_modifications(theory):
            classification = "modified_qm"
            analysis["confidence"] = 0.8
        elif relation_type == "extension":
            classification = "extended_qm"
            analysis["confidence"] = 0.7
        else:
            # 基于内容推断
            if analysis["uses_standard_equations"] and not analysis["has_modifications"]:
                classification = "standard_qm"
                analysis["confidence"] = 0.6
            else:
                classification = "modified_qm"
                analysis["confidence"] = 0.5
        
        return classification, analysis
    
    def _contains_standard_schrodinger(self, equation_text: str) -> bool:
        """检查是否包含标准薛定谔方程"""
        if not equation_text:
            return False
        
        # 标准薛定谔方程的模式
        patterns = [
            r"i\\?ħ.*∂.*ψ.*=.*H.*ψ",
            r"i\\?\\hbar.*\\partial.*\\psi.*=.*H.*\\psi",
            r"i\\?ħ.*d.*dt.*ψ.*=.*H.*ψ",
            r"iħ\\(d/dt\\)\\|Ψ.*=.*H\\|Ψ"
        ]
        
        for pattern in patterns:
            if re.search(pattern, equation_text, re.IGNORECASE):
                return True
        
        return False
    
    def _find_modification_evidence(self, theory: Dict[str, Any]) -> List[str]:
        """查找数学修改的证据"""
        evidence = []
        
        # 检查各个部分的文本
        text_sections = [
            ("formalism", theory.get("formalism", {})),
            ("core_principles", theory.get("core_principles", [])),
            ("predictions_and_verifiability", theory.get("predictions_and_verifiability", {}))
        ]
        
        for section_name, section_data in text_sections:
            section_text = json.dumps(section_data).lower()
            
            for indicator in self.modification_indicators:
                if indicator in section_text:
                    evidence.append(f"Found '{indicator}' in {section_name}")
        
        # 检查新参数
        formalism = theory.get("formalism", {})
        constants = formalism.get("constants_and_parameters", {})
        
        for param_name, param_data in constants.items():
            if isinstance(param_data, dict):
                param_type = param_data.get("type", "")
                if "new" in param_type.lower():
                    evidence.append(f"New parameter: {param_name}")
        
        return evidence
    
    def _filter_interpretation_evidence(self, evidence: List[str], theory: Dict[str, Any]) -> List[str]:
        """过滤掉纯诠释性的证据"""
        filtered = []
        
        # 获取理论名称以进行特殊处理
        theory_name = theory.get("name", "").lower()
        
        # 对于明确的诠释理论，过滤掉某些"假阳性"证据
        interpretation_theories = [
            "copenhagen", "many-worlds", "many worlds", "qbism", "relational", 
            "modal", "ensemble", "consistent histories", "transactional"
        ]
        
        is_interpretation_theory = any(interp in theory_name for interp in interpretation_theories)
        
        for ev in evidence:
            ev_lower = ev.lower()
            # 如果是诠释理论，过滤掉这些纯诠释性的证据
            if is_interpretation_theory:
                if ("found 'collapse'" in ev_lower or 
                    "found 'hidden variable'" in ev_lower or
                    "found 'observer'" in ev_lower or
                    "found 'measurement problem'" in ev_lower):
                    continue
            
            # 保留真正的数学修改证据
            if any(indicator in ev_lower for indicator in [
                "guidance equation", "pilot wave", "additional dynamics", 
                "new parameter", "non-linear", "stochastic", "localization",
                "spontaneous", "objective reduction"
            ]):
                filtered.append(ev)
            elif not is_interpretation_theory:
                # 对于非诠释理论，保留所有证据
                filtered.append(ev)
        
        return filtered
    
    def _has_genuine_modifications(self, theory: Dict[str, Any]) -> bool:
        """检查是否有真正的数学修改（而非仅仅是诠释性描述）"""
        formalism = theory.get("formalism", {})
        equations = formalism.get("equations", {})
        
        # 检查是否有额外的动力学方程
        additional_dynamics = equations.get("additional_dynamics", "")
        if additional_dynamics and additional_dynamics.strip() and additional_dynamics.lower() != "n/a":
            return True
        
        # 检查是否有新的参数
        constants = formalism.get("constants_and_parameters", {})
        for param_name, param_data in constants.items():
            if isinstance(param_data, dict):
                param_type = param_data.get("type", "")
                if "new" in param_type.lower():
                    return True
        
        # 检查是否有修改的状态演化方程
        state_evolution = equations.get("state_evolution", "")
        if state_evolution and not self._contains_standard_schrodinger(state_evolution):
            # 如果有状态演化方程但不是标准薛定谔方程，可能是修改
            if "stochastic" in state_evolution.lower() or "non-linear" in state_evolution.lower():
                return True
        
        return False
    
    def annotate_theory_with_classification(self, theory: Dict[str, Any]) -> Dict[str, Any]:
        """
        为理论添加数学分类标注
        
        Args:
            theory: 原理论数据
            
        Returns:
            Dict: 添加了分类标注的理论数据
        """
        classification, analysis = self.classify_theory_mathematics(theory)
        
        # 添加分类标注
        if "metadata" not in theory:
            theory["metadata"] = {}
        
        theory["metadata"]["mathematical_classification"] = {
            "type": classification,
            "uses_standard_qm_math": classification == "standard_qm",
            "analysis": analysis,
            "classifier_version": "1.0"
        }
        
        return theory
    
    def batch_classify_theories(self, theories: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        批量分类理论
        
        Args:
            theories: 理论名称到理论数据的映射
            
        Returns:
            Dict: 分类结果统计
        """
        results = {
            "standard_qm": [],
            "modified_qm": [],
            "extended_qm": [],
            "statistics": {}
        }
        
        for theory_name, theory_data in theories.items():
            classification, analysis = self.classify_theory_mathematics(theory_data)
            
            results[classification].append({
                "name": theory_name,
                "confidence": analysis["confidence"],
                "evidence": analysis.get("modification_evidence", [])
            })
        
        # 统计
        total = len(theories)
        results["statistics"] = {
            "total_theories": total,
            "standard_qm_count": len(results["standard_qm"]),
            "modified_qm_count": len(results["modified_qm"]),
            "extended_qm_count": len(results["extended_qm"]),
            "standard_qm_percentage": len(results["standard_qm"]) / total * 100 if total > 0 else 0
        }
        
        return results


def main():
    """测试函数"""
    import os
    import glob
    
    # 测试先验理论数据集
    theories_dir = "data/theories_v2.1"
    if os.path.exists(theories_dir):
        classifier = MathematicalClassifier()
        theories = {}
        
        for theory_file in glob.glob(os.path.join(theories_dir, "*.json")):
            with open(theory_file, 'r', encoding='utf-8') as f:
                theory_data = json.load(f)
                theory_name = theory_data.get("name", os.path.basename(theory_file))
                theories[theory_name] = theory_data
        
        results = classifier.batch_classify_theories(theories)
        
        print("数学形式分类结果:")
        print("=" * 50)
        print(f"总理论数: {results['statistics']['total_theories']}")
        print(f"标准量子力学: {results['statistics']['standard_qm_count']} ({results['statistics']['standard_qm_percentage']:.1f}%)")
        print(f"修改量子力学: {results['statistics']['modified_qm_count']}")
        print(f"扩展量子力学: {results['statistics']['extended_qm_count']}")
        
        print("\n标准量子力学理论:")
        for theory in results["standard_qm"]:
            print(f"  - {theory['name']} (置信度: {theory['confidence']:.2f})")
        
        print("\n修改量子力学理论:")
        for theory in results["modified_qm"]:
            print(f"  - {theory['name']} (置信度: {theory['confidence']:.2f})")
            if theory['evidence']:
                print(f"    证据: {', '.join(theory['evidence'][:2])}")


if __name__ == "__main__":
    main() 