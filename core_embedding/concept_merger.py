#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
概念融合器

融合从文献中提取的概念和公式，以及从理论生成器生成的概念和公式。
支持中英文概念的融合和对齐。
"""

import os
import json
import csv
import re
from typing import List, Dict, Any, Tuple, Set

class ConceptMerger:
    """融合不同来源的概念和公式"""
    
    def __init__(self):
        """初始化概念融合器"""
        self.extracted_concepts = []
        self.extracted_formulas = []
        self.generated_concepts = []
        self.generated_formulas = []
        self.merged_concepts = []
        self.merged_formulas = []
        
    def load_extracted_data(self, data_path: str) -> None:
        """
        加载从文献中提取的概念和公式
        
        Args:
            data_path: 提取数据的JSON文件路径
        """
        if not os.path.exists(data_path):
            print(f"[ERROR] 文件不存在: {data_path}")
            return
            
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.extracted_concepts = data.get('concepts', [])
            self.extracted_formulas = data.get('formulas', [])
            
            print(f"[INFO] 已加载 {len(self.extracted_concepts)} 个提取的概念和 {len(self.extracted_formulas)} 个提取的公式")
        except Exception as e:
            print(f"[ERROR] 加载提取数据时出错: {str(e)}")
            
    def load_generated_theories(self, data_path: str) -> None:
        """
        加载从理论生成器生成的理论、概念和公式
        
        Args:
            data_path: 生成数据的JSON文件路径
        """
        if not os.path.exists(data_path):
            print(f"[ERROR] 文件不存在: {data_path}")
            return
            
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理两种可能的数据格式
            if isinstance(data, list):
                # 新格式: 直接是理论列表
                theories = data
            else:
                # 旧格式: 包含theories键
                theories = data.get('theories', [])
            
            # 提取所有概念和公式
            for theory in theories:
                # 处理新格式的理论数据
                if "core_principles" in theory:
                    # 从core_principles创建一个概念
                    self.generated_concepts.append({
                        "name": f"{theory.get('name', '')}: Core Principles",
                        "description": theory.get('core_principles', ''),
                        "domain": "Physics",
                        "theory": theory.get('name', ''),
                        "source": "Theory Generation"
                    })
                    
                    # 处理量子现象解释
                    quantum_phenomena = theory.get('quantum_phenomena_explanation', {})
                    if quantum_phenomena:
                        for phenomenon, explanation in quantum_phenomena.items():
                            if phenomenon != "other" and explanation:
                                self.generated_concepts.append({
                                    "name": f"{phenomenon.replace('_', ' ').title()}",
                                    "description": explanation,
                                    "domain": "Physics",
                                    "theory": theory.get('name', ''),
                                    "source": "Theory Generation"
                                })
                    
                    # 处理数学公式
                    math_formulation = theory.get('mathematical_formulation', '')
                    if math_formulation and len(math_formulation) > 20:
                        # 尝试从数学公式文本中提取公式
                        formulas = self._extract_formulas_from_text(math_formulation)
                        if formulas:
                            self.generated_formulas.extend(formulas)
                        else:
                            # 如果无法提取具体公式，则作为一个整体处理
                            self.generated_formulas.append({
                                "name": f"{theory.get('name', '')}: Mathematical Formulation",
                                "expression": math_formulation,
                                "description": "Mathematical representation of the theory",
                                "variables": [],
                                "theory": theory.get('name', ''),
                                "source": "Theory Generation"
                            })
                else:
                    # 旧格式: 直接包含concepts和formulas
                    for concept in theory.get('concepts', []):
                        concept['theory'] = theory.get('theory_name', '')
                        concept['source'] = "Theory Generation"
                        self.generated_concepts.append(concept)
                        
                    for formula in theory.get('formulas', []):
                        formula['theory'] = theory.get('theory_name', '')
                        formula['source'] = "Theory Generation"
                        self.generated_formulas.append(formula)
            
            print(f"[INFO] 已加载 {len(self.generated_concepts)} 个生成的概念和 {len(self.generated_formulas)} 个生成的公式")
        except Exception as e:
            print(f"[ERROR] 加载生成数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _extract_formulas_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中提取数学公式
        
        Args:
            text: 包含数学公式的文本
            
        Returns:
            List[Dict]: 提取的公式列表
        """
        formulas = []
        
        # 尝试查找常见的公式模式，如等式，公式名称等
        # 1. 寻找等式模式: 左侧 = 右侧
        equation_pattern = r'([A-Za-z\d\s\^\{\}\(\)\[\]\\]+)\s*=\s*([A-Za-z\d\s\^\{\}\(\)\[\]\\]+)'
        for match in re.finditer(equation_pattern, text):
            left_side = match.group(1).strip()
            right_side = match.group(2).strip()
            expression = f"{left_side} = {right_side}"
            
            # 尝试识别公式名称
            name = self._guess_formula_name(left_side, text)
            
            formulas.append({
                "name": name,
                "expression": expression,
                "description": self._find_sentence_containing(expression, text),
                "variables": self._extract_variables(expression, text),
                "source": "Theory Generation"
            })
        
        return formulas
    
    def _guess_formula_name(self, expression: str, context: str) -> str:
        """
        尝试猜测公式名称
        
        Args:
            expression: 公式表达式
            context: 上下文文本
            
        Returns:
            str: 猜测的公式名称
        """
        # 一些常见的物理公式名称映射
        common_formulas = {
            r'E\s*=\s*mc': "Mass-Energy Equivalence",
            r'E\s*=\s*h\w*': "Planck's Energy Formula",
            r'\psi': "Wave Function",
            r'H\|\psi': "Schrödinger Equation",
            r'i\hbar': "Schrödinger Equation",
            r'\Delta x\Delta p': "Heisenberg Uncertainty Principle",
            r'\rho': "Density Matrix",
            r'P\(': "Probability Formula"
        }
        
        for pattern, name in common_formulas.items():
            if re.search(pattern, expression):
                return name
        
        # 尝试从上下文中找到名称
        sentences = context.split('.')
        for sentence in sentences:
            if expression in sentence and any(term in sentence.lower() for term in 
                                            ["called", "known as", "referred to as", "equation", "formula", "principle"]):
                for term in ["equation", "formula", "principle", "relation", "law"]:
                    match = re.search(f"([A-Za-z\s'-]+){term}", sentence, re.IGNORECASE)
                    if match:
                        return match.group(1).strip() + " " + term
        
        # 如果无法确定，返回默认名称
        return "Quantum Formula"
    
    def _find_sentence_containing(self, phrase: str, text: str) -> str:
        """
        找到包含特定短语的句子
        
        Args:
            phrase: 要查找的短语
            text: 要搜索的文本
            
        Returns:
            str: 包含该短语的句子，或空字符串
        """
        # 简化短语以提高匹配率
        simplified_phrase = re.sub(r'[\s\{\}\(\)\[\]\\]', '.', phrase)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if re.search(simplified_phrase, sentence):
                return sentence.strip()
        
        # 找不到具体句子，返回上下文中的一小段
        phrase_pos = text.find(phrase)
        if phrase_pos >= 0:
            start = max(0, phrase_pos - 100)
            end = min(len(text), phrase_pos + len(phrase) + 100)
            return text[start:end].strip()
        
        return ""
    
    def _extract_variables(self, expression: str, context: str) -> List[Dict[str, str]]:
        """
        提取公式中的变量及其含义
        
        Args:
            expression: 公式表达式
            context: 上下文文本
            
        Returns:
            List[Dict]: 变量列表
        """
        variables = []
        
        # 提取单字母变量（可能带有下标或上标）
        var_pattern = r'([A-Za-z])(?:_[A-Za-z\d]+|\^[A-Za-z\d]+)?'
        found_vars = set(re.findall(var_pattern, expression))
        
        # 常见量子力学变量的含义
        common_variables = {
            "ψ": "Wave function",
            "Ψ": "Wave function",
            "H": "Hamiltonian operator",
            "E": "Energy",
            "p": "Momentum",
            "x": "Position",
            "t": "Time",
            "ℏ": "Reduced Planck's constant",
            "h": "Planck's constant",
            "i": "Imaginary unit",
            "m": "Mass",
            "c": "Speed of light",
            "ρ": "Density matrix",
            "A": "Observable",
            "λ": "Wavelength",
            "Δ": "Uncertainty",
            "α": "Alpha parameter",
            "β": "Beta parameter",
            "γ": "Gamma parameter",
            "ω": "Angular frequency"
        }
        
        # 尝试从上下文中找出变量含义
        for var in found_vars:
            # 跳过一些常见的非变量
            if var in ['d', 'e'] and not re.search(f"{var}_", expression):
                continue
                
            meaning = ""
            
            # 尝试从上下文中解析变量含义
            var_pattern = f"{var}(?:\s+|\s*is\s+|\s*represents\s+|\s*denotes\s+)([^,.;:]+)"
            match = re.search(var_pattern, context, re.IGNORECASE)
            if match:
                meaning = match.group(1).strip()
            else:
                # 使用常见变量字典
                meaning = common_variables.get(var, "")
            
            if meaning:
                variables.append({
                    "symbol": var,
                    "meaning": meaning
                })
        
        return variables
    
    def merge_concepts(self) -> None:
        """融合提取的概念和生成的概念"""
        print("[INFO] 开始融合概念...")
        
        # 首先复制所有提取的概念
        self.merged_concepts = self.extracted_concepts.copy()
        
        # 添加生成的概念，避免重复
        added_concepts = set(c['name'].lower() for c in self.merged_concepts)
        
        for concept in self.generated_concepts:
            concept_name = concept.get('name', '').lower()
            
            # 检查是否已存在类似概念
            if concept_name in added_concepts:
                # 找到现有概念并增强描述
                existing_idx = next((i for i, c in enumerate(self.merged_concepts) 
                                     if c['name'].lower() == concept_name), -1)
                if existing_idx >= 0:
                    # 合并描述，增加理论来源
                    existing = self.merged_concepts[existing_idx]
                    merged_desc = f"{existing.get('description', '')}\n\n"
                    merged_desc += f"Additional perspective from {concept.get('theory', 'generated theory')}:\n"
                    merged_desc += concept.get('description', '')
                    
                    self.merged_concepts[existing_idx]['description'] = merged_desc
                    self.merged_concepts[existing_idx]['theories'] = list(set(
                        existing.get('theories', []) + [concept.get('theory', '')]
                    ))
            else:
                # 添加新概念
                concept_copy = concept.copy()
                concept_copy['theories'] = [concept.get('theory', '')]
                if 'theory' in concept_copy:
                    del concept_copy['theory']  # 移除单个theory字段
                
                self.merged_concepts.append(concept_copy)
                added_concepts.add(concept_name)
        
        print(f"[INFO] 概念融合完成，共有 {len(self.merged_concepts)} 个概念")
        
    def merge_formulas(self) -> None:
        """融合提取的公式和生成的公式"""
        print("[INFO] 开始融合公式...")
        
        # 首先复制所有提取的公式
        self.merged_formulas = self.extracted_formulas.copy()
        
        # 添加生成的公式，避免重复
        # 使用公式表达式作为去重的键
        added_expressions = set()
        for formula in self.merged_formulas:
            expr = formula.get('expression', '').replace(' ', '')
            if expr:
                added_expressions.add(expr)
        
        for formula in self.generated_formulas:
            expr = formula.get('expression', '').replace(' ', '')
            
            if not expr or expr in added_expressions:
                continue
                
            # 添加新公式
            formula_copy = formula.copy()
            formula_copy['theories'] = [formula.get('theory', '')]
            if 'theory' in formula_copy:
                del formula_copy['theory']  # 移除单个theory字段
                
            self.merged_formulas.append(formula_copy)
            added_expressions.add(expr)
        
        print(f"[INFO] 公式融合完成，共有 {len(self.merged_formulas)} 个公式")
    
    def merge_all(self) -> None:
        """执行所有融合操作"""
        self.merge_concepts()
        self.merge_formulas()
    
    def save_to_csv(self, output_dir: str) -> None:
        """
        将融合后的概念和公式保存到CSV文件
        
        Args:
            output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存概念
        concepts_path = os.path.join(output_dir, 'merged_concepts.csv')
        with open(concepts_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Description', 'Domain', 'Source', 'Theories'])
            
            for concept in self.merged_concepts:
                writer.writerow([
                    concept.get('name', ''),
                    concept.get('description', ''),
                    concept.get('domain', ''),
                    concept.get('source', ''),
                    '; '.join(concept.get('theories', []))
                ])
        
        # 保存公式
        formulas_path = os.path.join(output_dir, 'merged_formulas.csv')
        with open(formulas_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Expression', 'Description', 'Variables', 'Source', 'Theories'])
            
            for formula in self.merged_formulas:
                # 处理变量列表
                variables = formula.get('variables', [])
                var_text = '; '.join([f"{v.get('symbol', '')}: {v.get('meaning', '')}" for v in variables])
                
                writer.writerow([
                    formula.get('name', ''),
                    formula.get('expression', ''),
                    formula.get('description', ''),
                    var_text,
                    formula.get('source', ''),
                    '; '.join(formula.get('theories', []))
                ])
        
        print(f"[INFO] 融合后的概念保存到: {concepts_path}")
        print(f"[INFO] 融合后的公式保存到: {formulas_path}")
        
        # 也保存为JSON格式以保留完整数据结构
        json_path = os.path.join(output_dir, 'merged_data.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'concepts': self.merged_concepts,
                'formulas': self.merged_formulas
            }, f, ensure_ascii=False, indent=2)
            
        print(f"[INFO] 完整数据保存到: {json_path}") 