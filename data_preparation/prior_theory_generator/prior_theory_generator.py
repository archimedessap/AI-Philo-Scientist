#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
先验理论生成器

使用LLM直接生成量子理论的描述，包括自然语言概念和数学公式，
作为对从文献中提取的概念和公式的补充。
"""

import os
import json
import csv
from typing import Dict, List, Tuple, Any, Optional
from theory_generation.llm_interface import LLMInterface
import re

class PriorTheoryGenerator:
    """使用LLM生成先验理论的类"""
    
    def __init__(self, llm_interface: LLMInterface = None):
        """
        初始化先验理论生成器
        
        Args:
            llm_interface: LLM接口实例，如果为None则自动创建
        """
        # 如果未提供LLM接口，则创建一个
        if llm_interface is None:
            self.llm = LLMInterface(
                model_source="openai",
                model_name="gpt-4o-mini"
            )
        else:
            self.llm = llm_interface
            
        # 存储结果
        self.theories = []
        self.concepts = []
        self.formulas = []
    
    def generate_theory(self, theory_name: str) -> Dict:
        """
        生成指定量子理论的描述
        
        Args:
            theory_name: 理论名称
            
        Returns:
            Dict: 生成的理论描述
        """
        print(f"[INFO] 生成理论描述: {theory_name}")
        
        # 使用LLM接口直接生成已有理论描述
        theory_data = self.llm.generate_existing_theory(theory_name)
        
        if "error" in theory_data:
            print(f"[ERROR] 生成理论 {theory_name} 时出错: {theory_data['error']}")
            return theory_data
            
        # 添加到理论列表
        self.theories.append(theory_data)
        
        # 提取概念信息
        for concept in theory_data.get("key_concepts", []):
            concept_item = {
                "name": concept.get("name", ""),
                "description": concept.get("description", ""),
                "domain": "量子力学",
                "source": f"Theory: {theory_name}",
                "theory": theory_name
            }
            self.concepts.append(concept_item)
        
        # 如果理论数据中包含数学框架信息，添加为公式
        math_framework = theory_data.get("mathematical_framework", "")
        if math_framework:
            formula_item = {
                "name": f"{theory_name}的数学框架",
                "expression": "参见描述",
                "description": math_framework,
                "variables": [],
                "source": f"Theory: {theory_name}",
                "theory": theory_name
            }
            self.formulas.append(formula_item)
        
        print(f"[INFO] 已生成理论 {theory_name}，包含 {len(theory_data.get('key_concepts', []))} 个概念")
        return theory_data
    
    def generate_multiple_theories(self, theory_names: List[str]) -> List[Dict]:
        """
        生成多个量子理论的描述
        
        Args:
            theory_names: 理论名称列表
            
        Returns:
            List[Dict]: 生成的理论描述列表
        """
        results = []
        
        for theory_name in theory_names:
            theory_data = self.generate_theory(theory_name)
            results.append(theory_data)
        
        # 生成完成后，可以获取理论之间的比较分析
        if len(theory_names) > 1:
            print(f"[INFO] 生成理论之间的比较分析")
            comparison_data = self.llm.get_theory_comparison(theory_names)
            if "error" not in comparison_data:
                # 保存比较结果
                self.theory_comparisons = comparison_data
                results.append({"theory_name": "理论比较分析", "data": comparison_data})
        
        return results
    
    def save_to_csv(self, output_dir: str) -> None:
        """
        将生成的理论、概念和公式保存到CSV文件
        
        Args:
            output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存理论概述
        theories_path = os.path.join(output_dir, 'theories.csv')
        with open(theories_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Name', 'Description', 'Developers', 'Year', 'Source']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for theory in self.theories:
                writer.writerow({
                    'Name': theory.get('theory_name', ''),
                    'Description': theory.get('description', ''),
                    'Developers': ', '.join(theory.get('developers', [])),
                    'Year': theory.get('year', ''),
                    'Source': theory.get('source', '')
                })
        
        # 保存概念
        concepts_path = os.path.join(output_dir, 'concepts.csv')
        with open(concepts_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Name', 'Description', 'Domain', 'Theory', 'Source']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for concept in self.concepts:
                writer.writerow({
                    'Name': concept.get('name', ''),
                    'Description': concept.get('description', ''),
                    'Domain': concept.get('domain', ''),
                    'Theory': concept.get('theory', ''),
                    'Source': concept.get('source', '')
                })
        
        # 保存公式
        formulas_path = os.path.join(output_dir, 'formulas.csv')
        with open(formulas_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Name', 'Expression', 'Description', 'Variables', 'Theory', 'Source']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for formula in self.formulas:
                # 处理变量列表
                variables = formula.get('variables', [])
                var_text = '; '.join([f"{v.get('symbol', '')}: {v.get('meaning', '')}" for v in variables])
                
                writer.writerow({
                    'Name': formula.get('name', ''),
                    'Expression': formula.get('expression', ''),
                    'Description': formula.get('description', ''),
                    'Variables': var_text,
                    'Theory': formula.get('theory', ''),
                    'Source': formula.get('source', '')
                })
        
        print(f"[INFO] 理论概述保存到: {theories_path}")
        print(f"[INFO] 概念保存到: {concepts_path}")
        print(f"[INFO] 公式保存到: {formulas_path}")
        
        # 也保存为JSON格式以保留完整数据结构
        json_path = os.path.join(output_dir, 'generated_theories.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'theories': self.theories,
                'concepts': self.concepts,
                'formulas': self.formulas
            }, f, ensure_ascii=False, indent=2)
            
        print(f"[INFO] 完整数据保存到: {json_path}")

    def generate_quantum_interpretation_list(self, max_theories=10):
        """
        生成量子力学诠释理论的列表
        
        Args:
            max_theories: 生成理论的最大数量
            
        Returns:
            List[str]: 理论名称列表
        """
        prompt = f"""
        请列出所有主要的量子力学诠释理论。
        这些诠释理论尝试解释量子力学的基本原理和现象。
        请尽可能全面，不仅包括主流诠释，还包括非主流和历史上的诠释理论。
        只需提供理论名称列表，每行一个理论名称。
        最多列出{max_theories}个理论。
        """
        
        # 请求LLM生成理论列表
        try:
            response = self.llm.query(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            # 解析响应，提取每一行作为一个理论名称
            theory_names = [name.strip() for name in response.split('\n') if name.strip()]
            
            # 如果名称包含数字编号，则移除
            theory_names = [re.sub(r'^\d+\.\s*', '', name) for name in theory_names]
            
            # 移除可能的重复项
            theory_names = list(dict.fromkeys(theory_names))
            
            # 限制数量
            theory_names = theory_names[:max_theories]
            
            print(f"[INFO] 找到 {len(theory_names)} 个量子诠释理论")
            return theory_names
            
        except Exception as e:
            print(f"[ERROR] 生成理论列表失败: {str(e)}")
            # 返回一些默认理论
            default_theories = [
                "哥本哈根诠释", "多世界诠释", "玻姆力学", "关系量子力学", 
                "客观坍缩理论", "量子贝叶斯主义", "整体主义诠释"
            ]
            return default_theories

    def generate_quantum_interpretation_details(self, theory_name):
        """
        获取特定量子诠释理论的详细信息
        
        Args:
            theory_name: 理论名称
            
        Returns:
            Dict: 理论详细信息
        """
        # 直接使用LLM接口的方法生成已有理论
        return self.llm.generate_existing_theory(theory_name) 