#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
假说生成器

基于理论矛盾分析，直接利用LLM生成新的量子理论假说，
通过矛盾放松和概念综合创建创新的理论解释。
"""

import os
import json
import time
from typing import List, Dict, Any, Optional

class HypothesisGenerator:
    """从理论矛盾中生成新的理论假说"""
    
    def __init__(self, llm_interface):
        """
        初始化假说生成器
        
        Args:
            llm_interface: LLM接口实例
        """
        self.llm = llm_interface
        self.generated_hypotheses = []
    
    async def generate_from_contradiction(self, contradiction: Dict, generation_params: Optional[Dict] = None) -> Dict:
        """
        基于矛盾点生成新的理论假说
        
        Args:
            contradiction: 矛盾分析结果
            generation_params: 生成参数，控制创新度等
            
        Returns:
            Dict: 生成的新理论假说
        """
        # 设置默认参数
        if generation_params is None:
            generation_params = {
                "creativity_level": 0.7,  # 0.0-1.0, 值越高创新性越强
                "mathematical_rigor": 0.8,  # 0.0-1.0, 数学严谨程度
                "philosophical_depth": 0.6,  # 0.0-1.0, 哲学深度
                "emphasis_on_testability": 0.5,  # 0.0-1.0, 强调可测试性
            }
        
        # 使用的创新度影响温度参数
        temperature = 0.5 + generation_params["creativity_level"] * 0.5
        
        # 提取关键信息
        theory1 = contradiction.get("theory1", "理论1")
        theory2 = contradiction.get("theory2", "理论2")
        contradictions = contradiction.get("contradictions", [])
        
        # 如果没有矛盾点数据，返回错误
        if not contradictions:
            print(f"[ERROR] 没有矛盾点数据可供生成新假说")
            return {"error": "没有矛盾点数据"}
        
        # 构建提示
        prompt = self._build_generation_prompt(theory1, theory2, contradictions, generation_params)
        
        # 调用LLM生成新假说
        print(f"[INFO] 正在基于 {theory1} 和 {theory2} 的矛盾生成新假说...")
        response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        
        # 解析结果
        try:
            new_hypothesis = self.llm.extract_json(response)
            
            if not new_hypothesis:
                print(f"[ERROR] 无法解析LLM响应为有效JSON")
                return {"error": "无法解析响应", "raw_response": response}
            
            # 添加元数据
            new_hypothesis["generated_from"] = {
                "theory1": theory1,
                "theory2": theory2,
                "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "generation_parameters": generation_params
            }
            
            # 保存生成的假说
            self.generated_hypotheses.append(new_hypothesis)
            
            print(f"[INFO] 成功生成新假说: {new_hypothesis.get('name', '未命名理论')}")
            return new_hypothesis
        except Exception as e:
            print(f"[ERROR] 生成新假说失败: {str(e)}")
            return {"error": str(e), "theories": [theory1, theory2]}
    
    def _build_generation_prompt(self, theory1: str, theory2: str, contradictions: List[Dict], 
                               params: Dict) -> str:
        """
        构建生成新假说的提示
        
        Args:
            theory1: 第一个理论名称
            theory2: 第二个理论名称
            contradictions: 矛盾点列表
            params: 生成参数
            
        Returns:
            str: 生成提示
        """
        # 提取关键矛盾，按重要性排序
        sorted_contradictions = sorted(
            contradictions, 
            key=lambda x: x.get("importance_score", 5),
            reverse=True
        )
        
        # 只使用最重要的3个矛盾点
        top_contradictions = sorted_contradictions[:3]
        
        # 构建矛盾描述
        contradictions_text = ""
        for i, c in enumerate(top_contradictions, 1):
            dimension = c.get("dimension", f"Dimension {i}")
            theory1_pos = c.get("theory1_position", "Not specified")
            theory2_pos = c.get("theory2_position", "Not specified")
            core = c.get("core_tension", "")
            
            contradictions_text += f"""
Contradiction {i}: {dimension}
- {theory1}'s position: {theory1_pos}
- {theory2}'s position: {theory2_pos}
- Core tension: {core}
"""
        
        # 构建完整提示
        prompt = f"""
# Task
You are a theorist in the interdisciplinary field of quantum mechanics and philosophy. Based on the given analysis of contradictions between theories, propose a new interpretation of quantum mechanics. The theory should:

1. Have clear ontological or epistemological claims;
2. Provide a precise mathematical formulation (Hamiltonian or evolution equations), noting any "new physics" parameters (if any);
3. Include at least one state update/measurement rule (can be the Born rule or a modification);
4. Reconcile or transcend the contradictions described below.
5. Include detailed explanations of the meanings of all symbols and terms used (semantics).

# Contradiction Analysis
Theory comparison: {theory1} vs {theory2}

{contradictions_text}

# Evaluation Criteria
1. Philosophical depth: {params["philosophical_depth"] * 10}/10
2. Mathematical rigor: {params["mathematical_rigor"] * 10}/10
3. Innovation level: {params["creativity_level"] * 10}/10

# Output Format
Output only a JSON object, without code blocks or any additional text. Follow this exact format:

{{
  "name": "<theory name>",
  "summary": "<brief summary in less than 100 words>",
  "philosophy": {{
    "ontology": "<core ontological claims>",
    "measurement": "<view on measurement>"
  }},
  "parameters": {{
    "<symbol>": {{ "value": <number>, "unit": "<unit or empty string>", "role": "<brief explanation>" }},
    // If needed, add more parameters with the same structure
    // If no new parameters, use empty object {{}}
  }},
  "formalism": {{
    "hamiltonian": "<LaTeX equation or text>",
    "state_equation": "<LaTeX equation or text>",
    "measurement_rule": "<one sentence>"
  }},
  "semantics": {{
    "<symbol/term>": "<detailed explanation of what this symbol/term means in the theory>",
    "<symbol/term>": "<detailed explanation of what this symbol/term means in the theory>",
    // Add explanations for all important symbols in the Hamiltonian and state equations
    // Also explain any key concepts specific to this theory
    "overall_picture": "<concise description of how all the elements work together>"
  }}
}}

Ensure all mathematical expressions are properly escaped for JSON (e.g., \\\\ for a backslash in LaTeX).
Include detailed semantic explanations for all mathematical symbols and key concepts in your theory.
"""
        return prompt
    
    async def generate_multiple_hypotheses(self, contradiction: Dict, 
                                         variants_count: int = 3,
                                         diversity_level: float = 0.7) -> List[Dict]:
        """
        基于同一矛盾生成多个不同的假说变体
        
        Args:
            contradiction: 矛盾分析结果
            variants_count: 要生成的变体数量
            diversity_level: 变体多样性程度(0.0-1.0)
            
        Returns:
            List[Dict]: 生成的假说列表
        """
        generated_variants = []
        
        for i in range(variants_count):
            print(f"[INFO] 生成假说变体 {i+1}/{variants_count}")
            
            # 构建不同的生成参数，增加多样性
            params = {
                "creativity_level": 0.5 + (diversity_level * 0.5 * (i / variants_count)),
                "mathematical_rigor": 0.6 + 0.3 * ((i % 3) / 2),  # 循环变化数学严谨度
                "philosophical_depth": 0.5 + 0.4 * (((i+1) % 3) / 2),  # 循环变化哲学深度
                "emphasis_on_testability": 0.4 + 0.5 * (((i+2) % 3) / 2)  # 循环变化可测试性强调
            }
            
            # 生成一个变体
            variant = await self.generate_from_contradiction(contradiction, params)
            
            if "error" not in variant:
                # 添加变体标识
                variant["variant_id"] = i + 1
                
                # 将变体信息添加到理论名称中
                if "name" in variant:
                    original_name = variant["name"]
                    # 如果名称中已经包含变体信息则不添加
                    if f"(Variant {i+1})" not in original_name and f"（变种{i+1}）" not in original_name:
                        variant["name"] = f"{original_name} (Variant {i+1})"
                
                generated_variants.append(variant)
        
        return generated_variants
    
    def save_hypotheses(self, output_dir: str) -> None:
        """
        保存生成的假说到目录
        
        Args:
            output_dir: 输出目录
        """
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存每个假说到单独文件
        for i, hypothesis in enumerate(self.generated_hypotheses):
            theory_name = hypothesis.get("name", f"新理论_{i+1}")
            safe_name = theory_name.replace(" ", "_").replace("/", "_").lower()
            
            # 生成文件路径
            file_path = os.path.join(output_dir, f"{safe_name}.json")
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(hypothesis, f, ensure_ascii=False, indent=2)
                print(f"[INFO] 假说已保存到: {file_path}")
            except Exception as e:
                print(f"[ERROR] 保存假说失败: {str(e)}")
        
        # 保存所有假说到一个合并文件
        if self.generated_hypotheses:
            all_file_path = os.path.join(output_dir, "all_hypotheses.json")
            try:
                with open(all_file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.generated_hypotheses, f, ensure_ascii=False, indent=2)
                print(f"[INFO] 所有假说已合并保存到: {all_file_path}")
            except Exception as e:
                print(f"[ERROR] 保存合并假说失败: {str(e)}")
