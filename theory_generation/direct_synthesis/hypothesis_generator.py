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
            # 确保metadata字段存在
            if "metadata" not in new_hypothesis:
                new_hypothesis["metadata"] = {}
            
            new_hypothesis["metadata"]["generation_info"] = {
                "source": "direct_synthesis",
                "contradiction_base": {
                    "theory1": theory1,
                    "theory2": theory2,
                },
                "generation_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
        构建生成新假说的提示 (Schema v2.1)
        
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
### Contradiction {i}: {dimension}
- **{theory1}'s Position**: {theory1_pos}
- **{theory2}'s Position**: {theory2_pos}
- **Core Tension**: {core}
"""
        
        # 构建完整提示
        prompt = f"""
# TASK
As a theoretical physicist and philosopher of science, your task is to invent a novel quantum theory. This new theory must resolve or transcend the core contradictions identified between **{theory1}** and **{theory2}**.

# CONTRADICTION ANALYSIS
{contradictions_text}

# INSTRUCTIONS
Based on the analysis, construct a new theory. Your output **MUST** be a single, valid JSON object that strictly adheres to the "Quantum Theory Schema v2.1" provided below.

## Quantum Theory Schema v2.1

```json
{{
  "name": "string (A clear, descriptive name for the new theory)",
  "metadata": {{
    "uid": "string (A unique identifier, e.g., 'THEORY-YYYYMMDD-01')",
    "schema_version": "2.1",
    "author": "AI Physicist",
    "tags": ["array of strings (e.g., 'quantum interpretation', 'stochastic dynamics', 'non-local')"],
    "lineage": {{
      "method": "Direct Synthesis from Contradiction",
      "parents": ["{theory1}", "{theory2}"],
      "inspiration": "string (Briefly describe how the new theory resolves the core tensions)"
    }}
  }},
  "mathematical_relation_to_sqm": "string (Choose one: 'Interpretation', 'Modification', 'Extension')",
  "summary": "string (A concise, one-paragraph summary of the theory's main idea)",
  "core_principles": {{
    "ontological_commitments": "string (What does the theory claim exists fundamentally? e.g., wave function, particles, information)",
    "epistemological_stances": "string (What can be known and how? Role of the observer.)",
    "key_postulates": ["array of strings (List the core axioms or postulates of the theory)"]
  }},
  "formalism": {{
    "mathematical_objects": "string (List the primary mathematical objects, e.g., Hilbert space, configuration space)",
    "governing_equations": ["array of strings (Provide key equations in LaTeX, e.g., evolution equation, collapse dynamics)"],
    "comparison_with_sqm": {{
      "agreements": "string (What parts of SQM's formalism are retained?)",
      "modifications": "string (What parts are changed? e.g., adding non-linear terms to Schrödinger's equation)",
      "extensions": "string (What new mathematical structures are added?)"
    }}
  }},
  "predictions_and_verifiability": {{
    "reproduces_sqm_predictions": "string (Explain how the theory reproduces standard predictions in the appropriate limit)",
    "deviations_from_sqm": [
      {{
        "prediction_name": "string (e.g., 'Modified Double-Slit Interference')",
        "description": "string (Describe the new, testable prediction)",
        "mathematical_derivation": "string (Briefly explain how this deviation is derived from the new formalism)",
        "experimental_setup": "string (Suggest a feasible experimental setup to test this prediction)"
      }}
    ],
    "unanswered_questions": "string (What questions does this new theory open up?)"
  }}
}}
```

**CRITICAL**: 
- Populate every field of the JSON schema with detailed, scientifically-grounded content.
- Ensure all LaTeX expressions are correctly escaped for JSON (e.g., use `\\\\` for a single backslash).
- The `lineage.inspiration` field is crucial: explicitly state how your new theory's postulates resolve the specific contradictions listed.
- Do not add any text or explanation outside the single JSON object.
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
