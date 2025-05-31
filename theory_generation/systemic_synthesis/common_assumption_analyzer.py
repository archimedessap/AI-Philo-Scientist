#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
共同假设分析器

分析多个量子理论的共同依赖假设与概念网络，
识别隐含缺陷和框架限制。
"""

import os
import json
import time
from typing import List, Dict, Any, Optional

class CommonAssumptionAnalyzer:
    """多理论共同假设与概念网络分析器"""
    
    def __init__(self, llm_interface):
        """
        初始化共同假设分析器
        
        Args:
            llm_interface: LLM接口实例
        """
        self.llm = llm_interface
        self.theories = {}
        self.analysis_results = []
    
    def load_theories(self, theories_data: Dict[str, Dict]) -> None:
        """
        加载理论数据
        
        Args:
            theories_data: 理论名称到理论数据的映射
        """
        self.theories = theories_data
        print(f"[INFO] 已加载 {len(self.theories)} 个理论用于共同假设分析")
    
    async def analyze_common_assumptions(self, theory_subset: Optional[List[str]] = None) -> Dict:
        """
        分析多个理论的共同假设和概念网络
        
        Args:
            theory_subset: 要分析的理论子集，如果为None则分析所有理论
            
        Returns:
            Dict: 共同假设分析结果
        """
        # 确定要分析的理论
        if theory_subset is None:
            selected_theories = list(self.theories.keys())
        else:
            selected_theories = [name for name in theory_subset if name in self.theories]
        
        if len(selected_theories) < 2:
            return {"error": "至少需要2个理论进行共同假设分析"}
        
        print(f"[INFO] 正在分析 {len(selected_theories)} 个理论的共同假设...")
        
        # 构建分析提示
        prompt = self._build_analysis_prompt(selected_theories)
        
        # 调用LLM进行分析
        response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        # 解析结果
        try:
            analysis_result = self.llm.extract_json(response)
            
            if not analysis_result:
                print(f"[ERROR] 无法解析LLM响应为有效JSON")
                return {"error": "无法解析响应", "raw_response": response}
            
            # 添加元数据
            analysis_result["analyzed_theories"] = selected_theories
            analysis_result["analysis_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            analysis_result["theory_count"] = len(selected_theories)
            
            # 保存分析结果
            self.analysis_results.append(analysis_result)
            
            print(f"[INFO] 成功完成共同假设分析")
            return analysis_result
            
        except Exception as e:
            print(f"[ERROR] 共同假设分析失败: {str(e)}")
            return {"error": str(e), "theories": selected_theories}
    
    def _build_analysis_prompt(self, theory_names: List[str]) -> str:
        """
        构建共同假设分析的提示
        
        Args:
            theory_names: 要分析的理论名称列表
            
        Returns:
            str: 分析提示
        """
        # 构建理论信息
        theories_info = ""
        for name in theory_names:
            theory = self.theories[name]
            theories_info += f"\n## {name}\n"
            theories_info += f"**Summary**: {theory.get('summary', 'No summary')}\n"
            theories_info += f"**Philosophy**: {json.dumps(theory.get('philosophy', {}), indent=2)}\n"
            theories_info += f"**Parameters**: {json.dumps(theory.get('parameters', {}), indent=2)}\n"
            theories_info += f"**Formalism**: {json.dumps(theory.get('formalism', {}), indent=2)}\n"
            theories_info += f"**Semantics**: {json.dumps(theory.get('semantics', {}), indent=2)}\n"
        
        prompt = f"""
# Task: Multi-Theory Common Assumption Analysis

You are a theoretical physicist and philosopher of science specializing in quantum mechanics interpretations. Your task is to analyze the following quantum theories to identify their **common assumptions, shared conceptual networks, and systemic limitations**.

# Quantum Theories to Analyze
{theories_info}

# Analysis Requirements

## 1. Common Conceptual Foundation
Identify the **fundamental assumptions and concepts** that ALL or MOST of these theories share, including:
- Basic ontological commitments (what exists)
- Mathematical formalisms they all rely on
- Measurement concepts they all accept
- Causality assumptions
- Space-time assumptions
- Information-theoretic assumptions

## 2. Shared Limitations Analysis
For each identified common assumption, analyze:
- **Logical gaps**: Are there internal inconsistencies?
- **Physical limitations**: Are there unexplained physical phenomena?
- **Conceptual restrictions**: What possibilities are artificially excluded?
- **Framework boundaries**: What questions cannot be asked within this shared framework?

## 3. Systemic Deficiency Identification
Identify **systemic deficiencies** that emerge from their shared foundation:
- What fundamental questions do ALL these theories fail to address?
- What experimental predictions do they all struggle with?
- What conceptual paradoxes do they all inherit?
- What mathematical limitations do they all share?

# Output Format
Provide your analysis as a JSON object with the following structure:

{{
  "common_assumptions": {{
    "ontological": ["assumption 1", "assumption 2", ...],
    "mathematical": ["formalism 1", "formalism 2", ...],
    "measurement": ["concept 1", "concept 2", ...],
    "causality": ["principle 1", "principle 2", ...],
    "spacetime": ["assumption 1", "assumption 2", ...],
    "information": ["principle 1", "principle 2", ...]
  }},
  "shared_limitations": [
    {{
      "assumption": "specific common assumption",
      "logical_gaps": "description of logical inconsistencies",
      "physical_limitations": "unexplained phenomena or restrictions",
      "conceptual_restrictions": "artificially excluded possibilities",
      "framework_boundaries": "unaskable questions"
    }}
  ],
  "systemic_deficiencies": {{
    "unanswered_questions": ["fundamental question 1", "question 2", ...],
    "prediction_failures": ["experimental domain 1", "domain 2", ...],
    "inherited_paradoxes": ["paradox 1", "paradox 2", ...],
    "mathematical_constraints": ["limitation 1", "limitation 2", ...]
  }},
  "breakthrough_opportunities": {{
    "conceptual_reorganization": ["opportunity 1", "opportunity 2", ...],
    "framework_transcendence": ["possibility 1", "possibility 2", ...],
    "new_mathematical_tools": ["tool 1", "tool 2", ...]
  }},
  "meta_analysis": {{
    "analysis_confidence": "high/medium/low",
    "most_critical_deficiency": "single most important shared limitation",
    "revolutionary_potential": "assessment of potential for breakthrough"
  }}
}}

Focus on identifying **deep, structural limitations** rather than surface-level differences. Look for assumptions so fundamental that they are rarely questioned but may be the key to revolutionary breakthroughs.
"""
        return prompt
    
    async def identify_breakthrough_targets(self, analysis_result: Dict) -> Dict:
        """
        基于共同假设分析，识别突破性目标
        
        Args:
            analysis_result: 共同假设分析结果
            
        Returns:
            Dict: 突破性目标识别结果
        """
        if "error" in analysis_result:
            return {"error": "无法基于错误的分析结果识别突破目标"}
        
        print(f"[INFO] 正在识别突破性目标...")
        
        # 构建突破目标识别提示
        prompt = self._build_breakthrough_prompt(analysis_result)
        
        # 调用LLM
        response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8  # 更高的创新性
        )
        
        # 解析结果
        try:
            breakthrough_result = self.llm.extract_json(response)
            
            if not breakthrough_result:
                return {"error": "无法解析突破目标识别响应"}
            
            # 添加元数据
            breakthrough_result["based_on_analysis"] = analysis_result.get("analyzed_theories", [])
            breakthrough_result["identification_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            return breakthrough_result
            
        except Exception as e:
            print(f"[ERROR] 突破目标识别失败: {str(e)}")
            return {"error": str(e)}
    
    def _build_breakthrough_prompt(self, analysis_result: Dict) -> str:
        """
        构建突破目标识别提示
        
        Args:
            analysis_result: 共同假设分析结果
            
        Returns:
            str: 突破目标识别提示
        """
        # 提取关键信息
        systemic_deficiencies = analysis_result.get("systemic_deficiencies", {})
        shared_limitations = analysis_result.get("shared_limitations", [])
        most_critical = analysis_result.get("meta_analysis", {}).get("most_critical_deficiency", "")
        
        prompt = f"""
# Task: Breakthrough Target Identification

Based on the following analysis of common assumptions and systemic deficiencies in quantum theories, identify specific **breakthrough targets** for revolutionary theory development.

## Previous Analysis Results

### Most Critical Deficiency
{most_critical}

### Systemic Deficiencies
{json.dumps(systemic_deficiencies, indent=2)}

### Shared Limitations (first 3)
{json.dumps(shared_limitations[:3], indent=2)}

## Target Identification Requirements

Identify **3-5 specific breakthrough targets** that could lead to revolutionary new quantum theories. For each target, specify:

1. **Conceptual Reorganization Opportunities**: How existing concepts could be radically reorganized
2. **Framework Transcendence Strategies**: Specific ways to break current theoretical boundaries  
3. **Novel Mathematical Formulations**: New mathematical tools or structures needed
4. **Experimental Accessibility**: How the breakthrough could be tested

# Output Format

{{
  "breakthrough_targets": [
    {{
      "target_name": "concise name for the breakthrough",
      "target_description": "detailed description of what needs to be breakthrough",
      "current_limitation": "specific limitation this target addresses",
      "conceptual_reorganization": {{
        "key_concepts_to_modify": ["concept 1", "concept 2", ...],
        "reorganization_strategy": "how to reorganize these concepts",
        "expected_outcome": "what new understanding emerges"
      }},
      "framework_transcendence": {{
        "boundaries_to_break": ["boundary 1", "boundary 2", ...],
        "transcendence_method": "specific approach to break boundaries",
        "new_framework_features": ["feature 1", "feature 2", ...]
      }},
      "mathematical_innovation": {{
        "required_tools": ["tool 1", "tool 2", ...],
        "formalism_changes": "how mathematical formalism needs to change",
        "computational_implications": "computational consequences"
      }},
      "experimental_accessibility": {{
        "testable_predictions": ["prediction 1", "prediction 2", ...],
        "required_experiments": ["experiment type 1", "type 2", ...],
        "technological_requirements": ["requirement 1", "requirement 2", ...]
      }},
      "revolutionary_potential": "high/medium/low",
      "implementation_difficulty": "high/medium/low"
    }}
  ],
  "priority_ranking": [
    {{
      "target_name": "name matching above",
      "priority_score": "1-10 scale",
      "justification": "why this target has this priority"
    }}
  ],
  "synthesis_strategy": {{
    "multi_target_approach": "how to tackle multiple targets simultaneously",
    "sequential_dependencies": "which targets depend on others",
    "resource_allocation": "how to prioritize resources"
  }}
}}

Focus on targets that are **achievable but revolutionary** - not incremental improvements, but fundamental paradigm shifts that address the deepest structural limitations identified in the analysis.
"""
        return prompt
    
    def save_analysis_results(self, output_dir: str) -> None:
        """
        保存共同假设分析结果
        
        Args:
            output_dir: 输出目录
        """
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存所有分析结果
        if self.analysis_results:
            all_file_path = os.path.join(output_dir, "common_assumption_analyses.json")
            try:
                with open(all_file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.analysis_results, f, ensure_ascii=False, indent=2)
                print(f"[INFO] 共同假设分析结果已保存到: {all_file_path}")
            except Exception as e:
                print(f"[ERROR] 保存分析结果失败: {str(e)}")
        
        # 保存每个分析到单独文件
        for i, result in enumerate(self.analysis_results):
            theories_names = "_".join(result.get("analyzed_theories", [])[:3])
            if len(result.get("analyzed_theories", [])) > 3:
                theories_names += "_and_others"
            
            filename = f"analysis_{theories_names}_{i+1}.json"
            file_path = os.path.join(output_dir, filename)
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"[INFO] 分析结果已保存到: {file_path}")
            except Exception as e:
                print(f"[ERROR] 保存分析结果失败: {str(e)}") 