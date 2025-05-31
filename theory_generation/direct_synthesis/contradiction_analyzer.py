#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
矛盾分析器

分析不同量子诠释理论之间的矛盾点，
提取结构化的矛盾信息，为直接合成新理论提供基础。
"""

import os
import json
from typing import List, Dict, Any, Tuple

class ContradictionAnalyzer:
    """分析量子诠释理论之间的矛盾点"""
    
    def __init__(self, llm_interface):
        """
        初始化矛盾分析器
        
        Args:
            llm_interface: LLM接口实例
        """
        self.llm = llm_interface
        self.theories = {}  # 理论名称 -> 理论详情
        self.contradictions = []  # 存储找到的矛盾点
        
        # 定义比较维度
        self.key_dimensions = [
            "wave_function_reality",      # 波函数的实在性
            "measurement_process",         # 测量过程的本质
            "observer_role",               # 观察者的角色
            "determinism",                 # 确定性问题
            "non_locality",                # 非局域性解释
            "mathematical_formalism",      # 数学形式的解释
            "ontological_status",          # 本体论地位
            "quantum_classical_boundary"   # 量子经典边界
        ]
    
    def load_theories(self, theories_dir: str) -> None:
        """
        从目录加载理论详情
        
        Args:
            theories_dir: 理论JSON文件所在目录
        """
        if not os.path.exists(theories_dir):
            print(f"[ERROR] 目录不存在: {theories_dir}")
            return
            
        # 加载所有理论JSON文件
        theory_files = [f for f in os.listdir(theories_dir) if f.endswith('.json')]
        
        for file_name in theory_files:
            file_path = os.path.join(theories_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    theory_data = json.load(f)
                    
                if isinstance(theory_data, list):
                    # 处理包含多个理论的文件
                    for theory in theory_data:
                        theory_name = theory.get('theory_name', theory.get('name', ''))
                        if theory_name:
                            self.theories[theory_name] = theory
                            print(f"[INFO] 已加载理论: {theory_name}")
                else:
                    # 处理单个理论文件
                    theory_name = theory_data.get('theory_name', theory_data.get('name', ''))
                    if theory_name:
                        self.theories[theory_name] = theory_data
                        print(f"[INFO] 已加载理论: {theory_name}")
            except Exception as e:
                print(f"[ERROR] 加载理论文件 {file_name} 失败: {str(e)}")
        
        print(f"[INFO] 共加载了 {len(self.theories)} 个量子诠释理论")
    
    async def find_contradictions(self, theory1_name: str, theory2_name: str) -> Dict:
        """
        分析两个理论的矛盾点
        
        Args:
            theory1_name: 第一个理论名称
            theory2_name: 第二个理论名称
            
        Returns:
            Dict: 包含矛盾点的分析结果
        """
        # 获取理论详情
        theory1 = self.theories.get(theory1_name)
        theory2 = self.theories.get(theory2_name)
        
        if not theory1 or not theory2:
            missing = []
            if not theory1:
                missing.append(theory1_name)
            if not theory2:
                missing.append(theory2_name)
            print(f"[ERROR] 未找到理论: {', '.join(missing)}")
            return {"error": f"未找到理论: {', '.join(missing)}"}
        
        print(f"[INFO] 分析理论矛盾点: {theory1_name} vs {theory2_name}")
        
        # 构建提示
        prompt = self._build_analysis_prompt(theory1, theory2)
        
        # 调用LLM分析矛盾点
        response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2  # 使用较低的温度以获得确定性结果
        )
        
        # 解析结果
        try:
            analysis_result = self.llm.extract_json(response)
            
            if not analysis_result:
                print(f"[ERROR] 无法解析LLM响应为有效JSON")
                return {"error": "无法解析响应", "raw_response": response}
            
            # 添加理论信息
            analysis_result["theory1"] = theory1_name
            analysis_result["theory2"] = theory2_name
            
            # 保存结果
            self.contradictions.append(analysis_result)
            
            print(f"[INFO] 找到 {len(analysis_result.get('contradictions', []))} 个矛盾点")
            return analysis_result
        except Exception as e:
            print(f"[ERROR] 解析分析结果失败: {str(e)}")
            return {"error": str(e), "theory1": theory1_name, "theory2": theory2_name}
    
    def _build_analysis_prompt(self, theory1: Dict, theory2: Dict) -> str:
        """
        构建矛盾分析提示
        
        Args:
            theory1: 第一个理论详情
            theory2: 第二个理论详情
            
        Returns:
            str: 分析提示
        """
        # 提取理论名称
        theory1_name = theory1.get('theory_name', theory1.get('name', '未知理论1'))
        theory2_name = theory2.get('theory_name', theory2.get('name', '未知理论2'))
        
        # 提取核心内容
        theory1_core = theory1.get('core_principles', theory1.get('description', ''))
        theory2_core = theory2.get('core_principles', theory2.get('description', ''))
        
        # 构建完整提示
        prompt = f"""
        作为一位量子物理学的专家，请深入分析以下两种量子诠释理论之间的关键矛盾点。
        
        理论1: {theory1_name}
        核心原理: {theory1_core}
        
        理论2: {theory2_name}
        核心原理: {theory2_core}
        
        请分析这两种诠释理论之间的本质矛盾，特别关注以下方面：
        1. 波函数的本体论地位（实在性）
        2. 测量问题的处理
        3. 决定论与确定性
        4. 观察者的角色
        5. 非局域性的解释
        
        但不要局限于这些预设维度，请找出理论间真正的关键分歧。
        
        以JSON格式返回结果，格式如下:
        {{
          "theory1": "{theory1_name}",
          "theory2": "{theory2_name}",
          "contradictions": [
            {{
              "dimension": "矛盾维度名称",
              "theory1_position": "理论1在此维度的具体立场",
              "theory2_position": "理论2在此维度的具体立场",
              "core_tension": "矛盾的核心本质",
              "philosophical_implications": "该矛盾的哲学含义",
              "importance_score": 数值(1-10)
            }}
          ],
          "summary": "对两个理论矛盾的整体概括"
        }}
        
        请确保你的分析深入、准确，并精确捕捉两个理论之间的真正分歧，而不是表面差异。
        """
        return prompt
    
    def save_analyses(self, output_path: str) -> None:
        """
        保存所有矛盾分析结果
        
        Args:
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.contradictions, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 矛盾分析结果已保存到: {output_path}")
        except Exception as e:
            print(f"[ERROR] 保存矛盾分析结果失败: {str(e)}")
