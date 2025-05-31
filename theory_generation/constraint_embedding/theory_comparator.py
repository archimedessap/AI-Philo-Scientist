#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
量子诠释理论比较器

分析不同量子诠释理论之间的矛盾点，提取结构化的差异信息。
"""

import os
import json
from typing import List, Dict, Any, Tuple

class TheoryComparator:
    """比较量子诠释理论，找出关键矛盾点"""
    
    def __init__(self, llm_interface):
        """
        初始化理论比较器
        
        Args:
            llm_interface: LLM接口实例
        """
        self.llm = llm_interface
        self.theories = {}  # 理论名称 -> 理论详情
        self.contradictions = []  # 存储找到的矛盾点
        
        # 定义比较维度
        self.comparison_dimensions = [
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
                    
                # 提取理论名称
                theory_name = theory_data.get('name', '')
                if theory_name:
                    self.theories[theory_name] = theory_data
                    print(f"[INFO] 已加载理论: {theory_name}")
            except Exception as e:
                print(f"[ERROR] 加载理论文件 {file_name} 失败: {str(e)}")
        
        print(f"[INFO] 共加载了 {len(self.theories)} 个量子诠释理论")
    
    async def compare_theories(self, theory1_name: str, theory2_name: str) -> Dict:
        """
        比较两个理论，找出矛盾点
        
        Args:
            theory1_name: 第一个理论名称
            theory2_name: 第二个理论名称
            
        Returns:
            Dict: 包含矛盾点的比较结果
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
            return {}
        
        print(f"[INFO] 比较理论: {theory1_name} vs {theory2_name}")
        
        # 构建提示
        prompt = self._build_comparison_prompt(theory1, theory2)
        
        # 调用LLM分析矛盾点
        response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2  # 使用较低的温度以获得确定性结果
        )
        
        # 解析结果
        try:
            comparison_result = self.llm.extract_json(response)
            
            # 保存结果
            comparison_result["theory1"] = theory1_name
            comparison_result["theory2"] = theory2_name
            self.contradictions.append(comparison_result)
            
            print(f"[INFO] 找到 {len(comparison_result.get('contradictions', []))} 个矛盾点")
            return comparison_result
        except Exception as e:
            print(f"[ERROR] 解析比较结果失败: {str(e)}")
            return {"error": str(e), "theory1": theory1_name, "theory2": theory2_name}
    
    async def compare_all_pairs(self, max_pairs: int = None) -> List[Dict]:
        """
        比较所有理论对
        
        Args:
            max_pairs: 最大比较对数，None表示全部比较
            
        Returns:
            List[Dict]: 所有比较结果
        """
        theory_names = list(self.theories.keys())
        pairs = []
        
        # 生成所有可能的理论对
        for i in range(len(theory_names)):
            for j in range(i+1, len(theory_names)):
                pairs.append((theory_names[i], theory_names[j]))
        
        # 限制对数
        if max_pairs and max_pairs < len(pairs):
            pairs = pairs[:max_pairs]
            
        print(f"[INFO] 将比较 {len(pairs)} 对理论")
        
        # 比较每一对理论
        for theory1, theory2 in pairs:
            await self.compare_theories(theory1, theory2)
        
        return self.contradictions
    
    def _build_comparison_prompt(self, theory1: Dict, theory2: Dict) -> str:
        """
        构建比较提示
        
        Args:
            theory1: 第一个理论详情
            theory2: 第二个理论详情
            
        Returns:
            str: 比较提示
        """
        prompt = f"""
        作为一位量子物理学的专家，请分析以下两种量子诠释理论之间的关键矛盾点。
        
        理论1: {theory1['name']}
        核心原理: {theory1.get('core_principles', '')}
        量子现象解释: {json.dumps(theory1.get('quantum_phenomena_explanation', {}), ensure_ascii=False)}
        
        理论2: {theory2['name']}
        核心原理: {theory2.get('core_principles', '')}
        量子现象解释: {json.dumps(theory2.get('quantum_phenomena_explanation', {}), ensure_ascii=False)}
        
        请找出并分析这两种诠释理论之间所有本质性的矛盾点。不要局限于预设的维度，而是深入分析理论在各个方面的差异，找出真正的本质矛盾。
        
        以JSON格式返回结果，格式如下:
        {{
          "theory1": "{theory1['name']}",
          "theory2": "{theory2['name']}",
          "contradictions": [
            {{
              "dimension": "发现的矛盾维度名称",
              "theory1_position": "理论1在此维度的立场",
              "theory2_position": "理论2在此维度的立场",
              "contradiction_nature": "矛盾的本质描述",
              "potential_resolution": "可能的解决或调和方式"
            }}
          ],
          "compatible_aspects": [
            {{
              "dimension": "两个理论兼容的维度",
              "compatibility_description": "详细说明兼容性"
            }}
          ]
        }}
        
        重要提示：请自由识别所有可能的矛盾点，而不仅限于常见维度。查找深层次的哲学和物理学上的矛盾，以及理论解释能力的差异。
        """
        return prompt
    
    def save_contradictions(self, output_path: str) -> None:
        """
        保存矛盾点分析结果
        
        Args:
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.contradictions, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 矛盾点分析结果已保存到: {output_path}")
        except Exception as e:
            print(f"[ERROR] 保存矛盾点分析结果失败: {str(e)}") 