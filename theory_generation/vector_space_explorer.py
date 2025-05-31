#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
理论向量空间探索器

在嵌入空间中表示理论矛盾点，通过放松约束生成新的理论可能性。
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import pickle
import random
import asyncio

class VectorSpaceExplorer:
    """理论向量空间探索器"""
    
    def __init__(self, llm_interface):
        """
        初始化向量空间探索器
        
        Args:
            llm_interface: LLM接口实例
        """
        self.llm = llm_interface
        self.theory_embeddings = {}  # 理论名称 -> 嵌入向量
        self.contradictions = []     # 理论之间的矛盾点
        self.dimensions_map = {}     # 维度名称 -> 向量空间维度指标
        self.generated_theories = [] # 生成的新理论
        
    def load_embeddings(self, embeddings_path: str) -> bool:
        """
        加载理论嵌入向量
        
        Args:
            embeddings_path: 嵌入向量文件路径
            
        Returns:
            bool: 是否成功加载
        """
        if not os.path.exists(embeddings_path):
            print(f"[ERROR] 文件不存在: {embeddings_path}")
            return False
            
        try:
            with open(embeddings_path, 'rb') as f:
                self.theory_embeddings = pickle.load(f)
                
            print(f"[INFO] 已加载 {len(self.theory_embeddings)} 个理论嵌入向量")
            return True
        except Exception as e:
            print(f"[ERROR] 加载嵌入向量失败: {str(e)}")
            return False
    
    def load_contradictions(self, contradictions_path: str) -> bool:
        """
        加载理论矛盾点数据
        
        Args:
            contradictions_path: 矛盾点数据文件路径
            
        Returns:
            bool: 是否成功加载
        """
        if not os.path.exists(contradictions_path):
            print(f"[ERROR] 文件不存在: {contradictions_path}")
            return False
            
        try:
            with open(contradictions_path, 'r', encoding='utf-8') as f:
                self.contradictions = json.load(f)
                
            print(f"[INFO] 已加载 {len(self.contradictions)} 组理论矛盾点数据")
            
            # 初始化维度映射
            self._initialize_dimension_mapping()
                
            return True
        except Exception as e:
            print(f"[ERROR] 加载矛盾点数据失败: {str(e)}")
            return False
    
    def _initialize_dimension_mapping(self) -> None:
        """初始化维度映射，将理论比较维度映射到向量空间的维度"""
        # 收集所有矛盾点维度
        all_dimensions = set()
        for comparison in self.contradictions:
            for contradiction in comparison.get('contradictions', []):
                all_dimensions.add(contradiction.get('dimension', ''))
        
        # 创建维度映射
        self.dimensions_map = {dim: i for i, dim in enumerate(all_dimensions)}
        print(f"[INFO] 已映射 {len(self.dimensions_map)} 个理论维度到向量空间")
    
    def _get_contradiction_vector(self, contradiction: Dict) -> np.ndarray:
        """
        获取表示矛盾点的向量
        
        Args:
            contradiction: 矛盾点数据
            
        Returns:
            np.ndarray: 矛盾向量
        """
        # 获取理论嵌入向量
        theory1_name = contradiction.get('theory1', '')
        theory2_name = contradiction.get('theory2', '')
        
        theory1_vector = self.theory_embeddings.get(theory1_name)
        theory2_vector = self.theory_embeddings.get(theory2_name)
        
        if theory1_vector is None or theory2_vector is None:
            return None
            
        # 计算差异向量
        difference_vector = theory2_vector - theory1_vector
        
        return difference_vector
    
    async def generate_theory_variants(self, num_variants: int = 5) -> List[Dict]:
        """
        生成理论变体
        
        Args:
            num_variants: 每对理论生成的变体数量
            
        Returns:
            List[Dict]: 生成的理论变体
        """
        results = []
        
        for comparison in self.contradictions:
            theory1_name = comparison.get('theory1', '')
            theory2_name = comparison.get('theory2', '')
            
            # 获取理论向量
            if theory1_name not in self.theory_embeddings or theory2_name not in self.theory_embeddings:
                print(f"[WARN] 缺少理论嵌入: {theory1_name} 或 {theory2_name}")
                continue
                
            theory1_vec = self.theory_embeddings[theory1_name]
            theory2_vec = self.theory_embeddings[theory2_name]
            
            print(f"[INFO] 为理论对 {theory1_name} 和 {theory2_name} 生成变体")
            
            # 为每个矛盾点生成变体
            for _ in range(num_variants):
                # 选择放松策略
                strategy = random.choice(['interpolation', 'orthogonal', 'boundary_extension'])
                
                if strategy == 'interpolation':
                    # 插值策略：在两个理论之间创建中间点，但允许超出范围
                    alpha = random.uniform(-0.5, 1.5)  # 允许超出原始理论范围
                    new_vector = alpha * theory1_vec + (1 - alpha) * theory2_vec
                    strategy_desc = f"插值策略 (alpha={alpha:.2f})"
                    
                elif strategy == 'orthogonal':
                    # 正交扩展：添加垂直于差异向量的分量
                    diff_vec = theory2_vec - theory1_vec
                    # 创建随机正交向量
                    ortho_vec = np.random.randn(len(theory1_vec))
                    # 确保正交
                    ortho_vec = ortho_vec - np.dot(ortho_vec, diff_vec) * diff_vec / np.dot(diff_vec, diff_vec)
                    # 归一化
                    ortho_vec = ortho_vec / np.linalg.norm(ortho_vec)
                    
                    strength = random.uniform(0.1, 0.5)
                    base_vec = (theory1_vec + theory2_vec) / 2  # 使用中点作为基础
                    new_vector = base_vec + strength * ortho_vec
                    strategy_desc = f"正交扩展策略 (strength={strength:.2f})"
                    
                else:  # boundary_extension
                    # 边界扩展：沿着差异向量方向扩展边界
                    diff_vec = theory2_vec - theory1_vec
                    extension = random.uniform(1.2, 2.0)  # 扩展倍数
                    if random.choice([True, False]):
                        new_vector = theory1_vec - (extension - 1) * diff_vec  # 向theory1方向扩展
                    else:
                        new_vector = theory2_vec + (extension - 1) * diff_vec  # 向theory2方向扩展
                    strategy_desc = f"边界扩展策略 (extension={extension:.2f})"
                
                # 归一化新向量
                new_vector = new_vector / np.linalg.norm(new_vector)
                
                # 生成理论描述
                theory_desc = await self._vector_to_theory(
                    new_vector, 
                    theory1_name, 
                    theory2_name, 
                    comparison.get('contradictions', []),
                    strategy_desc
                )
                
                if theory_desc:
                    theory_desc["original_theories"] = [theory1_name, theory2_name]
                    theory_desc["generation_strategy"] = strategy_desc
                    results.append(theory_desc)
                    self.generated_theories.append(theory_desc)
        
        print(f"[INFO] 共生成 {len(results)} 个新理论变体")
        return results
    
    async def _vector_to_theory(self, 
                               vector: np.ndarray, 
                               theory1_name: str, 
                               theory2_name: str,
                               contradictions: List[Dict],
                               strategy_desc: str) -> Optional[Dict]:
        """
        将向量转换为理论描述
        
        Args:
            vector: 理论向量
            theory1_name: 第一个原始理论名称
            theory2_name: 第二个原始理论名称
            contradictions: 矛盾点列表
            strategy_desc: 生成策略描述
            
        Returns:
            Optional[Dict]: 生成的理论描述，失败则返回None
        """
        # 准备矛盾点描述
        contradiction_text = ""
        for i, c in enumerate(contradictions):
            contradiction_text += f"""
            矛盾点 {i+1}:
            - 维度: {c.get('dimension', '')}
            - {theory1_name} 的立场: {c.get('theory1_position', '')}
            - {theory2_name} 的立场: {c.get('theory2_position', '')}
            - 矛盾本质: {c.get('contradiction_nature', '')}
            """
        
        # 构建提示
        prompt = f"""
        作为量子物理学的理论创新者，你将创建一个新的量子诠释理论。这个理论来自于向量空间中对现有理论的扩展。

        原始理论:
        1. {theory1_name}
        2. {theory2_name}

        这些理论之间的主要矛盾点:
        {contradiction_text}

        生成策略: {strategy_desc}

        你的任务是创建一个新的量子诠释理论，它解决或重新概念化上述矛盾点。这个理论应该:
        1. 有自己独特的哲学立场
        2. 提供对量子现象的新解释
        3. 在数学上有一致性
        4. 包含创新性思想，超越原始理论的简单组合

        请以JSON格式返回新理论:
        {{
          "name": "新理论名称",
          "core_principles": "核心原理简要描述",
          "detailed_description": "详细描述，包括如何处理关键矛盾点",
          "quantum_phenomena_explanation": {{
            "wave_function_collapse": "这个理论如何解释波函数坍缩",
            "measurement_problem": "这个理论如何解释测量问题",
            "non_locality": "这个理论如何解释非局域性"
          }},
          "philosophical_stance": "理论的哲学立场",
          "mathematical_formulation": "简要的数学表述",
          "novel_predictions": "该理论可能做出的新预测"
        }}

        确保创建的理论在物理学上合理，并真正提供了原始理论矛盾点的新视角。
        """
        
        # 调用LLM生成理论描述
        try:
            response = await self.llm.query_async(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  
                model_source=model_source,
                model_name=model_name
            )
            
            # 解析结果
            theory_desc = self.llm.extract_json(response)
            
            # 添加向量信息
            if theory_desc:
                # 仅存储向量的哈希值，完整向量过大
                theory_desc["vector_hash"] = hash(tuple(vector[:10].tolist()))
                return theory_desc
            
            return None
        except Exception as e:
            print(f"[ERROR] 生成理论描述失败: {str(e)}")
            return None
    
    def save_generated_theories(self, output_path: str) -> None:
        """
        保存生成的理论
        
        Args:
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.generated_theories, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 生成的理论已保存到: {output_path}")
        except Exception as e:
            print(f"[ERROR] 保存生成的理论失败: {str(e)}")

    def cosine_similarity(self, vec1, vec2):
        """计算两个向量之间的余弦相似度"""
        # 转换为numpy数组以确保兼容性
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot_product / (norm_a * norm_b)

    def find_nearest_theories(self, vector, k=3):
        """找到与给定向量最相近的k个理论"""
        if not self.theory_embeddings:
            print("[错误] 未加载理论嵌入向量")
            return []
        
        similarities = []
        for theory_name, theory_vector in self.theory_embeddings.items():
            sim = self.cosine_similarity(vector, theory_vector)
            similarities.append((theory_name, sim))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

    async def generate_theory_with_nearest_neighbors(self, theory1, theory2, model_source="deepseek", model_name="deepseek-chat", relaxation_degree=0.5, num_variants=3):
        """使用最近邻方法生成基于矛盾点放松的理论"""
        results = []
        
        # 检查理论是否存在
        if theory1 not in self.theory_embeddings or theory2 not in self.theory_embeddings:
            print(f"[错误] 理论名称错误: {theory1} 或 {theory2} 不在嵌入向量库中")
            return results
        
        theory1_vec = self.theory_embeddings[theory1]
        theory2_vec = self.theory_embeddings[theory2]
        
        print(f"[INFO] 为理论对 {theory1} 和 {theory2} 生成变体")
        
        # 找到包含这两个理论的矛盾比较
        contradiction_data = None
        for comparison in self.contradictions:
            if (comparison.get('theory1') == theory1 and comparison.get('theory2') == theory2) or \
               (comparison.get('theory1') == theory2 and comparison.get('theory2') == theory1):
                contradiction_data = comparison
                break
        
        if not contradiction_data:
            print(f"[警告] 未找到 {theory1} 和 {theory2} 之间的矛盾数据")
        
        # 为不同放松程度生成向量
        alphas = [i/(num_variants+1) for i in range(1, num_variants+1)]
        
        for alpha in alphas:
            # 插值生成新向量
            new_vector = alpha * theory1_vec + (1 - alpha) * theory2_vec
            # 归一化
            new_vector = new_vector / np.linalg.norm(new_vector)
            
            # 找到最相似的理论
            nearest_theories = self.find_nearest_theories(new_vector, k=3)
            nearest_theories_text = "\n".join([f"{name}: 相似度 {sim:.4f}" for name, sim in nearest_theories])
            
            print(f"[INFO] 放松程度 {alpha:.2f} 的最相似理论: {nearest_theories_text}")
            
            # 准备矛盾点描述
            contradiction_text = ""
            if contradiction_data:
                for i, c in enumerate(contradiction_data.get('contradictions', [])):
                    contradiction_text += f"""
                    矛盾点 {i+1}:
                    - 维度: {c.get('dimension', '')}
                    - {theory1} 的立场: {c.get('theory1_position', '')}
                    - {theory2} 的立场: {c.get('theory2_position', '')}
                    - 矛盾本质: {c.get('contradiction_nature', '')}
                    """
            
            # 构建结合最近邻信息的提示
            prompt = f"""
            作为量子物理学的理论创新者，你将创建一个新的量子诠释理论。这个理论来自于向量空间中对现有理论的扩展。

            原始理论:
            1. {theory1}
            2. {theory2}

            这些理论之间的主要矛盾点:
            {contradiction_text}

            通过在嵌入空间中进行插值（放松程度 alpha={alpha:.2f}），生成了一个新的向量点。
            与这个新向量最相似的已知理论是:
            {nearest_theories_text}

            你的任务是创建一个新的量子诠释理论，它:
            1. 体现放松程度 alpha={alpha:.2f} 的矛盾点（0表示完全倾向{theory1}，1表示完全倾向{theory2}）
            2. 考虑与新向量最相似的几个理论的特点
            3. 有自己独特的哲学立场
            4. 提供对量子现象的新解释
            5. 在数学上有一致性

            请以JSON格式返回新理论:
            {{
              "name": "新理论名称",
              "core_principles": "核心原理简要描述",
              "detailed_description": "详细描述，包括如何处理关键矛盾点",
              "quantum_phenomena_explanation": {{
                "wave_function_collapse": "这个理论如何解释波函数坍缩",
                "measurement_problem": "这个理论如何解释测量问题",
                "non_locality": "这个理论如何解释非局域性"
              }},
              "philosophical_stance": "理论的哲学立场",
              "mathematical_formulation": "简要的数学表述",
              "novel_predictions": "该理论可能做出的新预测"
            }}

            确保创建的理论在物理学上合理，并真正提供了原始理论矛盾点的新视角。
            """
            
            # 调用LLM生成理论描述
            try:
                response = await self.llm.query_async(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,  
                    model_source=model_source,
                    model_name=model_name
                )
                
                # 解析结果
                theory_desc = self.llm.extract_json(response)
                
                # 添加向量信息和相似理论信息
                if theory_desc:
                    theory_desc["vector_hash"] = hash(tuple(new_vector[:10].tolist()))
                    theory_desc["alpha"] = alpha
                    theory_desc["nearest_theories"] = [name for name, _ in nearest_theories]
                    theory_desc["original_theories"] = [theory1, theory2]
                    theory_desc["generation_strategy"] = f"矛盾点放松插值 (alpha={alpha:.2f}) + 最近邻分析"
                    
                    results.append(theory_desc)
                    self.generated_theories.append(theory_desc)
            except Exception as e:
                print(f"[ERROR] 生成理论描述失败: {str(e)}")
        
        print(f"[INFO] 共生成 {len(results)} 个新理论变体")
        return results 

    async def generate_relaxed_theories(self, theory1, theory2, model_source="deepseek", model_name="deepseek-chat", num_variants=3, temperature=0.7):
        """使用最近邻方法生成基于矛盾点放松的理论，异步方法"""
        print(f"[INFO] 使用最近邻方法生成理论变体，参数：{theory1}, {theory2}")
        
        # 设置LLM模型信息
        self.llm.set_model(model_source, model_name)
        
        # 直接等待异步方法，不创建新的事件循环
        results = await self.generate_theory_with_nearest_neighbors(
            theory1, theory2,
            model_source=model_source,
            model_name=model_name,
            num_variants=num_variants
        )
        return results

    def print_available_theories(self):
        """打印所有可用的理论名称"""
        if not self.theory_embeddings:
            print("[错误] 未加载理论嵌入向量")
            return
        
        print("\n可用的理论名称:")
        for i, theory_name in enumerate(sorted(self.theory_embeddings.keys())):
            print(f"{i+1}. {theory_name}")
        print("\n") 