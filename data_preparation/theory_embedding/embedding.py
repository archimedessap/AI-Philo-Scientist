#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
概念和理论的嵌入(Embedding)与去嵌入(Deembedding)

将合并后的概念和公式嵌入到高维空间，并提供去嵌入API接口。
通过向量空间操作实现理论概念的分析、比较和生成。
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from theory_generation.llm_interface import LLMInterface

class ConceptEmbedder:
    """概念和理论的嵌入与去嵌入处理器"""
    
    def __init__(self, llm_interface: LLMInterface, embedding_dim: int = 1536):
        """
        初始化概念嵌入器
        
        Args:
            llm_interface: LLM接口
            embedding_dim: 嵌入向量维度
        """
        self.llm = llm_interface
        self.embedding_dim = embedding_dim
        self.concept_embeddings = {}  # 概念名称 -> 嵌入向量
        self.formula_embeddings = {}  # 公式名称 -> 嵌入向量
        self.theory_embeddings = {}   # 理论名称 -> 嵌入向量
        
    async def embed_concepts(self, concepts: List[Dict]) -> Dict[str, np.ndarray]:
        """
        将概念嵌入到高维空间
        
        Args:
            concepts: 概念列表
            
        Returns:
            Dict[str, np.ndarray]: 概念名称到嵌入向量的映射
        """
        print(f"[INFO] 嵌入 {len(concepts)} 个概念...")
        
        for concept in concepts:
            name = concept.get('name', '')
            if not name:
                continue
                
            description = concept.get('description', '')
            
            # 构建嵌入文本
            text = f"概念: {name}\n描述: {description}"
            
            # 获取嵌入向量
            try:
                embedding = await self._get_embedding(text)
                self.concept_embeddings[name] = embedding
            except Exception as e:
                print(f"[警告] 无法嵌入概念 '{name}': {e}")
        
        print(f"[INFO] 成功嵌入 {len(self.concept_embeddings)} 个概念")
        return self.concept_embeddings
    
    async def embed_formulas(self, formulas: List[Dict]) -> Dict[str, np.ndarray]:
        """
        将公式嵌入到高维空间
        
        Args:
            formulas: 公式列表
            
        Returns:
            Dict[str, np.ndarray]: 公式名称到嵌入向量的映射
        """
        print(f"[INFO] 嵌入 {len(formulas)} 个公式...")
        
        for formula in formulas:
            name = formula.get('name', '')
            if not name:
                continue
                
            expression = formula.get('expression', '')
            description = formula.get('description', '')
            
            # 构建嵌入文本
            text = f"公式: {name}\n表达式: {expression}\n描述: {description}"
            
            # 获取嵌入向量
            try:
                embedding = await self._get_embedding(text)
                self.formula_embeddings[name] = embedding
            except Exception as e:
                print(f"[警告] 无法嵌入公式 '{name}': {e}")
        
        print(f"[INFO] 成功嵌入 {len(self.formula_embeddings)} 个公式")
        return self.formula_embeddings
    
    async def embed_theories(self, theories: List[Dict]) -> Dict[str, np.ndarray]:
        """
        将理论嵌入到高维空间
        
        Args:
            theories: 理论列表
            
        Returns:
            Dict[str, np.ndarray]: 理论名称到嵌入向量的映射
        """
        print(f"[INFO] 嵌入 {len(theories)} 个理论...")
        
        for theory in theories:
            name = theory.get('name', '')
            if not name:
                continue
                
            description = theory.get('description', '')
            assumptions = theory.get('philosophical_assumptions', '')
            
            # 构建嵌入文本
            text = f"理论: {name}\n哲学假设: {assumptions}\n描述: {description}"
            
            # 获取嵌入向量
            try:
                embedding = await self._get_embedding(text)
                self.theory_embeddings[name] = embedding
            except Exception as e:
                print(f"[警告] 无法嵌入理论 '{name}': {e}")
        
        print(f"[INFO] 成功嵌入 {len(self.theory_embeddings)} 个理论")
        return self.theory_embeddings
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的嵌入向量
        
        Args:
            text: 要嵌入的文本
            
        Returns:
            np.ndarray: 嵌入向量
        """
        # 使用LLM接口获取嵌入
        embedding = await self.llm.get_embedding(text)
        return np.array(embedding)
    
    async def deembed_vector(self, vector: np.ndarray, vector_type: str = "concept") -> Dict:
        """
        将向量解嵌入(deembed)为自然语言描述
        
        Args:
            vector: 嵌入向量
            vector_type: 向量类型 ("concept", "formula", "theory")
            
        Returns:
            Dict: 解嵌入后的描述
        """
        # 标准化向量
        vector = vector / np.linalg.norm(vector)
        
        # 确定上下文
        context = "量子力学概念" if vector_type == "concept" else \
                 "量子力学公式" if vector_type == "formula" else \
                 "量子力学理论"
                 
        # 向量表示为文本
        vector_text = ",".join([str(round(x, 6)) for x in vector[:20]]) + "..."
        
        # 构建提示
        prompt = f"""
        你是一个专业的量子物理学家，精通量子理论和数学。
        
        我有一个表示{context}的向量(仅显示前20维):
        [{vector_text}]
        
        基于这个向量，请生成一个完整的{context}描述，包括:
        
        {
        "name": "名称",
        "description": "详细描述",
        }
        
        如果是公式，还需要包含:
        {
        "expression": "数学表达式",
        "variables": [{"symbol": "符号", "meaning": "含义"}]
        }
        
        如果是理论，还需要包含:
        {
        "philosophical_assumptions": "哲学假设",
        "key_predictions": "关键预测"
        }
        
        请保持科学准确性，确保描述与量子物理学知识一致。
        """
        
        # 使用LLM解释向量
        response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        # 从响应中提取JSON
        result = self.llm.extract_json(response)
        
        # 如果提取失败，使用默认结构
        if not result:
            result = {
                "name": f"未命名{context}",
                "description": "无法从向量解析出完整描述。"
            }
            
            if vector_type == "formula":
                result["expression"] = "未知表达式"
                result["variables"] = []
            elif vector_type == "theory":
                result["philosophical_assumptions"] = "未知假设"
                result["key_predictions"] = "未知预测"
        
        return result
    
    async def vector_operation(self, operation: str, vectors: List[Tuple[float, np.ndarray]], 
                              result_type: str = "concept") -> Dict:
        """
        执行向量运算并解释结果
        
        Args:
            operation: 操作类型 ("add", "subtract", "average")
            vectors: (权重, 向量)元组的列表
            result_type: 结果向量类型
            
        Returns:
            Dict: 解释的结果
        """
        if not vectors:
            return {"error": "没有提供向量"}
            
        # 执行向量运算
        result_vector = None
        
        if operation == "add":
            # 加权和
            result_vector = sum(weight * vector for weight, vector in vectors)
        elif operation == "subtract":
            # 减法（第一个减去其余的加权和）
            first_weight, first_vector = vectors[0]
            rest_sum = sum(weight * vector for weight, vector in vectors[1:]) if len(vectors) > 1 else 0
            result_vector = first_weight * first_vector - rest_sum
        elif operation == "average":
            # 加权平均
            total_weight = sum(weight for weight, _ in vectors)
            if total_weight == 0:
                total_weight = 1.0
            result_vector = sum(weight * vector for weight, vector in vectors) / total_weight
        else:
            return {"error": f"不支持的操作: {operation}"}
            
        # 标准化结果向量
        norm = np.linalg.norm(result_vector)
        if norm > 0:
            result_vector = result_vector / norm
            
        # 解释结果向量
        result = await self.deembed_vector(result_vector, result_type)
        result["operation"] = operation
        result["vectors_used"] = len(vectors)
        
        return result
    
    def visualize_embeddings(self, output_path: str, 
                             types: List[str] = ["concept", "formula", "theory"],
                             method: str = "tsne") -> str:
        """
        可视化嵌入向量
        
        Args:
            output_path: 输出文件路径
            types: 要可视化的类型
            method: 降维方法
            
        Returns:
            str: 输出文件路径
        """
        # 收集所有要可视化的向量和标签
        vectors = []
        labels = []
        colors = []
        
        color_map = {"concept": "blue", "formula": "red", "theory": "green"}
        
        if "concept" in types and self.concept_embeddings:
            for name, vector in self.concept_embeddings.items():
                vectors.append(vector)
                labels.append(name)
                colors.append(color_map["concept"])
                
        if "formula" in types and self.formula_embeddings:
            for name, vector in self.formula_embeddings.items():
                vectors.append(vector)
                labels.append(name)
                colors.append(color_map["formula"])
                
        if "theory" in types and self.theory_embeddings:
            for name, vector in self.theory_embeddings.items():
                vectors.append(vector)
                labels.append(name)
                colors.append(color_map["theory"])
        
        if not vectors:
            print("[警告] 没有可视化的嵌入向量")
            return None
            
        # 降维
        if method == "tsne":
            model = TSNE(n_components=2, random_state=42)
            embeddings_2d = model.fit_transform(np.array(vectors))
        else:
            raise ValueError(f"不支持的降维方法: {method}")
            
        # 绘制散点图
        plt.figure(figsize=(12, 10))
        
        # 为每种类型创建一个散点图
        for i, (x, y, label, color) in enumerate(zip(embeddings_2d[:, 0], embeddings_2d[:, 1], labels, colors)):
            plt.scatter(x, y, c=color, alpha=0.7)
            plt.annotate(label, (x, y), fontsize=8)
            
        # 添加图例
        if "concept" in types and self.concept_embeddings:
            plt.scatter([], [], c=color_map["concept"], label="概念")
        if "formula" in types and self.formula_embeddings:
            plt.scatter([], [], c=color_map["formula"], label="公式")
        if "theory" in types and self.theory_embeddings:
            plt.scatter([], [], c=color_map["theory"], label="理论")
            
        plt.legend()
        plt.title("概念、公式和理论的嵌入空间可视化")
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(output_path)
        print(f"[INFO] 可视化结果已保存到: {output_path}")
        
        return output_path
    
    def save_embeddings(self, output_dir: str) -> Tuple[str, str, str]:
        """
        保存嵌入向量
        
        Args:
            output_dir: 输出目录
            
        Returns:
            Tuple[str, str, str]: 保存的文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存概念嵌入
        concept_path = os.path.join(output_dir, "concept_embeddings.pkl")
        with open(concept_path, 'wb') as f:
            pickle.dump(self.concept_embeddings, f)
            
        # 保存公式嵌入
        formula_path = os.path.join(output_dir, "formula_embeddings.pkl")
        with open(formula_path, 'wb') as f:
            pickle.dump(self.formula_embeddings, f)
            
        # 保存理论嵌入
        theory_path = os.path.join(output_dir, "theory_embeddings.pkl")
        with open(theory_path, 'wb') as f:
            pickle.dump(self.theory_embeddings, f)
            
        print(f"[INFO] 嵌入向量已保存到: {output_dir}")
        
        return concept_path, formula_path, theory_path
    
    def load_embeddings(self, input_dir: str) -> bool:
        """
        加载嵌入向量
        
        Args:
            input_dir: 输入目录
            
        Returns:
            bool: 是否成功加载
        """
        # 加载概念嵌入
        concept_path = os.path.join(input_dir, "concept_embeddings.pkl")
        if os.path.exists(concept_path):
            with open(concept_path, 'rb') as f:
                self.concept_embeddings = pickle.load(f)
                
        # 加载公式嵌入
        formula_path = os.path.join(input_dir, "formula_embeddings.pkl")
        if os.path.exists(formula_path):
            with open(formula_path, 'rb') as f:
                self.formula_embeddings = pickle.load(f)
                
        # 加载理论嵌入
        theory_path = os.path.join(input_dir, "theory_embeddings.pkl")
        if os.path.exists(theory_path):
            with open(theory_path, 'rb') as f:
                self.theory_embeddings = pickle.load(f)
                
        loaded = (len(self.concept_embeddings) > 0 or 
                 len(self.formula_embeddings) > 0 or 
                 len(self.theory_embeddings) > 0)
                 
        if loaded:
            print(f"[INFO] 已加载 {len(self.concept_embeddings)} 个概念嵌入, " 
                 f"{len(self.formula_embeddings)} 个公式嵌入, "
                 f"{len(self.theory_embeddings)} 个理论嵌入")
        else:
            print("[警告] 未找到嵌入向量文件或文件为空")
            
        return loaded
