#!/usr/bin/env python3
"""
Physics Embedder - 物理领域特定嵌入器

提供针对物理概念和公式的特殊嵌入方法，结合语义和结构信息。
支持概念层次、公式结构和物理含义的综合表示。

Author: UniversalTheoryGen Team
Date: 2025-01-24
"""

import json
import os
import sys
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import asyncio
from collections import defaultdict

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_embedding.embedding import ConceptEmbedder
from utils.logging_config import get_logger
from theory_generation.llm_interface import LLMInterface

# 初始化日志
logger = get_logger('physics_embedder')

@dataclass
class PhysicsEmbedding:
    """物理概念嵌入"""
    vector: np.ndarray  # 主嵌入向量
    semantic_vector: np.ndarray  # 语义嵌入
    structural_vector: np.ndarray  # 结构嵌入
    domain_vector: np.ndarray  # 领域特征向量
    metadata: Dict[str, Any]

class PhysicsEmbedder:
    """物理领域特定嵌入器"""
    
    # 物理领域分类
    PHYSICS_DOMAINS = {
        'quantum_mechanics': ['wave function', 'superposition', 'entanglement', 'measurement', 'collapse'],
        'classical_mechanics': ['force', 'momentum', 'energy', 'conservation', 'lagrangian'],
        'statistical_mechanics': ['entropy', 'partition function', 'ensemble', 'temperature', 'phase transition'],
        'relativity': ['spacetime', 'metric', 'curvature', 'geodesic', 'light cone'],
        'field_theory': ['field', 'gauge', 'symmetry', 'action', 'propagator'],
        'thermodynamics': ['heat', 'work', 'temperature', 'entropy', 'free energy']
    }
    
    # 概念层次
    CONCEPT_HIERARCHY = {
        'fundamental': ['space', 'time', 'mass', 'charge', 'energy'],
        'derived': ['momentum', 'angular momentum', 'force', 'potential'],
        'emergent': ['temperature', 'entropy', 'phase', 'correlation'],
        'mathematical': ['operator', 'eigenvalue', 'hilbert space', 'manifold']
    }
    
    # 数学结构特征
    MATH_STRUCTURES = {
        'differential': ['∂', 'd/dt', '∇', 'Δ'],
        'integral': ['∫', '∮', '∬'],
        'operator': ['H', 'L', 'p̂', 'x̂'],
        'tensor': ['g_μν', 'T^μν', 'R_μνρσ'],
        'complex': ['i', 'e^i', '|ψ⟩', '⟨ψ|']
    }
    
    def __init__(self, 
                 base_embedder: Optional[ConceptEmbedder] = None,
                 embedding_dim: int = 1536,
                 domain_dim: int = 64):
        """初始化物理嵌入器"""
        self.base_embedder = base_embedder or ConceptEmbedder(
            llm_interface=LLMInterface(model_source="google", model_name="gemini-2.0-flash-exp"),
            embedding_dim=embedding_dim
        )
        
        self.embedding_dim = embedding_dim
        self.domain_dim = domain_dim
        
        # 领域编码器
        self.domain_encoders = self._initialize_domain_encoders()
        
        # 结构特征提取器
        self.structure_extractors = self._initialize_structure_extractors()
        
        # 缓存
        self.embedding_cache = {}
    
    def _initialize_domain_encoders(self) -> Dict[str, np.ndarray]:
        """初始化领域编码器"""
        encoders = {}
        
        # 为每个物理领域创建独特的编码向量
        for i, domain in enumerate(self.PHYSICS_DOMAINS.keys()):
            # 使用正交基向量
            vector = np.zeros(self.domain_dim)
            if i < self.domain_dim:
                vector[i] = 1.0
            else:
                # 如果领域数超过维度，使用随机正交向量
                vector = np.random.randn(self.domain_dim)
                vector = vector / np.linalg.norm(vector)
            
            encoders[domain] = vector
        
        return encoders
    
    def _initialize_structure_extractors(self) -> Dict[str, callable]:
        """初始化结构特征提取器"""
        return {
            'formula': self._extract_formula_features,
            'concept': self._extract_concept_features,
            'theory': self._extract_theory_features
        }
    
    async def embed_physics_concept(self, 
                                   text: str,
                                   concept_type: str = 'concept',
                                   metadata: Optional[Dict] = None) -> PhysicsEmbedding:
        """嵌入物理概念"""
        # 检查缓存
        cache_key = f"{concept_type}:{text[:100]}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # 1. 获取基础语义嵌入
        semantic_vector = await self._get_semantic_embedding(text)
        
        # 2. 提取结构特征
        structural_vector = self._extract_structural_features(text, concept_type)
        
        # 3. 提取领域特征
        domain_vector = self._extract_domain_features(text, metadata)
        
        # 4. 组合嵌入
        combined_vector = self._combine_embeddings(
            semantic_vector, 
            structural_vector, 
            domain_vector
        )
        
        # 创建嵌入对象
        embedding = PhysicsEmbedding(
            vector=combined_vector,
            semantic_vector=semantic_vector,
            structural_vector=structural_vector,
            domain_vector=domain_vector,
            metadata=metadata or {}
        )
        
        # 缓存
        self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    async def _get_semantic_embedding(self, text: str) -> np.ndarray:
        """获取语义嵌入"""
        # 使用基础嵌入器
        embeddings = await self.base_embedder.embed_texts([text])
        embedding = embeddings.get(text)
        
        # 确保返回numpy数组
        if embedding is None:
            logger.warning(f"No embedding found for text: {text[:50]}...")
            return np.zeros(self.base_embedder.dimension)
        
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        return embedding
    
    def _extract_structural_features(self, text: str, concept_type: str) -> np.ndarray:
        """提取结构特征"""
        if concept_type in self.structure_extractors:
            return self.structure_extractors[concept_type](text)
        else:
            return np.zeros(256)  # 默认结构向量
    
    def _extract_formula_features(self, formula: str) -> np.ndarray:
        """提取公式结构特征"""
        features = np.zeros(256)
        
        # 1. 数学结构特征
        structure_scores = {}
        for struct_type, patterns in self.MATH_STRUCTURES.items():
            score = sum(1 for p in patterns if p in formula)
            structure_scores[struct_type] = score
        
        # 编码到特征向量
        for i, (struct_type, score) in enumerate(structure_scores.items()):
            if i < 50:
                features[i] = score / (len(self.MATH_STRUCTURES[struct_type]) + 1)
        
        # 2. 公式复杂度特征
        features[50] = len(formula) / 100  # 长度
        features[51] = formula.count('=') / 10  # 等式数量
        features[52] = len(re.findall(r'[a-zA-Z]+', formula)) / 20  # 变量数量
        
        # 3. 特殊模式
        if 'ψ' in formula or 'Ψ' in formula:
            features[60] = 1.0  # 量子力学标记
        if 'H' in formula and ('ψ' in formula or 'Ψ' in formula):
            features[61] = 1.0  # 哈密顿量标记
        if '∂' in formula or 'partial' in formula:
            features[62] = 1.0  # 偏导数标记
        
        return features
    
    def _extract_concept_features(self, concept: str) -> np.ndarray:
        """提取概念结构特征"""
        features = np.zeros(256)
        
        # 1. 概念层次特征
        for i, (level, concepts) in enumerate(self.CONCEPT_HIERARCHY.items()):
            if any(c in concept.lower() for c in concepts):
                features[i*10] = 1.0
        
        # 2. 复合概念特征
        if ' ' in concept:
            features[100] = 1.0  # 复合概念标记
            features[101] = len(concept.split()) / 10  # 词数
        
        # 3. 特殊标记
        if concept.endswith('ism') or concept.endswith('ity'):
            features[110] = 1.0  # 理论/性质标记
        
        return features
    
    def _extract_theory_features(self, theory: str) -> np.ndarray:
        """提取理论结构特征"""
        features = np.zeros(256)
        
        # 简单的理论特征
        features[0] = len(theory) / 1000  # 理论描述长度
        features[1] = theory.count('.') / 100  # 句子数量近似
        
        return features
    
    def _extract_domain_features(self, text: str, metadata: Optional[Dict]) -> np.ndarray:
        """提取领域特征"""
        domain_vector = np.zeros(self.domain_dim)
        
        # 1. 基于文本内容判断领域
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.PHYSICS_DOMAINS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # 2. 结合元数据中的领域信息
        if metadata and 'domain' in metadata:
            declared_domain = metadata['domain']
            if declared_domain in self.domain_encoders:
                domain_scores[declared_domain] = domain_scores.get(declared_domain, 0) + 3
        
        # 3. 组合领域向量
        if domain_scores:
            total_score = sum(domain_scores.values())
            for domain, score in domain_scores.items():
                if domain in self.domain_encoders:
                    weight = score / total_score
                    domain_vector += weight * self.domain_encoders[domain]
        
        # 归一化
        norm = np.linalg.norm(domain_vector)
        if norm > 0:
            domain_vector = domain_vector / norm
        
        return domain_vector
    
    def _combine_embeddings(self, 
                           semantic: np.ndarray,
                           structural: np.ndarray,
                           domain: np.ndarray) -> np.ndarray:
        """组合多种嵌入"""
        # 调整维度
        if len(semantic) > self.embedding_dim - 320:
            semantic = semantic[:self.embedding_dim - 320]
        
        # 拼接向量
        combined = np.concatenate([
            semantic,  # 主要语义信息
            structural,  # 256维结构信息
            domain  # 64维领域信息
        ])
        
        # 填充到目标维度
        if len(combined) < self.embedding_dim:
            combined = np.pad(combined, (0, self.embedding_dim - len(combined)))
        elif len(combined) > self.embedding_dim:
            combined = combined[:self.embedding_dim]
        
        # 归一化
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        return combined
    
    async def embed_concept_with_context(self,
                                       concept: str,
                                       context: str,
                                       related_formulas: List[str] = None) -> PhysicsEmbedding:
        """带上下文的概念嵌入"""
        # 构建增强文本
        enhanced_text = f"{concept}: {context}"
        
        if related_formulas:
            enhanced_text += f"\nRelated formulas: {', '.join(related_formulas[:3])}"
        
        # 提取元数据
        metadata = {
            'has_context': True,
            'formula_count': len(related_formulas) if related_formulas else 0
        }
        
        return await self.embed_physics_concept(enhanced_text, 'concept', metadata)
    
    async def embed_formula_with_derivation(self,
                                          formula: str,
                                          derivation_context: str = None) -> PhysicsEmbedding:
        """带推导过程的公式嵌入"""
        if derivation_context:
            text = f"{formula}\nDerivation: {derivation_context}"
        else:
            text = formula
        
        metadata = {
            'has_derivation': bool(derivation_context)
        }
        
        return await self.embed_physics_concept(text, 'formula', metadata)
    
    def compute_physics_similarity(self, 
                                 emb1: PhysicsEmbedding,
                                 emb2: PhysicsEmbedding,
                                 weights: Dict[str, float] = None) -> float:
        """计算物理概念相似度"""
        if weights is None:
            weights = {
                'semantic': 0.6,
                'structural': 0.2,
                'domain': 0.2
            }
        
        # 计算各部分相似度
        semantic_sim = np.dot(emb1.semantic_vector, emb2.semantic_vector)
        structural_sim = np.dot(emb1.structural_vector, emb2.structural_vector)
        domain_sim = np.dot(emb1.domain_vector, emb2.domain_vector)
        
        # 加权组合
        total_sim = (
            weights['semantic'] * semantic_sim +
            weights['structural'] * structural_sim +
            weights['domain'] * domain_sim
        )
        
        return float(total_sim)
    
    def find_conceptual_bridges(self,
                               embeddings: Dict[str, PhysicsEmbedding],
                               threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """发现概念桥梁"""
        bridges = []
        
        concepts = list(embeddings.keys())
        for i in range(len(concepts)):
            for j in range(i+1, len(concepts)):
                c1, c2 = concepts[i], concepts[j]
                
                # 检查是否属于不同领域
                domain1 = embeddings[c1].metadata.get('domain', '')
                domain2 = embeddings[c2].metadata.get('domain', '')
                
                if domain1 != domain2:
                    # 计算相似度
                    sim = self.compute_physics_similarity(embeddings[c1], embeddings[c2])
                    
                    if sim > threshold:
                        bridges.append((c1, c2, sim))
        
        # 按相似度排序
        bridges.sort(key=lambda x: x[2], reverse=True)
        return bridges
    
    def identify_concept_gaps(self,
                            embeddings: Dict[str, PhysicsEmbedding],
                            min_gap_size: float = 0.5) -> List[Dict[str, Any]]:
        """识别概念空白"""
        gaps = []
        
        # 将嵌入转换为数组
        concepts = list(embeddings.keys())
        vectors = np.array([embeddings[c].vector for c in concepts])
        
        # 计算所有对之间的距离
        n = len(concepts)
        for i in range(n):
            for j in range(i+1, n):
                # 计算中点
                midpoint = (vectors[i] + vectors[j]) / 2
                
                # 检查中点附近是否有其他概念
                distances = np.linalg.norm(vectors - midpoint, axis=1)
                min_distance = np.min(distances)
                
                if min_distance > min_gap_size:
                    gap = {
                        'concepts': (concepts[i], concepts[j]),
                        'midpoint': midpoint,
                        'gap_size': min_distance,
                        'domains': (
                            embeddings[concepts[i]].metadata.get('domain', ''),
                            embeddings[concepts[j]].metadata.get('domain', '')
                        )
                    }
                    gaps.append(gap)
        
        # 按空白大小排序
        gaps.sort(key=lambda x: x['gap_size'], reverse=True)
        return gaps[:20]  # 返回前20个最大的空白


async def demo():
    """演示物理嵌入器的使用"""
    # 初始化嵌入器
    embedder = PhysicsEmbedder()
    
    # 测试概念
    test_concepts = {
        "wave function": {
            "context": "A mathematical description of the quantum state of a system",
            "domain": "quantum_mechanics",
            "formulas": ["ψ(x,t)", "iℏ∂ψ/∂t = Hψ"]
        },
        "entropy": {
            "context": "A measure of disorder or information content",
            "domain": "statistical_mechanics",
            "formulas": ["S = -k∑p_i ln p_i", "dS ≥ 0"]
        },
        "spacetime metric": {
            "context": "A tensor that describes the geometry of spacetime",
            "domain": "relativity",
            "formulas": ["ds² = g_μν dx^μ dx^ν"]
        }
    }
    
    # 嵌入概念
    embeddings = {}
    for concept, info in test_concepts.items():
        print(f"\nEmbedding: {concept}")
        embedding = await embedder.embed_concept_with_context(
            concept,
            info['context'],
            info.get('formulas', [])
        )
        embeddings[concept] = embedding
        
        # 显示领域特征
        print(f"Domain vector norm: {np.linalg.norm(embedding.domain_vector):.3f}")
        print(f"Structural features: {np.sum(embedding.structural_vector > 0)} active")
    
    # 计算相似度
    print("\n=== Concept Similarities ===")
    concepts = list(embeddings.keys())
    for i in range(len(concepts)):
        for j in range(i+1, len(concepts)):
            sim = embedder.compute_physics_similarity(
                embeddings[concepts[i]], 
                embeddings[concepts[j]]
            )
            print(f"{concepts[i]} <-> {concepts[j]}: {sim:.3f}")
    
    # 查找概念桥梁
    bridges = embedder.find_conceptual_bridges(embeddings, threshold=0.3)
    if bridges:
        print("\n=== Conceptual Bridges ===")
        for c1, c2, score in bridges:
            print(f"{c1} <-> {c2}: {score:.3f}")
    
    # 识别概念空白
    gaps = embedder.identify_concept_gaps(embeddings, min_gap_size=0.3)
    if gaps:
        print("\n=== Concept Gaps ===")
        for gap in gaps[:3]:
            print(f"Gap between {gap['concepts'][0]} and {gap['concepts'][1]}: {gap['gap_size']:.3f}")


if __name__ == "__main__":
    asyncio.run(demo())