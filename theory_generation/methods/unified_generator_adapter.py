#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_generator_adapter.py - 基于高维概念空间的统一生成器适配器
================================================================

真正的统一生成器：整合概念嵌入、向量空间探索和空间分析，
基于高维概念空间而非矛盾驱动来生成理论。

核心特点：
1. 多层次概念空间构建（概念、公式、理论）
2. 空间空白区域识别和分析
3. 向量空间驱动的理论生成
4. 维度语义映射和解释
"""

import json
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import sys
import os

# 导入基础适配器
try:
    from .base_adapter import TheoryGenerationMethod, GenerationResult
except ImportError:
    from base_adapter import TheoryGenerationMethod, GenerationResult

# 导入核心模块
try:
    # 相对导入
    from ...core_embedding.embedding import ConceptEmbedder
    from ...core_embedding.analyze_concept_space import compute_similarity_matrix
    from ..vector_space_explorer import VectorSpaceExplorer
    from ..llm_interface import LLMInterface
except ImportError:
    # 绝对导入（用于独立运行）
    current_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(current_dir))
    
    from core_embedding.embedding import ConceptEmbedder
    from core_embedding.analyze_concept_space import compute_similarity_matrix
    from theory_generation.vector_space_explorer import VectorSpaceExplorer
    from theory_generation.llm_interface import LLMInterface


class UnifiedSpaceBasedGenerator(TheoryGenerationMethod):
    """基于高维概念空间的统一理论生成器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 初始化LLM接口
        self.llm_interface = LLMInterface(
            model_source=self.model_source,
            model_name=self.model_name
        )
        
        # 初始化核心组件
        self.concept_embedder = ConceptEmbedder(
            llm_interface=self.llm_interface,
            embedding_dim=1536  # 使用标准嵌入维度
        )
        
        self.space_explorer = VectorSpaceExplorer(
            llm_interface=self.llm_interface
        )
        
        # 空间数据存储
        self.concept_space = {}      # 概念空间
        self.theory_space = {}       # 理论空间
        self.formula_space = {}      # 公式空间
        self.conceptual_gaps = []    # 概念空白区域
        self.semantic_dimensions = {}  # 维度语义映射
        
        self._log_info("初始化基于高维概念空间的统一生成器")
    
    def generate(self) -> Dict[str, Any]:
        """统一生成流程：空间构建 → 空白分析 → 空间生成（同步接口）"""
        try:
            # 使用asyncio.run来运行异步生成过程
            return asyncio.run(self._async_generate())
        except Exception as e:
            self._log_error(f"统一空间生成失败: {e}")
            return GenerationResult(
                success=False,
                error_message=str(e)
            ).to_dict()
    
    async def _async_generate(self) -> Dict[str, Any]:
        """异步统一生成流程：空间构建 → 空白分析 → 空间生成"""
        try:
            self._log_info("🚀 开始基于高维概念空间的统一理论生成")
            
            # 1. 加载和预处理理论
            theories = self._load_theories()
            if not theories:
                return GenerationResult(
                    success=False,
                    error_message="无法加载先验理论"
                ).to_dict()
            
            # 2. 构建多层次概念空间
            await self._build_multilevel_concept_space(theories)
            
            # 3. 分析概念空间结构
            self._analyze_space_structure()
            
            # 4. 识别概念空白区域
            self._identify_conceptual_gaps()
            
            # 5. 基于空间生成新理论
            generated_theories = await self._generate_from_space()
            
            # 6. 构建结果
            result = GenerationResult(
                success=True,
                theories=generated_theories,
                metadata={
                    'generation_method': 'unified_space_based',
                    'concept_space_size': len(self.concept_space),
                    'theory_space_size': len(self.theory_space),
                    'conceptual_gaps_found': len(self.conceptual_gaps),
                    'semantic_dimensions': len(self.semantic_dimensions),
                    'space_analysis': self._get_space_analysis_summary()
                },
                output_dir=str(self.output_dir)
            )
            
            # 保存空间数据和结果
            await self._save_space_data()
            self._save_result(result.to_dict())
            
            self._log_info(f"✅ 统一空间生成完成：生成 {len(generated_theories)} 个理论")
            return result.to_dict()
            
        except Exception as e:
            self._log_error(f"统一空间生成失败: {e}")
            return GenerationResult(
                success=False,
                error_message=str(e)
            ).to_dict()
    
    async def _build_multilevel_concept_space(self, theories: List[Dict]):
        """构建多层次概念空间"""
        self._log_info("🏗️ 构建多层次概念空间...")
        
        # 1. 构建理论空间
        self.theory_space = await self.concept_embedder.embed_theories(theories)
        self._log_info(f"理论空间维度: {len(self.theory_space)} × {self.concept_embedder.embedding_dim}")
        
        # 2. 提取和嵌入概念
        concepts = self._extract_concepts_from_theories(theories)
        if concepts:
            self.concept_space = await self.concept_embedder.embed_concepts(concepts)
            self._log_info(f"概念空间维度: {len(self.concept_space)} × {self.concept_embedder.embedding_dim}")
        
        # 3. 提取和嵌入公式（如果存在）
        formulas = self._extract_formulas_from_theories(theories)
        if formulas:
            self.formula_space = await self.concept_embedder.embed_formulas(formulas)
            self._log_info(f"公式空间维度: {len(self.formula_space)} × {self.concept_embedder.embedding_dim}")
    
    def _extract_concepts_from_theories(self, theories: List[Dict]) -> List[Dict]:
        """从理论中提取概念"""
        concepts = []
        seen_concepts = set()
        
        # 方法1: 从理论内容中提取（原有方法）
        for theory in theories:
            content = theory.get('content', '')
            
            # 简化的概念提取（基于关键词）
            key_concepts = self._extract_key_concepts(content)
            
            for concept in key_concepts:
                if concept not in seen_concepts:
                    concepts.append({
                        'name': concept,
                        'description': f"从理论 {theory.get('name', '')} 中提取的概念",
                        'source_theory': theory.get('name', ''),
                        'context': content[:500],  # 提供上下文
                        'extraction_method': 'theory_content'
                    })
                    seen_concepts.add(concept)
        
        # 方法2: 从文献数据库中加载概念（新增）
        literature_concepts = self._load_literature_concepts()
        for lit_concept in literature_concepts:
            concept_name = lit_concept.get('name', '').strip()
            if concept_name and concept_name not in seen_concepts:
                concepts.append({
                    'name': concept_name,
                    'description': lit_concept.get('description', ''),
                    'source_theory': lit_concept.get('source', ''),
                    'context': lit_concept.get('description', '')[:500],
                    'extraction_method': 'literature_analysis',
                    'domain': lit_concept.get('domain', ''),
                    'literature_source': lit_concept.get('source', '')
                })
                seen_concepts.add(concept_name)
        
        self._log_info(f"总共提取概念: 理论内容={len([c for c in concepts if c['extraction_method'] == 'theory_content'])}, "
                       f"文献分析={len([c for c in concepts if c['extraction_method'] == 'literature_analysis'])}")
        
        return concepts
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """提取关键概念（支持中英文）"""
        # 量子物理关键概念 - 中文
        quantum_concepts_zh = [
            '量子叠加', '波函数坍缩', '量子纠缠', '观测者效应', '测量问题',
            '波粒二象性', '不确定性原理', '量子隧道', '量子退相干', '多世界',
            '隐变量理论', '贝尔不等式', 'EPR佯谬', '量子态', '算符',
            '哈密顿量', '薛定谔方程', '波函数', '概率幅', '量子场'
        ]
        
        # 量子物理关键概念 - 英文
        quantum_concepts_en = [
            'quantum superposition', 'wave function collapse', 'quantum entanglement', 
            'observer effect', 'measurement problem', 'wave-particle duality',
            'uncertainty principle', 'quantum tunneling', 'quantum decoherence',
            'many worlds', 'hidden variables', 'Bell inequality', 'EPR paradox',
            'quantum state', 'operator', 'Hamiltonian', 'Schrödinger equation',
            'wave function', 'probability amplitude', 'quantum field',
            
            # 额外的英文概念
            'complementarity', 'Copenhagen interpretation', 'many-worlds interpretation',
            'pilot wave theory', 'de Broglie-Bohm theory', 'objective collapse',
            'GRW theory', 'spontaneous localization', 'quantum Bayesianism',
            'QBism', 'relational quantum mechanics', 'consistent histories',
            'modal interpretation', 'transactional interpretation', 'retrocausality',
            'nonlocality', 'locality', 'realism', 'instrumentalism',
            'determinism', 'indeterminism', 'measurement apparatus',
            'classical limit', 'quantum-classical boundary', 'Born rule',
            'unitary evolution', 'non-unitary evolution', 'state reduction',
            'quantum mechanics', 'quantum theory', 'quantum physics'
        ]
        
        # 合并所有概念
        all_concepts = quantum_concepts_zh + quantum_concepts_en
        
        found_concepts = []
        text_lower = text.lower()
        
        for concept in all_concepts:
            # 检查原文和小写文本
            if concept in text or concept.lower() in text_lower:
                found_concepts.append(concept)
        
        return found_concepts
    
    def _extract_formulas_from_theories(self, theories: List[Dict]) -> List[Dict]:
        """从理论中提取公式"""
        formulas = []
        
        for theory in theories:
            theory_name = theory.get('name', 'Unknown')
            content = theory.get('content', '')
            raw_data = theory.get('raw_data', {})
            
            # 方法1: 从JSON结构化数据中提取公式
            if raw_data:
                extracted_formulas = self._extract_formulas_from_json(raw_data, theory_name)
                formulas.extend(extracted_formulas)
            
            # 方法2: 从文本内容中提取公式（作为补充）
            text_formulas = self._extract_formulas_from_text(content, theory_name)
            formulas.extend(text_formulas)
        
        # 去重
        unique_formulas = []
        seen_expressions = set()
        for formula in formulas:
            expr = formula['expression'].strip()
            if expr and expr not in seen_expressions and len(expr) > 2:
                unique_formulas.append(formula)
                seen_expressions.add(expr)
        
        return unique_formulas
    
    def _extract_formulas_from_json(self, json_data: Dict, theory_name: str) -> List[Dict]:
        """从JSON结构化数据中提取公式"""
        formulas = []
        formula_counter = 1
        
        def extract_from_value(value, context_path):
            nonlocal formula_counter
            extracted = []
            
            if isinstance(value, str):
                # 检查是否包含数学符号和等号
                if any(symbol in value for symbol in ['=', 'Ψ', 'ψ', 'ħ', 'ℏ', '∂', '∇', '∫', '∆']):
                    extracted.append({
                        'name': f"{theory_name}_formula_{formula_counter}",
                        'expression': value.strip(),
                        'description': f"从理论 {theory_name} 的 {context_path} 中提取的公式",
                        'source_theory': theory_name,
                        'source_context': context_path
                    })
                    formula_counter += 1
            elif isinstance(value, dict):
                for k, v in value.items():
                    extracted.extend(extract_from_value(v, f"{context_path}.{k}"))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    extracted.extend(extract_from_value(item, f"{context_path}[{i}]"))
            
            return extracted
        
        # 重点关注可能包含公式的字段
        formula_fields = [
            'formalism', 'mathematical_formalism', 'equations', 'formulas', 
            'axioms', 'principles', 'mathematical_framework', 'dynamics',
            'mathematical_relation_to_sqm', 'postulates', 'mathematics'
        ]
        
        for field in formula_fields:
            if field in json_data:
                extracted = extract_from_value(json_data[field], field)
                formulas.extend(extracted)
        
        return formulas
    
    def _extract_formulas_from_text(self, content: str, theory_name: str) -> List[Dict]:
        """从文本内容中提取公式（改进版）"""
        formulas = []
        
        # 更宽松的公式模式
        import re
        formula_patterns = [
            # 基本等式模式
            r'[A-Za-z_]\w*\s*=\s*[^=\n]{5,}',  # 变量 = 表达式
            r'[Ψψ]\s*=\s*[^=\n]{3,}',          # 波函数等式
            r'H\s*=\s*[^=\n]{3,}',             # 哈密顿量
            r'E\s*=\s*[^=\n]{3,}',             # 能量公式
            r'P\([^)]+\)\s*=\s*[^=\n]{3,}',    # 概率公式
            r'ρ\s*=\s*[^=\n]{3,}',             # 密度矩阵
            r'd[A-Za-z_]\w*/dt\s*=\s*[^=\n]{3,}',  # 时间导数
            r'∂[^/]+/∂[^=\n]+\s*=\s*[^=\n]{3,}',  # 偏导数
            
            # 数学表达式（不一定有等号）
            r'\|[^|]+\|²',                     # 概率幅平方
            r'⟨[^⟩]+⟩',                       # 期望值
            r'∫[^∫]{5,}d[a-z]',               # 积分
            r'exp\[[^\]]{3,}\]',              # 指数函数
            r'[Ψψ]\([^)]{3,}\)',              # 波函数调用
            r'iħ[^=\n]{3,}',                  # 包含iħ的表达式
            r'ℏ[^=\n]{3,}',                   # 包含ℏ的表达式
        ]
        
        found_formulas = []
        for pattern in formula_patterns:
            try:
                matches = re.findall(pattern, content, re.UNICODE | re.MULTILINE)
                for match in matches:
                    # 清理匹配结果
                    cleaned = match.strip()
                    if len(cleaned) > 2 and cleaned not in found_formulas:
                        found_formulas.append(cleaned)
            except re.error:
                continue
        
        # 转换为标准格式
        for i, formula in enumerate(found_formulas):
            formulas.append({
                'name': f"{theory_name}_text_formula_{i+1}",
                'expression': formula,
                'description': f"从理论 {theory_name} 的文本内容中提取的公式",
                'source_theory': theory_name,
                'source_context': 'text_content'
            })
        
        return formulas
    
    def _load_literature_concepts(self) -> List[Dict]:
        """从文献分析数据中加载概念"""
        concepts = []
        
        # 加载extracted_concepts.csv
        concept_file = Path('data/extracted_concepts/concepts.csv')
        if concept_file.exists():
            try:
                import csv
                with open(concept_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('Name') and row.get('Description'):
                            concepts.append({
                                'name': row['Name'].strip(),
                                'description': row['Description'].strip(),
                                'domain': row.get('Domain', '').strip(),
                                'source': row.get('Source', '').strip()
                            })
                
                self._log_info(f"从文献数据库加载了 {len(concepts)} 个概念")
            except Exception as e:
                self._log_error(f"加载文献概念失败: {e}")
        else:
            self._log_warning(f"文献概念文件不存在: {concept_file}")
        
        # 加载merged_concepts.csv作为备选
        if not concepts:
            merged_file = Path('data/merged_concepts/merged_concepts.csv')
            if merged_file.exists():
                try:
                    import csv
                    with open(merged_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row.get('Name') and row.get('Description'):
                                concepts.append({
                                    'name': row['Name'].strip(),
                                    'description': row['Description'].strip(),
                                    'domain': row.get('Domain', '').strip(),
                                    'source': row.get('Source', '').strip()
                                })
                    
                    self._log_info(f"从合并概念文件加载了 {len(concepts)} 个概念")
                except Exception as e:
                    self._log_error(f"加载合并概念失败: {e}")
        
        # 过滤和清理概念
        filtered_concepts = []
        for concept in concepts:
            name = concept['name']
            # 过滤掉过短或无意义的概念名
            if len(name) > 2 and not name.isdigit() and 'theory' not in name.lower():
                filtered_concepts.append(concept)
        
        self._log_info(f"过滤后保留 {len(filtered_concepts)} 个高质量概念")
        return filtered_concepts[:50]  # 限制数量避免过多概念
    
    def _analyze_space_structure(self):
        """分析概念空间结构"""
        self._log_info("🔍 分析概念空间结构...")
        
        # 分析理论空间
        if self.theory_space:
            theory_similarity = compute_similarity_matrix(self.theory_space)
            self.semantic_dimensions['theory_clusters'] = self._find_clusters(self.theory_space)
            self._log_info(f"理论聚类数量: {len(self.semantic_dimensions['theory_clusters'])}")
        
        # 分析概念空间
        if self.concept_space:
            concept_similarity = compute_similarity_matrix(self.concept_space)
            self.semantic_dimensions['concept_clusters'] = self._find_clusters(self.concept_space)
            self._log_info(f"概念聚类数量: {len(self.semantic_dimensions['concept_clusters'])}")
        
        # 计算空间密度和分布
        self.semantic_dimensions['space_density'] = self._calculate_space_density()
    
    def _find_clusters(self, embeddings: Dict[str, np.ndarray], n_clusters: int = 5) -> List[List[str]]:
        """在嵌入空间中找到聚类"""
        if len(embeddings) < 2:
            return []
        
        from sklearn.cluster import KMeans
        
        names = list(embeddings.keys())
        vectors = np.array([embeddings[name] for name in names])
        
        # 调整聚类数量
        actual_clusters = min(n_clusters, len(names))
        
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(vectors)
        
        # 组织聚类结果
        clusters = [[] for _ in range(actual_clusters)]
        for name, label in zip(names, cluster_labels):
            clusters[label].append(name)
        
        return [cluster for cluster in clusters if cluster]  # 过滤空聚类
    
    def _calculate_space_density(self) -> Dict[str, float]:
        """计算空间密度"""
        density_info = {}
        
        # 理论空间密度
        if self.theory_space:
            theory_vectors = np.array(list(self.theory_space.values()))
            theory_distances = []
            for i in range(len(theory_vectors)):
                for j in range(i+1, len(theory_vectors)):
                    dist = np.linalg.norm(theory_vectors[i] - theory_vectors[j])
                    theory_distances.append(dist)
            
            if theory_distances:
                density_info['theory_avg_distance'] = np.mean(theory_distances)
                density_info['theory_std_distance'] = np.std(theory_distances)
        
        # 概念空间密度
        if self.concept_space:
            concept_vectors = np.array(list(self.concept_space.values()))
            if len(concept_vectors) > 1:
                concept_distances = []
                for i in range(len(concept_vectors)):
                    for j in range(i+1, len(concept_vectors)):
                        dist = np.linalg.norm(concept_vectors[i] - concept_vectors[j])
                        concept_distances.append(dist)
                
                if concept_distances:
                    density_info['concept_avg_distance'] = np.mean(concept_distances)
                    density_info['concept_std_distance'] = np.std(concept_distances)
        
        return density_info
    
    def _identify_conceptual_gaps(self):
        """识别概念空间中的空白区域"""
        self._log_info("🎯 识别概念空白区域...")
        
        if not self.theory_space:
            return
        
        theory_vectors = np.array(list(self.theory_space.values()))
        theory_names = list(self.theory_space.keys())
        
        # 1. 寻找理论向量间的中点作为潜在空白区域
        for i in range(len(theory_vectors)):
            for j in range(i+1, len(theory_vectors)):
                midpoint = (theory_vectors[i] + theory_vectors[j]) / 2
                
                # 检查中点是否远离所有现有理论
                min_distance = float('inf')
                for k, theory_vec in enumerate(theory_vectors):
                    distance = np.linalg.norm(midpoint - theory_vec)
                    min_distance = min(min_distance, distance)
                
                # 如果中点足够远离现有理论，认为是空白区域
                if min_distance > np.std([np.linalg.norm(v) for v in theory_vectors]) * 0.5:
                    self.conceptual_gaps.append({
                        'type': 'midpoint_gap',
                        'position': midpoint,
                        'source_theories': [theory_names[i], theory_names[j]],
                        'distance_to_nearest': min_distance,
                        'description': f"理论 {theory_names[i]} 和 {theory_names[j]} 之间的概念空白区域"
                    })
        
        # 2. 寻找聚类边界的空白区域
        if 'theory_clusters' in self.semantic_dimensions:
            clusters = self.semantic_dimensions['theory_clusters']
            for i, cluster1 in enumerate(clusters):
                for j, cluster2 in enumerate(clusters[i+1:], i+1):
                    # 计算聚类中心
                    center1 = np.mean([theory_vectors[theory_names.index(name)] 
                                     for name in cluster1 if name in theory_names], axis=0)
                    center2 = np.mean([theory_vectors[theory_names.index(name)] 
                                     for name in cluster2 if name in theory_names], axis=0)
                    
                    # 聚类间的中点
                    inter_cluster_gap = (center1 + center2) / 2
                    
                    self.conceptual_gaps.append({
                        'type': 'inter_cluster_gap',
                        'position': inter_cluster_gap,
                        'source_clusters': [cluster1, cluster2],
                        'description': f"聚类 {i+1} 和聚类 {j+1} 之间的概念空白区域"
                    })
        
        self._log_info(f"发现 {len(self.conceptual_gaps)} 个概念空白区域")
    
    async def _generate_from_space(self) -> List[Dict]:
        """基于概念空间生成新理论"""
        self._log_info("🎨 基于概念空间生成新理论...")
        
        generated_theories = []
        
        # 为每个空白区域生成理论
        for i, gap in enumerate(self.conceptual_gaps[:self.variants_per_contradiction]):
            self._log_info(f"为空白区域 {i+1} 生成理论...")
            
            try:
                # 构建基于空间的生成提示
                prompt = self._build_space_based_prompt(gap)
                
                # 调用LLM生成理论
                response = await self.llm_interface.query_async([
                    {"role": "user", "content": prompt}
                ], temperature=0.8)
                
                # 解析生成的理论
                theory_data = self._parse_generated_theory(response, gap, i+1)
                generated_theories.append(theory_data)
                
                self._log_info(f"✅ 空白区域 {i+1} 理论生成完成")
                
            except Exception as e:
                self._log_error(f"空白区域 {i+1} 生成失败: {e}")
                continue
        
        return generated_theories
    
    def _build_space_based_prompt(self, gap: Dict) -> str:
        """构建基于空间的生成提示"""
        gap_type = gap.get('type', 'unknown')
        description = gap.get('description', '')
        
        if gap_type == 'midpoint_gap':
            source_theories = gap.get('source_theories', [])
            prompt = f"""
基于概念空间分析，我们在量子理论的高维概念空间中发现了一个空白区域。
这个空白区域位于以下两个理论之间的概念空间中点：

理论A: {source_theories[0] if len(source_theories) > 0 else 'Unknown'}
理论B: {source_theories[1] if len(source_theories) > 1 else 'Unknown'}

请基于这个概念空间的位置，创造一个新的量子理论解释。这个新理论应该：

1. 融合上述两个理论的核心概念，但形成独特的理论框架
2. 在概念空间中占据这个空白区域，填补理论空白
3. 提供新的物理直觉和数学形式主义
4. 能够对量子现象给出独特的预测或解释

请提供：
- 理论名称
- 核心假设
- 主要概念
- 数学框架（如果适用）
- 与现有理论的区别
- 可能的实验预测

新理论：
"""
        
        elif gap_type == 'inter_cluster_gap':
            source_clusters = gap.get('source_clusters', [])
            prompt = f"""
基于概念空间的聚类分析，我们发现了两个理论聚类之间的概念空白区域：

聚类A包含理论: {', '.join(source_clusters[0]) if len(source_clusters) > 0 else 'Unknown'}
聚类B包含理论: {', '.join(source_clusters[1]) if len(source_clusters) > 1 else 'Unknown'}

请创造一个桥接这两个理论聚类的新量子理论。这个理论应该：

1. 结合两个聚类的核心思想
2. 在概念空间中形成连接桥梁
3. 提供跨聚类的统一理解框架
4. 开辟新的理论研究方向

请提供完整的理论描述，包括其在概念空间中的独特位置和价值。

桥接理论：
"""
        
        else:
            prompt = f"""
基于概念空间分析，发现了一个理论空白区域。请创造一个填补这个空白的新量子理论。

空白区域描述: {description}

请提供完整的理论框架。

新理论：
"""
        
        return prompt
    
    def _parse_generated_theory(self, response: str, gap: Dict, variant_id: int) -> Dict:
        """解析生成的理论"""
        # 提取理论名称
        theory_name = self._extract_theory_name(response)
        if not theory_name:
            theory_name = f"Space_Generated_Theory_{variant_id}"
        
        # 构建理论数据
        theory_data = {
            'name': theory_name,
            'content': response.strip(),
            'variant_info': {
                'generation_method': 'unified_space_based',
                'gap_type': gap.get('type', 'unknown'),
                'gap_description': gap.get('description', ''),
                'space_position': gap.get('position', []).tolist() if hasattr(gap.get('position', []), 'tolist') else [],
                'source_info': gap.get('source_theories', gap.get('source_clusters', [])),
                'variant_id': variant_id
            },
            'generation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'gap_analysis': self._get_space_analysis_summary(),
                'space_dimensions': self.concept_embedder.embedding_dim
            }
        }
        
        return theory_data
    
    def _extract_theory_name(self, text: str) -> Optional[str]:
        """从生成的文本中提取理论名称"""
        lines = text.split('\n')
        
        # 查找包含"理论"、"Theory"等关键词的行
        for line in lines:
            line = line.strip()
            if any(keyword in line for keyword in ['理论名称', '理论:', 'Theory:', 'Name:']):
                # 提取名称部分
                parts = line.split(':', 1)
                if len(parts) > 1:
                    return parts[1].strip()
            elif any(keyword in line for keyword in ['理论', 'Theory']) and len(line) < 100:
                return line.strip()
        
        return None
    
    def _get_space_analysis_summary(self) -> Dict[str, Any]:
        """获取空间分析摘要"""
        return {
            'total_theories': len(self.theory_space),
            'total_concepts': len(self.concept_space),
            'total_formulas': len(self.formula_space),
            'conceptual_gaps': len(self.conceptual_gaps),
            'space_density': self.semantic_dimensions.get('space_density', {}),
            'clustering_info': {
                'theory_clusters': len(self.semantic_dimensions.get('theory_clusters', [])),
                'concept_clusters': len(self.semantic_dimensions.get('concept_clusters', []))
            }
        }
    
    async def _save_space_data(self):
        """保存空间数据"""
        space_data = {
            'theory_space': {name: vector.tolist() for name, vector in self.theory_space.items()},
            'concept_space': {name: vector.tolist() for name, vector in self.concept_space.items()},
            'formula_space': {name: vector.tolist() for name, vector in self.formula_space.items()},
            'conceptual_gaps': [
                {
                    **gap,
                    'position': gap['position'].tolist() if hasattr(gap['position'], 'tolist') else gap['position']
                } for gap in self.conceptual_gaps
            ],
            'semantic_dimensions': self.semantic_dimensions,
            'analysis_summary': self._get_space_analysis_summary()
        }
        
        space_file = self.output_dir / "concept_space_data.json"
        with open(space_file, 'w', encoding='utf-8') as f:
            json.dump(space_data, f, ensure_ascii=False, indent=2)
        
        self._log_info(f"概念空间数据已保存: {space_file}")


# 为了兼容性，保持原有的适配器名称
class UnifiedGeneratorAdapter(UnifiedSpaceBasedGenerator):
    """统一生成器适配器（兼容性别名）"""
    pass


if __name__ == "__main__":
    # 测试代码
    import tempfile
    
    async def test_unified_generator():
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = UnifiedSpaceBasedGenerator(
                theories_dir="demo/theories",
                output_dir=temp_dir,
                model_source="google",
                model_name="gemini-2.5-flash"
            )
            
            result = await generator.generate()
            print(f"测试结果: {result['success']}")
            if result['success']:
                print(f"生成理论数量: {len(result['theories'])}")
                print(f"概念空间分析: {result['metadata']['space_analysis']}")
    
    # 如果直接运行此文件，执行测试
    asyncio.run(test_unified_generator()) 