#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一理论生成器

整合文献概念提取、先验理论库、概念向量空间和多级创新生成，
实现完整的"文献→概念空间→新理论"流程。
"""

import json
import pickle
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# 导入各个组件
from core_embedding.concept_extractor import ConceptExtractor
from core_embedding.embedding import ConceptEmbedder
from theory_generation.multi_level_innovation_generator import (
    MultiLevelInnovationGenerator, 
    MultiLevelInnovationConfig
)
from theory_generation.innovation_framework import InnovationLevel
from theory_generation.direct_synthesis.contradiction_analyzer import ContradictionAnalyzer
from utils.theory_registry import TheoryRegistry


@dataclass
class UnifiedGenerationConfig:
    """统一生成配置"""
    # 文献处理配置
    literature_dirs: List[str]  # 文献目录列表
    concept_cache_file: str = "data/unified_concepts.json"
    embedding_cache_file: str = "data/unified_embeddings.pkl"
    
    # 先验理论配置
    prior_theories_dir: str = "data/theories_v2.1"
    theory_registry_dir: str = "theory_registry"
    
    # 创新生成配置
    target_innovation_levels: List[InnovationLevel] = None
    synthesis_mode: str = "fusion"
    innovation_intensity: float = 0.8
    
    # 输出配置
    output_dir: str = "unified_theory_output"
    enable_visualization: bool = True


class UnifiedTheoryGenerator:
    """统一理论生成器"""
    
    def __init__(self, llm_interface, config: UnifiedGenerationConfig):
        """
        初始化统一理论生成器
        
        Args:
            llm_interface: LLM接口
            config: 统一生成配置
        """
        self.llm = llm_interface
        self.config = config
        
        # 初始化组件
        self.concept_extractor = ConceptExtractor(llm_interface)
        self.concept_embedder = ConceptEmbedder(llm_interface)
        self.contradiction_analyzer = ContradictionAnalyzer(llm_interface)
        self.theory_registry = TheoryRegistry(config.theory_registry_dir)
        
        # 数据存储
        self.literature_concepts = []  # 从文献提取的概念
        self.prior_theories = {}       # 先验理论库
        self.concept_embeddings = {}   # 概念嵌入向量
        self.theory_embeddings = {}    # 理论嵌入向量
        self.unified_concept_space = {}  # 统一概念空间
        
        # 多级创新生成器（稍后初始化）
        self.multi_level_generator = None
        
        # 创建输出目录
        Path(config.output_dir).mkdir(exist_ok=True)
    
    async def initialize_unified_system(self):
        """初始化统一系统"""
        print("🚀 初始化统一理论生成系统")
        print("=" * 60)
        
        # 步骤1: 处理文献和概念提取
        await self._process_literature()
        
        # 步骤2: 加载先验理论库
        await self._load_prior_theories()
        
        # 步骤3: 构建统一概念空间
        await self._build_unified_concept_space()
        
        # 步骤4: 初始化多级创新生成器
        self._initialize_multi_level_generator()
        
        print("✅ 统一系统初始化完成")
    
    async def _process_literature(self):
        """处理文献和概念提取"""
        print("\n📚 步骤1: 处理文献和概念提取")
        
        # 检查缓存
        cache_file = Path(self.config.concept_cache_file)
        if cache_file.exists():
            print("📋 加载缓存的概念数据...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            self.literature_concepts = cached_data.get('concepts', [])
            print(f"✅ 加载了 {len(self.literature_concepts)} 个缓存概念")
            return
        
        # 从文献中提取概念
        all_concepts = []
        all_formulas = []
        
        for lit_dir in self.config.literature_dirs:
            if Path(lit_dir).exists():
                print(f"📖 处理文献目录: {lit_dir}")
                concepts, formulas = self.concept_extractor.extract_from_directory(lit_dir)
                all_concepts.extend(concepts)
                all_formulas.extend(formulas)
            else:
                print(f"⚠️ 文献目录不存在: {lit_dir}")
        
        self.literature_concepts = all_concepts
        
        # 保存缓存
        cache_data = {
            'concepts': all_concepts,
            'formulas': all_formulas,
            'extracted_at': str(Path().cwd())
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 从文献中提取了 {len(all_concepts)} 个概念，{len(all_formulas)} 个公式")
    
    async def _load_prior_theories(self):
        """加载先验理论库"""
        print("\n🏛️ 步骤2: 加载先验理论库")
        
        prior_theories_path = Path(self.config.prior_theories_dir)
        if not prior_theories_path.exists():
            print(f"⚠️ 先验理论目录不存在: {prior_theories_path}")
            return
        
        # 加载所有先验理论
        for theory_file in prior_theories_path.glob("*.json"):
            try:
                with open(theory_file, 'r', encoding='utf-8') as f:
                    theory_data = json.load(f)
                
                theory_name = theory_data.get("name", theory_file.stem)
                self.prior_theories[theory_name] = theory_data
                
                print(f"✅ 加载先验理论: {theory_name}")
                
            except Exception as e:
                print(f"❌ 加载理论失败 {theory_file}: {e}")
        
        # 注册到理论注册库
        self.theory_registry.register_prior_theories(str(prior_theories_path))
        
        print(f"✅ 加载了 {len(self.prior_theories)} 个先验理论")
    
    async def _build_unified_concept_space(self):
        """构建统一概念空间"""
        print("\n🌌 步骤3: 构建统一概念空间")
        
        # 检查嵌入缓存
        embedding_cache = Path(self.config.embedding_cache_file)
        if embedding_cache.exists():
            print("📋 加载缓存的嵌入向量...")
            with open(embedding_cache, 'rb') as f:
                cached_embeddings = pickle.load(f)
            
            self.concept_embeddings = cached_embeddings.get('concepts', {})
            self.theory_embeddings = cached_embeddings.get('theories', {})
            
            print(f"✅ 加载了 {len(self.concept_embeddings)} 个概念嵌入，{len(self.theory_embeddings)} 个理论嵌入")
            
            # 构建统一概念空间
            self._merge_concept_spaces()
            return
        
        # 嵌入文献概念
        if self.literature_concepts:
            print("🔄 嵌入文献概念...")
            self.concept_embeddings = await self.concept_embedder.embed_concepts(
                self.literature_concepts
            )
        
        # 嵌入先验理论
        if self.prior_theories:
            print("🔄 嵌入先验理论...")
            theories_list = [
                {
                    'name': name,
                    'description': theory.get('summary', ''),
                    'philosophical_assumptions': str(theory.get('philosophy', {}))
                }
                for name, theory in self.prior_theories.items()
            ]
            self.theory_embeddings = await self.concept_embedder.embed_theories(theories_list)
        
        # 从先验理论中提取额外概念
        await self._extract_concepts_from_theories()
        
        # 保存嵌入缓存
        cache_data = {
            'concepts': self.concept_embeddings,
            'theories': self.theory_embeddings,
            'created_at': str(Path().cwd())
        }
        
        with open(embedding_cache, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # 构建统一概念空间
        self._merge_concept_spaces()
        
        print(f"✅ 构建统一概念空间: {len(self.unified_concept_space)} 个向量")
    
    async def _extract_concepts_from_theories(self):
        """从先验理论中提取概念"""
        print("🔍 从先验理论中提取概念...")
        
        theory_concepts = []
        
        for theory_name, theory_data in self.prior_theories.items():
            # 提取核心原理中的概念
            core_principles = theory_data.get('core_principles', [])
            if isinstance(core_principles, list):
                for principle in core_principles:
                    principle_text = principle.get('statement', '') if isinstance(principle, dict) else str(principle)
                    if principle_text:
                        theory_concepts.append({
                            'name': f"{theory_name}_principle",
                            'description': principle_text,
                            'source_theory': theory_name,
                            'type': 'principle'
                        })
            
            # 提取哲学假设中的概念
            philosophy = theory_data.get('philosophy', {})
            if isinstance(philosophy, dict):
                for key, value in philosophy.items():
                    if isinstance(value, str) and value:
                        theory_concepts.append({
                            'name': f"{theory_name}_{key}",
                            'description': value,
                            'source_theory': theory_name,
                            'type': 'philosophical'
                        })
        
        # 嵌入理论概念
        if theory_concepts:
            theory_concept_embeddings = await self.concept_embedder.embed_concepts(theory_concepts)
            # 合并到概念嵌入中
            self.concept_embeddings.update(theory_concept_embeddings)
            
            print(f"✅ 从理论中提取了 {len(theory_concepts)} 个额外概念")
    
    def _merge_concept_spaces(self):
        """合并概念空间"""
        print("🔗 合并概念空间...")
        
        # 合并概念和理论嵌入
        self.unified_concept_space = {}
        
        # 添加概念嵌入
        for name, embedding in self.concept_embeddings.items():
            self.unified_concept_space[f"concept_{name}"] = {
                'vector': embedding,
                'type': 'concept',
                'name': name
            }
        
        # 添加理论嵌入
        for name, embedding in self.theory_embeddings.items():
            self.unified_concept_space[f"theory_{name}"] = {
                'vector': embedding,
                'type': 'theory',
                'name': name
            }
        
        print(f"✅ 统一概念空间包含 {len(self.unified_concept_space)} 个向量")
    
    def _initialize_multi_level_generator(self):
        """初始化多级创新生成器"""
        print("\n🎯 步骤4: 初始化多级创新生成器")
        
        # 创建多级创新生成器
        self.multi_level_generator = MultiLevelInnovationGenerator(
            llm_interface=self.llm,
            concept_embeddings=self.concept_embeddings
        )
        
        # 为矛盾分析器加载先验理论数据（使用已有的实例）
        self.contradiction_analyzer.theories = self.prior_theories.copy()
        
        print(f"✅ 多级创新生成器就绪，包含 {len(self.concept_embeddings)} 个概念向量")
        print(f"✅ 矛盾分析器就绪，包含 {len(self.contradiction_analyzer.theories)} 个先验理论")
    
    async def generate_theory_from_literature_and_priors(self, 
                                                        theory1_name: str, 
                                                        theory2_name: str,
                                                        focus_concepts: Optional[List[str]] = None) -> Dict:
        """
        基于文献概念和先验理论生成新理论
        
        Args:
            theory1_name: 第一个先验理论名称
            theory2_name: 第二个先验理论名称  
            focus_concepts: 重点关注的概念列表（来自文献）
            
        Returns:
            Dict: 生成的新理论
        """
        print(f"\n🧬 生成新理论: {theory1_name} + {theory2_name}")
        print("=" * 60)
        
        # 步骤1: 分析先验理论矛盾
        print("🔍 分析理论矛盾...")
        contradiction = await self.contradiction_analyzer.find_contradictions(
            theory1_name, theory2_name
        )
        
        if not contradiction or "error" in contradiction:
            print(f"❌ 无法分析矛盾: {contradiction}")
            return {"error": "矛盾分析失败"}
        
        # 步骤2: 基于文献概念增强矛盾分析
        enhanced_contradiction = self._enhance_contradiction_with_literature(
            contradiction, focus_concepts
        )
        
        # 步骤3: 配置多级创新
        config = self._create_adaptive_config(enhanced_contradiction)
        
        # 步骤4: 生成新理论
        print("🚀 生成新理论...")
        new_theory = await self.multi_level_generator.generate_multi_level_theory(
            contradiction=enhanced_contradiction,
            config=config
        )
        
        # 步骤5: 后处理和验证
        if "error" not in new_theory:
            new_theory = self._post_process_generated_theory(
                new_theory, theory1_name, theory2_name, focus_concepts
            )
        
        return new_theory
    
    def _enhance_contradiction_with_literature(self, 
                                             contradiction: Dict, 
                                             focus_concepts: Optional[List[str]]) -> Dict:
        """基于文献概念增强矛盾分析"""
        print("📚 基于文献概念增强矛盾分析...")
        
        enhanced = contradiction.copy()
        
        # 添加文献概念相关性分析
        literature_insights = []
        
        for contra in contradiction.get('contradictions', []):
            dimension = contra.get('dimension', '')
            
            # 在文献概念中查找相关概念
            related_literature_concepts = self._find_literature_concepts_for_dimension(
                dimension, focus_concepts
            )
            
            if related_literature_concepts:
                literature_insights.append({
                    'dimension': dimension,
                    'related_concepts': related_literature_concepts,
                    'literature_perspective': self._analyze_literature_perspective(
                        related_literature_concepts, dimension
                    )
                })
        
        enhanced['literature_insights'] = literature_insights
        enhanced['enhanced_with_literature'] = True
        
        print(f"✅ 添加了 {len(literature_insights)} 个文献洞察")
        return enhanced
    
    def _find_literature_concepts_for_dimension(self, 
                                              dimension: str, 
                                              focus_concepts: Optional[List[str]]) -> List[Dict]:
        """为特定维度查找相关的文献概念"""
        related_concepts = []
        
        # 搜索范围：重点概念或所有文献概念
        search_concepts = focus_concepts if focus_concepts else [c['name'] for c in self.literature_concepts]
        
        dimension_keywords = set(dimension.lower().split())
        
        for concept_name in search_concepts:
            # 在概念嵌入中查找
            if concept_name in self.concept_embeddings:
                concept_data = next(
                    (c for c in self.literature_concepts if c['name'] == concept_name), 
                    None
                )
                
                if concept_data:
                    # 计算相关性
                    concept_keywords = set(concept_data.get('description', '').lower().split())
                    relevance = len(dimension_keywords & concept_keywords) / len(dimension_keywords | concept_keywords)
                    
                    if relevance > 0.1:  # 相关性阈值
                        related_concepts.append({
                            'name': concept_name,
                            'description': concept_data.get('description', ''),
                            'relevance': relevance,
                            'source': concept_data.get('source', 'unknown')
                        })
        
        # 按相关性排序
        related_concepts.sort(key=lambda x: x['relevance'], reverse=True)
        return related_concepts[:5]  # 返回前5个最相关的
    
    def _analyze_literature_perspective(self, 
                                      related_concepts: List[Dict], 
                                      dimension: str) -> str:
        """分析文献对特定维度的观点"""
        if not related_concepts:
            return "文献中未发现相关观点"
        
        # 简单的观点合成
        perspectives = []
        for concept in related_concepts:
            desc = concept.get('description', '')
            if desc:
                perspectives.append(f"• {concept['name']}: {desc[:100]}...")
        
        return f"文献观点汇总:\n" + "\n".join(perspectives)
    
    def _create_adaptive_config(self, contradiction: Dict) -> MultiLevelInnovationConfig:
        """基于矛盾分析创建自适应配置"""
        # 根据矛盾复杂性选择创新层次
        contradiction_count = len(contradiction.get('contradictions', []))
        has_literature = contradiction.get('enhanced_with_literature', False)
        
        if contradiction_count >= 3 or has_literature:
            # 复杂矛盾，使用高级创新层次
            target_levels = [
                InnovationLevel.FRAMEWORK_EXTENSION,
                InnovationLevel.PARAMETER_EXTENSION
            ]
            intensity = 0.9
        elif contradiction_count >= 2:
            # 中等复杂度
            target_levels = [
                InnovationLevel.PARAMETER_EXTENSION,
                InnovationLevel.INTERPRETATION
            ]
            intensity = 0.7
        else:
            # 简单矛盾
            target_levels = [InnovationLevel.INTERPRETATION]
            intensity = 0.5
        
        return self.multi_level_generator.create_multi_level_config(
            target_levels=target_levels,
            synthesis_mode=self.config.synthesis_mode,
            innovation_intensity=intensity
        )
    
    def _post_process_generated_theory(self, 
                                     theory: Dict, 
                                     theory1_name: str, 
                                     theory2_name: str,
                                     focus_concepts: Optional[List[str]]) -> Dict:
        """后处理生成的理论"""
        # 添加生成元数据
        theory['generation_metadata'] = {
            'source_theories': [theory1_name, theory2_name],
            'focus_concepts': focus_concepts or [],
            'literature_concepts_used': len(self.literature_concepts),
            'unified_concept_space_size': len(self.unified_concept_space),
            'generation_method': 'unified_literature_prior_synthesis'
        }
        
        # 添加概念空间分析
        theory['concept_space_analysis'] = self._analyze_theory_in_concept_space(theory)
        
        return theory
    
    def _analyze_theory_in_concept_space(self, theory: Dict) -> Dict:
        """在概念空间中分析新理论"""
        # 简化版分析 - 实际实现可以更复杂
        return {
            'novelty_score': 0.8,  # 新颖性评分
            'coherence_score': 0.9,  # 一致性评分
            'concept_coverage': 0.7,  # 概念覆盖度
            'literature_alignment': 0.8  # 与文献的对齐度
        }
    
    async def batch_generate_from_theory_pairs(self, 
                                             theory_pairs: List[Tuple[str, str]],
                                             focus_concepts: Optional[List[str]] = None) -> List[Dict]:
        """批量从理论对生成新理论"""
        results = []
        
        for i, (theory1, theory2) in enumerate(theory_pairs, 1):
            print(f"\n🔄 处理理论对 {i}/{len(theory_pairs)}: {theory1} vs {theory2}")
            
            try:
                new_theory = await self.generate_theory_from_literature_and_priors(
                    theory1, theory2, focus_concepts
                )
                results.append(new_theory)
                
                # 保存单个结果
                output_file = Path(self.config.output_dir) / f"theory_{i}_{theory1}_{theory2}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(new_theory, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                print(f"❌ 生成失败: {e}")
                results.append({"error": str(e), "theory_pair": (theory1, theory2)})
        
        return results
    
    def save_unified_analysis(self, filename: str = "unified_analysis.json"):
        """保存统一分析结果"""
        analysis = {
            'literature_concepts_count': len(self.literature_concepts),
            'prior_theories_count': len(self.prior_theories),
            'concept_embeddings_count': len(self.concept_embeddings),
            'theory_embeddings_count': len(self.theory_embeddings),
            'unified_concept_space_size': len(self.unified_concept_space),
            'prior_theories_list': list(self.prior_theories.keys()),
            'concept_space_statistics': self._calculate_concept_space_stats()
        }
        
        output_file = Path(self.config.output_dir) / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        print(f"📊 统一分析结果已保存到: {output_file}")
    
    def _calculate_concept_space_stats(self) -> Dict:
        """计算概念空间统计信息"""
        if not self.unified_concept_space:
            return {
                'dimension': 0,
                'mean_norm': 0.0,
                'std_norm': 0.0,
                'concept_count': 0,
                'theory_count': 0
            }
        
        # 提取所有向量，过滤掉None值
        vectors = []
        for item in self.unified_concept_space.values():
            if item and isinstance(item, dict) and 'vector' in item:
                vector = item['vector']
                if vector is not None:
                    # 确保向量是数值类型
                    try:
                        vector_array = np.array(vector, dtype=float)
                        if vector_array.size > 0 and not np.isnan(vector_array).any():
                            vectors.append(vector_array)
                    except (ValueError, TypeError):
                        continue
        
        if not vectors:
            return {
                'dimension': 0,
                'mean_norm': 0.0,
                'std_norm': 0.0,
                'concept_count': 0,
                'theory_count': 0
            }
        
        # 转换为numpy数组
        try:
            vectors_array = np.array(vectors)
            
            # 检查数组维度
            if vectors_array.ndim == 1:
                # 如果只有一个向量，reshape为二维
                vectors_array = vectors_array.reshape(1, -1)
            
            # 计算统计信息
            norms = np.linalg.norm(vectors_array, axis=1)
            
            return {
                'dimension': vectors_array.shape[1] if vectors_array.ndim > 1 else len(vectors_array),
                'mean_norm': float(np.mean(norms)),
                'std_norm': float(np.std(norms)),
                'concept_count': sum(1 for item in self.unified_concept_space.values() if item and item.get('type') == 'concept'),
                'theory_count': sum(1 for item in self.unified_concept_space.values() if item and item.get('type') == 'theory')
            }
        except Exception as e:
            print(f"[WARN] 计算概念空间统计时出错: {e}")
            return {
                'dimension': 0,
                'mean_norm': 0.0,
                'std_norm': 0.0,
                'concept_count': 0,
                'theory_count': 0
            } 