"""
unified_generator_adapter_optimized.py - 优化版统一生成器适配器

优化点：
1. 添加缓存机制，避免重复加载
2. 异步并行加载各种数据
3. 添加详细的进度日志
4. 可配置的加载选项
5. 更智能的嵌入批处理
6. 继承原始类，复用大部分逻辑
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime
import time
import pickle

try:
    from .unified_generator_adapter import UnifiedGeneratorAdapter, UnifiedSpaceBasedGenerator
    from .base_adapter import GenerationResult
except ImportError:
    from unified_generator_adapter import UnifiedGeneratorAdapter, UnifiedSpaceBasedGenerator
    from base_adapter import GenerationResult

from utils.logging_config import get_logger


class UnifiedGeneratorAdapterOptimized(UnifiedSpaceBasedGenerator):
    """
    优化版基于高维概念空间的统一理论生成器适配器
    继承自原始版本，仅优化性能瓶颈部分
    """
    
    def __init__(self, use_structured_format: bool = True, **kwargs):
        """初始化优化版生成器"""
        # 优化配置
        self.optimization_config = {
            'use_cache': kwargs.get('use_cache', True),
            'cache_dir': Path(kwargs.get('cache_dir', 'cache/unified_generator')),
            'batch_size': kwargs.get('batch_size', 50),
            'max_concepts': kwargs.get('max_concepts', 1000 if not kwargs.get('test_mode', False) else 50),
            'max_formulas': kwargs.get('max_formulas', 500 if not kwargs.get('test_mode', False) else 25),
            'max_literature_concepts': kwargs.get('max_literature_concepts', 500 if not kwargs.get('test_mode', False) else 20),
            'enable_physics_embedder': kwargs.get('enable_physics_embedder', False),
            'enable_knowledge_graph': kwargs.get('enable_knowledge_graph', False),
            'parallel_workers': kwargs.get('parallel_workers', 4),
            'progress_interval': kwargs.get('progress_interval', 10),
            'skip_visualization': kwargs.get('skip_visualization', True),  # 默认跳过可视化
            'lazy_load_knowledge_graph': kwargs.get('lazy_load_knowledge_graph', True)  # 延迟加载知识图谱
        }
        
        # 确保缓存目录存在
        self.optimization_config['cache_dir'].mkdir(parents=True, exist_ok=True)
        
        # 调用父类初始化
        super().__init__(use_structured_format=use_structured_format, **kwargs)
        
        # 添加优化相关的logger
        self.opt_logger = get_logger('unified_generator_optimized')
        
        # 缓存标记
        self._cache_loaded = False
        self._space_build_time = 0
        
    def _get_cache_path(self, cache_type: str) -> Path:
        """获取缓存文件路径"""
        theories_hash = hash(str(self.theories_dir))
        return self.optimization_config['cache_dir'] / f"{cache_type}_{theories_hash}.pkl"
    
    def _load_cache(self, cache_type: str) -> Optional[Any]:
        """加载缓存数据"""
        if not self.optimization_config['use_cache']:
            return None
            
        cache_path = self._get_cache_path(cache_type)
        if cache_path.exists():
            try:
                # 检查缓存是否过期（24小时）
                cache_age = time.time() - cache_path.stat().st_mtime
                if cache_age > 86400:  # 24 hours
                    self.opt_logger.info(f"Cache {cache_type} is too old, skipping")
                    return None
                
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                self.opt_logger.info(f"✅ Loaded {cache_type} from cache")
                self._cache_loaded = True
                return data
            except Exception as e:
                self.opt_logger.error(f"Failed to load cache {cache_type}: {e}")
        return None
    
    def _save_cache(self, cache_type: str, data: Any):
        """保存数据到缓存"""
        if not self.optimization_config['use_cache']:
            return
            
        cache_path = self._get_cache_path(cache_type)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            self.opt_logger.info(f"💾 Saved {cache_type} to cache")
        except Exception as e:
            self.opt_logger.error(f"Failed to save cache {cache_type}: {e}")
    
    def _initialize_enhanced_components(self):
        """重写：优化增强组件初始化"""
        # 物理嵌入器
        if self.optimization_config['enable_physics_embedder']:
            try:
                from physics_embedder import PhysicsEmbedder
                self.physics_embedder = PhysicsEmbedder(base_embedder=self.concept_embedder)
                self._log_info("✅ Physics embedder initialized")
            except ImportError:
                self._log_info("Physics embedder not available")
        else:
            self._log_info("Physics embedder disabled in optimization config")
        
        # 知识图谱（延迟加载）
        if self.optimization_config['enable_knowledge_graph'] and not self.optimization_config['lazy_load_knowledge_graph']:
            self._load_knowledge_graph_component()
        else:
            self._log_info("Knowledge graph loading deferred or disabled")
    
    def _load_knowledge_graph_component(self):
        """加载知识图谱组件"""
        try:
            # 首先检查缓存
            cached_kg = self._load_cache('knowledge_graph')
            if cached_kg:
                self.knowledge_graph = cached_kg
                return
            
            # 否则正常加载
            super()._initialize_enhanced_components()  # 调用父类的知识图谱加载逻辑
            
            # 保存到缓存
            if self.knowledge_graph:
                self._save_cache('knowledge_graph', self.knowledge_graph)
        except Exception as e:
            self._log_error(f"Failed to load knowledge graph: {e}")
    
    async def _async_generate(self) -> Dict[str, Any]:
        """重写：优化的异步生成流程"""
        try:
            start_time = time.time()
            self._log_info("🚀 Starting optimized unified theory generation")
            
            # 1. 加载和预处理理论（使用缓存）
            theories = self._load_theories_with_cache()
            if not theories:
                return GenerationResult(
                    success=False,
                    error_message="Unable to load prior theories"
                ).to_dict()
            
            # 2. 构建多层次概念空间（优化版）
            space_start = time.time()
            await self._build_multilevel_concept_space_optimized(theories)
            self._space_build_time = time.time() - space_start
            self._log_info(f"✅ Space construction completed in {self._space_build_time:.2f}s")
            
            # 3. 分析概念空间结构
            self._log_info("🔍 Analyzing space structure...")
            analysis_start = time.time()
            self._analyze_space_structure()
            self._log_info(f"Space analysis completed in {time.time() - analysis_start:.2f}s")
            
            # 4. 延迟加载知识图谱（如果需要）
            if self.optimization_config['lazy_load_knowledge_graph'] and self.optimization_config['enable_knowledge_graph']:
                self._log_info("Loading knowledge graph for gap analysis...")
                kg_start = time.time()
                self._load_knowledge_graph_component()
                self._log_info(f"Knowledge graph loaded in {time.time() - kg_start:.2f}s")
            
            # 5. 识别概念空白区域
            self._log_info("🎯 Identifying conceptual gaps...")
            gaps_start = time.time()
            if self.physics_embedder and self.knowledge_graph:
                self._identify_high_value_gaps()
            else:
                self._identify_conceptual_gaps()
            self._log_info(f"Gap identification completed in {time.time() - gaps_start:.2f}s")
            
            # 6. 基于空间生成新理论
            self._log_info("🧬 Generating theories from space...")
            gen_start = time.time()
            generated_theories = await self._generate_from_space()
            self._log_info(f"Theory generation completed in {time.time() - gen_start:.2f}s")
            
            # 7. 构建结果
            result = GenerationResult(
                success=True,
                theories=generated_theories,
                metadata={
                    'generation_method': 'unified_space_based_optimized',
                    'concept_space_size': len(self.concept_space),
                    'theory_space_size': len(self.theory_space),
                    'conceptual_gaps_found': len(self.conceptual_gaps),
                    'semantic_dimensions': len(self.semantic_dimensions),
                    'space_analysis': self._get_space_analysis_summary(),
                    'optimization_stats': {
                        'total_time': time.time() - start_time,
                        'space_build_time': self._space_build_time,
                        'cache_used': self._cache_loaded,
                        'concepts_limited': len(self.concept_space) == self.optimization_config['max_concepts'],
                        'formulas_limited': len(self.formula_space) == self.optimization_config['max_formulas']
                    }
                },
                output_dir=str(self.output_dir)
            )
            
            # 8. 保存空间数据和结果
            await self._save_space_data()
            self._save_result(result.to_dict())
            
            # 9. 生成概念空间可视化（可选）
            if not self.optimization_config['skip_visualization']:
                self._generate_space_visualization()
            
            total_time = time.time() - start_time
            self._log_info(f"✅ Optimized generation complete in {total_time:.2f}s: generated {len(generated_theories)} theories")
            return result.to_dict()
            
        except Exception as e:
            self._log_error(f"Unified space generation failed: {e}")
            import traceback
            traceback.print_exc()
            return GenerationResult(
                success=False,
                error_message=str(e)
            ).to_dict()
    
    def _load_theories_with_cache(self) -> List[Dict]:
        """使用缓存加载理论"""
        # 检查缓存
        cached_theories = self._load_cache('theories')
        if cached_theories:
            return cached_theories
        
        # 否则正常加载
        theories = self._load_theories()
        
        # 保存到缓存
        if theories:
            self._save_cache('theories', theories)
        
        return theories
    
    async def _build_multilevel_concept_space_optimized(self, theories: List[Dict]):
        """优化版构建多层次概念空间"""
        self._log_info("🏗️ Building multi-level conceptual space (optimized)...")
        
        # 并行执行所有空间构建任务
        tasks = []
        
        # 1. 理论空间构建任务
        tasks.append(self._build_theory_space_batch(theories))
        
        # 2. 概念空间构建任务
        tasks.append(self._build_concept_space_batch(theories))
        
        # 3. 公式空间构建任务（如果需要）
        if not self.test_mode:
            tasks.append(self._build_formula_space_batch(theories))
        
        # 4. 文献概念加载任务（如果启用）
        # 检查是否启用了文献概念
        use_raw_literature = getattr(self, 'use_raw_literature', self.kwargs.get('use_raw_literature', False))
        if use_raw_literature:
            tasks.append(self._load_literature_concepts_batch())
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理任何错误
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._log_error(f"Space building task {i} failed: {result}")
    
    async def _build_theory_space_batch(self, theories: List[Dict]):
        """批量构建理论空间"""
        self._log_info("📚 Building theory space (batch mode)...")
        
        # 检查缓存
        cached_space = self._load_cache('theory_space')
        if cached_space:
            self.theory_space = cached_space
            return
        
        # 批量嵌入
        batch_size = self.optimization_config['batch_size']
        self.theory_space = {}
        
        for i in range(0, len(theories), batch_size):
            batch = theories[i:i+batch_size]
            if i % self.optimization_config['progress_interval'] == 0:
                self._log_info(f"Embedding theory batch {i//batch_size + 1}/{(len(theories) + batch_size - 1)//batch_size}")
            
            batch_embeddings = await self.concept_embedder.embed_theories(batch)
            self.theory_space.update(batch_embeddings)
        
        # 保存到缓存
        self._save_cache('theory_space', self.theory_space)
        self._log_info(f"✅ Theory space built: {len(self.theory_space)} theories")
    
    async def _build_concept_space_batch(self, theories: List[Dict]):
        """批量构建概念空间"""
        self._log_info("💡 Building concept space (batch mode)...")
        
        # 检查缓存
        cached_space = self._load_cache('concept_space')
        if cached_space:
            self.concept_space = cached_space
            return
        
        # 提取概念
        concepts = self._extract_concepts_from_theories(theories)
        
        # 限制概念数量
        if len(concepts) > self.optimization_config['max_concepts']:
            self._log_info(f"Limiting concepts from {len(concepts)} to {self.optimization_config['max_concepts']}")
            # 按重要性排序（如果有重要性分数）
            if hasattr(self, 'concept_importance') and self.concept_importance:
                sorted_concepts = sorted(concepts, 
                                       key=lambda c: self.concept_importance.get(c.get('name', ''), 0), 
                                       reverse=True)
                concepts = sorted_concepts[:self.optimization_config['max_concepts']]
            else:
                concepts = concepts[:self.optimization_config['max_concepts']]
        
        if concepts:
            # 批量嵌入
            batch_size = self.optimization_config['batch_size']
            self.concept_space = {}
            
            for i in range(0, len(concepts), batch_size):
                batch = concepts[i:i+batch_size]
                if i % self.optimization_config['progress_interval'] == 0:
                    self._log_info(f"Embedding concept batch {i//batch_size + 1}/{(len(concepts) + batch_size - 1)//batch_size}")
                
                if self.physics_embedder and self.optimization_config['enable_physics_embedder']:
                    batch_embeddings = await self._embed_concepts_with_physics(batch)
                else:
                    batch_embeddings = await self.concept_embedder.embed_concepts(batch)
                
                self.concept_space.update(batch_embeddings)
            
            # 保存到缓存
            self._save_cache('concept_space', self.concept_space)
            self._log_info(f"✅ Concept space built: {len(self.concept_space)} concepts")
    
    async def _build_formula_space_batch(self, theories: List[Dict]):
        """批量构建公式空间"""
        self._log_info("🔢 Building formula space (batch mode)...")
        
        # 检查缓存
        cached_space = self._load_cache('formula_space')
        if cached_space:
            self.formula_space = cached_space
            return
        
        # 提取公式
        formulas = self._extract_formulas_from_theories(theories)
        
        # 限制公式数量
        if len(formulas) > self.optimization_config['max_formulas']:
            self._log_info(f"Limiting formulas from {len(formulas)} to {self.optimization_config['max_formulas']}")
            formulas = formulas[:self.optimization_config['max_formulas']]
        
        if formulas:
            # 批量嵌入
            batch_size = self.optimization_config['batch_size']
            self.formula_space = {}
            
            for i in range(0, len(formulas), batch_size):
                batch = formulas[i:i+batch_size]
                if i % self.optimization_config['progress_interval'] == 0:
                    self._log_info(f"Embedding formula batch {i//batch_size + 1}/{(len(formulas) + batch_size - 1)//batch_size}")
                
                if self.physics_embedder and self.optimization_config['enable_physics_embedder']:
                    batch_embeddings = await self._embed_formulas_with_physics(batch)
                else:
                    batch_embeddings = await self.concept_embedder.embed_formulas(batch)
                
                self.formula_space.update(batch_embeddings)
            
            # 保存到缓存
            self._save_cache('formula_space', self.formula_space)
            self._log_info(f"✅ Formula space built: {len(self.formula_space)} formulas")
    
    async def _load_literature_concepts_batch(self):
        """批量加载文献概念"""
        if self.test_mode:
            self._log_info("🧪 Test mode: using minimal literature concepts")
            return
        
        self._log_info("📖 Loading literature concepts (batch mode)...")
        
        # 检查缓存
        cached_concepts = self._load_cache('literature_concepts_embedded')
        if cached_concepts:
            # 直接添加到概念空间
            self.concept_space.update(cached_concepts)
            self._log_info(f"✅ Loaded {len(cached_concepts)} literature concepts from cache")
            return
        
        # 加载原始概念
        concepts = self._load_literature_concepts()
        
        if concepts:
            # 限制数量
            max_lit_concepts = self.optimization_config['max_literature_concepts']
            if len(concepts) > max_lit_concepts:
                self._log_info(f"Limiting literature concepts from {len(concepts)} to {max_lit_concepts}")
                concepts = concepts[:max_lit_concepts]
            
            # 批量嵌入
            batch_size = self.optimization_config['batch_size']
            literature_embeddings = {}
            
            for i in range(0, len(concepts), batch_size):
                batch = concepts[i:i+batch_size]
                if i % self.optimization_config['progress_interval'] == 0:
                    self._log_info(f"Embedding literature batch {i//batch_size + 1}/{(len(concepts) + batch_size - 1)//batch_size}")
                
                if self.physics_embedder and self.optimization_config['enable_physics_embedder']:
                    batch_embeddings = await self._embed_concepts_with_physics(batch)
                else:
                    batch_embeddings = await self.concept_embedder.embed_concepts(batch)
                
                # 添加前缀以区分文献概念
                for name, embedding in batch_embeddings.items():
                    literature_embeddings[f"lit_{name}"] = embedding
            
            # 添加到概念空间
            self.concept_space.update(literature_embeddings)
            
            # 保存到缓存
            self._save_cache('literature_concepts_embedded', literature_embeddings)
            self._log_info(f"✅ Added {len(literature_embeddings)} literature concepts to space")
    
    def _generate_space_visualization(self):
        """重写：可选的空间可视化"""
        if self.optimization_config['skip_visualization']:
            self._log_info("Skipping visualization (optimization enabled)")
            return
        
        # 调用父类的可视化方法
        super()._generate_space_visualization()
    
    # 添加一个方法来清理缓存
    def clear_cache(self):
        """清理所有缓存文件"""
        cache_types = ['theories', 'theory_space', 'concept_space', 'formula_space', 
                      'literature_concepts_embedded', 'knowledge_graph']
        
        for cache_type in cache_types:
            cache_path = self._get_cache_path(cache_type)
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    self.opt_logger.info(f"Cleared cache: {cache_type}")
                except Exception as e:
                    self.opt_logger.error(f"Failed to clear cache {cache_type}: {e}")


# 为了兼容性，保持原有的适配器名称
class UnifiedGeneratorAdapter(UnifiedGeneratorAdapterOptimized):
    """统一生成器适配器（使用优化版本）"""
    pass