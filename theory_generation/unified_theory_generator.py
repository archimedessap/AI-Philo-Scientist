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

# 导入日志系统
from utils.logging_config import get_logger, log_config, log_error_with_context, log_execution
# 导入缓存管理器
from utils.cache_manager import CacheManager

# 导入各个组件
from core_embedding.concept_extractor import ConceptExtractor
from core_embedding.embedding import ConceptEmbedder
from theory_generation.multi_level_innovation_generator import (
    MultiLevelInnovationGenerator, 
    MultiLevelInnovationConfig
)
from theory_generation.innovation_framework import InnovationLevel
from theory_generation.direct_synthesis.contradiction_analyzer import ContradictionAnalyzer
from pipelines.retrieve_topk import CardRetriever
from utils.theory_registry import TheoryRegistry

DEFAULT_CARD_CONSTRAINTS = {
    "hard_rules": [
        "Respect observed quantum data and no-signalling.",
        "Prefer minimal departures from standard quantum dynamics unless required.",
        "Ground any new mechanisms in clear math that extends the selected cards."
    ],
    "nice_to_have": [
        "Clarify the origin of the Born rule.",
        "Explain how classical objectivity emerges.",
        "Highlight discriminating experiments or predictions."
    ]
}



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
    cards_dir: str = "cards"
    card_schema_path: str = "schemas/card.schema.json"
    contradiction_schema_path: str = "schemas/contradiction.schema.json"
    new_interpretation_schema_path: str = "schemas/new_interpretation.schema.json"
    
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
        # 初始化日志器
        self.logger = get_logger('unified_theory_generator', module_specific=True)
        self.logger.info("初始化统一理论生成器")
        
        self.llm = llm_interface
        self.config = config
        
        # 记录配置
        log_config(config.__dict__, 'unified_config')
        
        # 初始化缓存管理器
        self.cache_manager = CacheManager(
            cache_dir="cache/unified_theory",
            version="1.1.0"  # 使用新版本号
        )
        
        try:
            # 初始化组件
            self.concept_extractor = ConceptExtractor(llm_interface)
            self.concept_embedder = ConceptEmbedder(llm_interface)
            self.theory_registry = TheoryRegistry(config.theory_registry_dir)

            # 短卡工作流组件
            self.cards_dir = Path(config.cards_dir)
            self.card_retriever = None
            self.card_analyzer = None
            self.new_interpretation_schema = None
            self.last_card_result = None
            if self.cards_dir.exists():
                try:
                    self.card_retriever = CardRetriever(
                        llm_interface,
                        cards_dir=str(self.cards_dir),
                        schema_path=config.card_schema_path
                    )
                    self.card_analyzer = ContradictionAnalyzer(
                        llm_interface,
                        schema_path=config.contradiction_schema_path
                    )
                    schema_text = Path(config.new_interpretation_schema_path).read_text(encoding='utf-8')
                    self.new_interpretation_schema = json.loads(schema_text)
                    self.logger.info("Short-card workflow initialized.")
                except Exception as exc:
                    self.logger.warning(f"Short-card workflow initialization failed: {exc}")
            else:
                self.logger.warning(f"Cards directory not found, short-card workflow disabled: {config.cards_dir}")

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
            self.logger.info(f"输出目录已创建: {config.output_dir}")
            
        except Exception as e:
            log_error_with_context(e, {
                'config': config.__dict__,
                'stage': 'initialization'
            })
            raise
    
    @log_execution('unified_theory_generator')
    async def initialize_unified_system(self):
        """初始化统一系统"""
        self.logger.info("🚀 开始初始化统一理论生成系统")
        print("🚀 初始化统一理论生成系统")
        print("=" * 60)
        
        try:
            # 步骤1: 处理文献和概念提取
            self.logger.info("步骤1: 处理文献和概念提取")
            await self._process_literature()
            
            # 步骤2: 加载先验理论库
            self.logger.info("步骤2: 加载先验理论库")
            await self._load_prior_theories()
            
            # 步骤3: 构建统一概念空间
            self.logger.info("步骤3: 构建统一概念空间")
            await self._build_unified_concept_space()
            
            # 步骤4: 初始化多级创新生成器
            self.logger.info("步骤4: 初始化多级创新生成器")
            self._initialize_multi_level_generator()
            
            self.logger.info("✅ 统一系统初始化完成")
            print("✅ 统一系统初始化完成")
            
            # 记录初始化统计
            self.logger.info(f"初始化统计:")
            self.logger.info(f"  - 文献概念数: {len(self.literature_concepts)}")
            self.logger.info(f"  - 先验理论数: {len(self.prior_theories)}")
            self.logger.info(f"  - 概念嵌入数: {len(self.concept_embeddings)}")
            self.logger.info(f"  - 统一概念空间大小: {len(self.unified_concept_space)}")
            
        except Exception as e:
            self.logger.error("系统初始化失败")
            log_error_with_context(e, {
                'stage': 'system_initialization',
                'literature_concepts': len(self.literature_concepts),
                'prior_theories': len(self.prior_theories)
            })
            raise
    
    async def _process_literature(self):
        """处理文献和概念提取"""
        print("\n📚 步骤1: 处理文献和概念提取")
        self.logger.info("开始处理文献和概念提取")
        
        # 尝试从新缓存系统加载
        cache_key = "literature_concepts"
        cached_data = self.cache_manager.load(cache_key, cache_type='json')
        
        if cached_data:
            self.literature_concepts = cached_data.get('concepts', [])
            self.logger.info(f"从缓存加载了 {len(self.literature_concepts)} 个概念")
            print(f"📋 从缓存加载了 {len(self.literature_concepts)} 个概念")
            return
        
        # 兼容旧缓存文件
        cache_file = Path(self.config.concept_cache_file)
        if cache_file.exists():
            print("📋 发现旧缓存文件，正在迁移...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # 保存到新缓存系统
            self.cache_manager.save(
                cache_key,
                cached_data,
                cache_type='json',
                ttl_hours=24*7  # 7天有效期
            )
            
            self.literature_concepts = cached_data.get('concepts', [])
            print(f"✅ 已迁移 {len(self.literature_concepts)} 个概念到新缓存系统")
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

        print(f"✅ 多级创新生成器就绪，包含 {len(self.concept_embeddings)} 个概念向量")
    
    async def generate_theory_from_literature_and_priors(self, 
                                                        theory1_name: str, 
                                                        theory2_name: str,
                                                        focus_concepts: Optional[List[str]] = None) -> Dict:
        """Compatibility wrapper that reuses the short-card workflow."""
        query = f"Resolve contradictions between {theory1_name} and {theory2_name}."
        result = await self.generate_card_driven_interpretation(query=query, top_k=max(2, 4))
        machine = result.get('machine_summary', {}) or {}
        interpretation_name = machine.get('name') or f"{theory1_name} vs {theory2_name} reinterpretation"
        return {
            'name': interpretation_name,
            'core_principles': result.get('writeup', ''),
            'generation_metadata': {
                'source_type': 'short_card_pipeline',
                'source_theories': [theory1_name, theory2_name],
                'selected_cards': result.get('selected_cards', []),
                'retrieval_scores': result.get('retrieval_scores', {}),
                'focus_concepts': focus_concepts or [],
            },
            'card_workflow': result
        }

    
    def _format_cards_for_prompt(self, cards: List[Dict]) -> str:
        lines: List[str] = []
        for card in cards:
            math = card.get('math_relation_to_SQM', {})
            block = "\n".join([
                f"id={card.get('id')} | name={card.get('name')}",
                f"one_line: {card.get('one_line')}",
                f"math: type={math.get('type')} change={math.get('math_change')} -> {math.get('equations_summary')}",
                f"key_claims: {'; '.join(card.get('key_claims', []))}",
                f"born_rule={card.get('born_rule')} | measurement={card.get('measurement_update')} | locality={card.get('locality_note')}",
                f"predictions: {'; '.join(card.get('predictions', [])) or 'none'}",
            ])
            lines.append(block)
        return "\n\n".join(lines)

    def _format_contradictions_for_prompt(self, table: Dict[str, Any]) -> str:
        rows = [f"{item['A']} vs {item['B']} [{item['issue']}]: {item['one_line']}" for item in table.get('contradictions', [])]
        return "\n".join(rows) if rows else "No contradictions returned."

    def _build_card_machine_messages(self, cards: List[Dict], table: Dict[str, Any], constraints: Dict[str, List[str]]) -> List[Dict[str, str]]:
        constraint_lines = ["Hard constraints:"] + [f"- {item}" for item in constraints.get('hard_rules', [])]
        constraint_lines.append('Nice-to-have goals:')
        constraint_lines.extend(f"- {item}" for item in constraints.get('nice_to_have', []))
        system_prompt = (
            "You design candidate quantum interpretations. Use the contradictions to extend theory space while respecting the constraints. "
            "Output only JSON that matches the provided schema."
        )
        user_prompt = (
            f"Selected cards:\n{self._format_cards_for_prompt(cards)}\n\n"
            f"Contradictions:\n{self._format_contradictions_for_prompt(table)}\n\n"
            + "\n".join(constraint_lines)
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_card_human_messages(self, cards: List[Dict], table: Dict[str, Any], constraints: Dict[str, List[str]]) -> List[Dict[str, str]]:
        reminders = constraints.get('hard_rules', []) + constraints.get('nice_to_have', [])
        reminder_text = "\n- ".join(reminders) if reminders else "None"
        system_prompt = (
            "Write a six-section human-readable proposal for a new quantum interpretation that resolves the listed contradictions. "
            "Each section should be one short paragraph (4-6 sentences) following this order: "
            "(1) Core commitments, (2) Relation to SQM mathematics, (3) Measurement and Born rule, "
            "(4) Ontology, (5) Distinct empirical or operational consequences, (6) Attitude toward Bell/Kochen-Specker/PBR."
        )
        user_prompt = (
            f"Cards considered:\n{self._format_cards_for_prompt(cards)}\n\n"
            f"Key contradictions:\n{self._format_contradictions_for_prompt(table)}\n\n"
            f"Constraints to respect:\n- {reminder_text}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def generate_card_driven_interpretation(self, query: str, top_k: int = 6, constraints: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """Use short-card retrieval and contradiction analysis to synthesize a new interpretation."""
        if not self.card_retriever or not self.card_analyzer or not self.new_interpretation_schema:
            raise RuntimeError("Short-card workflow is not available. Ensure cards and schemas are configured.")
        await self.card_retriever.ensure_index()
        retrieval_results = await self.card_retriever.top_k(query, k=top_k)
        cards = [result.card for result in retrieval_results]
        if not cards:
            raise ValueError("No cards retrieved for the provided query.")
        contradiction_table = await self.card_analyzer.build_table(cards, task_hint=query)
        constraint_bundle = constraints or DEFAULT_CARD_CONSTRAINTS
        machine_messages = self._build_card_machine_messages(cards, contradiction_table, constraint_bundle)
        machine_summary = await self.llm.query_structured_json(
            messages=machine_messages,
            schema=self.new_interpretation_schema,
            schema_name="new_interpretation",
            temperature=0.4,
        )
        if not machine_summary:
            raise ValueError('Structured summary generation failed.')
        human_messages = self._build_card_human_messages(cards, contradiction_table, constraint_bundle)
        writeup = await self.llm.query_async(human_messages, temperature=0.6)
        result = {
            "query": query,
            "selected_cards": [res.card_id for res in retrieval_results],
            "retrieval_scores": {res.card_id: res.score for res in retrieval_results},
            "contradictions": contradiction_table.get('contradictions', []),
            "machine_summary": machine_summary,
            "writeup": writeup,
        }
        self.last_card_result = result
        return result

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
        self.logger.info("保存统一分析结果")
        
        analysis = {
            'literature_concepts_count': len(self.literature_concepts),
            'prior_theories_count': len(self.prior_theories),
            'concept_embeddings_count': len(self.concept_embeddings),
            'theory_embeddings_count': len(self.theory_embeddings),
            'unified_concept_space_size': len(self.unified_concept_space),
            'prior_theories_list': list(self.prior_theories.keys()),
            'concept_space_statistics': self._calculate_concept_space_stats()
        }
        if self.last_card_result:
            analysis['last_card_result'] = self.last_card_result

        output_file = Path(self.config.output_dir) / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        print(f"📊 统一分析结果已保存到: {output_file}")
        
        # 生成概念空间可视化（如果启用）
        if self.config.enable_visualization:
            self._generate_concept_space_visualization()
    
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
    
    def _generate_concept_space_visualization(self):
        """生成概念空间可视化"""
        try:
            from utils.concept_space_visualizer import ConceptSpaceVisualizer
            
            print("\n📊 生成概念空间可视化...")
            self.logger.info("开始生成概念空间可视化")
            
            # 创建可视化器
            viz_dir = Path(self.config.output_dir) / "concept_space_visualization"
            visualizer = ConceptSpaceVisualizer(str(viz_dir))
            
            # 准备数据
            all_embeddings = {}
            all_labels = {}
            
            # 添加概念嵌入
            for name, embedding in self.concept_embeddings.items():
                if embedding is not None and isinstance(embedding, np.ndarray):
                    all_embeddings[f"概念: {name}"] = embedding
                    all_labels[f"概念: {name}"] = "文献概念"
            
            # 添加理论嵌入
            for name, embedding in self.theory_embeddings.items():
                if embedding is not None and isinstance(embedding, np.ndarray):
                    all_embeddings[f"理论: {name}"] = embedding
                    all_labels[f"理论: {name}"] = "先验理论"
            
            # 确保有足够的数据进行可视化
            if len(all_embeddings) < 3:
                self.logger.warning("嵌入数量不足，跳过可视化")
                print("⚠️ 嵌入数量不足（需要至少3个），跳过可视化")
                return
            
            # 生成综合报告
            visualizer.create_comprehensive_report(
                all_embeddings,
                labels=all_labels,
                output_prefix="unified_concept_space"
            )
            
            print(f"✅ 概念空间可视化已生成: {viz_dir}")
            self.logger.info(f"概念空间可视化完成: {viz_dir}")
            
        except ImportError:
            self.logger.warning("无法导入可视化模块，跳过可视化")
            print("⚠️ 可视化模块未安装，跳过概念空间可视化")
        except Exception as e:
            self.logger.error(f"生成可视化时出错: {e}")
            print(f"⚠️ 生成可视化时出错: {e}") 