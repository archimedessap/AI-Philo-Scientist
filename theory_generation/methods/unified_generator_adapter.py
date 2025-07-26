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

# 导入日志系统
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logging_config import get_logger, log_error_with_context, log_execution

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
    
    def __init__(self, use_structured_format: bool = True, **kwargs):
        super().__init__(**kwargs)
        
        # 添加参数以控制文献概念加载
        self.force_load_literature = kwargs.get('force_load_literature', False)
        
        # 初始化日志器
        self.logger = get_logger('unified_generator_adapter', module_specific=True)
        self.logger.info("Initializing unified generator based on high-dimensional conceptual space")
        
        # 自动检测测试模式：如果理论目录是测试目录或理论数量很少，启用测试模式
        self.test_mode = (
            'test' in str(self.theories_dir).lower() or 
            'mini' in str(self.theories_dir).lower() or
            len(list(self.theories_dir.glob('*.json'))) <= 5
        )
        
        # 格式选项
        self.use_structured_format = use_structured_format
        
        if self.test_mode:
            self._log_info("🧪 Auto-enabled test mode - will reduce concept loading")
            self.logger.info("Test mode enabled")
        
        if self.use_structured_format:
            self._log_info("📋 Using structured format - consistent with other generation methods")
            self.logger.info("Using structured format output")
        
        # Initialize LLM interface
        self.logger.debug("Initializing LLM interface")
        self.logger.debug(f"Model source: {self.model_source}, Model name: {self.model_name}")
        try:
            self.llm_interface = LLMInterface(
                model_source=self.model_source,
                model_name=self.model_name
            )
            self.logger.debug("LLM interface initialized successfully")
        except Exception as e:
            self.logger.error(f"LLM interface initialization failed: {e}")
            raise
        
        # 初始化核心组件
        self.concept_embedder = ConceptEmbedder(
            llm_interface=self.llm_interface,
            embedding_dim=1536  # 使用标准嵌入维度
        )
        
        self.space_explorer = VectorSpaceExplorer(
            llm_interface=self.llm_interface
        )
        
        # 初始化增强组件（如果可用）
        self.physics_embedder = None
        self.knowledge_graph = None
        self._initialize_enhanced_components()
        
        # 空间数据存储
        self.concept_space = {}      # 概念空间
        self.theory_space = {}       # 理论空间
        self.formula_space = {}      # 公式空间
        self.conceptual_gaps = []    # 概念空白区域
        self.semantic_dimensions = {}  # 维度语义映射
        self.concept_importance = {}  # 概念重要性分数
        
        self._log_info("Initializing unified generator based on high-dimensional conceptual space")
    
    def _initialize_enhanced_components(self):
        """初始化增强组件（如果可用）"""
        # 尝试导入物理嵌入器
        try:
            from physics_embedder import PhysicsEmbedder
            self.physics_embedder = PhysicsEmbedder(base_embedder=self.concept_embedder)
            self._log_info("✅ Physics embedder initialized")
        except ImportError:
            self._log_info("Physics embedder not available")
        
        # 尝试加载知识图谱
        try:
            from knowledge_graph_builder import KnowledgeGraph
            kg_dir = Path('data/knowledge_graph')
            if kg_dir.exists():
                # 查找所有知识图谱文件（包括增强版本）
                kg_patterns = ['enhanced_kg_*.json', 'test_kg_*.json', 'knowledge_graph_*.json']
                kg_files = []
                for pattern in kg_patterns:
                    kg_files.extend(list(kg_dir.glob(pattern)))
                
                if kg_files:
                    latest_kg = max(kg_files, key=os.path.getctime)
                    self.knowledge_graph = self._load_knowledge_graph(latest_kg)
                    self._log_info(f"✅ Knowledge graph loaded from {latest_kg.name}")
                else:
                    self._log_info("No knowledge graph files found")
            else:
                self._log_info("Knowledge graph directory not found")
        except Exception as e:
            self._log_info(f"Knowledge graph not loaded: {e}")
    
    def _load_knowledge_graph(self, kg_file: Path) -> Optional['KnowledgeGraph']:
        """加载知识图谱"""
        try:
            from knowledge_graph_builder import KnowledgeGraph, Node, Edge
            
            with open(kg_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            kg = KnowledgeGraph()
            
            # 加载节点
            for node_id, node_data in data.get('nodes', {}).items():
                node = Node(
                    id=node_data['id'],
                    type=node_data['type'],
                    name=node_data['name'],
                    attributes=node_data['attributes']
                )
                kg.add_node(node)
            
            # 加载边
            for edge_data in data.get('edges', []):
                edge = Edge(
                    source=edge_data['source'],
                    target=edge_data['target'],
                    relation=edge_data['relation'],
                    weight=edge_data.get('weight', 1.0),
                    attributes=edge_data.get('attributes')
                )
                kg.add_edge(edge)
            
            return kg
        except Exception as e:
            self._log_error(f"Failed to load knowledge graph: {e}")
            return None
    
    def generate(self) -> Dict[str, Any]:
        """统一生成流程：空间构建 → 空白分析 → 空间生成（同步接口）"""
        try:
            # 使用asyncio.run来运行异步生成过程
            return asyncio.run(self._async_generate())
        except Exception as e:
            self._log_error(f"Unified space generation failed: {e}")
            return GenerationResult(
                success=False,
                error_message=str(e)
            ).to_dict()
    
    async def _async_generate(self) -> Dict[str, Any]:
        """异步统一生成流程：空间构建 → 空白分析 → 空间生成"""
        try:
            self._log_info("🚀 Starting unified theory generation based on high-dimensional conceptual space")
            
            # 1. 加载和预处理理论
            theories = self._load_theories()
            if not theories:
                return GenerationResult(
                    success=False,
                    error_message="Unable to load prior theories"
                ).to_dict()
            
            # 2. 构建多层次概念空间
            await self._build_multilevel_concept_space(theories)
            
            # 3. 分析概念空间结构
            self._analyze_space_structure()
            
            # 4. 识别概念空白区域
            if self.physics_embedder and self.knowledge_graph:
                self._identify_high_value_gaps()
            else:
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
            
            # 生成概念空间可视化
            self._generate_space_visualization()
            
            self._log_info(f"✅ Unified space generation complete: generated {len(generated_theories)} theories")
            return result.to_dict()
            
        except Exception as e:
            self._log_error(f"Unified space generation failed: {e}")
            return GenerationResult(
                success=False,
                error_message=str(e)
            ).to_dict()
    
    async def _build_multilevel_concept_space(self, theories: List[Dict]):
        """构建多层次概念空间"""
        self._log_info("🏗️ Building multi-level conceptual space...")
        
        # 1. 构建理论空间
        self.theory_space = await self.concept_embedder.embed_theories(theories)
        self._log_info(f"Theory space dimensions: {len(self.theory_space)} × {self.concept_embedder.embedding_dim}")
        
        # 2. 提取和嵌入概念
        concepts = self._extract_concepts_from_theories(theories)
        if concepts:
            # 如果有物理嵌入器，使用增强嵌入
            if self.physics_embedder:
                await self._embed_concepts_with_physics(concepts)
            else:
                self.concept_space = await self.concept_embedder.embed_concepts(concepts)
            self._log_info(f"Concept space dimensions: {len(self.concept_space)} × {self.concept_embedder.embedding_dim}")
        
        # 3. 提取和嵌入公式（如果存在）
        formulas = self._extract_formulas_from_theories(theories)
        if formulas:
            # 如果有物理嵌入器，使用增强嵌入
            if self.physics_embedder:
                await self._embed_formulas_with_physics(formulas)
            else:
                self.formula_space = await self.concept_embedder.embed_formulas(formulas)
            self._log_info(f"Formula space dimensions: {len(self.formula_space)} × {self.concept_embedder.embedding_dim}")
        
        # 4. 如果有知识图谱，计算概念重要性
        if self.knowledge_graph:
            self._compute_concept_importance()
    
    def _extract_concepts_from_theories(self, theories: List[Dict]) -> List[Dict]:
        """从理论中提取概念"""
        concepts = []
        seen_concepts = set()
        
        # 方法1: 从理论内容中提取（原有方法）
        for theory in theories:
            content = theory.get('content', '')
            
            # 简化的概念提取（基于关键词）
            key_concepts = self._extract_key_concepts(content)
            
            # 测试模式下限制每个理论提取的概念数量
            max_concepts_per_theory = 3 if self.test_mode else 10
            
            for concept in key_concepts[:max_concepts_per_theory]:
                if concept not in seen_concepts:
                    concepts.append({
                        'name': concept,
                        'description': f"Concept extracted from theory {theory.get('name', '')}",
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
        
        # 测试模式下进一步限制总概念数量
        max_total_concepts = 15 if self.test_mode else len(concepts)
        
        self._log_info(f"Total concepts extracted: theory_content={len([c for c in concepts if c.get('extraction_method') == 'theory_content'])}, literature_analysis={len([c for c in concepts if c.get('extraction_method') == 'literature_analysis'])}")
        
        return concepts[:max_total_concepts]
    
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
        
        # 方法3: 加载文献中提取的公式
        literature_formulas = self._load_literature_formulas()
        formulas.extend(literature_formulas)
        
        # 去重
        unique_formulas = []
        seen_expressions = set()
        for formula in formulas:
            expr = formula.get('expression', '').strip()
            if expr and expr not in seen_expressions and len(expr) > 2:
                unique_formulas.append(formula)
                seen_expressions.add(expr)
        
        self._log_info(f"Total formulas collected: {len(unique_formulas)} (from theories: {len(formulas) - len(literature_formulas)}, from literature: {len(literature_formulas)})")
        
        # 限制公式数量（测试模式）
        max_formulas = 20 if self.test_mode else len(unique_formulas)
        
        return unique_formulas[:max_formulas]
    
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
                        'description': f"Formula extracted from {context_path} of theory {theory_name}",
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
                'description': f"Formula extracted from text content of theory {theory_name}",
                'source_theory': theory_name,
                'source_context': 'text_content'
            })
        
        return formulas
    
    def _load_literature_concepts(self) -> List[Dict]:
        """从文献分析中加载概念"""
        if self.test_mode and not self.force_load_literature:
            # 测试模式：只返回几个基础概念，避免大量加载（除非强制加载）
            self._log_info("🧪 Test mode: skipping literature concept loading")
            return [
                {'name': 'Quantum Superposition', 'description': 'Quantum system in multiple states simultaneously', 'domain': 'quantum', 'source': 'test'},
                {'name': 'Wave Function Collapse', 'description': 'Instantaneous change of quantum state', 'domain': 'quantum', 'source': 'test'},
                {'name': 'Quantum Entanglement', 'description': 'Correlation between quantum particles', 'domain': 'quantum', 'source': 'test'}
            ]
        
        concepts = []
        
        # 优先加载增强的概念文件
        enhanced_concepts_dir = Path('data/enhanced_concepts')
        if enhanced_concepts_dir.exists():
            # 查找最新的增强概念文件（修正文件名模式）
            concept_files = list(enhanced_concepts_dir.glob('enhanced_concepts_*.json'))
            # 排除关系文件
            concept_files = [f for f in concept_files if 'relations' not in f.name]
            
            if concept_files:
                latest_file = max(concept_files, key=os.path.getctime)
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        enhanced_concepts = json.load(f)
                    
                    for concept_data in enhanced_concepts:
                        # 处理不同的数据格式
                        if isinstance(concept_data, dict):
                            concepts.append({
                                'name': concept_data.get('name', ''),
                                'description': concept_data.get('description', ''),
                                'domain': concept_data.get('domain', concept_data.get('category', '')),
                                'source': concept_data.get('source', ''),
                                'category': concept_data.get('category', 'fundamental'),
                                'confidence': concept_data.get('confidence', 0.8),
                                'prerequisites': concept_data.get('prerequisites', []),
                                'related_formulas': concept_data.get('related_formulas', [])
                            })
                    
                    self._log_info(f"Loaded {len(concepts)} enhanced concepts from {latest_file.name}")
                    
                    # 如果成功加载了增强概念，直接返回
                    if concepts:
                        return concepts
                        
                except Exception as e:
                    self._log_error(f"Failed to load enhanced concepts: {e}")
        
        # 如果没有增强概念，加载传统的extracted_concepts.csv
        if not concepts:
            csv_file = Path('data/extracted_concepts/concepts.csv')
            if csv_file.exists():
                try:
                    import csv
                    with open(csv_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row.get('Name') and row.get('Description'):
                                concepts.append({
                                    'name': row['Name'].strip(),
                                    'description': row['Description'].strip(),
                                    'domain': row.get('Domain', '').strip(),
                                    'source': row.get('Source', '').strip()
                                })
                    
                    self._log_info(f"📚 Loaded {len(concepts)} concepts from extracted concepts file")
                    if self.force_load_literature:
                        self._log_info("✅ Force-loaded literature concepts in test mode")
                except Exception as e:
                    self._log_error(f"Failed to load extracted concepts: {e}")
        
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
                    
                    self._log_info(f"Loaded {len(concepts)} concepts from merged concepts file")
                except Exception as e:
                    self._log_error(f"Failed to load merged concepts: {e}")
        
        # 过滤和清理概念
        filtered_concepts = []
        for concept in concepts:
            name = concept['name']
            # 过滤掉过短或无意义的概念名
            if len(name) > 2 and not name.isdigit() and 'theory' not in name.lower():
                filtered_concepts.append(concept)
        
        # 根据模式限制数量
        max_concepts = 10 if self.test_mode else 50
        self._log_info(f"After filtering, retained {len(filtered_concepts)} high-quality concepts, limited to {max_concepts}")
        return filtered_concepts[:max_concepts]
    
    def _load_literature_formulas(self) -> List[Dict]:
        """从文献分析中加载公式"""
        formulas = []
        
        # 加载增强的公式文件
        enhanced_formulas_dir = Path('data/extracted_formulas')
        if enhanced_formulas_dir.exists():
            # 查找最新的增强公式文件
            formula_files = list(enhanced_formulas_dir.glob('enhanced_formulas_*.json'))
            
            if formula_files:
                latest_file = max(formula_files, key=os.path.getctime)
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        enhanced_formulas = json.load(f)
                    
                    for formula_data in enhanced_formulas:
                        if isinstance(formula_data, dict):
                            formulas.append({
                                'name': formula_data.get('name', ''),
                                'expression': formula_data.get('expression', ''),
                                'description': formula_data.get('description', ''),
                                'category': formula_data.get('category', 'physical_law'),
                                'source': formula_data.get('source', ''),
                                'variables': formula_data.get('variables', [])
                            })
                    
                    self._log_info(f"Loaded {len(formulas)} enhanced formulas from {latest_file.name}")
                    
                except Exception as e:
                    self._log_error(f"Failed to load enhanced formulas: {e}")
        
        # 如果没有增强公式，加载传统文件
        if not formulas:
            traditional_formulas_file = Path('data/extracted_formulas/formulas_20250123_103806.json')
            if traditional_formulas_file.exists():
                try:
                    with open(traditional_formulas_file, 'r', encoding='utf-8') as f:
                        formula_list = json.load(f)
                    
                    for formula in formula_list:
                        formulas.append({
                            'name': formula.get('name', ''),
                            'expression': formula.get('expression', ''),
                            'description': formula.get('description', ''),
                            'category': 'extracted',
                            'source': formula.get('source', '')
                        })
                    
                    self._log_info(f"Loaded {len(formulas)} formulas from traditional file")
                    
                except Exception as e:
                    self._log_error(f"Failed to load traditional formulas: {e}")
        
        return formulas
    
    def _analyze_space_structure(self):
        """分析概念空间结构"""
        self._log_info("🔍 Analyzing conceptual space structure...")
        
        # 分析理论空间
        if self.theory_space:
            theory_similarity = compute_similarity_matrix(self.theory_space)
            self.semantic_dimensions['theory_clusters'] = self._find_clusters(self.theory_space)
            self._log_info(f"Theory cluster count: {len(self.semantic_dimensions['theory_clusters'])}")
        
        # 分析概念空间
        if self.concept_space:
            concept_similarity = compute_similarity_matrix(self.concept_space)
            self.semantic_dimensions['concept_clusters'] = self._find_clusters(self.concept_space)
            self._log_info(f"Concept cluster count: {len(self.semantic_dimensions['concept_clusters'])}")
        
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
        self._log_info("🎯 Identifying conceptual gap regions...")
        
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
                # 计算理论向量的标准差作为阈值
                theory_norms = [np.linalg.norm(v) for v in theory_vectors]
                if len(theory_norms) > 1:
                    threshold = np.std(theory_norms) * 0.5
                else:
                    # 如果向量数量不足，使用固定阈值
                    threshold = 0.1
                
                if min_distance > threshold:
                    self.conceptual_gaps.append({
                        'type': 'midpoint_gap',
                        'position': midpoint,
                        'source_theories': [theory_names[i], theory_names[j]],
                        'distance_to_nearest': min_distance,
                        'description': f"Conceptual gap region between theory {theory_names[i]} and {theory_names[j]}"
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
                        'description': f"Conceptual gap region between cluster {i+1} and cluster {j+1}"
                    })
        
        self._log_info(f"Found {len(self.conceptual_gaps)} conceptual gap regions")
    
    async def _generate_from_space(self) -> List[Dict]:
        """基于概念空间生成新理论"""
        self._log_info("🎨 Generating new theories based on conceptual space...")
        
        generated_theories = []
        
        # 为每个空白区域生成理论
        for i, gap in enumerate(self.conceptual_gaps[:self.variants_per_contradiction]):
            self._log_info(f"Generating theory for gap region {i+1}...")
            
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
                
                self._log_info(f"✅ Gap region {i+1} theory generation complete")
                
            except Exception as e:
                self._log_error(f"Gap region {i+1} generation failed: {e}")
                continue
        
        return generated_theories
    
    def _build_space_based_prompt(self, gap: Dict) -> str:
        """构建基于空间的生成提示"""
        gap_type = gap.get('type', 'unknown')
        description = gap.get('description', '')
        
        # 添加概念重要性和关系信息
        context_info = self._build_gap_context(gap)
        
        if gap_type == 'conceptual_bridge':
            # 概念桥梁类型的提示
            source_concepts = gap.get('source_theories', [])
            prompt = f"""
Based on advanced conceptual space analysis, we have identified a high-value conceptual bridge opportunity in quantum physics.

This bridge connects two important concepts from different domains:
- Concept A: {source_concepts[0] if len(source_concepts) > 0 else 'Unknown'}
- Concept B: {source_concepts[1] if len(source_concepts) > 1 else 'Unknown'}

{context_info}

Please create a new quantum theory that bridges these concepts, forming a unified framework that:
1. Explains how these seemingly disparate concepts are fundamentally related
2. Provides new insights into quantum phenomena by connecting different domains
3. Offers novel predictions based on this conceptual synthesis

Please answer strictly in the following format:

**Theory Name:** [Provide a concise theory name here, no more than 50 characters]

**Core Assumptions:**
[List 3-5 core assumptions that bridge the concepts]

**Bridging Mechanism:**
[Explain how the theory connects the two concepts]

**Mathematical Framework:**
[Provide mathematical description that unifies both domains]

**Novel Predictions:**
[List specific, testable predictions arising from this bridge]

**Implications:**
[Describe broader implications for quantum physics]

New Theory:
"""
        elif gap_type == 'midpoint_gap':
            source_theories = gap.get('source_theories', [])
            prompt = f"""
Based on conceptual space analysis, we have discovered a gap region in the high-dimensional conceptual space of quantum theories.
This gap region is located at the midpoint in the conceptual space between the following two theories:

Theory A: {source_theories[0] if len(source_theories) > 0 else 'Unknown'}
Theory B: {source_theories[1] if len(source_theories) > 1 else 'Unknown'}

Please create a new quantum theory interpretation based on this position in the conceptual space. Please answer strictly in the following format:

**Theory Name:** [Provide a concise theory name here, no more than 50 characters]

**Core Assumptions:**
[List 3-5 core assumptions]

**Key Concepts:**
[Describe the key concepts]

**Mathematical Framework:**
[Provide mathematical description if applicable]

**Differences from Existing Theories:**
[Explain how it differs from existing theories]

**Possible Experimental Predictions:**
[Provide testable predictions]

Please ensure:
1. The theory name is concise and follows academic naming conventions
2. It integrates core concepts from both theories above while forming a unique theoretical framework
3. It occupies this gap region in the conceptual space, filling the theoretical void
4. It provides new physical intuition and mathematical formalism
5. It can offer unique predictions or explanations for quantum phenomena

New Theory:
"""
        
        elif gap_type == 'inter_cluster_gap':
            source_clusters = gap.get('source_clusters', [])
            prompt = f"""
Based on clustering analysis of the conceptual space, we have discovered a conceptual gap region between two theory clusters:

Cluster A contains theories: {', '.join(source_clusters[0]) if len(source_clusters) > 0 else 'Unknown'}
Cluster B contains theories: {', '.join(source_clusters[1]) if len(source_clusters) > 1 else 'Unknown'}

Please create a new quantum theory that bridges these two theory clusters. Please answer strictly in the following format:

**Theory Name:** [Provide a concise theory name here, no more than 50 characters]

**Core Assumptions:**
[List 3-5 core assumptions]

**Key Concepts:**
[Describe the key concepts]

**Mathematical Framework:**
[Provide mathematical description if applicable]

**Differences from Existing Theories:**
[Explain how it differs from existing theories]

**Bridging Mechanism:**
[Explain how it bridges the two clusters]

Please ensure this theory:
1. Combines core ideas from both clusters
2. Forms a connecting bridge in the conceptual space
3. Provides a unified understanding framework across clusters
4. Opens new theoretical research directions

Bridging Theory:
"""
        
        else:
            prompt = f"""
Based on conceptual space analysis, a theoretical gap region has been discovered.

Gap region description: {description}

Please create a new quantum theory that fills this gap. Please answer strictly in the following format:

**Theory Name:** [Provide a concise theory name here, no more than 50 characters]

**Core Assumptions:**
[List 3-5 core assumptions]

**Key Concepts:**
[Describe the key concepts]

**Mathematical Framework:**
[Provide mathematical description if applicable]

**Differences from Existing Theories:**
[Explain how it differs from existing theories]

New Theory:
"""
        
        return prompt
    
    def _parse_generated_theory(self, response: str, gap: Dict, variant_id: int) -> Dict:
        """解析生成的理论，输出标准化格式"""
        # 提取理论名称
        theory_name = self._extract_theory_name(response)
        if not theory_name:
            theory_name = f"Space_Generated_Theory_{variant_id}"
        
        # 解析结构化内容
        parsed_content = self._parse_theory_structure(response)
        
        # 构建标准化理论数据（按照选项1格式）
        theory_data = {
            'name': theory_name,
            'description': parsed_content.get('description', 'Quantum theory interpretation generated based on conceptual space'),
            'core_assumptions': parsed_content.get('core_assumptions', []),
            'key_concepts': parsed_content.get('key_concepts', []),
            'mathematical_formalism': parsed_content.get('mathematical_formalism', ''),
            'empirical_predictions': parsed_content.get('empirical_predictions', []),
            'differences_from_existing': parsed_content.get('differences_from_existing', ''),
            'content': response.strip(),  # 完整原文作为备份
            'variant_info': {
                'generation_method': 'unified_space_based',
                'gap_type': gap.get('type', 'unknown'),
                'gap_description': gap.get('description', ''),
                # 简化空间信息存储
                'space_position_sample': self._get_position_sample(gap.get('position')),
                'space_position_dim': self._get_safe_length(gap.get('position')),
                'source_info': gap.get('source_theories', gap.get('source_clusters', [])),
                'variant_id': variant_id
            },
            'generation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'gap_analysis': self._get_space_analysis_summary(),
                'space_dimensions': self.concept_embedder.embedding_dim,
                'format_version': '2.0_structured'
            }
        }
        
        return theory_data
    
    def _parse_theory_structure(self, response: str) -> Dict[str, Any]:
        """从响应中解析结构化的理论内容"""
        parsed = {}
        
        # 提取描述（通常在开头）
        lines = response.split('\n')
        description_lines = []
        for line in lines[:10]:  # 检查前10行
            line = line.strip()
            if line and not line.startswith('**') and not line.startswith('#'):
                if not any(keyword in line for keyword in ['理论名称', '核心假设', '主要概念', '数学框架']):
                    description_lines.append(line)
        
        if description_lines:
            parsed['description'] = ' '.join(description_lines)
        
        # 解析核心假设
        core_assumptions = self._extract_numbered_list(response, ['**核心假设：**', '核心假设:', '**Core Assumptions:**'])
        if core_assumptions:
            parsed['core_assumptions'] = core_assumptions
        
        # 解析主要概念
        key_concepts = self._extract_concept_list(response, ['**主要概念：**', '主要概念:', '**Key Concepts:**'])
        if key_concepts:
            parsed['key_concepts'] = key_concepts
        
        # 提取数学框架
        math_formalism = self._extract_section_content(response, ['**数学框架：**', '数学框架:', '**Mathematical Framework:**'])
        if math_formalism:
            parsed['mathematical_formalism'] = math_formalism
        
        # 提取实验预测
        predictions = self._extract_numbered_list(response, ['**可能的实验预测：**', '**实验预测：**', '实验预测:', '**Experimental Predictions:**'])
        if predictions:
            parsed['empirical_predictions'] = predictions
        
        # 提取与现有理论的区别
        differences = self._extract_section_content(response, ['**与现有理论的区别：**', '与现有理论的区别:', '**Differences:**'])
        if differences:
            parsed['differences_from_existing'] = differences
        
        return parsed
    
    def _extract_numbered_list(self, text: str, section_markers: List[str]) -> List[str]:
        """提取编号列表内容"""
        items = []
        
        for marker in section_markers:
            if marker in text:
                # 找到标记位置
                start_idx = text.find(marker)
                if start_idx == -1:
                    continue
                
                # 从标记后开始解析
                section_text = text[start_idx + len(marker):]
                
                # 按行分割并查找编号项
                lines = section_text.split('\n')
                current_item = ""
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        if current_item:
                            items.append(current_item.strip())
                            current_item = ""
                        continue
                    
                    # 检查是否是下一个section的开始
                    if line.startswith('**') and '：' in line and line != marker:
                        break
                    
                    # 检查是否是编号项
                    import re
                    if re.match(r'^\d+\.', line):
                        if current_item:
                            items.append(current_item.strip())
                        # 移除编号，保留内容
                        current_item = re.sub(r'^\d+\.\s*', '', line)
                    else:
                        if current_item:
                            current_item += " " + line
                
                # 添加最后一项
                if current_item:
                    items.append(current_item.strip())
                
                if items:
                    break
        
        # 清理items，移除markdown格式
        cleaned_items = []
        for item in items:
            cleaned = item.replace('**', '').replace('*', '').strip()
            if cleaned and len(cleaned) > 5:  # 过滤太短的项
                cleaned_items.append(cleaned[:200])  # 限制长度
        
        return cleaned_items[:10]  # 最多10项
    
    def _extract_concept_list(self, text: str, section_markers: List[str]) -> List[str]:
        """提取概念列表"""
        concepts = []
        
        for marker in section_markers:
            if marker in text:
                start_idx = text.find(marker)
                if start_idx == -1:
                    continue
                
                section_text = text[start_idx + len(marker):]
                
                # 查找概念项（通常以*开头或有冒号）
                lines = section_text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 检查是否是下一个section
                    if line.startswith('**') and '：' in line and line != marker:
                        break
                    
                    # 提取概念名称
                    import re
                    # 匹配 "* **概念名：**" 格式
                    concept_match = re.match(r'\*\s*\*\*(.+?)\*\*[：:]', line)
                    if concept_match:
                        concept_name = concept_match.group(1).strip()
                        if concept_name:
                            concepts.append(concept_name)
                    # 匹配 "**概念名 (英文名)：**" 格式
                    elif re.match(r'\*\*(.+?)\*\*[：:]', line):
                        concept_match = re.match(r'\*\*(.+?)\*\*[：:]', line)
                        concept_name = concept_match.group(1).strip()
                        if concept_name:
                            concepts.append(concept_name)
                
                if concepts:
                    break
        
        return concepts[:15]  # 最多15个概念
    
    def _extract_section_content(self, text: str, section_markers: List[str]) -> str:
        """提取章节内容"""
        for marker in section_markers:
            if marker in text:
                start_idx = text.find(marker)
                if start_idx == -1:
                    continue
                
                # 找到section的结束位置
                section_start = start_idx + len(marker)
                section_text = text[section_start:]
                
                # 查找下一个section的开始
                lines = section_text.split('\n')
                content_lines = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        content_lines.append('')
                        continue
                    
                    # 检查是否是下一个section
                    if line.startswith('**') and '：' in line:
                        break
                    
                    content_lines.append(line)
                
                content = '\n'.join(content_lines).strip()
                if content:
                    return content[:1000]  # 限制长度
        
        return ""
    
    def _extract_theory_name(self, text: str) -> Optional[str]:
        """从生成的文本中提取理论名称"""
        lines = text.split('\n')
        
        # 查找包含"理论"、"Theory"等关键词的行
        for line in lines:
            line = line.strip()
            
            # 优先查找明确的理论名称标记
            if any(keyword in line for keyword in ['**理论名称：**', '理论名称:', 'Theory Name:', '**Theory Name:**']):
                # 提取名称部分
                if '：' in line:
                    parts = line.split('：', 1)
                elif ':' in line:
                    parts = line.split(':', 1)
                else:
                    continue
                    
                if len(parts) > 1:
                    name = parts[1].strip()
                    # 清理markdown格式和方括号
                    name = name.replace('**', '').replace('*', '').replace('[', '').replace(']', '').strip()
                    if name and len(name) < 100:
                        return name
            
            # 查找包含"解释"、"Interpretation"的理论名称
            elif any(keyword in line for keyword in ['解释', 'Interpretation', '理论']) and len(line) < 150:
                # 排除明显不是理论名称的行
                if not any(exclude in line for exclude in ['与现有理论', '基于', '好的', '我们', '这个', '它', '请', '在此处']):
                    name = line.strip()
                    # 清理markdown格式
                    name = name.replace('**', '').replace('*', '').strip()
                    if name and not name.startswith(('好的', '基于', '我们', '请')):
                        return name
        
        # 如果没找到，返回一个默认名称
        return None
    
    def _generate_safe_filename(self, theory_name: str, variant_id: int) -> str:
        """生成安全的文件名"""
        if not theory_name:
            return f"unified_theory_{variant_id}"
        
        # 清理理论名称
        safe_name = theory_name.replace('**', '').replace('*', '').strip()
        
        # 如果名称太长，截取前30个字符
        if len(safe_name) > 30:
            safe_name = safe_name[:30]
        
        # 移除或替换特殊字符
        safe_chars = []
        for char in safe_name:
            if char.isalnum() or char in (' ', '-', '_', '(', ')'):
                safe_chars.append(char)
            elif char in ('，', ',', '：', ':'):
                safe_chars.append('_')
        
        safe_name = ''.join(safe_chars).strip()
        
        # 清理多余的空格和下划线
        safe_name = ' '.join(safe_name.split())  # 移除多余空格
        safe_name = safe_name.replace(' ', '_')  # 空格替换为下划线
        
        # 如果清理后为空，使用默认名称
        if not safe_name:
            safe_name = f"unified_theory_{variant_id}"
        
        return safe_name
    
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
        try:
            # 安全处理概念空白区域的位置信息
            safe_gaps = []
            for gap in self.conceptual_gaps:
                safe_gap = dict(gap)  # 复制原始gap
                position = gap.get('position')
                if position is not None:
                    try:
                        if hasattr(position, 'tolist'):
                            safe_gap['position'] = position.tolist()
                        elif hasattr(position, '__iter__'):
                            safe_gap['position'] = list(position)
                        else:
                            safe_gap['position'] = position
                    except Exception as e:
                        self._log_error(f"Failed to convert position data: {e}")
                        safe_gap['position'] = []
                else:
                    safe_gap['position'] = []
                safe_gaps.append(safe_gap)
            
            space_data = {
                'theory_space': {name: vector.tolist() for name, vector in self.theory_space.items()},
                'concept_space': {name: vector.tolist() for name, vector in self.concept_space.items()},
                'formula_space': {name: vector.tolist() for name, vector in self.formula_space.items()},
                'conceptual_gaps': safe_gaps,
                'semantic_dimensions': self.semantic_dimensions,
                'analysis_summary': self._get_space_analysis_summary()
            }
            
            space_file = self.output_dir / "concept_space_data.json"
            with open(space_file, 'w', encoding='utf-8') as f:
                json.dump(space_data, f, ensure_ascii=False, indent=2)
            
            self._log_info(f"Conceptual space data saved: {space_file}")
            
        except Exception as e:
            self._log_error(f"Failed to save space data: {e}")
            # 保存一个简化版本
            try:
                simplified_data = {
                    'error': str(e),
                    'theory_count': len(self.theory_space),
                    'concept_count': len(self.concept_space),
                    'formula_count': len(self.formula_space),
                    'gaps_count': len(self.conceptual_gaps)
                }
                space_file = self.output_dir / "concept_space_data.json"
                with open(space_file, 'w', encoding='utf-8') as f:
                    json.dump(simplified_data, f, ensure_ascii=False, indent=2)
                self._log_info(f"Saved simplified space data: {space_file}")
            except Exception as e2:
                self._log_error(f"Even simplified version failed to save: {e2}")

    def _save_result(self, result: Dict[str, Any]):
        """保存生成结果"""
        try:
            # 保存主要结果文件
            result_file = self.output_dir / "generation_result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            self._log_info(f"Saving results to: {result_file}")
            
            # 创建eval_ready_theories目录并保存理论文件供后续评估使用
            eval_theories_dir = self.output_dir / "eval_ready_theories"
            eval_theories_dir.mkdir(exist_ok=True)
            
            # 导入数学分类器（如果需要）
            try:
                import sys
                if '.' not in sys.path:
                    sys.path.append('.')
                from utils.mathematical_classifier import MathematicalClassifier
                classifier = MathematicalClassifier()
            except ImportError:
                classifier = None
                self._log_info("Math classifier not found, skipping mathematical classification")
            
            # 保存每个理论为单独的文件
            theories = result.get('theories', [])
            for i, theory in enumerate(theories):
                # 确保理论有必要的字段
                if 'name' not in theory and 'theory_name' not in theory:
                    theory['theory_name'] = f"unified_theory_{i+1}"
                
                # 添加生成元数据
                if 'metadata' not in theory:
                    theory['metadata'] = {}
                
                theory['metadata'].update({
                    'generation_method': 'unified_space_based',
                    'generator_version': '1.0',
                    'concept_space_size': len(self.concept_space),
                    'theory_space_size': len(self.theory_space)
                })
                
                # 进行数学分类标注
                if classifier:
                    try:
                        theory = classifier.annotate_theory_with_classification(theory)
                    except Exception as e:
                        self._log_error(f"Mathematical classification failed: {e}")
                
                # 保存理论文件
                theory_name = theory.get('name', theory.get('theory_name', f'theory_{i+1}'))
                # 清理文件名中的特殊字符
                safe_name = self._generate_safe_filename(theory_name, i+1)
                
                theory_file = eval_theories_dir / f"{safe_name}.json"
                with open(theory_file, 'w', encoding='utf-8') as f:
                    json.dump(theory, f, ensure_ascii=False, indent=2)
                
                self._log_info(f"Saving theory: {theory_file}")
            
            self._log_info(f"✅ Saved {len(theories)} theories to eval_ready_theories directory")
            
        except Exception as e:
            self._log_error(f"Failed to save results: {e}")
            raise

    def _get_position_sample(self, position) -> List[float]:
        """获取空间位置的采样（只保留前10维）"""
        try:
            if position is None:
                return []
            
            # 检查是否是numpy数组或类似结构
            if hasattr(position, '__getitem__') and hasattr(position, '__len__'):
                if len(position) > 0:
                    # 转换为Python列表并取前10维
                    if hasattr(position, 'tolist'):
                        pos_list = position.tolist()
                    else:
                        pos_list = list(position)
                    return pos_list[:10]
            return []
        except Exception as e:
            self._log_error(f"Position sampling failed: {e}")
            return []
    
    def _get_safe_length(self, position) -> int:
        """安全获取位置向量的长度"""
        try:
            if position is None:
                return 0
            if hasattr(position, '__len__'):
                return len(position)
            return 0
        except Exception as e:
            self._log_error(f"Failed to get position length: {e}")
            return 0
    
    def _generate_space_visualization(self):
        """生成概念空间可视化"""
        try:
            # 动态导入可视化模块
            from utils.concept_space_visualizer import ConceptSpaceVisualizer
            
            self._log_info("📊 Generating conceptual space visualization...")
            
            # 创建可视化器
            viz_dir = self.output_dir / "unified_space_visualization"
            visualizer = ConceptSpaceVisualizer(str(viz_dir))
            
            # 准备数据
            all_embeddings = {}
            all_labels = {}
            
            # 添加概念嵌入
            for name, embedding in self.concept_space.items():
                if embedding is not None:
                    all_embeddings[f"Concept: {name}"] = embedding
                    all_labels[f"Concept: {name}"] = "Concept"
            
            # 添加理论嵌入
            for name, embedding in self.theory_space.items():
                if embedding is not None:
                    all_embeddings[f"Theory: {name}"] = embedding
                    all_labels[f"Theory: {name}"] = "Theory"
            
            # 添加公式嵌入（如果有）
            for name, embedding in self.formula_space.items():
                if embedding is not None:
                    all_embeddings[f"Formula: {name}"] = embedding
                    all_labels[f"Formula: {name}"] = "Formula"
            
            # 确保有足够的数据
            if len(all_embeddings) < 3:
                self._log_warning("Insufficient embeddings, skipping visualization")
                return
            
            # 生成可视化报告
            visualizer.create_comprehensive_report(
                all_embeddings,
                labels=all_labels,
                output_prefix="unified_concept_space"
            )
            
            # 如果有概念空白区域，生成专门的可视化
            if self.conceptual_gaps:
                self._visualize_conceptual_gaps(visualizer)
            
            self._log_info(f"✅ Conceptual space visualization generated: {viz_dir}")
            
        except ImportError:
            self._log_warning("Visualization module not installed, skipping visualization")
        except Exception as e:
            self._log_error(f"Error generating visualization: {e}")
    
    def _visualize_conceptual_gaps(self, visualizer):
        """可视化概念空白区域"""
        try:
            # 创建一个特殊的可视化，标记概念空白区域
            import matplotlib.pyplot as plt
            import numpy as np
            from sklearn.decomposition import PCA
            
            # 收集所有嵌入
            all_embeddings = []
            all_names = []
            all_types = []
            
            for name, emb in self.concept_space.items():
                if emb is not None:
                    all_embeddings.append(emb)
                    all_names.append(f"Concept: {name}")
                    all_types.append("concept")
            
            for name, emb in self.theory_space.items():
                if emb is not None:
                    all_embeddings.append(emb)
                    all_names.append(f"Theory: {name}")
                    all_types.append("theory")
            
            if len(all_embeddings) < 3:
                return
            
            # PCA降维
            pca = PCA(n_components=2, random_state=42)
            coords_2d = pca.fit_transform(all_embeddings)
            
            # 创建图形
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # 绘制现有概念和理论
            colors = {'concept': 'blue', 'theory': 'green'}
            for i, (name, type_) in enumerate(zip(all_names, all_types)):
                ax.scatter(coords_2d[i, 0], coords_2d[i, 1], 
                          c=colors[type_], s=100, alpha=0.7,
                          edgecolors='black', linewidth=0.5)
                ax.annotate(name, (coords_2d[i, 0], coords_2d[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
            
            # 标记概念空白区域
            for i, gap in enumerate(self.conceptual_gaps[:5]):  # 最多显示5个
                if 'position' in gap:
                    # 将高维位置投影到2D
                    gap_2d = pca.transform([gap['position']])[0]
                    ax.scatter(gap_2d[0], gap_2d[1], 
                              c='red', marker='x', s=200, linewidth=3,
                              label='Conceptual Gap' if i == 0 else '')
                    ax.annotate(f"Gap{i+1}", (gap_2d[0], gap_2d[1]),
                               xytext=(5, -15), textcoords='offset points',
                               fontsize=10, fontweight='bold', color='red')
            
            ax.set_xlabel('Principal Component 1', fontsize=12)
            ax.set_ylabel('Principal Component 2', fontsize=12)
            ax.set_title('Conceptual Space and Gap Region Analysis', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = self.output_dir / "unified_space_visualization" / "conceptual_gaps.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._log_info(f"Conceptual gap region visualization saved: {save_path}")
            
        except Exception as e:
            self._log_error(f"Error visualizing conceptual gap regions: {e}")
    
    async def _embed_concepts_with_physics(self, concepts: List[Dict]):
        """使用物理嵌入器嵌入概念"""
        self._log_info("Using physics embedder for enhanced concept embedding")
        
        for concept in concepts:
            try:
                # 构建增强文本
                text = f"{concept['name']}: {concept['description']}"
                
                # 获取相关公式（如果有）
                related_formulas = concept.get('related_formulas', [])
                
                # 使用物理嵌入器
                embedding = await self.physics_embedder.embed_concept_with_context(
                    concept['name'],
                    concept['description'],
                    related_formulas
                )
                
                # 存储嵌入
                self.concept_space[concept['name']] = embedding.vector
                
            except Exception as e:
                self._log_error(f"Failed to embed concept {concept['name']}: {e}")
    
    async def _embed_formulas_with_physics(self, formulas: List[Dict]):
        """使用物理嵌入器嵌入公式"""
        self._log_info("Using physics embedder for enhanced formula embedding")
        
        for formula in formulas:
            try:
                # 使用物理嵌入器
                embedding = await self.physics_embedder.embed_formula_with_derivation(
                    formula['expression'],
                    formula.get('description', '')
                )
                
                # 存储嵌入
                self.formula_space[formula['expression']] = embedding.vector
                
            except Exception as e:
                self._log_error(f"Failed to embed formula: {e}")
    
    def _compute_concept_importance(self):
        """使用知识图谱计算概念重要性"""
        self._log_info("Computing concept importance from knowledge graph")
        
        try:
            # 获取概念中心性分数
            centrality = self.knowledge_graph.compute_centrality()
            
            # 匹配概念空间中的概念
            for concept_name in self.concept_space:
                node = self.knowledge_graph.find_node_by_name(concept_name)
                if node:
                    self.concept_importance[concept_name] = centrality.get(node.id, 0.5)
                else:
                    self.concept_importance[concept_name] = 0.5  # 默认重要性
            
            # 归一化
            if self.concept_importance:
                max_importance = max(self.concept_importance.values())
                for concept in self.concept_importance:
                    self.concept_importance[concept] /= max_importance
                    
        except Exception as e:
            self._log_error(f"Failed to compute concept importance: {e}")
    
    def _identify_high_value_gaps(self):
        """识别高价值的概念空白"""
        if not self.physics_embedder or not self.knowledge_graph:
            return self._identify_conceptual_gaps()
        
        self._log_info("🎯 Identifying high-value conceptual gaps using enhanced analysis")
        
        # 使用物理嵌入器识别概念桥梁
        embeddings = {}
        for name, vector in self.concept_space.items():
            embeddings[name] = type('PhysicsEmbedding', (), {
                'vector': vector,
                'metadata': {'domain': self._infer_domain(name)}
            })()
        
        # 查找概念桥梁（跨领域的高相似度概念对）
        bridges = self.physics_embedder.find_conceptual_bridges(embeddings, threshold=0.6)
        
        # 将桥梁转换为空白区域
        for concept1, concept2, similarity in bridges[:10]:
            if concept1 in self.concept_space and concept2 in self.concept_space:
                midpoint = (self.concept_space[concept1] + self.concept_space[concept2]) / 2
                
                self.conceptual_gaps.append({
                    'type': 'conceptual_bridge',
                    'position': midpoint,
                    'source_theories': [concept1, concept2],
                    'description': f"Conceptual bridge between {concept1} and {concept2}",
                    'value_score': similarity * self.concept_importance.get(concept1, 0.5) * self.concept_importance.get(concept2, 0.5)
                })
        
        # 使用传统方法识别其他空白
        traditional_gaps = self._identify_conceptual_gaps()
        
        # 合并并排序
        all_gaps = self.conceptual_gaps + traditional_gaps
        all_gaps.sort(key=lambda x: x.get('value_score', 0), reverse=True)
        
        self.conceptual_gaps = all_gaps[:20]  # 保留前20个最有价值的空白
        self._log_info(f"Found {len(self.conceptual_gaps)} high-value conceptual gaps")
    
    def _infer_domain(self, concept_name: str) -> str:
        """推断概念所属领域"""
        # 简单的领域推断
        concept_lower = concept_name.lower()
        if any(kw in concept_lower for kw in ['quantum', 'wave', 'superposition', 'entangle']):
            return 'quantum_mechanics'
        elif any(kw in concept_lower for kw in ['entropy', 'temperature', 'statistical']):
            return 'statistical_mechanics'
        elif any(kw in concept_lower for kw in ['space', 'time', 'relativity', 'metric']):
            return 'relativity'
        else:
            return 'general_physics'
    
    def _build_gap_context(self, gap: Dict) -> str:
        """构建空白区域的上下文信息"""
        context_parts = []
        
        # 添加概念重要性信息
        if self.concept_importance and 'source_theories' in gap:
            importance_info = []
            for concept in gap['source_theories']:
                if concept in self.concept_importance:
                    importance = self.concept_importance[concept]
                    importance_info.append(f"- {concept} (importance: {importance:.2f})")
            
            if importance_info:
                context_parts.append("Concept Importance Scores:\n" + "\n".join(importance_info))
        
        # 添加相关概念信息（从知识图谱）
        if self.knowledge_graph and 'source_theories' in gap:
            related_concepts = []
            for concept in gap['source_theories']:
                node = self.knowledge_graph.find_node_by_name(concept)
                if node:
                    neighbors = self.knowledge_graph.get_neighbors(node.id)[:5]
                    for neighbor_id in neighbors:
                        neighbor_node = self.knowledge_graph.nodes.get(neighbor_id)
                        if neighbor_node:
                            related_concepts.append(neighbor_node.name)
            
            if related_concepts:
                unique_related = list(set(related_concepts))[:10]
                context_parts.append("Related Concepts in Knowledge Graph:\n" + ", ".join(unique_related))
        
        # 添加价值分数信息
        if 'value_score' in gap:
            context_parts.append(f"Gap Value Score: {gap['value_score']:.3f}")
        
        return "\n\n".join(context_parts) if context_parts else ""


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
            print(f"Test result: {result['success']}")
            if result['success']:
                print(f"Number of theories generated: {len(result['theories'])}")
                print(f"Conceptual space analysis: {result['metadata']['space_analysis']}")
    
    # 如果直接运行此文件，执行测试
    asyncio.run(test_unified_generator()) 