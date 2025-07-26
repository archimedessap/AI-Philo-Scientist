#!/usr/bin/env python3
"""
Enhanced Concept Extractor - 增强的概念提取器

该模块从科学文献中提取高质量的概念、定义和关系。
支持多轮提取、概念分类、关系识别和置信度评分。

Author: UniversalTheoryGen Team
Date: 2025-01-24
"""

import json
import os
import sys
import csv
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import asyncio
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from theory_generation.llm_interface import LLMInterface
from utils.logging_config import get_logger, log_execution
from utils.retry_decorator import retry_with_exponential_backoff

# 初始化日志
logger = get_logger('enhanced_concept_extractor')

@dataclass
class Concept:
    """概念数据结构"""
    name: str
    description: str
    category: str  # fundamental, derived, experimental, theoretical_framework
    domain: str
    source: str
    definition: str
    related_formulas: List[str]
    prerequisites: List[str]  # 前置概念
    applications: List[str]
    confidence: float  # 0-1
    context: str  # 原文上下文
    
@dataclass
class ConceptRelation:
    """概念关系"""
    concept1: str
    relation_type: str  # depends_on, contradicts, generalizes, specializes, equivalent
    concept2: str
    confidence: float
    evidence: str

class EnhancedConceptExtractor:
    """增强的概念提取器"""
    
    # 概念类别定义
    CONCEPT_CATEGORIES = {
        'fundamental': 'Basic concepts that form the foundation of the theory',
        'derived': 'Concepts derived from fundamental principles',
        'experimental': 'Concepts related to experimental observations',
        'theoretical_framework': 'High-level theoretical constructs'
    }
    
    # 关系类型定义
    RELATION_TYPES = {
        'depends_on': 'Concept A requires understanding of Concept B',
        'contradicts': 'Concepts are mutually exclusive or in opposition',
        'generalizes': 'Concept A is a generalization of Concept B',
        'specializes': 'Concept A is a special case of Concept B',
        'equivalent': 'Concepts are different names for the same thing'
    }
    
    def __init__(self, 
                 model_source: str = "google",
                 model_name: str = "gemini-2.0-flash-exp",
                 output_dir: str = "data/enhanced_concepts"):
        """初始化增强概念提取器"""
        self.llm = LLMInterface(model_source=model_source, model_name=model_name)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 统计信息
        self.stats = {
            'total_documents': 0,
            'total_concepts': 0,
            'total_relations': 0,
            'categories': defaultdict(int),
            'relation_types': defaultdict(int)
        }
    
    @retry_with_exponential_backoff(retries=3)
    async def extract_concepts_multi_pass(self, text: str, source: str) -> Tuple[List[Concept], List[ConceptRelation]]:
        """多轮概念提取"""
        logger.info(f"Starting multi-pass extraction for {source}")
        
        # 第一轮：提取概念列表
        concept_names = await self._extract_concept_list(text)
        logger.info(f"Found {len(concept_names)} potential concepts")
        
        # 第二轮：深入分析每个概念
        concepts = []
        for name in concept_names[:20]:  # 限制数量避免token超限
            concept = await self._extract_concept_details(name, text, source)
            if concept and concept.confidence > 0.6:
                concepts.append(concept)
        
        # 第三轮：提取概念关系
        relations = await self._extract_concept_relations(concepts, text)
        
        logger.info(f"Extracted {len(concepts)} high-quality concepts and {len(relations)} relations")
        return concepts, relations
    
    async def _extract_concept_list(self, text: str) -> List[str]:
        """提取概念名称列表"""
        prompt = f"""
        You are analyzing a scientific text on quantum physics. Extract all important concepts, terms, and principles mentioned.
        
        Text: {text[:3000]}
        
        List all significant concepts (one per line, no descriptions):
        """
        
        response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        concepts = [line.strip() for line in response.strip().split('\n') if line.strip()]
        return concepts[:30]  # 限制数量
    
    async def _extract_concept_details(self, concept_name: str, text: str, source: str) -> Optional[Concept]:
        """提取概念的详细信息"""
        # 查找概念在文本中的上下文
        context = self._find_context(concept_name, text, window=500)
        if not context:
            return None
        
        prompt = f"""
        Analyze the concept "{concept_name}" from this quantum physics text.
        
        Context: {context}
        
        Provide the following in JSON format:
        {{
            "name": "exact concept name",
            "description": "brief description (1-2 sentences)",
            "category": "one of: fundamental, derived, experimental, theoretical_framework",
            "domain": "specific physics domain (e.g., quantum_mechanics, measurement_theory)",
            "definition": "formal definition if available",
            "related_formulas": ["list of related mathematical formulas"],
            "prerequisites": ["concepts that must be understood first"],
            "applications": ["practical or theoretical applications"],
            "confidence": 0.0-1.0 (your confidence in this extraction)
        }}
        """
        
        try:
            response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
            data = json.loads(self._extract_json(response))
            
            return Concept(
                name=data.get('name', concept_name),
                description=data.get('description', ''),
                category=data.get('category', 'fundamental'),
                domain=data.get('domain', 'quantum_physics'),
                source=source,
                definition=data.get('definition', ''),
                related_formulas=data.get('related_formulas', []),
                prerequisites=data.get('prerequisites', []),
                applications=data.get('applications', []),
                confidence=float(data.get('confidence', 0.5)),
                context=context[:500]
            )
        except Exception as e:
            logger.warning(f"Failed to extract details for {concept_name}: {e}")
            return None
    
    async def _extract_concept_relations(self, concepts: List[Concept], text: str) -> List[ConceptRelation]:
        """提取概念之间的关系"""
        if len(concepts) < 2:
            return []
        
        # 构建概念对
        concept_pairs = []
        for i in range(len(concepts)):
            for j in range(i+1, min(i+5, len(concepts))):  # 只检查相近的概念
                concept_pairs.append((concepts[i], concepts[j]))
        
        relations = []
        for c1, c2 in concept_pairs[:20]:  # 限制数量
            relation = await self._analyze_concept_pair(c1, c2, text)
            if relation and relation.confidence > 0.6:
                relations.append(relation)
        
        return relations
    
    async def _analyze_concept_pair(self, c1: Concept, c2: Concept, text: str) -> Optional[ConceptRelation]:
        """分析一对概念的关系"""
        prompt = f"""
        Analyze the relationship between these quantum physics concepts:
        
        Concept 1: {c1.name} - {c1.description}
        Concept 2: {c2.name} - {c2.description}
        
        Context: {text[:1000]}
        
        Determine their relationship:
        - depends_on: one requires the other
        - contradicts: mutually exclusive
        - generalizes: one is more general
        - specializes: one is a special case
        - equivalent: same concept, different names
        - none: no clear relationship
        
        Response format:
        {{
            "relation_type": "type or none",
            "direction": "1to2 or 2to1",
            "confidence": 0.0-1.0,
            "evidence": "brief explanation"
        }}
        """
        
        try:
            response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
            data = json.loads(self._extract_json(response))
            
            if data.get('relation_type') == 'none':
                return None
            
            # 确定方向
            if data.get('direction') == '2to1':
                concept1, concept2 = c2.name, c1.name
            else:
                concept1, concept2 = c1.name, c2.name
            
            return ConceptRelation(
                concept1=concept1,
                relation_type=data.get('relation_type', 'depends_on'),
                concept2=concept2,
                confidence=float(data.get('confidence', 0.5)),
                evidence=data.get('evidence', '')
            )
        except Exception as e:
            logger.warning(f"Failed to analyze relation between {c1.name} and {c2.name}: {e}")
            return None
    
    def _find_context(self, concept: str, text: str, window: int = 500) -> Optional[str]:
        """查找概念在文本中的上下文"""
        # 查找概念出现的位置
        pattern = re.compile(r'\b' + re.escape(concept) + r'\b', re.IGNORECASE)
        match = pattern.search(text)
        
        if match:
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            return text[start:end]
        return None
    
    def _extract_json(self, text: str) -> str:
        """从响应中提取JSON"""
        # 查找JSON块
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # 尝试直接查找JSON对象
        json_match = re.search(r'(\{.*?\})', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        return text
    
    async def process_document(self, doc_path: str) -> Tuple[List[Concept], List[ConceptRelation]]:
        """处理单个文档"""
        logger.info(f"Processing document: {doc_path}")
        
        # 读取预处理的文档
        with open(doc_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_concepts = []
        all_relations = []
        
        # 判断文档格式
        if isinstance(data, list):
            # 新格式：列表形式
            for i, doc in enumerate(data):
                # 合并title和content
                text = doc.get('title', '') + '\n\n' + doc.get('content', '')
                if len(text) < 100:
                    continue
                
                concepts, relations = await self.extract_concepts_multi_pass(
                    text[:5000],  # 限制长度避免超时
                    f"{os.path.basename(doc_path)}#doc_{i}"
                )
                
                all_concepts.extend(concepts)
                all_relations.extend(relations)
                break  # 只处理第一个文档以加快测试
        else:
            # 旧格式：字典形式
            for i, segment in enumerate(data.get('segments', [])):
                text = segment.get('text', '')
                if len(text) < 100:
                    continue
                
                concepts, relations = await self.extract_concepts_multi_pass(
                    text, 
                    f"{os.path.basename(doc_path)}#segment_{i}"
                )
                
                all_concepts.extend(concepts)
                all_relations.extend(relations)
        
        # 去重和合并
        concepts = self._deduplicate_concepts(all_concepts)
        relations = self._deduplicate_relations(all_relations)
        
        # 更新统计
        self._update_stats(concepts, relations)
        
        return concepts, relations
    
    def _deduplicate_concepts(self, concepts: List[Concept]) -> List[Concept]:
        """去重概念，保留最高置信度的版本"""
        concept_map = {}
        for concept in concepts:
            key = concept.name.lower()
            if key not in concept_map or concept.confidence > concept_map[key].confidence:
                concept_map[key] = concept
        return list(concept_map.values())
    
    def _deduplicate_relations(self, relations: List[ConceptRelation]) -> List[ConceptRelation]:
        """去重关系"""
        relation_set = set()
        unique_relations = []
        
        for rel in relations:
            key = (rel.concept1, rel.relation_type, rel.concept2)
            if key not in relation_set:
                relation_set.add(key)
                unique_relations.append(rel)
        
        return unique_relations
    
    def _update_stats(self, concepts: List[Concept], relations: List[ConceptRelation]):
        """更新统计信息"""
        self.stats['total_documents'] += 1
        self.stats['total_concepts'] += len(concepts)
        self.stats['total_relations'] += len(relations)
        
        for concept in concepts:
            self.stats['categories'][concept.category] += 1
        
        for relation in relations:
            self.stats['relation_types'][relation.relation_type] += 1
    
    def save_results(self, concepts: List[Concept], relations: List[ConceptRelation], 
                     output_prefix: str = "enhanced_concepts"):
        """保存提取结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存概念
        concepts_file = os.path.join(self.output_dir, f"{output_prefix}_concepts_{timestamp}.json")
        with open(concepts_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(c) for c in concepts], f, ensure_ascii=False, indent=2)
        
        # 保存CSV格式（兼容现有系统）
        csv_file = os.path.join(self.output_dir, f"{output_prefix}_concepts_{timestamp}.csv")
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['name', 'description', 'category', 
                                                   'domain', 'source', 'confidence'])
            writer.writeheader()
            for concept in concepts:
                writer.writerow({
                    'name': concept.name,
                    'description': concept.description,
                    'category': concept.category,
                    'domain': concept.domain,
                    'source': concept.source,
                    'confidence': concept.confidence
                })
        
        # 保存关系
        relations_file = os.path.join(self.output_dir, f"{output_prefix}_relations_{timestamp}.json")
        with open(relations_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in relations], f, ensure_ascii=False, indent=2)
        
        # 保存统计信息
        stats_file = os.path.join(self.output_dir, f"{output_prefix}_stats_{timestamp}.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"Concepts: {concepts_file}")
        logger.info(f"Relations: {relations_file}")
        logger.info(f"Stats: {stats_file}")
        
        return concepts_file, relations_file, stats_file


async def main():
    """主函数 - 测试增强概念提取器"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Concept Extractor')
    parser.add_argument('--input_dir', type=str, default='data/preprocessed_documents',
                       help='Directory containing preprocessed documents')
    parser.add_argument('--output_dir', type=str, default='data/enhanced_concepts',
                       help='Output directory for extracted concepts')
    parser.add_argument('--model_source', type=str, default='google',
                       help='LLM model source')
    parser.add_argument('--model_name', type=str, default='gemini-2.0-flash-exp',
                       help='LLM model name')
    parser.add_argument('--max_docs', type=int, default=5,
                       help='Maximum number of documents to process')
    
    args = parser.parse_args()
    
    # 初始化提取器
    extractor = EnhancedConceptExtractor(
        model_source=args.model_source,
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    # 获取所有预处理文档
    doc_files = []
    if os.path.exists(args.input_dir):
        for file in os.listdir(args.input_dir):
            if file.endswith('.json'):
                doc_files.append(os.path.join(args.input_dir, file))
    
    if not doc_files:
        logger.warning(f"No preprocessed documents found in {args.input_dir}")
        return
    
    # 处理文档
    all_concepts = []
    all_relations = []
    
    for i, doc_path in enumerate(doc_files[:args.max_docs]):
        logger.info(f"Processing document {i+1}/{min(len(doc_files), args.max_docs)}")
        concepts, relations = await extractor.process_document(doc_path)
        all_concepts.extend(concepts)
        all_relations.extend(relations)
    
    # 最终去重
    final_concepts = extractor._deduplicate_concepts(all_concepts)
    final_relations = extractor._deduplicate_relations(all_relations)
    
    # 保存结果
    extractor.save_results(final_concepts, final_relations)
    
    # 打印统计
    print("\n=== Extraction Statistics ===")
    print(f"Total documents processed: {extractor.stats['total_documents']}")
    print(f"Total concepts extracted: {len(final_concepts)}")
    print(f"Total relations found: {len(final_relations)}")
    print("\nConcept categories:")
    for cat, count in extractor.stats['categories'].items():
        print(f"  {cat}: {count}")
    print("\nRelation types:")
    for rel, count in extractor.stats['relation_types'].items():
        print(f"  {rel}: {count}")


if __name__ == "__main__":
    asyncio.run(main())