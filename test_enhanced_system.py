#!/usr/bin/env python3
"""
Test Enhanced System - 测试增强的概念提取和理论生成系统

该脚本演示和测试改进后的系统，包括：
1. 增强的概念提取
2. 公式提取和分类
3. 知识图谱构建
4. 基于增强概念空间的理论生成

Author: UniversalTheoryGen Team
Date: 2025-01-24
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logging_config import get_logger

# 初始化日志
logger = get_logger('test_enhanced_system')

async def test_enhanced_concept_extraction():
    """测试增强的概念提取"""
    logger.info("\n=== Testing Enhanced Concept Extraction ===")
    
    from enhanced_concept_extractor import EnhancedConceptExtractor
    
    # 初始化提取器
    extractor = EnhancedConceptExtractor(
        model_source="google",
        model_name="gemini-2.0-flash-exp"
    )
    
    # 测试文档
    test_doc = Path('data/preprocessed')
    if test_doc.exists():
        docs = list(test_doc.glob('*.json'))[:2]  # 只处理2个文档
        
        all_concepts = []
        all_relations = []
        
        for doc in docs:
            logger.info(f"Processing {doc.name}...")
            concepts, relations = await extractor.process_document(str(doc))
            all_concepts.extend(concepts)
            all_relations.extend(relations)
        
        # 保存结果
        extractor.save_results(all_concepts, all_relations, "test_enhanced")
        
        logger.info(f"✅ Extracted {len(all_concepts)} concepts and {len(all_relations)} relations")
        
        # 显示一些示例
        if all_concepts:
            logger.info("\nSample concepts:")
            for concept in all_concepts[:3]:
                logger.info(f"  - {concept.name} ({concept.category}): {concept.description[:100]}...")
        
        return all_concepts, all_relations
    else:
        logger.warning("No preprocessed documents found")
        return [], []

async def test_formula_extraction():
    """测试公式提取"""
    logger.info("\n=== Testing Formula Extraction ===")
    
    from formula_extractor import FormulaExtractor
    
    # 初始化提取器
    extractor = FormulaExtractor(
        model_source="google",
        model_name="gemini-2.0-flash-exp"
    )
    
    # 测试文档
    test_doc = Path('data/preprocessed')
    if test_doc.exists():
        docs = list(test_doc.glob('*.json'))[:1]  # 只处理1个文档
        
        all_formulas = []
        all_relations = []
        
        for doc in docs:
            logger.info(f"Processing {doc.name} for formulas...")
            formulas, relations = await extractor.process_document(str(doc))
            all_formulas.extend(formulas)
            all_relations.extend(relations)
        
        # 保存结果
        extractor.save_results(all_formulas, all_relations, "test_formulas")
        
        logger.info(f"✅ Extracted {len(all_formulas)} formulas")
        
        # 显示一些示例
        if all_formulas:
            logger.info("\nSample formulas:")
            for formula in all_formulas[:3]:
                logger.info(f"  - {formula.name}: {formula.expression}")
        
        return all_formulas
    else:
        logger.warning("No preprocessed documents found")
        return []

def test_knowledge_graph_building():
    """测试知识图谱构建"""
    logger.info("\n=== Testing Knowledge Graph Building ===")
    
    from knowledge_graph_builder import KnowledgeGraphBuilder
    
    # 初始化构建器
    builder = KnowledgeGraphBuilder()
    
    # 查找最新的概念文件
    concepts_dir = Path('data/enhanced_concepts')
    if concepts_dir.exists():
        concept_files = list(concepts_dir.glob('test_enhanced_concepts_*.json'))
        if concept_files:
            latest_concepts = max(concept_files, key=os.path.getctime)
            builder.load_concepts(str(latest_concepts))
            logger.info(f"Loaded concepts from {latest_concepts.name}")
        
        # 查找关系文件
        relation_files = list(concepts_dir.glob('test_enhanced_relations_*.json'))
        if relation_files:
            latest_relations = max(relation_files, key=os.path.getctime)
            builder.load_concept_relations(str(latest_relations))
            logger.info(f"Loaded relations from {latest_relations.name}")
    
    # 查找公式文件
    formulas_dir = Path('data/extracted_formulas')
    if formulas_dir.exists():
        formula_files = list(formulas_dir.glob('test_formulas_*.json'))
        if formula_files:
            latest_formulas = max(formula_files, key=os.path.getctime)
            builder.load_formulas(str(latest_formulas))
            logger.info(f"Loaded formulas from {latest_formulas.name}")
    
    # 加载理论
    theories_dir = Path('data/theories_test')
    if theories_dir.exists():
        builder.load_theories(str(theories_dir))
        logger.info(f"Loaded theories from {theories_dir}")
    
    # 推理关系
    builder.infer_implicit_relations()
    
    # 保存图谱
    graph_file = builder.save_graph("test_kg")
    
    # 显示统计
    stats = builder.kg.get_statistics()
    logger.info("\nKnowledge Graph Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    return graph_file

async def test_enhanced_theory_generation():
    """测试增强的理论生成"""
    logger.info("\n=== Testing Enhanced Theory Generation ===")
    
    from theory_generation.methods.unified_generator_adapter import UnifiedGeneratorAdapter
    
    # 初始化生成器
    generator = UnifiedGeneratorAdapter(
        theories_dir='data/theories_test',
        output_dir='output_enhanced_test',
        model_source='google',
        model_name='gemini-2.0-flash-exp',
        use_structured_format=True
    )
    
    logger.info("Generating theory with enhanced conceptual space...")
    
    # 生成理论
    # 由于generate()内部使用了asyncio.run()，我们需要在同步环境中调用
    import nest_asyncio
    nest_asyncio.apply()
    result = generator.generate()
    
    if result['success']:
        theories = result.get('theories', [])
        logger.info(f"✅ Generated {len(theories)} theories")
        
        if theories:
            theory = theories[0]
            logger.info(f"\nGenerated Theory: {theory['name']}")
            logger.info(f"Description: {theory['description'][:200]}...")
            
            # 检查是否使用了增强特性
            metadata = result.get('metadata', {})
            if 'used_physics_embedder' in metadata:
                logger.info("✅ Physics embedder was used")
            if 'used_knowledge_graph' in metadata:
                logger.info("✅ Knowledge graph was used")
        
        # 保存结果
        output_file = Path('output_enhanced_test') / 'test_result.json'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Result saved to {output_file}")
        
    else:
        logger.error(f"Theory generation failed: {result.get('error_message', 'Unknown error')}")
    
    return result

async def run_full_test_pipeline():
    """运行完整的测试流程"""
    logger.info("=== Running Full Test Pipeline ===")
    logger.info(f"Start time: {datetime.now()}")
    
    # 1. 测试概念提取
    concepts, relations = await test_enhanced_concept_extraction()
    
    # 2. 测试公式提取
    formulas = await test_formula_extraction()
    
    # 3. 测试知识图谱构建
    kg_file = test_knowledge_graph_building()
    
    # 4. 测试增强的理论生成
    result = await test_enhanced_theory_generation()
    
    # 总结
    logger.info("\n=== Test Summary ===")
    logger.info(f"✅ Concepts extracted: {len(concepts)}")
    logger.info(f"✅ Relations found: {len(relations)}")
    logger.info(f"✅ Formulas extracted: {len(formulas)}")
    logger.info(f"✅ Knowledge graph built: {kg_file}")
    logger.info(f"✅ Theory generation: {'Success' if result.get('success') else 'Failed'}")
    logger.info(f"End time: {datetime.now()}")

async def test_individual_component(component: str):
    """测试单个组件"""
    if component == 'concepts':
        await test_enhanced_concept_extraction()
    elif component == 'formulas':
        await test_formula_extraction()
    elif component == 'graph':
        test_knowledge_graph_building()
    elif component == 'generation':
        await test_enhanced_theory_generation()
    else:
        logger.error(f"Unknown component: {component}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Test Enhanced Theory Generation System')
    parser.add_argument('--component', type=str, 
                       choices=['concepts', 'formulas', 'graph', 'generation', 'all'],
                       default='all',
                       help='Which component to test')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Skip extraction steps (use existing data)')
    
    args = parser.parse_args()
    
    if args.component == 'all':
        if args.skip_extraction:
            # 只测试知识图谱和理论生成
            kg_file = test_knowledge_graph_building()
            asyncio.run(test_enhanced_theory_generation())
        else:
            # 运行完整测试
            asyncio.run(run_full_test_pipeline())
    else:
        asyncio.run(test_individual_component(args.component))

if __name__ == "__main__":
    main()