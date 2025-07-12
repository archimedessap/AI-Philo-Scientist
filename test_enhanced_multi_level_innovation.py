#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试增强的多级创新生成器

演示如何结合概念向量空间和多级创新来生成新理论
"""

import asyncio
import json
import os
from pathlib import Path

from theory_generation.llm_interface import LLMInterface
from theory_generation.multi_level_innovation_generator import (
    MultiLevelInnovationGenerator, 
    MultiLevelInnovationConfig
)
from theory_generation.innovation_framework import InnovationLevel
from core_embedding.concept_extractor import ConceptExtractor
from core_embedding.embedding import ConceptEmbedder

async def test_enhanced_multi_level_innovation():
    """测试增强的多级创新生成"""
    
    print("🧪 测试增强的多级创新生成器")
    print("=" * 60)
    
    # 1. 初始化LLM接口
    llm = LLMInterface(
        model_source="google",
        model_name="gemini-2.5-pro"
    )
    
    # 2. 模拟概念嵌入数据（实际应用中从文件加载）
    concept_embeddings = await create_mock_concept_embeddings(llm)
    
    # 3. 初始化多级创新生成器
    generator = MultiLevelInnovationGenerator(
        llm_interface=llm,
        concept_embeddings=concept_embeddings
    )
    
    # 4. 创建多级创新配置
    config = generator.create_multi_level_config(
        target_levels=[
            InnovationLevel.PARAMETER_EXTENSION,
            InnovationLevel.INTERPRETATION
        ],
        weights={
            InnovationLevel.PARAMETER_EXTENSION: 0.6,
            InnovationLevel.INTERPRETATION: 0.4
        },
        synthesis_mode="fusion",
        innovation_intensity=0.8
    )
    config.use_concept_space = True
    
    # 5. 模拟理论矛盾
    contradiction = {
        "theory1": "Copenhagen Interpretation",
        "theory2": "Many-Worlds Interpretation",
        "contradictions": [
            {
                "dimension": "wave_function_collapse",
                "theory1_position": "波函数在测量时坍缩",
                "theory2_position": "波函数从不坍缩，所有可能性都实现",
                "contradiction_nature": "测量过程的本质完全不同",
                "importance": 9
            },
            {
                "dimension": "reality_of_possibilities",
                "theory1_position": "只有一个结果是真实的",
                "theory2_position": "所有可能的结果都是真实的",
                "contradiction_nature": "对现实本质的根本分歧",
                "importance": 8
            }
        ]
    }
    
    # 6. 生成新理论
    print("\n🚀 开始生成增强的多级创新理论...")
    new_theory = await generator.generate_multi_level_theory(
        contradiction=contradiction,
        config=config
    )
    
    # 7. 输出结果
    if "error" in new_theory:
        print(f"❌ 生成失败: {new_theory['error']}")
    else:
        print("\n🎉 成功生成新理论!")
        print(f"理论名称: {new_theory.get('name', 'Unknown')}")
        print(f"核心原理: {new_theory.get('core_principles', 'N/A')[:200]}...")
        
        # 检查是否有概念空间分析
        if new_theory.get('concept_space_enhanced'):
            print("✅ 理论生成过程使用了概念向量空间增强")
        
        # 保存结果
        output_file = "enhanced_multi_level_theory_test.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_theory, f, ensure_ascii=False, indent=2)
        print(f"📄 结果已保存到: {output_file}")

async def create_mock_concept_embeddings(llm):
    """创建模拟的概念嵌入数据"""
    print("🔧 创建模拟概念嵌入数据...")
    
    # 定义一些关键概念
    concepts = [
        "wave function",
        "measurement",
        "quantum superposition",
        "decoherence",
        "observer effect",
        "quantum entanglement",
        "probability amplitude",
        "collapse",
        "many worlds",
        "reality",
        "determinism",
        "locality"
    ]
    
    # 为每个概念生成嵌入向量
    concept_embeddings = {}
    for concept in concepts:
        try:
            # 使用LLM的嵌入功能
            embedding = await llm.get_embedding(f"quantum mechanics concept: {concept}")
            concept_embeddings[concept] = embedding
            print(f"  ✓ {concept}")
        except Exception as e:
            print(f"  ✗ {concept}: {e}")
    
    print(f"📊 创建了 {len(concept_embeddings)} 个概念嵌入")
    return concept_embeddings

async def test_concept_space_analysis():
    """测试概念空间分析功能"""
    print("\n🔍 测试概念空间分析...")
    
    # 初始化LLM
    llm = LLMInterface(
        model_source="google",
        model_name="gemini-2.5-pro"
    )
    
    # 创建概念嵌入
    concept_embeddings = await create_mock_concept_embeddings(llm)
    
    # 初始化生成器
    generator = MultiLevelInnovationGenerator(
        llm_interface=llm,
        concept_embeddings=concept_embeddings
    )
    
    # 测试概念相关性分析
    related_concepts = generator._find_related_concepts(
        "measurement", 
        "wave function collapses",
        "all possibilities exist"
    )
    
    print(f"找到 {len(related_concepts)} 个相关概念:")
    for concept_info in related_concepts:
        print(f"  - {concept_info['concept']}: 相关性 {concept_info['relevance_to_dimension']:.3f}")
        for related_name, similarity in concept_info['related_concepts']:
            print(f"    → {related_name}: {similarity:.3f}")

async def main():
    """主函数"""
    try:
        await test_enhanced_multi_level_innovation()
        await test_concept_space_analysis()
        
        print("\n🎯 测试完成！")
        print("=" * 60)
        print("主要改进:")
        print("1. ✅ 集成了概念向量空间分析")
        print("2. ✅ 增强了矛盾检测能力")
        print("3. ✅ 支持基于概念相似度的理论生成")
        print("4. ✅ 提供了概念张力计算")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 