#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行统一理论生成系统

演示完整的"文献→概念空间→新理论"流程
"""

import asyncio
import argparse
import json
from pathlib import Path
from theory_generation.llm_interface import LLMInterface
from theory_generation.unified_theory_generator import (
    UnifiedTheoryGenerator, 
    UnifiedGenerationConfig
)
from theory_generation.innovation_framework import InnovationLevel


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="统一理论生成系统")
    parser.add_argument("--literature_dirs", nargs='+', 
                       default=["data/literature", "demo/papers"],
                       help="文献目录列表")
    parser.add_argument("--prior_theories_dir", 
                       default="data/theories_v2.1",
                       help="先验理论目录")
    parser.add_argument("--output_dir", 
                       default="unified_theory_output",
                       help="输出目录")
    parser.add_argument("--model_source", 
                       default="openai",
                       help="LLM模型源")
    parser.add_argument("--model_name", 
                       default="gpt-4o-mini",
                       help="LLM模型名称")
    parser.add_argument("--demo_mode", 
                       action="store_true",
                       help="演示模式（使用模拟数据）")
    parser.add_argument("--card_query",
                       default=None,
                       help="使用短卡工作流时的任务描述")
    parser.add_argument("--card_topk",
                       type=int,
                       default=6,
                       help="短卡工作流检索的卡片数量")
    parser.add_argument("--card_only",
                       action="store_true",
                       help="仅运行短卡工作流，跳过旧流程")
    
    args = parser.parse_args()
    
    print("🌟 统一理论生成系统")
    print("=" * 60)
    print("整合：文献概念提取 + 先验理论库 + 概念向量空间 + 多级创新")
    print("=" * 60)
    
    # 1. 初始化LLM接口
    print("\n🔧 初始化LLM接口...")
    llm = LLMInterface(
        model_source=args.model_source,
        model_name=args.model_name
    )
    
    # 2. 配置统一生成系统
    config = UnifiedGenerationConfig(
        literature_dirs=args.literature_dirs,
        prior_theories_dir=args.prior_theories_dir,
        output_dir=args.output_dir,
        target_innovation_levels=[
            InnovationLevel.PARAMETER_EXTENSION,
            InnovationLevel.INTERPRETATION
        ],
        synthesis_mode="fusion",
        innovation_intensity=0.8
    )
    
    # 3. 初始化统一理论生成器
    print("\n🚀 初始化统一理论生成器...")
    generator = UnifiedTheoryGenerator(llm, config)

    card_result = None
    if args.card_query:
        if not generator.card_analyzer:
            print('[WARN] Short-card workflow disabled (missing cards or schema).')
        else:
            card_result = await generator.generate_card_driven_interpretation(
                query=args.card_query,
                top_k=args.card_topk
            )
            card_output = Path(args.output_dir) / "card_workflow_result.json"
            card_output.parent.mkdir(parents=True, exist_ok=True)
            card_output.write_text(json.dumps(card_result, indent=2, ensure_ascii=False), encoding="utf-8")
            print("\n[Card] Short-card workflow completed")
            print(f"选取卡片: {', '.join(card_result.get('selected_cards', []))}")
            machine = card_result.get('machine_summary', {}) or {}
            print(f"新诠释: {machine.get('name', 'unnamed interpretation')}")
            preview = card_result.get('writeup', '')
            if preview:
                print(f"写作预览: {preview[:200]}...")
            print(f"结果已保存: {card_output}")
            if args.card_only:
                generator.save_unified_analysis()
                print("\n[Card] Workflow finished")
                print(f"📁 结果保存在: {args.output_dir}")
                return

    if not args.card_only:
        # 4. 初始化系统（加载文献、理论、构建概念空间）
        await generator.initialize_unified_system()
        
        # 5. 演示理论生成
        if args.demo_mode:
            await demo_theory_generation(generator)
        else:
            await interactive_theory_generation(generator)

    # 6. 保存分析结果
    generator.save_unified_analysis()

    print("\n[Done] Unified generation finished")
    print(f"📁 结果保存在: {args.output_dir}")


async def demo_theory_generation(generator: UnifiedTheoryGenerator):
    """演示模式的理论生成"""
    print("\n🎭 演示模式：生成示例理论")
    print("-" * 40)
    
    # 获取可用的先验理论
    available_theories = list(generator.prior_theories.keys())
    print(f"📚 可用先验理论: {len(available_theories)} 个")
    for i, theory in enumerate(available_theories[:5], 1):
        print(f"  {i}. {theory}")
    
    # 选择理论对进行演示
    demo_pairs = [
        ("Copenhagen Interpretation", "Many-Worlds Interpretation"),
        ("de Broglie-Bohm Theory", "Copenhagen Interpretation"),
    ]
    
    # 如果有足够的理论，使用实际理论对
    if len(available_theories) >= 4:
        demo_pairs = [
            (available_theories[0], available_theories[1]),
            (available_theories[2], available_theories[3])
        ]
    
    print(f"\n🔬 将生成 {len(demo_pairs)} 个演示理论...")
    
    # 模拟重点概念（如果有文献概念的话）
    focus_concepts = None
    if generator.literature_concepts:
        focus_concepts = [c['name'] for c in generator.literature_concepts[:5]]
        print(f"🎯 重点关注概念: {focus_concepts}")
    
    # 批量生成理论
    results = await generator.batch_generate_from_theory_pairs(
        demo_pairs, 
        focus_concepts
    )
    
    # 显示结果摘要
    print(f"\n📊 生成结果摘要:")
    for i, result in enumerate(results, 1):
        if "error" in result:
            print(f"  {i}. ❌ 生成失败: {result.get('error', 'Unknown error')}")
        else:
            theory_name = result.get('name', 'Unknown Theory')
            metadata = result.get('generation_metadata', {})
            print(f"  {i}. ✅ {theory_name}")
            print(f"     源理论: {metadata.get('source_theories', [])}")
            print(f"     生成方法: {metadata.get('generation_method', 'unknown')}")


async def interactive_theory_generation(generator: UnifiedTheoryGenerator):
    """交互模式的理论生成"""
    print("\n💬 交互模式：自定义理论生成")
    print("-" * 40)
    
    available_theories = list(generator.prior_theories.keys())
    
    print(f"📚 可用先验理论 ({len(available_theories)} 个):")
    for i, theory in enumerate(available_theories, 1):
        print(f"  {i}. {theory}")
    
    # 获取用户输入
    try:
        print(f"\n请选择两个理论进行组合:")
        
        # 简化版：自动选择前两个理论
        if len(available_theories) >= 2:
            theory1 = available_theories[0]
            theory2 = available_theories[1]
            
            print(f"自动选择: {theory1} + {theory2}")
            
            # 生成理论
            result = await generator.generate_theory_from_literature_and_priors(
                theory1, theory2
            )
            
            # 显示结果
            if "error" in result:
                print(f"❌ 生成失败: {result['error']}")
            else:
                print(f"\n🎉 成功生成新理论!")
                print(f"理论名称: {result.get('name', 'Unknown')}")
                print(f"核心原理: {result.get('core_principles', 'N/A')[:200]}...")
                
                metadata = result.get('generation_metadata', {})
                print(f"\n📊 生成统计:")
                print(f"  文献概念数: {metadata.get('literature_concepts_used', 0)}")
                print(f"  概念空间大小: {metadata.get('unified_concept_space_size', 0)}")
                print(f"  生成方法: {metadata.get('generation_method', 'unknown')}")
        else:
            print("❌ 可用理论不足，无法进行组合")
            
    except KeyboardInterrupt:
        print("\n👋 用户取消操作")
    except Exception as e:
        print(f"❌ 交互过程出错: {e}")


async def test_unified_system():
    """测试统一系统的各个组件"""
    print("\n🧪 测试统一系统组件")
    print("-" * 40)
    
    # 初始化LLM
    llm = LLMInterface(
        model_source="google",
        model_name="gemini-2.5-pro"
    )
    
    # 创建测试配置
    config = UnifiedGenerationConfig(
        literature_dirs=["demo/papers"],  # 使用demo目录
        prior_theories_dir="data/theories_v2.1",
        output_dir="test_unified_output"
    )
    
    # 初始化生成器
    generator = UnifiedTheoryGenerator(llm, config)
    
    print("✅ 统一生成器初始化成功")
    
    # 测试系统初始化
    try:
        await generator.initialize_unified_system()
        print("✅ 系统初始化成功")
        
        # 显示统计信息
        print(f"\n📊 系统统计:")
        print(f"  文献概念: {len(generator.literature_concepts)}")
        print(f"  先验理论: {len(generator.prior_theories)}")
        print(f"  概念嵌入: {len(generator.concept_embeddings)}")
        print(f"  理论嵌入: {len(generator.theory_embeddings)}")
        print(f"  统一空间: {len(generator.unified_concept_space)}")
        
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc() 
