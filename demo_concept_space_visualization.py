#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
概念空间可视化演示

演示如何使用概念空间可视化工具
"""

import asyncio
import argparse
import json
from pathlib import Path
import numpy as np

from theory_generation.llm_interface import LLMInterface
from theory_generation.unified_theory_generator import UnifiedTheoryGenerator, UnifiedGenerationConfig
from theory_generation.innovation_framework import InnovationLevel
from utils.concept_space_visualizer import ConceptSpaceVisualizer


async def demo_unified_theory_visualization():
    """演示统一理论生成器的概念空间可视化"""
    print("🎯 演示1: 统一理论生成器的概念空间可视化")
    print("=" * 60)
    
    # 初始化LLM
    llm = LLMInterface(
        model_source="google",
        model_name="gemini-2.5-flash"  # 使用快速模型进行演示
    )
    
    # 配置
    config = UnifiedGenerationConfig(
        literature_dirs=["demo/papers"],  # 使用demo目录
        prior_theories_dir="data/theories_v2.1",
        output_dir="demo_unified_output",
        enable_visualization=True  # 启用可视化
    )
    
    # 创建生成器
    generator = UnifiedTheoryGenerator(llm, config)
    
    # 初始化系统
    print("\n🚀 初始化统一系统...")
    await generator.initialize_unified_system()
    
    # 保存分析（会自动生成可视化）
    generator.save_unified_analysis()
    
    print("\n✅ 统一理论空间可视化完成！")
    print(f"📁 查看结果: {config.output_dir}/concept_space_visualization/")


async def demo_custom_concept_space():
    """演示自定义概念空间可视化"""
    print("\n🎯 演示2: 自定义概念空间可视化")
    print("=" * 60)
    
    # 创建一些示例概念嵌入
    np.random.seed(42)
    
    # 生成具有聚类结构的概念
    concepts = {}
    
    # 聚类1: 量子测量相关
    cluster1_center = np.random.randn(512)
    concepts["波函数坍缩"] = cluster1_center + np.random.randn(512) * 0.1
    concepts["测量问题"] = cluster1_center + np.random.randn(512) * 0.1
    concepts["观测者效应"] = cluster1_center + np.random.randn(512) * 0.1
    
    # 聚类2: 量子纠缠相关
    cluster2_center = np.random.randn(512) * 2
    concepts["量子纠缠"] = cluster2_center + np.random.randn(512) * 0.1
    concepts["贝尔不等式"] = cluster2_center + np.random.randn(512) * 0.1
    concepts["EPR佯谬"] = cluster2_center + np.random.randn(512) * 0.1
    
    # 聚类3: 诠释理论
    cluster3_center = np.random.randn(512) * 1.5
    concepts["哥本哈根诠释"] = cluster3_center + np.random.randn(512) * 0.1
    concepts["多世界诠释"] = cluster3_center + np.random.randn(512) * 0.1
    concepts["德布罗意-玻姆理论"] = cluster3_center + np.random.randn(512) * 0.1
    
    # 离群概念
    concepts["量子计算"] = np.random.randn(512) * 3
    concepts["量子隧道"] = np.random.randn(512) * 3
    
    # 创建标签
    labels = {
        "波函数坍缩": "测量相关",
        "测量问题": "测量相关",
        "观测者效应": "测量相关",
        "量子纠缠": "纠缠相关",
        "贝尔不等式": "纠缠相关",
        "EPR佯谬": "纠缠相关",
        "哥本哈根诠释": "诠释理论",
        "多世界诠释": "诠释理论",
        "德布罗意-玻姆理论": "诠释理论",
        "量子计算": "应用",
        "量子隧道": "量子现象"
    }
    
    # 创建可视化器
    visualizer = ConceptSpaceVisualizer("demo_custom_visualization")
    
    # 生成各种可视化
    print("\n📊 生成自定义概念空间可视化...")
    
    # 1. PCA投影
    visualizer.visualize_concept_space_2d(
        concepts, labels, method='pca',
        title="量子物理概念空间 - PCA投影",
        save_path="demo_custom_visualization/pca_projection.png"
    )
    
    # 2. t-SNE投影
    visualizer.visualize_concept_space_2d(
        concepts, labels, method='tsne',
        title="量子物理概念空间 - t-SNE投影",
        save_path="demo_custom_visualization/tsne_projection.png"
    )
    
    # 3. 聚类分析
    visualizer.visualize_concept_clusters(
        concepts, n_clusters=4,
        save_path="demo_custom_visualization/cluster_analysis.png"
    )
    
    # 4. 概念网络
    visualizer.visualize_concept_network(
        concepts, threshold=0.7,
        save_path="demo_custom_visualization/concept_network.png"
    )
    
    # 5. 密度分析
    visualizer.visualize_concept_density(
        concepts,
        save_path="demo_custom_visualization/density_analysis.png"
    )
    
    # 6. 生成综合报告
    visualizer.create_comprehensive_report(
        concepts, labels,
        output_prefix="quantum_concepts"
    )
    
    print("\n✅ 自定义概念空间可视化完成！")
    print("📁 查看结果: demo_custom_visualization/")


async def demo_concept_evolution():
    """演示概念空间演化可视化"""
    print("\n🎯 演示3: 概念空间演化可视化")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 创建时间线上的概念演化
    timeline = []
    
    # 时间点1: 早期量子力学
    early_concepts = {
        "波粒二象性": np.random.randn(512),
        "不确定性原理": np.random.randn(512),
        "量子态": np.random.randn(512)
    }
    timeline.append(("早期量子力学 (1920s)", early_concepts))
    
    # 时间点2: 中期发展
    mid_concepts = early_concepts.copy()
    mid_concepts.update({
        "量子纠缠": np.random.randn(512),
        "EPR佯谬": np.random.randn(512),
        "贝尔定理": np.random.randn(512)
    })
    timeline.append(("中期发展 (1960s)", mid_concepts))
    
    # 时间点3: 现代量子理论
    modern_concepts = mid_concepts.copy()
    modern_concepts.update({
        "量子计算": np.random.randn(512),
        "量子信息": np.random.randn(512),
        "量子密码学": np.random.randn(512),
        "量子隐形传态": np.random.randn(512)
    })
    timeline.append(("现代量子理论 (2000s)", modern_concepts))
    
    # 创建可视化器
    visualizer = ConceptSpaceVisualizer("demo_evolution_visualization")
    
    # 生成演化可视化
    print("\n📊 生成概念空间演化可视化...")
    visualizer.visualize_concept_evolution(
        timeline,
        save_path="demo_evolution_visualization/concept_evolution.png"
    )
    
    print("\n✅ 概念空间演化可视化完成！")
    print("📁 查看结果: demo_evolution_visualization/")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="概念空间可视化演示")
    parser.add_argument("--demo", type=int, choices=[1, 2, 3],
                       help="选择演示: 1=统一理论, 2=自定义空间, 3=概念演化")
    parser.add_argument("--all", action="store_true",
                       help="运行所有演示")
    
    args = parser.parse_args()
    
    print("🌟 概念空间可视化演示")
    print("=" * 60)
    
    if args.all or args.demo == 1:
        await demo_unified_theory_visualization()
    
    if args.all or args.demo == 2:
        await demo_custom_concept_space()
    
    if args.all or args.demo == 3:
        await demo_concept_evolution()
    
    if not args.all and not args.demo:
        print("请指定演示编号 (--demo 1/2/3) 或运行所有演示 (--all)")
        print("\n可用演示:")
        print("  1. 统一理论生成器的概念空间可视化")
        print("  2. 自定义概念空间可视化")
        print("  3. 概念空间演化可视化")


if __name__ == "__main__":
    asyncio.run(main())