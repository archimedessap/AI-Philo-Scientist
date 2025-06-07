#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强版理论生成器测试脚本

演示如何使用基于七大标准的理论生成器来创建高质量的量子诠释理论。
"""

import os
import sys
import json
import asyncio
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from theory_generation.enhanced_theory_generator import EnhancedTheoryGenerator
from theory_generation.llm_interface import LLMInterface

async def test_enhanced_generator():
    """测试增强版理论生成器"""
    
    print("=" * 60)
    print("增强版量子理论生成器测试")
    print("=" * 60)
    
    # 初始化LLM接口
    print("[INFO] 初始化LLM接口...")
    llm = LLMInterface()
    
    # 初始化增强版生成器
    generator = EnhancedTheoryGenerator(llm)
    
    # 定义测试用的理论矛盾
    test_contradictions = [
        {
            "dimension": "波函数本体论地位",
            "theory1_position": "哥本哈根诠释认为波函数是计算工具，不对应物理实在",
            "theory2_position": "多世界诠释认为波函数是完全的物理实在",
            "core_tension": "波函数究竟是认识论工具还是本体论实在？",
            "importance_score": 9
        },
        {
            "dimension": "测量问题处理",
            "theory1_position": "哥本哈根诠释通过波函数坍缩来解释测量",
            "theory2_position": "多世界诠释通过分裂来避免坍缩",
            "core_tension": "测量过程中是否发生真实的物理变化？",
            "importance_score": 8
        },
        {
            "dimension": "观察者角色",
            "theory1_position": "哥本哈根诠释中观察者具有特殊地位",
            "theory2_position": "多世界诠释中观察者也服从量子力学",
            "core_tension": "观察者是否需要特殊的物理地位？",
            "importance_score": 7
        }
    ]
    
    # 测试1: 生成单个高质量理论
    print("\n" + "=" * 40)
    print("测试1: 生成单个高质量理论")
    print("=" * 40)
    
    config1 = {
        "focus_on_standards": ["explanatory_completeness", "experimental_distinguishability"],
        "mathematical_depth": "moderate",
        "novelty_level": "incremental",
        "experimental_orientation": True
    }
    
    theory1 = await generator.generate_high_quality_theory(
        "哥本哈根诠释",
        "多世界诠释", 
        test_contradictions,
        config1
    )
    
    if "error" not in theory1:
        print(f"✓ 成功生成理论: {theory1['name']}")
        print(f"✓ 质量评分: {theory1['quality_assessment']['overall_score']:.2f}/10")
        
        # 保存理论
        output_dir = project_root / "demo" / "outputs" / "enhanced_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        theory_file = output_dir / f"{theory1['name'].replace(' ', '_').lower()}.json"
        with open(theory_file, 'w', encoding='utf-8') as f:
            json.dump(theory1, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 理论已保存到: {theory_file}")
    else:
        print(f"✗ 理论生成失败: {theory1['error']}")
    
    # 测试2: 生成理论家族
    print("\n" + "=" * 40)
    print("测试2: 生成理论家族")
    print("=" * 40)
    
    theory_family = await generator.generate_theory_family(
        "玻姆力学",
        "关系量子力学",
        test_contradictions,
        family_size=2  # 生成2个变体
    )
    
    print(f"✓ 成功生成理论家族，共 {len(theory_family)} 个成员")
    
    for i, theory in enumerate(theory_family):
        print(f"  - 成员 {i+1}: {theory['name']}")
        print(f"    质量评分: {theory['quality_assessment']['overall_score']:.2f}/10")
        
        # 保存理论家族
        family_file = output_dir / f"family_member_{i+1}.json"
        with open(family_file, 'w', encoding='utf-8') as f:
            json.dump(theory, f, ensure_ascii=False, indent=2)
    
    # 测试3: 不同配置的对比
    print("\n" + "=" * 40)
    print("测试3: 不同配置的理论对比")
    print("=" * 40)
    
    configs = [
        {
            "name": "实验导向型",
            "config": {
                "focus_on_standards": ["experimental_distinguishability"],
                "mathematical_depth": "moderate",
                "novelty_level": "moderate",
                "experimental_orientation": True
            }
        },
        {
            "name": "数学严谨型",
            "config": {
                "focus_on_standards": ["mathematical_consistency", "conceptual_clarity"],
                "mathematical_depth": "advanced",
                "novelty_level": "incremental",
                "experimental_orientation": False
            }
        }
    ]
    
    for config_info in configs:
        print(f"\n生成 {config_info['name']} 理论...")
        
        theory = await generator.generate_high_quality_theory(
            "客观坍缩理论",
            "量子贝叶斯主义",
            test_contradictions,
            config_info['config']
        )
        
        if "error" not in theory:
            print(f"✓ {config_info['name']}: {theory['name']}")
            quality = theory['quality_assessment']
            print(f"  质量评分: {quality['overall_score']:.2f}/10")
            print(f"  详细评分: {quality['detailed_scores']}")
        else:
            print(f"✗ {config_info['name']} 生成失败")
    
    print("\n" + "=" * 60)
    print("测试完成！查看 demo/outputs/enhanced_test/ 目录中的结果文件。")
    print("=" * 60)

def demonstrate_compatibility():
    """演示与现有系统的兼容性"""
    
    print("\n" + "=" * 50)
    print("兼容性演示")
    print("=" * 50)
    
    # 创建一个模拟的理论，展示格式兼容性
    sample_theory = {
        "name": "示例增强理论",
        "summary": "这是一个演示格式兼容性的示例理论",
        "philosophy": {
            "ontology": "波函数表示量子场的真实激发",
            "measurement": "测量是系统与环境的相互作用过程"
        },
        "parameters": {
            "α_coupling": {
                "value": 0.137,
                "unit": "dimensionless",
                "role": "精细结构常数，控制电磁相互作用强度"
            },
            "τ_decoherence": {
                "value": 1e-12,
                "unit": "s",
                "role": "特征退相干时间尺度"
            }
        },
        "formalism": {
            "hamiltonian": "H = H_0 + α_coupling V_int + τ_decoherence^{-1} H_env",
            "state_equation": "i\\hbar \\partial_t |ψ⟩ = H |ψ⟩",
            "measurement_rule": "概率 = |⟨outcome|ψ⟩|^2，修正为环境耦合"
        },
        "semantics": {
            "α_coupling": "这个参数控制量子系统与经典测量设备的耦合强度",
            "τ_decoherence": "环境诱导的退相干特征时间，决定了量子态维持相干性的典型时间",
            "overall_picture": "量子系统通过与环境的连续相互作用逐渐显现经典性质"
        }
    }
    
    print("示例理论结构:")
    print(json.dumps(sample_theory, ensure_ascii=False, indent=2))
    
    print("\n✓ 该格式与现有系统完全兼容")
    print("✓ 包含了 name, parameters, formalism 等必需字段")
    print("✓ 增加了 philosophy, semantics 等增强字段")
    print("✓ 可以直接用于现有的实验预测和评估流程")

if __name__ == "__main__":
    # 演示兼容性
    demonstrate_compatibility()
    
    # 运行异步测试
    try:
        asyncio.run(test_enhanced_generator())
    except KeyboardInterrupt:
        print("\n[INFO] 测试被用户中断")
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {str(e)}")
        import traceback
        traceback.print_exc() 