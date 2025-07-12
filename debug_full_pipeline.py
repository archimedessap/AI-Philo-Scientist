#!/usr/bin/env python3
"""
debug_full_pipeline.py - 测试完整的多层次演进管道
"""

import asyncio
import json
from pathlib import Path
from theory_generation.llm_interface import LLMInterface
from theory_generation.innovation_framework import InnovationFramework, InnovationLevel
from theory_generation.multi_level_innovation_generator import (
    MultiLevelInnovationGenerator, 
    MultiLevelInnovationConfig
)

async def debug_full_pipeline():
    # 初始化组件
    llm = LLMInterface(model_source="google", model_name="gemini-2.5-flash")
    framework = InnovationFramework()
    generator = MultiLevelInnovationGenerator(llm, framework)
    
    print("开始完整管道测试...")
    
    # 1. 加载真实的先验理论
    print("1. 加载先验理论...")
    theories_dir = Path("data/theories_v2.1")
    theories = {}
    
    for theory_file in theories_dir.glob("*.json"):
        try:
            with open(theory_file, 'r', encoding='utf-8') as f:
                theory_data = json.load(f)
            theory_name = theory_data.get('name', theory_file.stem)
            theories[theory_name] = theory_data
            print(f"  加载: {theory_name}")
        except Exception as e:
            print(f"  错误加载 {theory_file}: {e}")
    
    print(f"总共加载了 {len(theories)} 个理论")
    
    # 2. 创建真实的矛盾分析
    print("\n2. 创建矛盾分析...")
    theory_names = list(theories.keys())[:2]  # 取前两个理论
    
    if len(theory_names) >= 2:
        t1_name, t2_name = theory_names[0], theory_names[1]
        t1_data, t2_data = theories[t1_name], theories[t2_name]
        
        print(f"  分析矛盾: {t1_name} vs {t2_name}")
        
        # 检查这两个理论的数据结构
        print(f"  {t1_name} 结构:")
        for key in t1_data.keys():
            print(f"    {key}: {type(t1_data[key])}")
        
        print(f"  {t2_name} 结构:")
        for key in t2_data.keys():
            print(f"    {key}: {type(t2_data[key])}")
        
        # 创建矛盾对象
        contradiction = {
            "theory1": t1_name,
            "theory2": t2_name,
            "contradictions": [
                {
                    "contradiction": "理论框架差异",
                    "theory1_position": f"{t1_name}的理论框架", 
                    "theory2_position": f"{t2_name}的理论框架"
                }
            ]
        }
        
        # 3. 测试理论评估（使用真实数据）
        print("\n3. 测试真实理论评估...")
        
        try:
            print(f"  评估 {t1_name}...")
            level1, scores1 = framework.assess_theory_innovation_level(t1_data)
            print(f"    层次: {level1.value}, 分数: {scores1}")
            
            print(f"  评估 {t2_name}...")
            level2, scores2 = framework.assess_theory_innovation_level(t2_data)
            print(f"    层次: {level2.value}, 分数: {scores2}")
            
        except Exception as e:
            print(f"  理论评估错误: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 4. 测试多层次生成（不实际调用LLM）
        print("\n4. 测试多层次生成配置...")
        
        config = generator.create_multi_level_config(
            target_levels=[InnovationLevel.INTERPRETATION, InnovationLevel.PARAMETER_EXTENSION],
            synthesis_mode="fusion",
            innovation_intensity=0.6
        )
        
        # 构建所有组件但不调用LLM
        level_targets = {}
        for level in config.target_levels:
            targets = framework.generate_innovation_targets(level)
            level_targets[level] = targets
        
        synthesis_strategy = await generator._fusion_synthesis(level_targets, config, contradiction)
        prompt = generator._build_multi_level_prompt(contradiction, synthesis_strategy, config)
        
        print(f"  生成配置成功")
        print(f"  提示长度: {len(prompt)} 字符")
        
        # 5. 测试模拟理论后处理
        print("\n5. 测试理论后处理...")
        
        # 创建一个模拟的生成理论
        mock_theory = {
            "name": "模拟多层次理论",
            "summary": "这是一个测试生成的多层次理论",
            "mathematical_relation_to_sqm": "interpretation",
            "formalism": {
                "mathematical_objects": "扩展希尔伯特空间",
                "governing_equations": ["修改的薛定谔方程", "新的演化方程"]
            },
            "core_principles": {
                "ontological_commitments": "信息理论基础",
                "key_postulates": ["信息守恒", "量子信息处理"]
            }
        }
        
        try:
            assessment = generator._assess_multi_level_innovation(mock_theory, config)
            processed_theory = generator._post_process_multi_level_theory(mock_theory, config, assessment)
            
            print(f"  后处理成功")
            print(f"  多层次评分: {assessment['overall_multi_level_score']:.3f}")
            print(f"  成功层次: {processed_theory['metadata']['successful_innovation_levels']}")
            
        except Exception as e:
            print(f"  后处理错误: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        print("  理论数量不足，无法创建矛盾分析")

if __name__ == "__main__":
    asyncio.run(debug_full_pipeline()) 