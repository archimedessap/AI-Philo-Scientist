#!/usr/bin/env python3
"""
debug_multi_level.py - 调试多层次生成器
"""

import asyncio
from theory_generation.llm_interface import LLMInterface
from theory_generation.innovation_framework import InnovationFramework, InnovationLevel
from theory_generation.multi_level_innovation_generator import (
    MultiLevelInnovationGenerator, 
    MultiLevelInnovationConfig
)

async def debug_multi_level():
    # 初始化组件
    llm = LLMInterface(model_source="google", model_name="gemini-2.5-flash")
    framework = InnovationFramework()
    generator = MultiLevelInnovationGenerator(llm, framework)
    
    # 创建简单的矛盾
    contradiction = {
        "theory1": "Test Theory 1",
        "theory2": "Test Theory 2", 
        "contradictions": [
            {
                "contradiction": "测试矛盾",
                "theory1_position": "位置1",
                "theory2_position": "位置2"
            }
        ]
    }
    
    # 创建配置
    config = generator.create_multi_level_config(
        target_levels=[InnovationLevel.INTERPRETATION, InnovationLevel.PARAMETER_EXTENSION],
        synthesis_mode="fusion",
        innovation_intensity=0.6
    )
    
    print("开始测试多层次生成...")
    
    try:
        # 1. 测试目标生成
        print("1. 测试创新目标生成...")
        level_targets = {}
        for level in config.target_levels:
            targets = framework.generate_innovation_targets(level)
            level_targets[level] = targets
            print(f"  {level.value}: OK")
        
        # 2. 测试综合策略
        print("2. 测试综合策略...")
        synthesis_strategy = await generator._fusion_synthesis(level_targets, config, contradiction)
        print("  综合策略: OK")
        
        # 3. 测试提示构建
        print("3. 测试提示构建...")
        prompt = generator._build_multi_level_prompt(contradiction, synthesis_strategy, config)
        print(f"  提示长度: {len(prompt)} 字符")
        
        # 4. 测试简单理论评估
        print("4. 测试理论评估...")
        simple_theory = {
            "name": "测试理论",
            "summary": "这是一个测试理论",
            "mathematical_relation_to_sqm": "interpretation",
            "formalism": {
                "mathematical_objects": "标准希尔伯特空间",
                "governing_equations": ["薛定谔方程"]
            },
            "core_principles": {
                "ontological_commitments": "量子态作为基本实体",
                "key_postulates": ["量子叠加", "概率诠释"]
            }
        }
        
        assessment = generator._assess_multi_level_innovation(simple_theory, config)
        print("  理论评估: OK")
        print(f"  总体分数: {assessment['overall_multi_level_score']:.3f}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_multi_level()) 