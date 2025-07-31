#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_feedback_loop.py - 测试评估反馈循环
======================================

演示如何使用简单的反馈循环来改进理论
"""

import asyncio
import json
from pathlib import Path

async def test_simple_feedback_loop():
    """测试简单的反馈循环"""
    print("🔄 测试简单反馈循环")
    print("="*60)
    
    # 使用之前生成的理论作为测试
    theory_path = "output_clean_evolution/run_20250725_104606/generation_0/synthesis/eval_ready_theories"
    
    # 运行反馈循环
    from simple_feedback_loop import SimpleFeedbackLoop
    
    feedback_loop = SimpleFeedbackLoop(
        model_source="google",
        model_name="gemini-2.5-flash"  # 使用更快的模型进行测试
    )
    
    await feedback_loop.run_feedback_loop(
        theory_path=theory_path,
        output_dir="output_feedback_test",
        iterations=1  # 只运行一次迭代进行测试
    )
    
    print("\n✅ 简单反馈循环测试完成！")


async def test_feedback_aware_generation():
    """测试基于反馈的理论生成"""
    print("\n🎯 测试基于反馈的理论生成")
    print("="*60)
    
    # 假设我们已经有了一些评估结果
    # 这里我们模拟一些反馈数据
    mock_feedback = {
        "Coherent Reality Interpretation": {
            "evaluations": {
                "physicist": {
                    "weaknesses": [
                        "缺乏C-Field的具体数学描述",
                        "Reality Weight的量化方法不明确",
                        "未提供与标准量子力学的定量差异预测"
                    ],
                    "improvement_suggestions": "需要建立C-Field的拉格朗日量，定义Reality Weight的演化方程"
                },
                "mathematician": {
                    "weaknesses": [
                        "数学形式化不完整",
                        "缺少严格的数学证明"
                    ]
                }
            }
        }
    }
    
    # 使用feedback_aware生成器
    from theory_generation.methods.feedback_aware_generator import FeedbackAwareGenerator
    
    generator = FeedbackAwareGenerator(
        feedback_data=mock_feedback,
        theories_dir="data/theories_test",
        output_dir="output_feedback_aware_test",
        model_source="google",
        model_name="gemini-2.5-flash",
        force_load_literature=True,
        num_theories_to_generate=1
    )
    
    # 生成基于反馈的新理论
    result = await generator._async_generate()
    
    if result['success']:
        print(f"\n✅ 成功生成 {len(result['theories'])} 个基于反馈的理论")
        
        # 保存结果
        output_path = Path("output_feedback_aware_test")
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / "feedback_aware_result.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        print(f"\n❌ 生成失败: {result.get('error_message')}")


async def main():
    """运行所有测试"""
    # 测试1：简单反馈循环
    # await test_simple_feedback_loop()
    
    # 测试2：反馈感知生成
    await test_feedback_aware_generation()


if __name__ == "__main__":
    print("🚀 开始测试评估反馈系统")
    asyncio.run(main())