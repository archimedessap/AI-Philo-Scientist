#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础的评估反馈循环测试
"""

import json
import asyncio
from theory_generation.llm_interface import LLMInterface
from theory_validation.agent_validation.theory_evaluator import TheoryEvaluator

async def test_basic_feedback():
    """测试基础的评估反馈循环"""
    
    # 1. 初始化
    llm = LLMInterface(model_source="google", model_name="gemini-2.5-flash")
    evaluator = TheoryEvaluator(llm)
    
    # 2. 测试理论
    test_theory = {
        "name": "Coherent Reality Interpretation (CRI)",
        "content": """
        核心假设：
        1. 波函数演化是幺正的，没有坍缩
        2. 现实通过相干性固化过程涌现
        3. 存在一个相干场与量子态相互作用
        
        数学形式：
        |Ψ(t)⟩ 演化遵循薛定谔方程
        ρ_eff(t) = Tr_env[|Ψ⟩⟨Ψ|]
        
        预测：
        - 宏观叠加态衰减速度超过标准退相干
        - 存在相干现实形成的阈值
        """,
        "philosophy": "现实通过量子叠加的相干性固化动态涌现",
        "core_assumptions": [
            "波函数幺正演化",
            "相干性固化过程", 
            "相干场存在"
        ],
        "mathematical_formalism": "ρ_eff演化方程：dρ/dt = -i[H,ρ] + L_decoherence + L_consolidation",
        "empirical_predictions": [
            "宏观叠加态加速衰减",
            "相干现实形成阈值"
        ]
    }
    
    # 3. 执行评估
    print("执行三角色评估...")
    evaluation_results = {
        'theory_name': test_theory['name'],
        'evaluations': {}
    }
    
    # 直接调用角色评估
    for role_id, role_info in evaluator.evaluation_roles.items():
        print(f"  - {role_info['name']}评估中...")
        eval_result = await evaluator._evaluate_as_role(test_theory, role_id, role_info)
        evaluation_results['evaluations'][role_id] = eval_result
        print(f"    得分: {eval_result.get('score', '未知')}/10")
    
    # 4. 提取反馈
    print("\n提取评估反馈...")
    feedback = {
        'strengths': [],
        'weaknesses': [],
        'suggestions': []
    }
    
    if 'evaluations' in evaluation_results:
        for role, eval_data in evaluation_results['evaluations'].items():
            if 'strengths' in eval_data:
                feedback['strengths'].extend([f"[{role}] {s}" for s in eval_data['strengths']])
            if 'weaknesses' in eval_data:
                feedback['weaknesses'].extend([f"[{role}] {w}" for w in eval_data['weaknesses']])
            if 'improvement_suggestions' in eval_data:
                feedback['suggestions'].append(f"[{role}] {eval_data['improvement_suggestions']}")
    
    print(f"\n反馈统计：")
    print(f"- 优点: {len(feedback['strengths'])}条")
    print(f"- 缺点: {len(feedback['weaknesses'])}条") 
    print(f"- 建议: {len(feedback['suggestions'])}条")
    
    # 5. 基于反馈生成改进
    if feedback['weaknesses'] or feedback['suggestions']:
        print("\n基于反馈生成改进版本...")
        
        improvement_prompt = f"""基于以下评估反馈，改进量子力学诠释理论。

原始理论：{test_theory['name']}

主要问题：
{chr(10).join(feedback['weaknesses'][:5])}

改进建议：
{chr(10).join(feedback['suggestions'])}

请生成改进后的理论，输出JSON格式：
{{
  "name": "{test_theory['name']} v2",
  "description": "改进的描述",
  "core_assumptions": ["假设1", "假设2"],
  "mathematical_formalism": "完整数学描述",
  "empirical_predictions": ["预测1", "预测2"],
  "improvements_made": "主要改进"
}}"""

        messages = [{"role": "user", "content": improvement_prompt}]
        response = await llm.query_async(messages)
        
        print("\nLLM响应预览：")
        print(response[:500] + "..." if len(response) > 500 else response)
        
        # 解析响应
        if response.startswith("```json"):
            response = response[7:-3].strip()
        
        try:
            improved_theory = json.loads(response)
            print("\n✅ 成功生成改进理论！")
            print(f"改进后理论名称: {improved_theory.get('name')}")
        except Exception as e:
            print(f"\n❌ 解析失败: {e}")
    
    # 关闭客户端
    await llm.aclose()

if __name__ == "__main__":
    asyncio.run(test_basic_feedback())