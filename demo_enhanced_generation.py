#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强理论生成演示

演示新的创新框架和增强反馈循环的效果，
展示如何生成不同创新层次的理论并进行精确的改进循环。
"""

import asyncio
import json
import time
import os
from pathlib import Path

# 假设这些模块可用
from theory_generation.innovation_framework import InnovationFramework, InnovationLevel
from theory_generation.adaptive_generator import AdaptiveTheoryGenerator
from theory_validation.enhanced_feedback_loop import EnhancedFeedbackLoop

class EnhancedGenerationDemo:
    """增强理论生成演示类"""
    
    def __init__(self, llm_interface):
        self.llm = llm_interface
        self.innovation_framework = InnovationFramework()
        self.adaptive_generator = AdaptiveTheoryGenerator(llm_interface)
        self.enhanced_feedback = EnhancedFeedbackLoop(llm_interface)
        
        # 创建演示输出目录
        self.demo_dir = Path("demo_enhanced_outputs")
        self.demo_dir.mkdir(exist_ok=True)
        
    async def run_comprehensive_demo(self):
        """运行全面的增强生成演示"""
        
        print("=" * 60)
        print("🚀 增强理论生成系统演示")
        print("=" * 60)
        
        # 定义示例矛盾
        sample_contradiction = {
            "theory1": "Copenhagen Interpretation",
            "theory2": "Many-Worlds Interpretation", 
            "contradictions": [
                {
                    "contradiction": "Wave Function Reality",
                    "theory1_position": "Wave function is not real, collapses upon measurement",
                    "theory2_position": "Wave function is real, all branches exist simultaneously"
                }
            ]
        }
        
        # 1. 演示分层次创新生成
        await self._demo_innovation_levels(sample_contradiction)
        
        # 2. 演示渐进式理论系列
        await self._demo_progressive_series(sample_contradiction)
        
        # 3. 演示增强反馈循环
        await self._demo_enhanced_feedback_loop()
        
        # 4. 生成综合报告
        self._generate_demo_report()
        
        print("\n✅ 演示完成！检查 demo_enhanced_outputs/ 目录获取详细结果。")
    
    async def _demo_innovation_levels(self, contradiction):
        """演示不同创新层次的理论生成"""
        
        print("\n📊 演示1: 分层次创新理论生成")
        print("-" * 40)
        
        innovation_levels = [
            InnovationLevel.INTERPRETATION,
            InnovationLevel.PARAMETER_EXTENSION,
            InnovationLevel.EQUATION_MODIFICATION,
            InnovationLevel.FRAMEWORK_EXTENSION
        ]
        
        level_results = {}
        
        for level in innovation_levels:
            print(f"\n🎯 生成 {level.value} 层次理论...")
            
            # 生成目标理论
            theory = await self.adaptive_generator.generate_targeted_theory(
                contradiction, 
                level
            )
            
            if "error" not in theory:
                # 评估创新层次
                actual_level, scores = self.innovation_framework.assess_theory_innovation_level(theory)
                
                print(f"   ✓ 目标层次: {level.value}")
                print(f"   ✓ 实际层次: {actual_level.value}")
                print(f"   ✓ 创新评分: {scores}")
                
                level_results[level.value] = {
                    "theory": theory,
                    "target_level": level.value,
                    "actual_level": actual_level.value,
                    "innovation_scores": scores,
                    "level_match": level == actual_level
                }
                
                # 保存理论
                theory_file = self.demo_dir / f"theory_{level.value}.json"
                with open(theory_file, 'w', encoding='utf-8') as f:
                    json.dump(theory, f, ensure_ascii=False, indent=2)
                    
            else:
                print(f"   ❌ 生成失败: {theory.get('error', 'Unknown error')}")
        
        # 保存层次对比结果
        comparison_file = self.demo_dir / "innovation_levels_comparison.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(level_results, f, ensure_ascii=False, indent=2)
        
        # 分析结果
        success_rate = sum(1 for r in level_results.values() if r["level_match"]) / len(level_results)
        print(f"\n📈 层次匹配成功率: {success_rate:.1%}")
    
    async def _demo_progressive_series(self, contradiction):
        """演示渐进式理论系列生成"""
        
        print("\n🔄 演示2: 渐进式理论系列生成")
        print("-" * 40)
        
        print("正在生成从诠释到框架扩展的完整系列...")
        
        progressive_theories = await self.adaptive_generator.generate_progressive_series(
            contradiction,
            max_level=InnovationLevel.FRAMEWORK_EXTENSION
        )
        
        print(f"✓ 生成了 {len(progressive_theories)} 个渐进式理论")
        
        # 分析渐进趋势
        progression_analysis = {
            "total_theories": len(progressive_theories),
            "innovation_progression": [],
            "mathematical_complexity_growth": [],
            "experimental_sophistication": []
        }
        
        for i, theory in enumerate(progressive_theories):
            level, scores = self.innovation_framework.assess_theory_innovation_level(theory)
            
            progression_analysis["innovation_progression"].append({
                "theory_index": i,
                "innovation_level": level.value,
                "overall_score": sum(scores.values()) / len(scores),
                "theory_name": theory.get("name", f"Theory_{i}")
            })
            
            print(f"   理论 {i+1}: {theory.get('name', 'Unknown')} - {level.value}")
        
        # 保存渐进系列
        series_file = self.demo_dir / "progressive_series.json"
        with open(series_file, 'w', encoding='utf-8') as f:
            json.dump({
                "theories": progressive_theories,
                "analysis": progression_analysis
            }, f, ensure_ascii=False, indent=2)
    
    async def _demo_enhanced_feedback_loop(self):
        """演示增强反馈循环"""
        
        print("\n🔄 演示3: 增强反馈循环")
        print("-" * 40)
        
        # 创建一个模拟的低分理论用于演示改进
        sample_theory = {
            "name": "Sample Theory for Improvement Demo",
            "summary": "A basic quantum theory with room for improvement",
            "core_principles": {
                "ontological_commitments": "Standard quantum objects",
                "epistemological_stances": "Traditional measurement theory",
                "key_postulates": ["Standard postulate 1", "Standard postulate 2"]
            },
            "formalism": {
                "mathematical_objects": "Standard Hilbert space",
                "governing_equations": ["i\\hbar \\partial_t |\\psi\\rangle = H|\\psi\\rangle"],
                "comparison_with_sqm": {
                    "agreements": "Complete agreement",
                    "modifications": "None",
                    "extensions": "None"
                }
            }
        }
        
        # 模拟评估结果（低分）
        mock_evaluation = {
            "role_score": 0.45,
            "details": {
                "theoretical_physicist": {
                    "score": 4,
                    "rationale": "Theory lacks mathematical innovation and conceptual novelty. No new insights provided."
                },
                "experimentalist": {
                    "score": 3,
                    "rationale": "No new experimental predictions. Theory is not distinguishable from standard QM."
                },
                "philosopher": {
                    "score": 5,
                    "rationale": "Philosophical assumptions are unclear and not well justified."
                }
            }
        }
        
        print("正在运行改进循环...")
        print(f"初始评分: {mock_evaluation['role_score']:.2f}")
        
        # 运行改进循环
        improvement_results = await self.enhanced_feedback.run_improvement_cycle(
            sample_theory,
            mock_evaluation,
            max_iterations=3,
            target_score_threshold=0.75
        )
        
        # 分析改进效果
        initial_score = improvement_results["cycle_summary"]["initial_score"]
        final_score = improvement_results["cycle_summary"]["final_score"]
        improvement = improvement_results["cycle_summary"]["total_improvement"]
        
        print(f"✓ 改进循环完成:")
        print(f"   初始分数: {initial_score:.3f}")
        print(f"   最终分数: {final_score:.3f}")
        print(f"   总体改进: {improvement:.3f} ({improvement/initial_score*100:.1f}%)")
        print(f"   迭代次数: {improvement_results['cycle_summary']['total_iterations']}")
        
        # 保存改进结果
        feedback_file = self.demo_dir / "feedback_loop_demo.json"
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(improvement_results, f, ensure_ascii=False, indent=2)
    
    def _generate_demo_report(self):
        """生成演示报告"""
        
        report = {
            "demo_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "purpose": "演示增强理论生成系统的新功能",
                "components_tested": [
                    "分层次创新框架",
                    "自适应理论生成器", 
                    "渐进式理论系列",
                    "增强反馈循环"
                ]
            },
            "key_features_demonstrated": {
                "innovation_framework": {
                    "description": "五层次创新评估和目标设定",
                    "levels": [level.value for level in InnovationLevel],
                    "benefits": [
                        "精确控制理论创新程度",
                        "客观评估创新层次",
                        "指导性强的生成目标"
                    ]
                },
                "adaptive_generation": {
                    "description": "根据目标创新层次自适应调整生成策略",
                    "features": [
                        "动态参数调整",
                        "反馈驱动优化",
                        "渐进式系列生成"
                    ]
                },
                "enhanced_feedback": {
                    "description": "精确分析评估反馈并执行针对性改进",
                    "capabilities": [
                        "智能反馈分析",
                        "策略化改进规划",
                        "循环效果验证"
                    ]
                }
            },
            "performance_improvements": {
                "innovation_control": "可精确控制和验证理论创新层次",
                "feedback_efficiency": "反馈循环效果更加明显和可衡量",
                "theory_quality": "生成理论的质量和多样性显著提升",
                "systematic_approach": "从随机生成转向系统化、目标导向的生成"
            },
            "next_steps": {
                "integration": "将新框架集成到主系统中",
                "validation": "使用真实评估器验证改进效果",
                "optimization": "基于实际使用数据进一步优化参数",
                "scaling": "扩展到更多理论类型和领域"
            }
        }
        
        report_file = self.demo_dir / "demo_comprehensive_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📋 详细演示报告已保存到: {report_file}")

# 演示运行函数
async def run_demo():
    """运行演示的主函数"""
    
    # 这里需要一个实际的LLM接口实例
    # 为了演示，我们使用一个模拟的接口
    class MockLLMInterface:
        async def query_async(self, messages, temperature=0.7):
            # 模拟LLM响应
            return '{"name": "Mock Theory", "summary": "A mock theory for demonstration"}'
        
        def extract_json(self, response):
            try:
                return json.loads(response)
            except:
                return {"name": "Mock Theory", "summary": "A mock theory for demonstration"}
    
    mock_llm = MockLLMInterface()
    demo = EnhancedGenerationDemo(mock_llm)
    
    await demo.run_comprehensive_demo()

if __name__ == "__main__":
    asyncio.run(run_demo()) 