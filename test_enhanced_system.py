#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强理论生成系统测试

测试新实施的创新框架、自适应生成器和增强反馈循环的功能
"""

import asyncio
import json
import time
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append('.')

# 导入新的组件
try:
    from theory_generation.innovation_framework import InnovationFramework, InnovationLevel
    from theory_generation.adaptive_generator import AdaptiveTheoryGenerator
    from theory_validation.enhanced_feedback_loop import EnhancedFeedbackLoop
    from theory_generation.llm_interface import LLMInterface
    print("✅ 所有新组件导入成功")
except ImportError as e:
    print(f"❌ 组件导入失败: {e}")
    sys.exit(1)

class EnhancedSystemTester:
    """增强系统测试器"""
    
    def __init__(self):
        # 使用deepseek-reasoner作为测试模型
        self.llm = LLMInterface(model_source="deepseek", model_name="deepseek-reasoner")
        self.innovation_framework = InnovationFramework()
        self.adaptive_generator = AdaptiveTheoryGenerator(self.llm)
        self.enhanced_feedback = EnhancedFeedbackLoop(self.llm)
        
        # 创建测试输出目录
        self.test_dir = Path("test_enhanced_outputs")
        self.test_dir.mkdir(exist_ok=True)
        
    async def test_innovation_framework(self):
        """测试创新框架"""
        print("\n🔬 测试1: 创新层次框架")
        print("-" * 40)
        
        # 创建一个示例理论用于测试
        sample_theory = {
            "name": "Test Quantum Theory",
            "mathematical_relation_to_sqm": "Modification",
            "summary": "A novel quantum theory with emergent spacetime and holographic information processing",
            "core_principles": {
                "ontological_commitments": "Information is fundamental, spacetime emerges from quantum entanglement",
                "epistemological_stances": "Knowledge is relational and observer-dependent",
                "key_postulates": [
                    "Reality consists of discrete information units",
                    "Spacetime emerges from entanglement networks",
                    "Consciousness plays a fundamental role in measurement"
                ]
            },
            "formalism": {
                "mathematical_objects": "Extended Hilbert spaces with holographic boundary conditions",
                "governing_equations": [
                    "i\\hbar \\partial_t |\\psi\\rangle = H_{emergent}|\\psi\\rangle + \\gamma \\mathcal{I}[\\psi]",
                    "\\mathcal{I}[\\psi] = \\sum_i \\alpha_i \\langle\\psi|O_i|\\psi\\rangle \\log\\langle\\psi|O_i|\\psi\\rangle"
                ],
                "comparison_with_sqm": {
                    "agreements": "Maintains unitarity in emergent limit",
                    "modifications": "Adds information-theoretic terms and emergent spacetime",
                    "extensions": "Includes holographic boundary dynamics"
                }
            },
            "predictions_and_verifiability": {
                "reproduces_sqm_predictions": "In low-energy, classical spacetime limit",
                "deviations_from_sqm": [
                    {
                        "prediction_name": "Quantum Spacetime Fluctuations",
                        "description": "Detectable deviations in high-precision interferometry",
                        "mathematical_derivation": "From emergent metric fluctuations",
                        "experimental_setup": "Advanced gravitational wave detectors"
                    }
                ]
            }
        }
        
        # 测试创新层次评估
        level, scores = self.innovation_framework.assess_theory_innovation_level(sample_theory)
        
        print(f"📊 理论创新层次: {level.value}")
        print(f"📈 创新评分详情:")
        for dimension, score in scores.items():
            print(f"   - {dimension}: {score:.3f}")
        
        overall_score = sum(scores.values()) / len(scores)
        print(f"🎯 总体创新分数: {overall_score:.3f}")
        
        # 测试不同层次的目标生成
        print(f"\n🎯 测试创新目标生成:")
        for test_level in [InnovationLevel.INTERPRETATION, InnovationLevel.EQUATION_MODIFICATION, InnovationLevel.FRAMEWORK_EXTENSION]:
            targets = self.innovation_framework.generate_innovation_targets(test_level)
            print(f"   {test_level.value}: {len(targets['mathematical_targets'])} 个数学目标")
        
        return {"level": level.value, "scores": scores, "overall_score": overall_score}
    
    async def test_adaptive_generator(self):
        """测试自适应理论生成器"""
        print("\n🤖 测试2: 自适应理论生成器")
        print("-" * 40)
        
        # 创建测试矛盾
        test_contradiction = {
            "theory1": "Copenhagen Interpretation",
            "theory2": "Many-Worlds Interpretation",
            "contradictions": [
                {
                    "contradiction": "Wave Function Reality",
                    "theory1_position": "Wave function is epistemic, collapses upon measurement",
                    "theory2_position": "Wave function is ontic, all branches are real"
                }
            ]
        }
        
        print("🎯 测试目标创新层次生成...")
        
        # 测试不同创新层次的理论生成
        target_level = InnovationLevel.PARAMETER_EXTENSION
        print(f"目标层次: {target_level.value}")
        
        try:
            theory = await self.adaptive_generator.generate_targeted_theory(
                test_contradiction,
                target_level
            )
            
            if "error" not in theory:
                print(f"✅ 成功生成理论: {theory.get('name', 'Unknown')}")
                
                # 验证创新层次
                actual_level, innovation_scores = self.innovation_framework.assess_theory_innovation_level(theory)
                print(f"📊 实际创新层次: {actual_level.value}")
                print(f"🎯 层次匹配: {'✅' if actual_level == target_level else '❌'}")
                
                # 保存生成的理论
                theory_file = self.test_dir / f"generated_theory_{target_level.value}.json"
                with open(theory_file, 'w', encoding='utf-8') as f:
                    json.dump(theory, f, ensure_ascii=False, indent=2)
                
                return {
                    "success": True,
                    "theory_name": theory.get('name'),
                    "target_level": target_level.value,
                    "actual_level": actual_level.value,
                    "level_match": actual_level == target_level,
                    "innovation_scores": innovation_scores
                }
            else:
                print(f"❌ 理论生成失败: {theory.get('error')}")
                return {"success": False, "error": theory.get('error')}
                
        except Exception as e:
            print(f"❌ 生成过程出错: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def test_enhanced_feedback_loop(self):
        """测试增强反馈循环"""
        print("\n🔄 测试3: 增强反馈循环")
        print("-" * 40)
        
        # 创建一个模拟的低分理论
        low_score_theory = {
            "name": "Basic Interpretation Test Theory",
            "mathematical_relation_to_sqm": "Interpretation",
            "summary": "A simple interpretation with standard quantum mechanics",
            "core_principles": {
                "ontological_commitments": "Standard quantum objects",
                "epistemological_stances": "Traditional measurement approach",
                "key_postulates": ["Standard QM postulates"]
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
        
        # 模拟低分评估结果
        mock_evaluation = {
            "role_score": 0.35,
            "details": {
                "theoretical_physicist": {
                    "score": 3,
                    "rationale": "Theory lacks mathematical innovation. No new insights beyond standard QM."
                },
                "experimentalist": {
                    "score": 2,
                    "rationale": "No experimental predictions. Theory is indistinguishable from standard QM."
                },
                "philosopher": {
                    "score": 4,
                    "rationale": "Philosophical position is unclear and not well motivated."
                }
            }
        }
        
        print(f"📊 初始评分: {mock_evaluation['role_score']:.3f}")
        print("🔄 开始改进循环...")
        
        try:
            # 运行改进循环
            improvement_results = await self.enhanced_feedback.run_improvement_cycle(
                low_score_theory,
                mock_evaluation,
                max_iterations=2,  # 减少迭代次数以加快测试
                target_score_threshold=0.7
            )
            
            # 分析结果
            initial_score = improvement_results["cycle_summary"]["initial_score"]
            final_score = improvement_results["cycle_summary"]["final_score"]
            total_improvement = improvement_results["cycle_summary"]["total_improvement"]
            iterations = improvement_results["cycle_summary"]["total_iterations"]
            
            print(f"✅ 改进循环完成:")
            print(f"   📈 初始分数: {initial_score:.3f}")
            print(f"   📈 最终分数: {final_score:.3f}")
            print(f"   📈 总体改进: {total_improvement:.3f} ({total_improvement/initial_score*100:.1f}%)")
            print(f"   🔄 迭代次数: {iterations}")
            
            # 保存改进结果
            feedback_file = self.test_dir / "feedback_loop_test.json"
            with open(feedback_file, 'w', encoding='utf-8') as f:
                json.dump(improvement_results, f, ensure_ascii=False, indent=2)
            
            return {
                "success": True,
                "initial_score": initial_score,
                "final_score": final_score,
                "improvement": total_improvement,
                "improvement_percentage": total_improvement/initial_score*100,
                "iterations": iterations
            }
            
        except Exception as e:
            print(f"❌ 反馈循环测试失败: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def run_comprehensive_test(self):
        """运行完整的系统测试"""
        print("🚀 增强理论生成系统测试开始")
        print("=" * 60)
        
        test_results = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_results": {}
        }
        
        # 测试1: 创新框架
        framework_result = await self.test_innovation_framework()
        test_results["test_results"]["innovation_framework"] = framework_result
        
        # 测试2: 自适应生成器
        generator_result = await self.test_adaptive_generator()
        test_results["test_results"]["adaptive_generator"] = generator_result
        
        # 测试3: 增强反馈循环
        feedback_result = await self.test_enhanced_feedback_loop()
        test_results["test_results"]["enhanced_feedback"] = feedback_result
        
        # 生成测试报告
        await self.generate_test_report(test_results)
        
        print("\n" + "=" * 60)
        print("🎉 系统测试完成！")
        print(f"📁 测试结果保存在: {self.test_dir}")
        
        # 关闭LLM连接
        try:
            await self.llm.aclose()
        except:
            pass
    
    async def generate_test_report(self, test_results):
        """生成测试报告"""
        
        # 分析测试结果
        framework_success = test_results["test_results"]["innovation_framework"]["overall_score"] > 0.5
        generator_success = test_results["test_results"]["adaptive_generator"]["success"]
        feedback_success = test_results["test_results"]["enhanced_feedback"]["success"]
        
        overall_success = framework_success and generator_success and feedback_success
        
        report = {
            "test_summary": {
                "overall_success": overall_success,
                "components_tested": 3,
                "components_passed": sum([framework_success, generator_success, feedback_success]),
                "test_timestamp": test_results["test_timestamp"]
            },
            "component_results": {
                "innovation_framework": {
                    "status": "✅ PASSED" if framework_success else "❌ FAILED",
                    "innovation_score": test_results["test_results"]["innovation_framework"]["overall_score"],
                    "detected_level": test_results["test_results"]["innovation_framework"]["level"]
                },
                "adaptive_generator": {
                    "status": "✅ PASSED" if generator_success else "❌ FAILED",
                    "theory_generated": generator_success,
                    "level_match": test_results["test_results"]["adaptive_generator"].get("level_match", False) if generator_success else False
                },
                "enhanced_feedback": {
                    "status": "✅ PASSED" if feedback_success else "❌ FAILED",
                    "improvement_achieved": test_results["test_results"]["enhanced_feedback"].get("improvement", 0) if feedback_success else 0
                }
            },
            "recommendations": self._generate_recommendations(test_results),
            "detailed_results": test_results["test_results"]
        }
        
        # 保存报告
        report_file = self.test_dir / "test_comprehensive_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 打印摘要
        print(f"\n📋 测试摘要:")
        print(f"   整体状态: {'✅ 成功' if overall_success else '❌ 失败'}")
        print(f"   通过组件: {sum([framework_success, generator_success, feedback_success])}/3")
        
        if generator_success:
            theory_name = test_results["test_results"]["adaptive_generator"]["theory_name"]
            print(f"   生成理论: {theory_name}")
        
        if feedback_success:
            improvement = test_results["test_results"]["enhanced_feedback"]["improvement_percentage"]
            print(f"   改进效果: {improvement:.1f}%")
    
    def _generate_recommendations(self, test_results):
        """生成改进建议"""
        recommendations = []
        
        framework_result = test_results["test_results"]["innovation_framework"]
        generator_result = test_results["test_results"]["adaptive_generator"]
        feedback_result = test_results["test_results"]["enhanced_feedback"]
        
        if framework_result["overall_score"] < 0.7:
            recommendations.append("创新框架评分偏低，建议优化评估标准")
        
        if not generator_result["success"]:
            recommendations.append("自适应生成器失败，检查LLM连接和提示模板")
        elif not generator_result.get("level_match", False):
            recommendations.append("创新层次匹配度不高，需要调整生成参数")
        
        if not feedback_result["success"]:
            recommendations.append("反馈循环失败，检查改进策略生成逻辑")
        elif feedback_result.get("improvement", 0) < 0.1:
            recommendations.append("改进效果有限，需要增强反馈分析精度")
        
        if not recommendations:
            recommendations.append("所有组件运行正常，可以进行生产环境集成")
        
        return recommendations

async def main():
    """主测试函数"""
    tester = EnhancedSystemTester()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main()) 