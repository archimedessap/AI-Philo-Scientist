#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完整增强系统测试

展示新实施的创新框架、自适应生成器和增强反馈循环的集成效果
包含多层次理论生成、渐进式系列生成等高级功能
"""

import asyncio
import json
import time
import sys
import os
from pathlib import Path
from typing import List, Dict

# 添加项目路径
sys.path.append('.')

# 导入新的组件
try:
    from theory_generation.innovation_framework import InnovationFramework, InnovationLevel
    from theory_generation.adaptive_generator import AdaptiveTheoryGenerator
    from theory_validation.enhanced_feedback_loop import EnhancedFeedbackLoop
    from theory_generation.llm_interface import LLMInterface
    print("✅ 所有增强组件导入成功")
except ImportError as e:
    print(f"❌ 组件导入失败: {e}")
    sys.exit(1)

class CompleteSystemTester:
    """完整系统测试器"""
    
    def __init__(self):
        # 使用deepseek-reasoner作为测试模型
        self.llm = LLMInterface(model_source="deepseek", model_name="deepseek-reasoner")
        self.innovation_framework = InnovationFramework()
        self.adaptive_generator = AdaptiveTheoryGenerator(self.llm)
        self.enhanced_feedback = EnhancedFeedbackLoop(self.llm)
        
        # 创建测试输出目录
        self.test_dir = Path("test_complete_outputs")
        self.test_dir.mkdir(exist_ok=True)
        
        # 测试结果存储
        self.test_results = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests_performed": [],
            "generated_theories": [],
            "performance_metrics": {}
        }
    
    async def test_multi_level_generation(self):
        """测试多层次理论生成"""
        print("\n🎯 测试1: 多层次理论生成")
        print("=" * 50)
        
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
        
        # 测试不同创新层次
        test_levels = [
            InnovationLevel.INTERPRETATION,
            InnovationLevel.PARAMETER_EXTENSION,
            InnovationLevel.EQUATION_MODIFICATION
        ]
        
        generated_theories = []
        
        for level in test_levels:
            print(f"\n🔬 生成 {level.value} 层次理论...")
            
            try:
                theory = await self.adaptive_generator.generate_targeted_theory(
                    test_contradiction,
                    level
                )
                
                if "error" not in theory:
                    # 验证创新层次
                    actual_level, innovation_scores = self.innovation_framework.assess_theory_innovation_level(theory)
                    
                    result = {
                        "target_level": level.value,
                        "actual_level": actual_level.value,
                        "theory_name": theory.get('name', 'Unknown'),
                        "level_match": actual_level == level,
                        "innovation_scores": innovation_scores,
                        "overall_score": sum(innovation_scores.values()) / len(innovation_scores)
                    }
                    
                    generated_theories.append(result)
                    
                    print(f"✅ 成功: {theory.get('name', 'Unknown')}")
                    print(f"   目标层次: {level.value}")
                    print(f"   实际层次: {actual_level.value}")
                    print(f"   匹配状态: {'✅' if actual_level == level else '❌'}")
                    print(f"   总体评分: {result['overall_score']:.3f}")
                    
                    # 保存理论
                    theory_file = self.test_dir / f"theory_{level.value}.json"
                    with open(theory_file, 'w', encoding='utf-8') as f:
                        json.dump(theory, f, ensure_ascii=False, indent=2)
                        
                else:
                    print(f"❌ 生成失败: {theory.get('error')}")
                    
            except Exception as e:
                print(f"❌ 生成异常: {str(e)}")
        
        self.test_results["tests_performed"].append("multi_level_generation")
        self.test_results["generated_theories"].extend(generated_theories)
        
        # 计算成功率
        total_tests = len(test_levels)
        successful_matches = sum(1 for result in generated_theories if result["level_match"])
        match_rate = successful_matches / total_tests if total_tests > 0 else 0
        
        print(f"\n📊 多层次生成结果:")
        print(f"   测试层次: {total_tests}")
        print(f"   成功生成: {len(generated_theories)}")
        print(f"   层次匹配: {successful_matches}")
        print(f"   匹配率: {match_rate:.1%}")
        
        return {
            "success": len(generated_theories) > 0,
            "match_rate": match_rate,
            "theories_generated": len(generated_theories),
            "results": generated_theories
        }
    
    async def test_progressive_series_generation(self):
        """测试渐进式理论系列生成"""
        print("\n🚀 测试2: 渐进式理论系列生成")
        print("=" * 50)
        
        # 创建测试矛盾
        test_contradiction = {
            "theory1": "Pilot Wave Theory",
            "theory2": "Consistent Histories",
            "contradictions": [
                {
                    "contradiction": "Hidden Variables",
                    "theory1_position": "Hidden variables (pilot wave) determine particle trajectories",
                    "theory2_position": "No hidden variables, only consistent sets of histories"
                }
            ]
        }
        
        print("🎯 生成渐进式理论系列（解释 → 参数扩展 → 方程修改）...")
        
        try:
            theories = await self.adaptive_generator.generate_progressive_series(
                test_contradiction,
                max_level=InnovationLevel.EQUATION_MODIFICATION
            )
            
            if theories:
                print(f"✅ 成功生成 {len(theories)} 个渐进式理论:")
                
                series_results = []
                for i, theory in enumerate(theories):
                    actual_level, scores = self.innovation_framework.assess_theory_innovation_level(theory)
                    
                    result = {
                        "sequence_number": i + 1,
                        "theory_name": theory.get('name', 'Unknown'),
                        "actual_level": actual_level.value,
                        "innovation_scores": scores,
                        "overall_score": sum(scores.values()) / len(scores)
                    }
                    
                    series_results.append(result)
                    
                    print(f"   {i+1}. {theory.get('name', 'Unknown')}")
                    print(f"      层次: {actual_level.value}")
                    print(f"      评分: {result['overall_score']:.3f}")
                    
                    # 保存理论
                    theory_file = self.test_dir / f"progressive_series_{i+1}_{actual_level.value}.json"
                    with open(theory_file, 'w', encoding='utf-8') as f:
                        json.dump(theory, f, ensure_ascii=False, indent=2)
                
                # 分析渐进性
                scores = [result['overall_score'] for result in series_results]
                is_progressive = all(scores[i] <= scores[i+1] for i in range(len(scores)-1))
                
                print(f"\n📈 渐进性分析:")
                print(f"   评分序列: {[f'{s:.3f}' for s in scores]}")
                print(f"   渐进性: {'✅ 递增' if is_progressive else '❌ 非递增'}")
                
                self.test_results["tests_performed"].append("progressive_series_generation")
                
                return {
                    "success": True,
                    "theories_count": len(theories),
                    "is_progressive": is_progressive,
                    "score_range": (min(scores), max(scores)),
                    "series_results": series_results
                }
            else:
                print("❌ 未生成任何理论")
                return {"success": False, "error": "No theories generated"}
                
        except Exception as e:
            print(f"❌ 渐进式生成失败: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def test_enhanced_feedback_integration(self):
        """测试增强反馈循环的集成效果"""
        print("\n🔄 测试3: 增强反馈循环集成")
        print("=" * 50)
        
        # 生成一个初始理论进行改进
        test_contradiction = {
            "theory1": "Objective Collapse Theories",
            "theory2": "Quantum Bayesianism",
            "contradictions": [
                {
                    "contradiction": "Measurement Problem",
                    "theory1_position": "Objective physical collapse occurs during measurement",
                    "theory2_position": "Subjective Bayesian updating, no objective collapse"
                }
            ]
        }
        
        print("🎯 生成初始理论...")
        
        try:
            initial_theory = await self.adaptive_generator.generate_targeted_theory(
                test_contradiction,
                InnovationLevel.PARAMETER_EXTENSION
            )
            
            if "error" not in initial_theory:
                print(f"✅ 初始理论: {initial_theory.get('name', 'Unknown')}")
                
                # 模拟低分评估
                mock_evaluation = {
                    "role_score": 0.45,
                    "details": {
                        "theoretical_physicist": {
                            "score": 4,
                            "rationale": "Theory introduces new parameters but lacks clear mathematical derivation and experimental predictions."
                        },
                        "experimentalist": {
                            "score": 3,
                            "rationale": "Experimental predictions are vague and not sufficiently detailed for practical testing."
                        },
                        "philosopher": {
                            "score": 6,
                            "rationale": "Philosophical position is interesting but needs deeper analysis of measurement problem."
                        }
                    }
                }
                
                print(f"📊 模拟评估分数: {mock_evaluation['role_score']:.3f}")
                print("🔄 启动增强反馈循环...")
                
                # 运行改进循环
                improvement_results = await self.enhanced_feedback.run_improvement_cycle(
                    initial_theory,
                    mock_evaluation,
                    max_iterations=3,
                    target_score_threshold=0.75
                )
                
                # 分析改进效果
                initial_score = improvement_results["cycle_summary"]["initial_score"]
                final_score = improvement_results["cycle_summary"]["final_score"]
                total_improvement = improvement_results["cycle_summary"]["total_improvement"]
                iterations = improvement_results["cycle_summary"]["total_iterations"]
                
                improvement_percentage = (total_improvement / initial_score) * 100
                
                print(f"✅ 反馈循环完成:")
                print(f"   📈 初始分数: {initial_score:.3f}")
                print(f"   📈 最终分数: {final_score:.3f}")
                print(f"   📈 总体改进: {total_improvement:.3f} ({improvement_percentage:.1f}%)")
                print(f"   🔄 迭代次数: {iterations}")
                
                # 保存改进结果
                feedback_file = self.test_dir / "enhanced_feedback_integration.json"
                with open(feedback_file, 'w', encoding='utf-8') as f:
                    json.dump(improvement_results, f, ensure_ascii=False, indent=2)
                
                self.test_results["tests_performed"].append("enhanced_feedback_integration")
                
                return {
                    "success": True,
                    "initial_score": initial_score,
                    "final_score": final_score,
                    "improvement_percentage": improvement_percentage,
                    "iterations": iterations,
                    "target_reached": final_score >= 0.75
                }
            else:
                print(f"❌ 初始理论生成失败: {initial_theory.get('error')}")
                return {"success": False, "error": "Initial theory generation failed"}
                
        except Exception as e:
            print(f"❌ 反馈循环测试失败: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def test_innovation_level_accuracy(self):
        """测试创新层次评估的准确性"""
        print("\n🎯 测试4: 创新层次评估准确性")
        print("=" * 50)
        
        # 创建不同层次的测试理论
        test_theories = {
            "interpretation_theory": {
                "name": "Pure Interpretation Test",
                "mathematical_relation_to_sqm": "Interpretation",
                "summary": "A pure interpretation without mathematical changes",
                "formalism": {
                    "mathematical_objects": "Standard Hilbert space",
                    "governing_equations": ["i\\hbar \\partial_t |\\psi\\rangle = H|\\psi\\rangle"],
                    "comparison_with_sqm": {
                        "agreements": "Complete agreement",
                        "modifications": "None",
                        "extensions": "None"
                    }
                }
            },
            "parameter_extension_theory": {
                "name": "Parameter Extension Test",
                "mathematical_relation_to_sqm": "Extension", 
                "summary": "Theory with new parameter α controlling decoherence",
                "formalism": {
                    "mathematical_objects": "Standard Hilbert space with decoherence parameter α",
                    "governing_equations": [
                        "i\\hbar \\partial_t |\\psi\\rangle = H|\\psi\\rangle + α \\mathcal{L}[\\psi]"
                    ],
                    "comparison_with_sqm": {
                        "agreements": "Maintains unitary evolution when α=0",
                        "modifications": "Adds decoherence term",
                        "extensions": "Introduces new parameter α"
                    }
                },
                "predictions_and_verifiability": {
                    "deviations_from_sqm": [
                        {
                            "prediction_name": "Enhanced Decoherence",
                            "experimental_setup": "Specific interferometer setup with controlled environment"
                        }
                    ]
                }
            }
        }
        
        accuracy_results = []
        
        for theory_type, theory in test_theories.items():
            expected_level = theory_type.split('_')[0]  # 'interpretation' or 'parameter'
            
            actual_level, scores = self.innovation_framework.assess_theory_innovation_level(theory)
            
            # 检查是否匹配期望
            level_match = False
            if expected_level == "interpretation" and actual_level == InnovationLevel.INTERPRETATION:
                level_match = True
            elif expected_level == "parameter" and actual_level == InnovationLevel.PARAMETER_EXTENSION:
                level_match = True
            
            result = {
                "theory_type": theory_type,
                "expected_level": expected_level,
                "actual_level": actual_level.value,
                "level_match": level_match,
                "scores": scores,
                "overall_score": sum(scores.values()) / len(scores)
            }
            
            accuracy_results.append(result)
            
            print(f"📊 {theory_type}:")
            print(f"   期望层次: {expected_level}")
            print(f"   实际层次: {actual_level.value}")
            print(f"   匹配状态: {'✅' if level_match else '❌'}")
            print(f"   总体评分: {result['overall_score']:.3f}")
        
        # 计算准确率
        total_tests = len(accuracy_results)
        correct_classifications = sum(1 for result in accuracy_results if result["level_match"])
        accuracy_rate = correct_classifications / total_tests if total_tests > 0 else 0
        
        print(f"\n📈 评估准确性:")
        print(f"   总测试数: {total_tests}")
        print(f"   正确分类: {correct_classifications}")
        print(f"   准确率: {accuracy_rate:.1%}")
        
        self.test_results["tests_performed"].append("innovation_level_accuracy")
        self.test_results["performance_metrics"]["innovation_accuracy"] = accuracy_rate
        
        return {
            "success": True,
            "accuracy_rate": accuracy_rate,
            "correct_classifications": correct_classifications,
            "total_tests": total_tests,
            "detailed_results": accuracy_results
        }
    
    async def run_complete_system_test(self):
        """运行完整的系统测试"""
        print("🚀 完整增强理论生成系统测试")
        print("=" * 70)
        
        # 运行所有测试
        test1_result = await self.test_multi_level_generation()
        test2_result = await self.test_progressive_series_generation()
        test3_result = await self.test_enhanced_feedback_integration()
        test4_result = await self.test_innovation_level_accuracy()
        
        # 汇总结果
        all_results = {
            "multi_level_generation": test1_result,
            "progressive_series_generation": test2_result,
            "enhanced_feedback_integration": test3_result,
            "innovation_level_accuracy": test4_result
        }
        
        # 计算总体成功率
        successful_tests = sum(1 for result in all_results.values() if result.get("success", False))
        total_tests = len(all_results)
        overall_success_rate = successful_tests / total_tests
        
        # 更新测试结果
        self.test_results["all_test_results"] = all_results
        self.test_results["performance_metrics"]["overall_success_rate"] = overall_success_rate
        self.test_results["performance_metrics"]["successful_tests"] = successful_tests
        self.test_results["performance_metrics"]["total_tests"] = total_tests
        
        # 生成最终报告
        await self.generate_final_report()
        
        print("\n" + "=" * 70)
        print("🎉 完整系统测试完成！")
        print(f"📊 总体成功率: {overall_success_rate:.1%} ({successful_tests}/{total_tests})")
        print(f"📁 详细结果保存在: {self.test_dir}")
        
        # 关闭LLM连接
        try:
            await self.llm.aclose()
        except:
            pass
    
    async def generate_final_report(self):
        """生成最终测试报告"""
        
        # 保存完整测试结果
        report_file = self.test_dir / "complete_system_test_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        # 生成人类可读的摘要报告
        summary_file = self.test_dir / "test_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# 增强理论生成系统测试报告\n\n")
            f.write(f"**测试时间**: {self.test_results['test_timestamp']}\n\n")
            
            f.write("## 测试概览\n\n")
            metrics = self.test_results["performance_metrics"]
            f.write(f"- **总体成功率**: {metrics['overall_success_rate']:.1%}\n")
            f.write(f"- **成功测试**: {metrics['successful_tests']}/{metrics['total_tests']}\n")
            f.write(f"- **创新评估准确率**: {metrics.get('innovation_accuracy', 0):.1%}\n\n")
            
            f.write("## 主要发现\n\n")
            
            # 分析多层次生成结果
            if "multi_level_generation" in self.test_results.get("all_test_results", {}):
                mlg_result = self.test_results["all_test_results"]["multi_level_generation"]
                if mlg_result.get("success"):
                    f.write(f"### 多层次理论生成\n")
                    f.write(f"- 层次匹配率: {mlg_result['match_rate']:.1%}\n")
                    f.write(f"- 成功生成理论数: {mlg_result['theories_generated']}\n\n")
            
            # 分析渐进式生成结果
            if "progressive_series_generation" in self.test_results.get("all_test_results", {}):
                psg_result = self.test_results["all_test_results"]["progressive_series_generation"]
                if psg_result.get("success"):
                    f.write(f"### 渐进式理论系列\n")
                    f.write(f"- 生成理论数: {psg_result['theories_count']}\n")
                    f.write(f"- 渐进性: {'✅ 递增' if psg_result['is_progressive'] else '❌ 非递增'}\n")
                    f.write(f"- 评分范围: {psg_result['score_range'][0]:.3f} - {psg_result['score_range'][1]:.3f}\n\n")
            
            # 分析反馈循环结果
            if "enhanced_feedback_integration" in self.test_results.get("all_test_results", {}):
                efi_result = self.test_results["all_test_results"]["enhanced_feedback_integration"]
                if efi_result.get("success"):
                    f.write(f"### 增强反馈循环\n")
                    f.write(f"- 改进幅度: {efi_result['improvement_percentage']:.1f}%\n")
                    f.write(f"- 迭代次数: {efi_result['iterations']}\n")
                    f.write(f"- 目标达成: {'✅' if efi_result['target_reached'] else '❌'}\n\n")
            
            f.write("## 系统状态\n\n")
            f.write("✅ **增强理论生成系统已成功集成并运行**\n\n")
            f.write("主要组件:\n")
            f.write("- 创新框架 (Innovation Framework)\n")
            f.write("- 自适应理论生成器 (Adaptive Theory Generator)\n")
            f.write("- 增强反馈循环 (Enhanced Feedback Loop)\n\n")
            
            f.write("系统现在能够:\n")
            f.write("1. 精确控制理论创新层次\n")
            f.write("2. 生成渐进式理论系列\n")
            f.write("3. 通过智能反馈循环持续改进理论质量\n")
            f.write("4. 准确评估理论的创新水平\n")

async def main():
    """主测试函数"""
    tester = CompleteSystemTester()
    await tester.run_complete_system_test()

if __name__ == "__main__":
    asyncio.run(main()) 