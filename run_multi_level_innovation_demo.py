#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多层次综合创新演示

展示如何使用多层次创新生成器同时在多个创新层次上进行理论创新。
"""

import asyncio
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List

from theory_generation.llm_interface import LLMInterface
from theory_generation.innovation_framework import InnovationLevel
from theory_generation.multi_level_innovation_generator import (
    MultiLevelInnovationGenerator, 
    MultiLevelInnovationConfig
)


class MultiLevelInnovationDemo:
    """多层次创新演示系统"""
    
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir) / f"multi_level_demo_{time.strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化LLM接口
        self.llm = LLMInterface(
            model_source=args.model_source,
            model_name=args.model_name
        )
        
        # 初始化多层次创新生成器
        self.multi_generator = MultiLevelInnovationGenerator(self.llm)
        
        print(f"[SETUP] 多层次创新演示启动")
        print(f"[SETUP] 输出目录: {self.output_dir}")
        print(f"[SETUP] 使用模型: {args.model_source} {args.model_name}")
    
    async def run_multi_level_demos(self):
        """运行多层次创新演示"""
        
        # 示例矛盾：哥本哈根诠释 vs 多世界诠释
        sample_contradiction = {
            "theory1": "Copenhagen Interpretation",
            "theory2": "Many-Worlds Interpretation", 
            "contradictions": [
                {
                    "contradiction": "测量坍缩机制",
                    "theory1_position": "测量导致波函数坍缩到确定本征态",
                    "theory2_position": "测量导致观察者与系统纠缠，形成分支宇宙"
                },
                {
                    "contradiction": "物理实在性",
                    "theory1_position": "测量前物理量没有确定值",
                    "theory2_position": "所有可能结果都是真实的，存在于不同分支中"
                },
                {
                    "contradiction": "概率解释",
                    "theory1_position": "Born规则给出测量结果的内禀概率",
                    "theory2_position": "概率反映观察者在分支中的相对频率"
                }
            ]
        }
        
        # 定义多种创新组合方案
        innovation_scenarios = [
            {
                "name": "双层次并行创新",
                "levels": [InnovationLevel.INTERPRETATION, InnovationLevel.PARAMETER_EXTENSION],
                "weights": {InnovationLevel.INTERPRETATION: 0.4, InnovationLevel.PARAMETER_EXTENSION: 0.6},
                "mode": "parallel"
            },
            {
                "name": "三层次分层创新", 
                "levels": [InnovationLevel.INTERPRETATION, InnovationLevel.PARAMETER_EXTENSION, InnovationLevel.EQUATION_MODIFICATION],
                "weights": {InnovationLevel.INTERPRETATION: 0.3, InnovationLevel.PARAMETER_EXTENSION: 0.4, InnovationLevel.EQUATION_MODIFICATION: 0.3},
                "mode": "hierarchical"
            },
            {
                "name": "全层次融合创新",
                "levels": [InnovationLevel.PARAMETER_EXTENSION, InnovationLevel.EQUATION_MODIFICATION, InnovationLevel.FRAMEWORK_EXTENSION],
                "weights": {InnovationLevel.PARAMETER_EXTENSION: 0.4, InnovationLevel.EQUATION_MODIFICATION: 0.3, InnovationLevel.FRAMEWORK_EXTENSION: 0.3},
                "mode": "fusion"
            },
            {
                "name": "激进四层次创新",
                "levels": [InnovationLevel.INTERPRETATION, InnovationLevel.PARAMETER_EXTENSION, InnovationLevel.FRAMEWORK_EXTENSION, InnovationLevel.PARADIGM_REVOLUTION],
                "weights": {InnovationLevel.INTERPRETATION: 0.2, InnovationLevel.PARAMETER_EXTENSION: 0.3, InnovationLevel.FRAMEWORK_EXTENSION: 0.3, InnovationLevel.PARADIGM_REVOLUTION: 0.2},
                "mode": "fusion"
            }
        ]
        
        results = []
        
        for scenario in innovation_scenarios:
            print(f"\n{'='*60}")
            print(f"🚀 开始 {scenario['name']} 演示")
            print(f"{'='*60}")
            
            result = await self.run_single_scenario(sample_contradiction, scenario)
            results.append(result)
            
            # 保存单个场景结果
            scenario_file = self.output_dir / f"{scenario['name']}.json"
            with open(scenario_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"✅ {scenario['name']} 完成，结果保存至 {scenario_file}")
        
        # 生成对比分析报告
        comparison_report = self.generate_comparison_report(results)
        
        # 保存完整结果
        complete_results = {
            "demo_info": {
                "timestamp": time.time(),
                "contradiction": sample_contradiction,
                "scenarios_tested": len(innovation_scenarios)
            },
            "individual_results": results,
            "comparison_analysis": comparison_report
        }
        
        results_file = self.output_dir / "complete_multi_level_demo.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 多层次创新演示完成！")
        print(f"📊 完整结果保存至: {results_file}")
        
        # 打印总结报告
        self.print_summary_report(comparison_report)
    
    async def run_single_scenario(self, contradiction: dict, scenario: dict) -> dict:
        """运行单个创新场景"""
        
        start_time = time.time()
        
        # 创建多层次配置
        config = self.multi_generator.create_multi_level_config(
            target_levels=scenario["levels"],
            weights=scenario["weights"],
            synthesis_mode=scenario["mode"],
            innovation_intensity=0.8
        )
        
        print(f"[CONFIG] 目标层次: {[level.value for level in config.target_levels]}")
        print(f"[CONFIG] 权重分配: {[(k.value, f'{v:.1f}') for k, v in config.level_weights.items()]}")
        print(f"[CONFIG] 综合模式: {config.synthesis_mode}")
        
        # 生成多层次创新理论
        try:
            theory = await self.multi_generator.generate_multi_level_theory(
                contradiction=contradiction,
                config=config
            )
            
            generation_time = time.time() - start_time
            
            result = {
                "scenario_info": scenario,
                "configuration": {
                    "target_levels": [level.value for level in config.target_levels],
                    "level_weights": {k.value: v for k, v in config.level_weights.items()},
                    "synthesis_mode": config.synthesis_mode,
                    "innovation_intensity": config.innovation_intensity
                },
                "generated_theory": theory,
                "generation_time": generation_time,
                "success": "error" not in theory,
                "error_message": theory.get("error", None)
            }
            
            if "error" not in theory:
                # 分析创新成就
                innovation_analysis = self.analyze_innovation_achievement(theory, config)
                result["innovation_analysis"] = innovation_analysis
                
                print(f"✅ 理论生成成功: {theory.get('name', 'Unknown')}")
                print(f"⏱️  生成耗时: {generation_time:.2f}秒")
                print(f"🎯 多层次成功率: {innovation_analysis['multi_level_success_rate']:.1%}")
            else:
                print(f"❌ 理论生成失败: {theory['error']}")
            
            return result
            
        except Exception as e:
            print(f"❌ 场景执行异常: {str(e)}")
            return {
                "scenario_info": scenario,
                "success": False,
                "error_message": str(e),
                "generation_time": time.time() - start_time
            }
    
    def analyze_innovation_achievement(self, theory: dict, config: MultiLevelInnovationConfig) -> dict:
        """分析创新成就"""
        
        multi_level_data = theory.get("metadata", {}).get("multi_level_innovation", {})
        
        analysis = {
            "target_levels": [level.value for level in config.target_levels],
            "achieved_levels": multi_level_data.get("assessment", {}).get("level_specific_scores", {}).keys(),
            "multi_level_success_rate": len(multi_level_data.get("assessment", {}).get("level_specific_scores", {})) / len(config.target_levels),
            "overall_score": multi_level_data.get("assessment", {}).get("overall_multi_level_score", 0.0),
            "synthesis_effectiveness": multi_level_data.get("assessment", {}).get("synthesis_effectiveness", 0.0),
            "emergent_properties": multi_level_data.get("assessment", {}).get("emergent_properties_detected", False),
            "coherence_score": multi_level_data.get("assessment", {}).get("coherence_score", 0.0)
        }
        
        return analysis
    
    def generate_comparison_report(self, results: List[dict]) -> dict:
        """生成对比分析报告"""
        
        successful_results = [r for r in results if r.get("success", False)]
        
        if not successful_results:
            return {"error": "没有成功的结果用于对比分析"}
        
        # 按不同维度分析
        comparison = {
            "success_rate_by_mode": {},
            "performance_by_level_count": {},
            "synthesis_effectiveness_ranking": [],
            "innovation_achievement_comparison": {},
            "generation_time_analysis": {}
        }
        
        # 按综合模式分析成功率
        mode_stats = {}
        for result in results:
            mode = result.get("configuration", {}).get("synthesis_mode", "unknown")
            if mode not in mode_stats:
                mode_stats[mode] = {"total": 0, "successful": 0}
            mode_stats[mode]["total"] += 1
            if result.get("success", False):
                mode_stats[mode]["successful"] += 1
        
        comparison["success_rate_by_mode"] = {
            mode: stats["successful"] / stats["total"] if stats["total"] > 0 else 0
            for mode, stats in mode_stats.items()
        }
        
        # 按层次数量分析性能
        level_count_stats = {}
        for result in successful_results:
            level_count = len(result.get("configuration", {}).get("target_levels", []))
            if level_count not in level_count_stats:
                level_count_stats[level_count] = []
            
            level_count_stats[level_count].append(
                result.get("innovation_analysis", {}).get("overall_score", 0.0)
            )
        
        comparison["performance_by_level_count"] = {
            count: {
                "average_score": sum(scores) / len(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "sample_count": len(scores)
            }
            for count, scores in level_count_stats.items()
        }
        
        # 综合效果排名
        synthesis_ranking = []
        for result in successful_results:
            analysis = result.get("innovation_analysis", {})
            synthesis_ranking.append({
                "scenario": result.get("scenario_info", {}).get("name", "Unknown"),
                "synthesis_effectiveness": analysis.get("synthesis_effectiveness", 0.0),
                "overall_score": analysis.get("overall_score", 0.0),
                "multi_level_success_rate": analysis.get("multi_level_success_rate", 0.0)
            })
        
        synthesis_ranking.sort(key=lambda x: x["synthesis_effectiveness"], reverse=True)
        comparison["synthesis_effectiveness_ranking"] = synthesis_ranking
        
        # 生成时间分析
        generation_times = [r.get("generation_time", 0) for r in successful_results]
        if generation_times:
            comparison["generation_time_analysis"] = {
                "average": sum(generation_times) / len(generation_times),
                "min": min(generation_times),
                "max": max(generation_times),
                "total": sum(generation_times)
            }
        
        return comparison
    
    def print_summary_report(self, comparison_report: dict):
        """打印总结报告"""
        
        print(f"\n{'='*60}")
        print("📊 多层次创新演示总结报告")
        print(f"{'='*60}")
        
        # 成功率分析
        print("\n🎯 按综合模式的成功率:")
        for mode, success_rate in comparison_report.get("success_rate_by_mode", {}).items():
            print(f"  • {mode}: {success_rate:.1%}")
        
        # 性能分析
        print("\n📈 按层次数量的性能:")
        for count, stats in comparison_report.get("performance_by_level_count", {}).items():
            print(f"  • {count}层次: 平均分 {stats['average_score']:.2f}, 最高分 {stats['max_score']:.2f}")
        
        # 综合效果排名
        print("\n🏆 综合效果排名:")
        for i, item in enumerate(comparison_report.get("synthesis_effectiveness_ranking", [])[:3], 1):
            print(f"  {i}. {item['scenario']}: 综合效果 {item['synthesis_effectiveness']:.2f}")
        
        # 时间分析
        time_analysis = comparison_report.get("generation_time_analysis", {})
        if time_analysis:
            print(f"\n⏱️  生成时间统计:")
            print(f"  • 平均耗时: {time_analysis['average']:.2f}秒")
            print(f"  • 总耗时: {time_analysis['total']:.2f}秒")


async def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description="多层次综合创新演示")
    parser.add_argument("--model_source", default="deepseek", choices=["openai", "anthropic", "deepseek"],
                       help="模型来源")
    parser.add_argument("--model_name", default="deepseek-reasoner", 
                       help="模型名称")
    parser.add_argument("--output_dir", default="multi_level_outputs",
                       help="输出目录")
    
    args = parser.parse_args()
    
    demo = MultiLevelInnovationDemo(args)
    await demo.run_multi_level_demos()


if __name__ == "__main__":
    asyncio.run(main()) 