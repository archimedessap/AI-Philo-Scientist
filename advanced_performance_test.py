#!/usr/bin/env python3
"""
高级性能测试脚本
测试不同创新层次和并发配置的组合效果
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List
import argparse

class AdvancedPerformanceTest:
    """高级性能测试器"""
    
    def __init__(self):
        self.results = {}
        self.base_cmd = [
            "python", "run_theory_evolution.py",
            "--initial_theories_dir", "data/theories_v2.1",
            "--experiment_dir", "demo/experiments/",
            "--max_generations", "1",
            "--refinement_iterations", "1",
            "--model_source", "google",
            "--model_name", "gemini-2.5-pro",
            "--use_instrument_correction",
            "--target_score", "0.85"
        ]
    
    async def run_comprehensive_test(self):
        """运行全面的性能测试"""
        print("🧪 高级性能测试：创新层次 × 并发配置")
        print("="*80)
        
        # 测试配置矩阵
        innovation_levels = [
            "interpretation", 
            "parameter_ext", 
            "equation_mod", 
            "framework_ext"
        ]
        
        parallel_configs = [
            {"name": "单理论", "args": []},
            {"name": "并发2变体", "args": ["--enable_parallel_generation", "--theory_variants", "2"]},
            {"name": "智能并发", "args": ["--enable_parallel_generation"]}  # 自动选择变体数量
        ]
        
        # 运行所有组合
        for innovation in innovation_levels:
            for config in parallel_configs:
                test_name = f"{innovation}_{config['name']}"
                
                cmd_args = self.base_cmd + [
                    "--initial_innovation_level", innovation
                ] + config['args']
                
                print(f"\n🔬 测试: {test_name}")
                result = await self._run_single_test(test_name, cmd_args)
                self.results[test_name] = result
                
                # 短暂休息
                await asyncio.sleep(3)
        
        # 生成分析报告
        self._generate_advanced_report()
    
    async def _run_single_test(self, test_name: str, cmd_args: List[str]) -> Dict:
        """运行单个测试"""
        start_time = time.time()
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            end_time = time.time()
            runtime = end_time - start_time
            
            output_text = stdout.decode('utf-8')
            
            # 解析结果
            result = {
                'test_name': test_name,
                'runtime': runtime,
                'success': process.returncode == 0,
                'output_lines': len(output_text.split('\n'))
            }
            
            # 提取关键指标
            if "⭐ 综合评分:" in output_text:
                for line in output_text.split('\n'):
                    if "⭐ 综合评分:" in line:
                        try:
                            score = float(line.split(':')[1].strip())
                            result['final_score'] = score
                        except:
                            pass
            
            # 提取并发信息
            if "🏆 从" in output_text:
                for line in output_text.split('\n'):
                    if "🏆 从" in line and "个候选中选择" in line:
                        try:
                            candidates = int(line.split('从')[1].split('个候选')[0].strip())
                            result['num_candidates'] = candidates
                        except:
                            pass
            
            # 提取创新层次匹配度
            if "🎯 创新层次控制:" in output_text:
                for line in output_text.split('\n'):
                    if "目标" in line and "实际" in line:
                        result['innovation_match'] = "精确匹配" in line
            
            print(f"✅ {test_name}: {runtime:.1f}s, 分数: {result.get('final_score', 'N/A')}")
            return result
            
        except Exception as e:
            end_time = time.time()
            runtime = end_time - start_time
            print(f"❌ {test_name} 失败: {e}")
            return {
                'test_name': test_name,
                'runtime': runtime,
                'success': False,
                'error': str(e)
            }
    
    def _generate_advanced_report(self):
        """生成高级分析报告"""
        print(f"\n{'='*80}")
        print("📊 高级性能测试报告")
        print(f"{'='*80}")
        
        # 按创新层次分组
        innovation_groups = {}
        for test_name, result in self.results.items():
            if result.get('success', False):
                parts = test_name.split('_', 1)
                innovation = parts[0]
                config = parts[1] if len(parts) > 1 else "unknown"
                
                if innovation not in innovation_groups:
                    innovation_groups[innovation] = {}
                innovation_groups[innovation][config] = result
        
        # 生成每个创新层次的报告
        for innovation, configs in innovation_groups.items():
            print(f"\n🎯 创新层次: {innovation.upper()}")
            print("-" * 60)
            
            for config_name, result in configs.items():
                runtime = result.get('runtime', 0)
                score = result.get('final_score', 0)
                candidates = result.get('num_candidates', 1)
                
                print(f"  {config_name:<15}: {runtime:>6.1f}s  分数:{score:>6.3f}  候选:{candidates}")
        
        # 找出最佳配置
        print(f"\n{'='*80}")
        print("🏆 最佳配置分析")
        print(f"{'='*80}")
        
        successful_results = {k: v for k, v in self.results.items() if v.get('success', False)}
        
        if successful_results:
            # 最高分数
            best_score = max(successful_results.items(), key=lambda x: x[1].get('final_score', 0))
            print(f"🥇 最高分数: {best_score[0]} ({best_score[1].get('final_score', 0):.3f})")
            
            # 最快速度
            fastest = min(successful_results.items(), key=lambda x: x[1].get('runtime', float('inf')))
            print(f"🏃 最快速度: {fastest[0]} ({fastest[1].get('runtime', 0):.1f}秒)")
            
            # 最佳性价比 (分数/时间)
            best_efficiency = max(successful_results.items(), 
                                key=lambda x: x[1].get('final_score', 0) / max(x[1].get('runtime', 1), 1))
            efficiency_score = best_efficiency[1].get('final_score', 0) / max(best_efficiency[1].get('runtime', 1), 1)
            print(f"⚡ 最佳效率: {best_efficiency[0]} ({efficiency_score:.6f} 分/秒)")
        
        # 创新层次分析
        print(f"\n📈 创新层次效果分析")
        print("-" * 40)
        
        innovation_stats = {}
        for test_name, result in successful_results.items():
            innovation = test_name.split('_')[0]
            if innovation not in innovation_stats:
                innovation_stats[innovation] = []
            innovation_stats[innovation].append(result.get('final_score', 0))
        
        for innovation, scores in innovation_stats.items():
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            print(f"  {innovation:<15}: 平均 {avg_score:.3f}, 最高 {max_score:.3f}")
        
        # 保存详细结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"advanced_test_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 详细结果已保存到: {results_file}")

async def main():
    parser = argparse.ArgumentParser(description="高级性能测试")
    parser.add_argument("--innovation_only", nargs='+', 
                       help="只测试指定的创新层次", 
                       choices=["interpretation", "parameter_ext", "equation_mod", "framework_ext"])
    
    args = parser.parse_args()
    
    tester = AdvancedPerformanceTest()
    
    if args.innovation_only:
        print(f"🎯 只测试指定创新层次: {args.innovation_only}")
        # 这里可以添加过滤逻辑
    
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main()) 