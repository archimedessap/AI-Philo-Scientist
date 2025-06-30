#!/usr/bin/env python3
"""
性能基准测试脚本
比较不同配置下的理论演进系统性能
"""

import asyncio
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List
import argparse

class PerformanceBenchmark:
    """性能基准测试器"""
    
    def __init__(self):
        self.results = {}
        self.base_cmd = [
            "python", "run_theory_evolution.py",
            "--initial_theories_dir", "data/theories_v2.1",
            "--experiment_dir", "demo/experiments/",
            "--max_generations", "2",
            "--refinement_iterations", "1",
            "--initial_innovation_level", "parameter_ext",
            "--model_source", "google",
            "--model_name", "gemini-2.5-pro",
            "--use_instrument_correction",
            "--target_score", "0.85"
        ]
    
    async def run_configuration(self, config_name: str, additional_args: List[str]) -> Dict:
        """运行特定配置并收集性能数据"""
        print(f"\n{'='*60}")
        print(f"🧪 测试配置: {config_name}")
        print(f"{'='*60}")
        
        cmd = self.base_cmd + additional_args
        print(f"📋 命令: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            # 运行命令
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            end_time = time.time()
            runtime = end_time - start_time
            
            # 解析输出
            output_text = stdout.decode('utf-8')
            
            # 提取关键信息
            result = {
                'config_name': config_name,
                'runtime': runtime,
                'success': process.returncode == 0,
                'additional_args': additional_args,
                'output_lines': len(output_text.split('\n')),
                'error_output': stderr.decode('utf-8') if stderr else None
            }
            
            # 尝试从输出中提取性能统计
            if "📊 性能统计摘要" in output_text:
                result.update(self._parse_performance_stats(output_text))
            
            # 尝试提取最终理论信息
            if "⭐ 最佳理论:" in output_text:
                result.update(self._parse_final_theory(output_text))
            
            print(f"✅ 配置 {config_name} 完成 (耗时: {runtime:.2f}秒)")
            return result
            
        except Exception as e:
            end_time = time.time()
            runtime = end_time - start_time
            print(f"❌ 配置 {config_name} 失败: {e}")
            return {
                'config_name': config_name,
                'runtime': runtime,
                'success': False,
                'error': str(e),
                'additional_args': additional_args
            }
    
    def _parse_performance_stats(self, output: str) -> Dict:
        """解析性能统计信息"""
        stats = {}
        lines = output.split('\n')
        
        for line in lines:
            if "总运行时间:" in line:
                try:
                    runtime = float(line.split(':')[1].strip().replace('秒', ''))
                    stats['total_runtime'] = runtime
                except:
                    pass
            elif "缓存命中率:" in line:
                try:
                    hit_rate = line.split(':')[1].strip().split()[0].replace('%', '')
                    stats['cache_hit_rate'] = float(hit_rate) / 100
                except:
                    pass
            elif "API调用次数:" in line:
                try:
                    api_calls = int(line.split(':')[1].strip())
                    stats['api_calls'] = api_calls
                except:
                    pass
        
        return stats
    
    def _parse_final_theory(self, output: str) -> Dict:
        """解析最终理论信息"""
        theory_info = {}
        lines = output.split('\n')
        
        for line in lines:
            if "⭐ 最佳理论:" in line:
                try:
                    parts = line.split('(综合评分:')
                    if len(parts) == 2:
                        theory_name = parts[0].split(':')[1].strip()
                        score = float(parts[1].strip().replace(')', ''))
                        theory_info['best_theory_name'] = theory_name
                        theory_info['best_theory_score'] = score
                except:
                    pass
            elif "🏆 从" in line and "个候选中选择最佳理论" in line:
                try:
                    num_candidates = int(line.split('从')[1].split('个候选')[0].strip())
                    theory_info['num_candidates'] = num_candidates
                except:
                    pass
        
        return theory_info
    
    async def run_benchmark(self):
        """运行完整的基准测试"""
        print("🚀 开始性能基准测试")
        print("="*80)
        
        # 定义测试配置
        configurations = [
            {
                'name': '单理论传统模式',
                'args': []
            },
            {
                'name': '并发生成2变体',
                'args': ['--enable_parallel_generation', '--theory_variants', '2']
            },
            {
                'name': '并发生成3变体',
                'args': ['--enable_parallel_generation', '--theory_variants', '3']
            },
            {
                'name': '并发生成4变体',
                'args': ['--enable_parallel_generation', '--theory_variants', '4']
            }
        ]
        
        # 运行所有配置
        for config in configurations:
            result = await self.run_configuration(config['name'], config['args'])
            self.results[config['name']] = result
            
            # 短暂休息，避免API限制
            await asyncio.sleep(5)
        
        # 生成比较报告
        self._generate_comparison_report()
    
    def _generate_comparison_report(self):
        """生成性能比较报告"""
        print(f"\n{'='*80}")
        print("📊 性能基准测试报告")
        print(f"{'='*80}")
        
        # 表格头
        print(f"{'配置名称':<20} {'成功':<6} {'运行时间(秒)':<12} {'最佳分数':<10} {'候选数':<8} {'缓存命中率':<12}")
        print("-" * 80)
        
        for config_name, result in self.results.items():
            success = "✅" if result.get('success', False) else "❌"
            runtime = f"{result.get('runtime', 0):.1f}"
            score = f"{result.get('best_theory_score', 0):.3f}" if result.get('best_theory_score') else "N/A"
            candidates = str(result.get('num_candidates', 1))
            cache_rate = f"{result.get('cache_hit_rate', 0):.1%}" if result.get('cache_hit_rate') is not None else "N/A"
            
            print(f"{config_name:<20} {success:<6} {runtime:<12} {score:<10} {candidates:<8} {cache_rate:<12}")
        
        # 性能分析
        print(f"\n{'='*80}")
        print("📈 性能分析")
        print(f"{'='*80}")
        
        successful_results = {k: v for k, v in self.results.items() if v.get('success', False)}
        
        if len(successful_results) > 1:
            # 找出最快和最慢的配置
            fastest = min(successful_results.items(), key=lambda x: x[1].get('runtime', float('inf')))
            slowest = max(successful_results.items(), key=lambda x: x[1].get('runtime', 0))
            
            print(f"🏃 最快配置: {fastest[0]} ({fastest[1].get('runtime', 0):.1f}秒)")
            print(f"🐌 最慢配置: {slowest[0]} ({slowest[1].get('runtime', 0):.1f}秒)")
            
            if fastest[1].get('runtime', 0) > 0:
                speedup = slowest[1].get('runtime', 0) / fastest[1].get('runtime', 1)
                print(f"⚡ 加速比: {speedup:.2f}x")
            
            # 找出最高分数
            best_score = max(successful_results.items(), 
                           key=lambda x: x[1].get('best_theory_score', 0))
            print(f"🏆 最高分数: {best_score[0]} ({best_score[1].get('best_theory_score', 0):.3f})")
        
        # 保存详细结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"benchmark_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 详细结果已保存到: {results_file}")

async def main():
    parser = argparse.ArgumentParser(description="理论演进系统性能基准测试")
    parser.add_argument("--quick", action="store_true", help="快速测试模式（减少变体数量）")
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark()
    
    if args.quick:
        # 快速测试模式，只测试关键配置
        print("🏃 快速测试模式")
        benchmark.base_cmd.extend(["--max_generations", "1"])
    
    await benchmark.run_benchmark()

if __name__ == "__main__":
    asyncio.run(main()) 