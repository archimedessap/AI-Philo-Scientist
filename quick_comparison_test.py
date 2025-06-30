#!/usr/bin/env python3
"""
快速对比测试：单理论 vs 并发模式
"""

import asyncio
import time
import subprocess
from pathlib import Path

async def run_test(test_name, args):
    """运行单个测试"""
    print(f"\n🧪 {test_name}")
    print("="*50)
    
    cmd = [
        "python", "run_theory_evolution.py",
        "--initial_theories_dir", "data/theories_v2.1",
        "--experiment_dir", "demo/experiments/",
        "--max_generations", "1",
        "--initial_innovation_level", "parameter_ext",
        "--model_source", "google",
        "--model_name", "gemini-2.5-pro",
        "--use_instrument_correction",
        "--target_score", "0.85"
    ] + args
    
    start_time = time.time()
    
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        end_time = time.time()
        runtime = end_time - start_time
        
        output = stdout.decode('utf-8')
        
        # 提取关键指标
        api_calls = 0
        final_score = 0
        candidates = 1
        
        for line in output.split('\n'):
            if "📡 API调用次数:" in line:
                try:
                    api_calls = int(line.split(':')[1].strip())
                except:
                    pass
            elif "⭐ 综合评分:" in line:
                try:
                    final_score = float(line.split(':')[1].strip())
                except:
                    pass
            elif "🏆 从" in line and "个候选中选择" in line:
                try:
                    candidates = int(line.split('从')[1].split('个候选')[0].strip())
                except:
                    pass
        
        print(f"✅ 完成: {runtime:.1f}秒")
        print(f"📊 API调用: {api_calls}")
        print(f"⭐ 最终分数: {final_score:.3f}")
        print(f"🎯 候选数: {candidates}")
        
        return {
            'test_name': test_name,
            'runtime': runtime,
            'api_calls': api_calls,
            'final_score': final_score,
            'candidates': candidates,
            'success': process.returncode == 0
        }
        
    except Exception as e:
        print(f"❌ 失败: {e}")
        return {
            'test_name': test_name,
            'runtime': time.time() - start_time,
            'success': False,
            'error': str(e)
        }

async def main():
    """运行对比测试"""
    print("🚀 快速对比测试：单理论 vs 并发模式")
    print("="*60)
    
    tests = [
        ("单理论模式", []),
        ("并发2变体", ["--enable_parallel_generation", "--theory_variants", "2"])
    ]
    
    results = []
    for test_name, args in tests:
        result = await run_test(test_name, args)
        results.append(result)
        
        # 短暂休息
        await asyncio.sleep(5)
    
    # 生成对比报告
    print(f"\n{'='*60}")
    print("📊 对比结果")
    print(f"{'='*60}")
    
    print(f"{'模式':<12} {'时间(秒)':<10} {'API调用':<8} {'分数':<8} {'候选数':<6}")
    print("-" * 50)
    
    for result in results:
        if result.get('success', False):
            print(f"{result['test_name']:<12} {result['runtime']:<10.1f} {result['api_calls']:<8} {result['final_score']:<8.3f} {result['candidates']:<6}")
    
    # 计算效率提升
    if len(results) == 2 and all(r.get('success', False) for r in results):
        single_time = results[0]['runtime']
        parallel_time = results[1]['runtime']
        
        if parallel_time > 0:
            speedup = single_time / parallel_time
            print(f"\n⚡ 加速比: {speedup:.2f}x")
            
            # 质量对比
            single_score = results[0]['final_score']
            parallel_score = results[1]['final_score']
            score_improvement = (parallel_score - single_score) / single_score * 100
            print(f"📈 质量提升: {score_improvement:.1f}%")

if __name__ == "__main__":
    asyncio.run(main()) 