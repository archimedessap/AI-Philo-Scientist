#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_theory_evolution.py - 混合增强型理论演进系统
==================================================
结合了 run_clean_evolution.py 的成熟组件调用模式
和 run_theory_evolution.py 的高级智能增强功能。
"""

import asyncio
import argparse
import json
import subprocess
import shutil
import random
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

# 导入所有需要的模块
from theory_generation.llm_interface import LLMInterface
from theory_generation.innovation_framework import InnovationFramework, InnovationLevel
from theory_generation.adaptive_generator import AdaptiveTheoryGenerator
from theory_validation.enhanced_feedback_loop import EnhancedFeedbackLoop


class ExternalEvaluator:
    """一个包装器类，使外部评估脚本能被内部反馈循环使用"""
    def __init__(self, orchestrator, gen_dir):
        self.orchestrator = orchestrator
        self.gen_dir = gen_dir
        self.eval_count = 0

    async def evaluate_theory(self, theory: Dict) -> Dict:
        """调用外部评估脚本"""
        self.eval_count += 1
        eval_subdir = self.gen_dir / f"feedback_eval_{self.eval_count}"
        eval_subdir.mkdir(exist_ok=True)
        
        print(f"[FeedbackLoop] 正在评估变体: {theory.get('name', 'Unknown')}")
        results = await self.orchestrator._call_evaluation(theory, eval_subdir)
        # 确保返回一个字典，即使评估失败
        return results if results else {}


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_time = time.time()
        self.stage_times = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls = 0
    
    def start_stage(self, stage_name: str):
        """开始计时一个阶段"""
        self.stage_times[stage_name] = {'start': time.time()}
    
    def end_stage(self, stage_name: str):
        """结束计时一个阶段"""
        if stage_name in self.stage_times:
            self.stage_times[stage_name]['end'] = time.time()
            self.stage_times[stage_name]['duration'] = (
                self.stage_times[stage_name]['end'] - self.stage_times[stage_name]['start']
            )
    
    def record_cache_hit(self):
        """记录缓存命中"""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """记录缓存未命中"""
        self.cache_misses += 1
    
    def record_api_call(self):
        """记录API调用"""
        self.api_calls += 1
    
    def get_summary(self) -> Dict:
        """获取性能摘要"""
        total_time = time.time() - self.start_time
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return {
            'total_runtime': total_time,
            'stage_times': {k: v.get('duration', 0) for k, v in self.stage_times.items()},
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'api_calls': self.api_calls,
            'avg_time_per_api_call': total_time / self.api_calls if self.api_calls > 0 else 0
        }
    
    def print_summary(self):
        """打印性能摘要"""
        summary = self.get_summary()
        print("\n" + "="*70)
        print("📊 性能统计摘要")
        print("="*70)
        print(f"⏱️  总运行时间: {summary['total_runtime']:.2f}秒")
        print(f"🎯 缓存命中率: {summary['cache_hit_rate']:.1%} ({summary['cache_hits']}/{summary['cache_hits'] + summary['cache_misses']})")
        print(f"📡 API调用次数: {summary['api_calls']}")
        print(f"⚡ 平均API耗时: {summary['avg_time_per_api_call']:.2f}秒")
        
        if summary['stage_times']:
            print("\n🔍 各阶段耗时:")
            for stage, duration in summary['stage_times'].items():
                print(f"  • {stage}: {duration:.2f}秒")
        print("="*70)


class TheoryEvolutionOrchestrator:
    """混合增强型理论演进调度器"""

    def __init__(self, args):
        self.args = args
        self.run_dir = Path(args.output_dir) / f"evolution_run_{time.strftime('%Y%m%d_%H%M%S')}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.llm = LLMInterface(
            model_source=args.model_source,
            model_name=args.model_name
        )
        
        # 添加评估缓存和性能监控
        self.evaluation_cache = {}
        self.performance_monitor = PerformanceMonitor()
        
        self.adaptive_generator = AdaptiveTheoryGenerator(
            self.llm, 
            temperature=args.temperature,
            performance_monitor=self.performance_monitor
        )
        
        # 加载先验理论
        self.prior_theories = self._load_prior_theories(args.initial_theories_dir)
        
        print(f"[SETUP] 演进结果将保存在: {self.run_dir}")
        print(f"[INFO] 使用{args.model_source.title()} {args.model_name.replace('-', ' ').title()}模型")
        print(f"[SETUP] 加载了 {len(self.prior_theories)} 个先验理论")

    def _load_prior_theories(self, theories_dir: str) -> Dict:
        """加载先验理论"""
        theories = {}
        theories_path = Path(theories_dir)
        
        if not theories_path.exists():
            raise FileNotFoundError(f"先验理论目录不存在: {theories_dir}")
            
        theory_files = list(theories_path.glob("*.json"))
        print(f"[INFO] 在目录 {theories_dir} 中找到 {len(theory_files)} 个理论文件")
        
        for file_path in theory_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    theory_data = json.load(f)
                theory_name = theory_data.get('name', file_path.stem)
                theories[theory_name] = theory_data
            except Exception as e:
                print(f"[WARN] 无法加载理论文件 {file_path}: {e}")
                
        return theories

    def _analyze_multiple_theory_contradictions(self, num_pairs: int = 3) -> List[Dict]:
        """分析多对理论矛盾 (并发优化)"""
        print(f"[PARALLEL] 分析 {num_pairs} 对理论矛盾...")
        
        theory_names = list(self.prior_theories.keys())
        if len(theory_names) < 2:
            return [{"theory1": "Unknown", "theory2": "Unknown", "contradictions": []}]
        
        contradictions_list = []
        used_theories = set()
        
        for i in range(num_pairs):
            # 避免重复选择相同的理论对
            available_theories = [t for t in theory_names if t not in used_theories]
            if len(available_theories) < 2:
                # 如果可用理论不够，重置已使用集合
                used_theories = set()
                available_theories = theory_names
            
            selected_theories = random.sample(available_theories, 2)
            t1_name, t2_name = selected_theories[0], selected_theories[1]
            used_theories.update(selected_theories)
            
            t1_data, t2_data = self.prior_theories[t1_name], self.prior_theories[t2_name]
            
            print(f"[PARALLEL] 理论对 {i+1}: {t1_name} vs {t2_name}")
            
            contradictions = []
            # 本体论矛盾
            t1_ontology = t1_data.get("philosophy", {}).get("ontology", {}).get("fundamental_entities", [""])[0]
            t2_ontology = t2_data.get("philosophy", {}).get("ontology", {}).get("fundamental_entities", [""])[0]
            if t1_ontology and t2_ontology and t1_ontology != t2_ontology:
                contradictions.append({
                    "contradiction": "Ontological Reality", 
                    "theory1_position": f"{t1_name}: {t1_ontology[:100]}...", 
                    "theory2_position": f"{t2_name}: {t2_ontology[:100]}..."
                })

            # 数学框架矛盾
            t1_math = t1_data.get("mathematical_relation_to_sqm", {}).get("type", "")
            t2_math = t2_data.get("mathematical_relation_to_sqm", {}).get("type", "")
            if t1_math != t2_math:
                contradictions.append({
                    "contradiction": "Mathematical Framework", 
                    "theory1_position": f"{t1_name}: {t1_math}", 
                    "theory2_position": f"{t2_name}: {t2_math}"
                })
            
            # 认识论矛盾
            t1_epistem = t1_data.get("philosophy", {}).get("epistemology", {}).get("role_of_observer", "")
            t2_epistem = t2_data.get("philosophy", {}).get("epistemology", {}).get("role_of_observer", "")
            if t1_epistem and t2_epistem and t1_epistem != t2_epistem:
                contradictions.append({
                    "contradiction": "Observer Role", 
                    "theory1_position": f"{t1_name}: {t1_epistem[:100]}...", 
                    "theory2_position": f"{t2_name}: {t2_epistem[:100]}..."
                })
            
            if not contradictions:
                contradictions = [{
                    "contradiction": "General Interpretational Differences", 
                    "theory1_position": "Different philosophical stances", 
                    "theory2_position": "Varying predictions or structures"
                }]
            
            contradictions_list.append({
                "theory1": t1_name, 
                "theory2": t2_name, 
                "contradictions": contradictions,
                "contradiction_count": len(contradictions)
            })
            print(f"[PARALLEL] 发现 {len(contradictions)} 个矛盾维度")
        
        return contradictions_list

    async def generate_theories_from_multiple_contradictions(self, contradictions_list: List[Dict], innovation_level: InnovationLevel) -> List[Dict]:
        """基于多个矛盾对并发生成理论"""
        print(f"[PARALLEL] 基于 {len(contradictions_list)} 个矛盾对并发生成理论...")
        self.performance_monitor.start_stage("parallel_multi_contradiction_generation")
        
        # 创建生成任务
        tasks = []
        for i, contradiction_data in enumerate(contradictions_list):
            print(f"[PARALLEL] 创建生成任务 {i+1}: {contradiction_data['theory1']} vs {contradiction_data['theory2']}")
            task = self.adaptive_generator.generate_targeted_theory(contradiction_data, innovation_level)
            tasks.append(task)
        
        # 并行执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        valid_theories = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[PARALLEL] 理论生成 {i+1} 失败: {result}")
            elif "error" not in result:
                valid_theories.append(result)
                print(f"[PARALLEL] 理论生成 {i+1} 成功: {result.get('name', 'Unknown')}")
        
        self.performance_monitor.end_stage("parallel_multi_contradiction_generation")
        print(f"[PARALLEL] 成功生成 {len(valid_theories)}/{len(contradictions_list)} 个理论")
        
        return valid_theories

    async def run_evolution_cycle(self):
        """运行完整的演进循环"""
        print("\n" + "="*70)
        print("🚀 开始理论演进...")
        print("="*70)
        
        evolution_history = []
        current_best_theory = None
        no_improvement_count = 0  # 连续无改进计数
        min_improvement_threshold = 0.05  # 最小改进阈值
        max_no_improvement = 2  # 最大连续无改进次数
        
        for generation in range(self.args.max_generations):
            print(f"\n🔄 Generation {generation + 1}/{self.args.max_generations}")
            print("-" * 50)
            
            gen_dir = self.run_dir / f"generation_{generation}"
            gen_dir.mkdir(exist_ok=True)
            
            if generation == 0:
                print("🌱 Generation 0: 基于矛盾和创新控制生成初始理论")
                
                # 智能选择变体数量
                optimal_variants = self._get_optimal_variant_count()
                
                if getattr(self.args, 'enable_parallel_generation', False) and optimal_variants > 1:
                    # 并发生成模式
                    print(f"[PARALLEL] 启用并发生成模式，变体数量: {optimal_variants}")
                    contradictions_list = self._analyze_multiple_theory_contradictions(optimal_variants)
                    candidate_theories = await self.generate_theories_from_multiple_contradictions(
                        contradictions_list, 
                        InnovationLevel(self.args.initial_innovation_level)
                    )
                    
                    if not candidate_theories:
                        print(f"❌ 并发理论生成失败")
                        break
                    
                    # 并发评估所有候选理论
                    evaluations = await self._call_evaluation_batch(candidate_theories, gen_dir)
                    current_best_theory, evaluation_results = self._select_best_theory(candidate_theories, evaluations)
                    
                    if not current_best_theory:
                        print(f"❌ 无法找到有效的候选理论")
                        break
                    
                    print(f"🏆 从 {len(candidate_theories)} 个候选中选择最佳理论: {current_best_theory.get('name', 'Unknown')}")
                    
                    # 跳过后续的单独评估，直接使用已有结果
                    composite_score = evaluation_results.get('combined_score', 0)
                    print(f"⭐ 综合评分: {composite_score:.3f}")
                    
                else:
                    # 传统单理论生成模式
                    contradictions_list = self._analyze_multiple_theory_contradictions(1)
                    initial_theory = await self.adaptive_generator.generate_targeted_theory(
                        contradictions_list[0],
                        InnovationLevel(self.args.initial_innovation_level)
                    )
                    if "error" in initial_theory:
                        print(f"❌ 初始理论生成失败: {initial_theory['error']}")
                        break
                    current_best_theory = initial_theory
                    evaluation_results = None  # 需要后续评估
            
            # 如果还没有评估结果，进行评估
            if evaluation_results is None:
                print(f"🔬 评估理论: {current_best_theory.get('name', 'Unknown')}")
                evaluation_results = await self._call_evaluation(current_best_theory, gen_dir)
                if not evaluation_results:
                    print("❌ 理论评估失败")
                    break
                    
                composite_score = evaluation_results.get('combined_score', 0)
                print(f"⭐ 综合评分: {composite_score:.3f}")
            else:
                composite_score = evaluation_results.get('combined_score', 0)
            
            # 检查是否有显著改进
            if evolution_history:
                last_score = evolution_history[-1].get('composite_score', 0)
                improvement = composite_score - last_score
                
                if improvement < min_improvement_threshold:
                    no_improvement_count += 1
                    print(f"📈 改进幅度: {improvement:.3f} (低于阈值 {min_improvement_threshold})")
                    print(f"⚠️ 连续无显著改进: {no_improvement_count}/{max_no_improvement}")
                else:
                    no_improvement_count = 0
                    print(f"📈 显著改进: {improvement:.3f}")
            
            evolution_history.append({
                "generation": generation,
                "theory_name": current_best_theory.get('name'),
                "evaluation_results": evaluation_results,
                "composite_score": composite_score,
                "theory_data": current_best_theory
            })
            
            if composite_score >= self.args.target_score:
                print(f"🎉 达到目标分数 {self.args.target_score}，演进完成！")
                break
            
            # 智能早停检查
            if no_improvement_count >= max_no_improvement:
                print(f"🛑 连续{max_no_improvement}代无显著改进，智能早停")
                break
            
            if generation < self.args.max_generations - 1:
                print(f"🧠 演进理论 (使用增强反馈循环)...")
                evaluator = ExternalEvaluator(self, gen_dir)
                feedback_loop = EnhancedFeedbackLoop(self.llm, evaluator=evaluator)
                
                improvement_results = await feedback_loop.run_improvement_cycle(
                    theory=current_best_theory,
                    evaluation_results=evaluation_results,
                    max_iterations=self.args.refinement_iterations
                )
                
                # 检查是否有任何改进的理论
                final_theory = improvement_results.get("final_theory")
                final_evaluation = improvement_results.get("final_evaluation")
                
                if final_theory and final_evaluation:
                    # 比较分数，如果改进了就使用新理论
                    old_score = evaluation_results.get('combined_score', 0)
                    new_score = final_evaluation.get('combined_score', 0)
                    
                    if new_score > old_score:
                        current_best_theory = final_theory
                        print(f"✅ 理论改进成功: {old_score:.3f} → {new_score:.3f}")
                        print(f"📝 新理论: {current_best_theory.get('name', 'Unknown')}")
                        
                        # 需要重新评估新理论以更新evolution_history
                        print(f"🔬 重新评估改进后的理论...")
                        continue  # 继续下一代，重新评估改进后的理论
                    else:
                        print(f"⚠️ 理论改进失败: {old_score:.3f} → {new_score:.3f}")
                        break
                else:
                    print("⚠️ 演进未产生有效的理论，终止演进。")
                    break

        print("\n" + "="*70 + "\n🏁 理论演进结束\n" + "="*70)
        
        # 保存演进历史和性能数据
        final_report_file = self.run_dir / "evolution_summary.json"
        performance_summary = self.performance_monitor.get_summary()
        
        full_report = {
            'evolution_history': evolution_history,
            'performance_summary': performance_summary,
            'configuration': {
                'max_generations': self.args.max_generations,
                'target_score': self.args.target_score,
                'model_source': self.args.model_source,
                'model_name': self.args.model_name,
                'initial_innovation_level': self.args.initial_innovation_level
            }
        }
        
        with open(final_report_file, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2)
        print(f"📜 演进历史已保存到: {final_report_file}")
        
        self._print_final_summary(evolution_history)
        self.performance_monitor.print_summary()

    def _get_theory_hash(self, theory: Dict) -> str:
        """计算理论的哈希值用于缓存"""
        # 提取关键字段用于哈希计算
        key_fields = {
            'name': theory.get('name', ''),
            'formalism': theory.get('formalism', {}),
            'philosophy': theory.get('philosophy', {}),
            'predictions_and_verifiability': theory.get('predictions_and_verifiability', {})
        }
        theory_str = json.dumps(key_fields, sort_keys=True)
        return hashlib.md5(theory_str.encode()).hexdigest()

    async def _call_evaluation(self, theory, gen_dir):
        """调用外部评估脚本 (带缓存优化)"""
        self.performance_monitor.start_stage(f"evaluation_{theory.get('name', 'Unknown')}")
        
        # 检查缓存
        theory_hash = self._get_theory_hash(theory)
        if theory_hash in self.evaluation_cache:
            self.performance_monitor.record_cache_hit()
            print(f"[CACHE] 使用缓存的评估结果: {theory.get('name', 'Unknown')}")
            self.performance_monitor.end_stage(f"evaluation_{theory.get('name', 'Unknown')}")
            return self.evaluation_cache[theory_hash]
        
        self.performance_monitor.record_cache_miss()
        
        theory_dir = gen_dir / "theory_to_evaluate"
        theory_dir.mkdir(exist_ok=True)
        
        theory_file = theory_dir / f"{theory.get('name', 'theory').replace(' ', '_')}.json"
        with open(theory_file, 'w', encoding='utf-8') as f:
            json.dump(theory, f, ensure_ascii=False, indent=2)
        
        eval_output_dir = gen_dir / "evaluation"
        cmd = [
            "python", "demo/demo_1.py",
            "--theory_path", str(theory_dir),
            "--experiment_dir", self.args.experiment_dir,
            "--output_dir", str(eval_output_dir),
            "--model_source", self.args.model_source,
            "--model_name", self.args.model_name,
            "--run_role_evaluation",
            "--role_success_threshold", "0.3"
        ]
        
        if getattr(self.args, 'use_instrument_correction', False):
            cmd.append("--use_instrument_correction")
            
        success = await self._run_command("理论评估", cmd)
        if success:
            results = self._load_evaluation_results(eval_output_dir)
            if results:
                # 缓存结果
                self.evaluation_cache[theory_hash] = results
                print(f"[CACHE] 缓存评估结果: {theory.get('name', 'Unknown')}")
                self.performance_monitor.end_stage(f"evaluation_{theory.get('name', 'Unknown')}")
                return results
        
        self.performance_monitor.end_stage(f"evaluation_{theory.get('name', 'Unknown')}")
        return None

    def _load_evaluation_results(self, eval_output_dir):
        """加载评估结果"""
        # 查找combined_rankings.json文件 - 它位于嵌套的run_*目录中
        ranking_files = list(eval_output_dir.rglob("combined_rankings.json"))
        
        if ranking_files:
            try:
                with open(ranking_files[0], 'r', encoding='utf-8') as f:
                    rankings = json.load(f)
                if rankings:
                    print(f"[INFO] 成功加载评估结果: {rankings[0]}")
                    return rankings[0]
            except Exception as e:
                print(f"[WARN] 无法加载评估结果: {e}")
        else:
            print(f"[WARN] 在 {eval_output_dir} 中未找到 combined_rankings.json 文件")
            
        return None

    async def _run_command(self, stage_name, cmd):
        """异步运行命令"""
        print(f"[INFO] 执行 {stage_name}: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            print(f"[✅] {stage_name} 成功完成")
            return True
        else:
            print(f"[❌] {stage_name} 失败 (返回码: {process.returncode})")
            if stderr:
                print(f"错误信息: {stderr.decode()}")
            return False
        
    def _print_final_summary(self, history):
        """打印最终的演进总结"""
        if not history:
            print("未完成任何演进。")
            return
            
        print("\n🏆 最终演进结果总结 🏆")
        print("="*80)
        print(f"{'代数':<8} {'理论名称':<50} {'综合评分':<15}")
        print("-"*80)
        
        for record in history:
            score = record.get('composite_score', 0)
            print(f"{record['generation']:<8} {record['theory_name']:<50} {score:<15.3f}")
            
        best_record = max(history, key=lambda x: x.get('composite_score', 0))
        print("-"*80)
        print(f"\n⭐ 最佳理论: {best_record['theory_name']} (综合评分: {best_record.get('composite_score', 0):.3f})")

    async def _call_evaluation_batch(self, theories: List[Dict], gen_dir: Path) -> List[Optional[Dict]]:
        """并行评估多个理论 (性能优化)"""
        if not theories:
            return []
        
        print(f"[PARALLEL] 并行评估 {len(theories)} 个理论...")
        
        # 创建评估任务
        tasks = []
        for i, theory in enumerate(theories):
            theory_gen_dir = gen_dir / f"theory_{i}"
            theory_gen_dir.mkdir(exist_ok=True)
            task = self._call_evaluation(theory, theory_gen_dir)
            tasks.append(task)
        
        # 并行执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[PARALLEL] 理论 {i} 评估失败: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        successful_count = sum(1 for r in processed_results if r is not None)
        print(f"[PARALLEL] 成功评估 {successful_count}/{len(theories)} 个理论")
        
        return processed_results

    def _select_best_theory(self, theories: List[Dict], evaluations: List[Optional[Dict]]) -> tuple[Dict, Dict]:
        """从评估结果中选择最佳理论"""
        best_theory = None
        best_evaluation = None
        best_score = -1
        
        for theory, evaluation in zip(theories, evaluations):
            if evaluation and evaluation.get('combined_score', 0) > best_score:
                best_score = evaluation.get('combined_score', 0)
                best_theory = theory
                best_evaluation = evaluation
        
        return best_theory, best_evaluation

    async def generate_theory_variants(self, contradiction: Dict, innovation_level: InnovationLevel, num_variants: int = 3) -> List[Dict]:
        """并发生成多个理论变体"""
        print(f"[PARALLEL] 并发生成 {num_variants} 个理论变体...")
        self.performance_monitor.start_stage("parallel_generation")
        
        # 创建生成任务
        tasks = []
        for i in range(num_variants):
            task = self.adaptive_generator.generate_targeted_theory(contradiction, innovation_level)
            tasks.append(task)
        
        # 并行执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        valid_theories = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[PARALLEL] 变体 {i+1} 生成失败: {result}")
            elif "error" not in result:
                valid_theories.append(result)
                print(f"[PARALLEL] 变体 {i+1} 生成成功: {result.get('name', 'Unknown')}")
        
        self.performance_monitor.end_stage("parallel_generation")
        print(f"[PARALLEL] 成功生成 {len(valid_theories)}/{num_variants} 个理论变体")
        
        return valid_theories

    def _get_optimal_variant_count(self) -> int:
        """根据可用理论数量和性能目标智能选择变体数量"""
        total_theories = len(self.prior_theories)
        
        # 基于基准测试结果的智能选择
        if getattr(self.args, 'theory_variants', 1) > 1:
            # 用户明确指定了变体数量
            return min(self.args.theory_variants, total_theories // 2)
        
        # 自动选择最优配置
        if total_theories >= 8:
            # 足够的理论，使用最优的2变体配置
            optimal_count = 2
        elif total_theories >= 4:
            # 中等数量理论，使用保守配置
            optimal_count = 2
        else:
            # 理论数量有限，使用单变体
            optimal_count = 1
        
        print(f"[AUTO] 基于 {total_theories} 个先验理论，自动选择 {optimal_count} 个变体")
        return optimal_count

async def main():
    parser = argparse.ArgumentParser(description="混合增强型理论演进系统")
    
    # 演进参数
    evo_group = parser.add_argument_group('Evolution Parameters')
    evo_group.add_argument("--output_dir", type=str, default="theory_evolution_outputs", help="保存演进结果的总目录")
    evo_group.add_argument("--experiment_dir", type=str, default="demo/experiments", help="实验数据目录")
    evo_group.add_argument("--initial_theories_dir", type=str, default="data/theories_v2.1", help="先验理论目录")
    evo_group.add_argument("--max_generations", type=int, default=3, help="最大演进代数")
    evo_group.add_argument("--refinement_iterations", type=int, default=2, help="每代理论的改进迭代次数")
    evo_group.add_argument("--target_score", type=float, default=0.85, help="演进停止的目标综合评分")
    
    # 增强功能参数
    enh_group = parser.add_argument_group('Enhanced Features')
    enh_group.add_argument("--initial_innovation_level", type=str, default="parameter_ext", choices=[e.value for e in InnovationLevel], help="初始理论的创新层次")
    enh_group.add_argument("--theory_variants", type=int, default=1, help="每代生成的理论变体数量（并发生成）")
    enh_group.add_argument("--enable_parallel_generation", action="store_true", help="启用并发理论生成")

    # LLM 参数
    model_group = parser.add_argument_group('LLM Configuration')
    model_group.add_argument("--model_source", type=str, default="google", choices=["openai", "deepseek", "xai", "google"], help="LLM provider.")
    model_group.add_argument("--model_name", type=str, default="gemini-2.5-flash", help="Specific model name.")
    model_group.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for the LLM.")
    
    # 评估参数
    eval_group = parser.add_argument_group('Evaluation Parameters')
    eval_group.add_argument("--use_instrument_correction", action="store_true", default=True, help="启用仪器修正评估（默认开启）")
    eval_group.add_argument("--disable_instrument_correction", action="store_true", help="禁用仪器修正评估")

    args = parser.parse_args()
    
    if args.disable_instrument_correction:
        args.use_instrument_correction = False

    orchestrator = TheoryEvolutionOrchestrator(args)
    await orchestrator.run_evolution_cycle()

if __name__ == "__main__":
    asyncio.run(main()) 