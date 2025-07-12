#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_theory_evolution_multi_level.py - 多层次增强型理论演进系统
============================================================
基于完整的 run_theory_evolution.py 系统，集成多层次综合创新功能。
保留所有现有功能：仪器修正、外部评估、并发处理、性能监控等。
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
from theory_generation.multi_level_innovation_generator import (
    MultiLevelInnovationGenerator, 
    MultiLevelInnovationConfig
)
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
        self.multi_level_stats = {
            'parallel_calls': 0,
            'fusion_calls': 0,
            'hierarchical_calls': 0,
            'total_levels_generated': 0
        }
    
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
    
    def record_multi_level_generation(self, synthesis_mode: str, level_count: int):
        """记录多层次生成统计"""
        self.multi_level_stats[f'{synthesis_mode}_calls'] += 1
        self.multi_level_stats['total_levels_generated'] += level_count
    
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
            'avg_time_per_api_call': total_time / self.api_calls if self.api_calls > 0 else 0,
            'multi_level_stats': self.multi_level_stats
        }
    
    def print_summary(self):
        """打印性能摘要"""
        summary = self.get_summary()
        print("\n" + "="*70)
        print("📊 多层次演进性能统计摘要")
        print("="*70)
        print(f"⏱️  总运行时间: {summary['total_runtime']:.2f}秒")
        print(f"🎯 缓存命中率: {summary['cache_hit_rate']:.1%} ({summary['cache_hits']}/{summary['cache_hits'] + summary['cache_misses']})")
        print(f"📡 API调用次数: {summary['api_calls']}")
        print(f"⚡ 平均API耗时: {summary['avg_time_per_api_call']:.2f}秒")
        
        # 多层次统计
        ml_stats = summary['multi_level_stats']
        total_ml_calls = ml_stats['parallel_calls'] + ml_stats['fusion_calls'] + ml_stats['hierarchical_calls']
        if total_ml_calls > 0:
            print(f"\n🔬 多层次创新统计:")
            print(f"  • 并行创新: {ml_stats['parallel_calls']} 次")
            print(f"  • 融合创新: {ml_stats['fusion_calls']} 次")
            print(f"  • 分层创新: {ml_stats['hierarchical_calls']} 次")
            print(f"  • 总创新层次数: {ml_stats['total_levels_generated']}")
            print(f"  • 平均层次数: {ml_stats['total_levels_generated'] / total_ml_calls:.1f}")
        
        if summary['stage_times']:
            print("\n🔍 各阶段耗时:")
            for stage, duration in summary['stage_times'].items():
                print(f"  • {stage}: {duration:.2f}秒")
        print("="*70)


class MultiLevelEvolutionOrchestrator:
    """多层次增强型理论演进调度器"""

    def __init__(self, args):
        self.args = args
        self.run_dir = Path(args.output_dir) / f"ml_evolution_run_{time.strftime('%Y%m%d_%H%M%S')}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.llm = LLMInterface(
            model_source=args.model_source,
            model_name=args.model_name
        )
        
        # 添加评估缓存和性能监控
        self.evaluation_cache = {}
        self.performance_monitor = PerformanceMonitor()
        
        # 初始化传统生成器
        self.adaptive_generator = AdaptiveTheoryGenerator(
            self.llm, 
            temperature=args.temperature,
            performance_monitor=self.performance_monitor
        )
        
        # 初始化多层次生成器
        self.innovation_framework = InnovationFramework()
        self.multi_level_generator = MultiLevelInnovationGenerator(
            self.llm, 
            self.innovation_framework,
            self.performance_monitor
        )
        
        # 加载先验理论
        self.prior_theories = self._load_prior_theories(args.initial_theories_dir)
        
        print(f"[SETUP] 多层次演进结果将保存在: {self.run_dir}")
        print(f"[INFO] 使用{args.model_source.title()} {args.model_name.replace('-', ' ').title()}模型")
        print(f"[SETUP] 加载了 {len(self.prior_theories)} 个先验理论")
        print(f"[INFO] 多层次创新模式: {args.multi_level_mode}")
        print(f"[INFO] 创新强度: {args.innovation_intensity}")
        if args.use_instrument_correction:
            print(f"[INFO] 仪器修正: 开启")

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
        """分析多对理论矛盾 (保持原有逻辑)"""
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
                    "contradiction": "本体论基础",
                    "theory1_position": f"{t1_name}: {t1_ontology}",
                    "theory2_position": f"{t2_name}: {t2_ontology}"
                })
            
            # 数学框架矛盾
            t1_equations = t1_data.get("formalism", {}).get("equations", t1_data.get("formalism", {}).get("governing_equations", []))
            t2_equations = t2_data.get("formalism", {}).get("equations", t2_data.get("formalism", {}).get("governing_equations", []))
            
            if t1_equations and t2_equations:
                if isinstance(t1_equations, dict):
                    t1_eq_text = str(list(t1_equations.values())[0]) if t1_equations else ""
                elif isinstance(t1_equations, list):
                    t1_eq_text = str(t1_equations[0]) if t1_equations else ""
                else:
                    t1_eq_text = str(t1_equations)
                    
                if isinstance(t2_equations, dict):
                    t2_eq_text = str(list(t2_equations.values())[0]) if t2_equations else ""
                elif isinstance(t2_equations, list):
                    t2_eq_text = str(t2_equations[0]) if t2_equations else ""
                else:
                    t2_eq_text = str(t2_equations)
                
                if t1_eq_text and t2_eq_text and t1_eq_text != t2_eq_text:
                    contradictions.append({
                        "contradiction": "数学框架",
                        "theory1_position": f"{t1_name}: {t1_eq_text[:100]}...",
                        "theory2_position": f"{t2_name}: {t2_eq_text[:100]}..."
                    })
            
            # 实验预测矛盾
            t1_predictions = t1_data.get("experimental_predictions", {}).get("novel_phenomena", [])
            t2_predictions = t2_data.get("experimental_predictions", {}).get("novel_phenomena", [])
            if t1_predictions and t2_predictions:
                contradictions.append({
                    "contradiction": "实验预测",
                    "theory1_position": f"{t1_name}: {t1_predictions[0] if t1_predictions else '无特殊预测'}",
                    "theory2_position": f"{t2_name}: {t2_predictions[0] if t2_predictions else '无特殊预测'}"
                })
            
            # 如果没有找到具体矛盾，添加一个通用矛盾
            if not contradictions:
                contradictions.append({
                    "contradiction": "理论框架差异",
                    "theory1_position": f"{t1_name}的理论框架",
                    "theory2_position": f"{t2_name}的理论框架"
                })
            
            contradictions_list.append({
                "theory1": t1_name,
                "theory2": t2_name,
                "contradictions": contradictions
            })
        
        return contradictions_list

    def _select_multi_level_strategy(self, generation: int) -> Dict:
        """根据演进代数智能选择多层次策略"""
        
        # 基于命令行参数或智能选择
        if hasattr(self.args, 'target_levels') and self.args.target_levels:
            # 用户指定了目标层次
            target_levels = [InnovationLevel(level) for level in self.args.target_levels.split(',')]
        else:
            # 智能选择策略
            if generation == 0:
                # 第一代：保守的双层次并行
                target_levels = [InnovationLevel.INTERPRETATION, InnovationLevel.PARAMETER_EXTENSION]
            elif generation == 1:
                # 第二代：平衡的三层次融合
                target_levels = [InnovationLevel.PARAMETER_EXTENSION, InnovationLevel.EQUATION_MODIFICATION, InnovationLevel.FRAMEWORK_EXTENSION]
            else:
                # 后续代数：激进的多层次创新
                target_levels = [InnovationLevel.FRAMEWORK_EXTENSION, InnovationLevel.PARADIGM_REVOLUTION]
        
        # 权重配置
        if len(target_levels) == 2:
            weights = {target_levels[0]: 0.6, target_levels[1]: 0.4}
        elif len(target_levels) == 3:
            weights = {target_levels[0]: 0.4, target_levels[1]: 0.3, target_levels[2]: 0.3}
        else:
            # 均等分配
            weight_value = 1.0 / len(target_levels)
            weights = {level: weight_value for level in target_levels}
        
        strategy = {
            'target_levels': target_levels,
            'weights': weights,
            'synthesis_mode': getattr(self.args, 'multi_level_mode', 'fusion'),
            'innovation_intensity': getattr(self.args, 'innovation_intensity', 0.7)
        }
        
        print(f"[STRATEGY] 第{generation}代多层次策略:")
        print(f"  • 层次: {[level.value for level in target_levels]}")
        print(f"  • 权重: {[(k.value, f'{v:.1f}') for k, v in weights.items()]}")
        print(f"  • 模式: {strategy['synthesis_mode']}")
        print(f"  • 强度: {strategy['innovation_intensity']}")
        
        return strategy

    async def generate_multi_level_theories_from_contradictions(self, contradictions_list: List[Dict], generation: int) -> List[Dict]:
        """从多个矛盾生成多层次创新理论"""
        
        # 选择多层次策略
        strategy = self._select_multi_level_strategy(generation)
        
        print(f"[MULTI-LEVEL] 第{generation}代：生成多层次创新理论...")
        self.performance_monitor.start_stage(f"multi_level_generation_gen_{generation}")
        
        # 创建多层次配置
        config = self.multi_level_generator.create_multi_level_config(
            target_levels=strategy['target_levels'],
            weights=strategy['weights'],
            synthesis_mode=strategy['synthesis_mode'],
            innovation_intensity=strategy['innovation_intensity']
        )
        
        # 记录统计信息
        self.performance_monitor.record_multi_level_generation(
            config.synthesis_mode, 
            len(config.target_levels)
        )
        
        theories = []
        
        # 并发生成理论
        if self.args.enable_parallel_generation and len(contradictions_list) > 1:
            print(f"[PARALLEL] 并发生成 {len(contradictions_list)} 个多层次理论...")
            
            # 创建生成任务
            tasks = []
            for contradiction in contradictions_list:
                task = self.multi_level_generator.generate_multi_level_theory(
                    contradiction=contradiction,
                    config=config
                )
                tasks.append(task)
            
            # 并行执行
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"[PARALLEL] 矛盾 {i+1} 理论生成失败: {result}")
                elif "error" not in result:
                    theories.append(result)
                    print(f"[PARALLEL] 矛盾 {i+1} 生成成功: {result.get('name', 'Unknown')}")
                else:
                    print(f"[PARALLEL] 矛盾 {i+1} 生成失败: {result['error']}")
        else:
            # 顺序生成
            for i, contradiction in enumerate(contradictions_list):
                print(f"[SEQUENTIAL] 处理矛盾 {i+1}: {contradiction['theory1']} vs {contradiction['theory2']}")
                
                try:
                    theory = await self.multi_level_generator.generate_multi_level_theory(
                        contradiction=contradiction,
                        config=config
                    )
                    
                    if "error" not in theory:
                        theories.append(theory)
                        print(f"[SUCCESS] 生成理论: {theory.get('name', 'Unknown')}")
                    else:
                        print(f"[ERROR] 生成失败: {theory['error']}")
                        
                except Exception as e:
                    print(f"[EXCEPTION] 生成异常: {str(e)}")
        
        self.performance_monitor.end_stage(f"multi_level_generation_gen_{generation}")
        print(f"[MULTI-LEVEL] 第{generation}代成功生成 {len(theories)} 个多层次理论")
        
        return theories

    async def run_evolution_cycle(self):
        """运行多层次增强演进循环 (保持原有框架结构)"""
        
        print(f"\n{'='*70}")
        print("🧬 开始多层次增强型理论演进")
        print(f"{'='*70}")
        
        self.performance_monitor.start_stage("total_evolution")
        
        # 分析理论矛盾
        self.performance_monitor.start_stage("contradiction_analysis")
        contradictions_list = self._analyze_multiple_theory_contradictions(
            num_pairs=getattr(self.args, 'theory_pairs', 2)
        )
        self.performance_monitor.end_stage("contradiction_analysis")
        
        evolution_history = []
        
        for generation in range(self.args.max_generations):
            print(f"\n{'='*60}")
            print(f"🧬 第 {generation} 代演进 - 多层次创新")
            print(f"{'='*60}")
            
            gen_dir = self.run_dir / f"generation_{generation}"
            gen_dir.mkdir(exist_ok=True)
            
            # 生成多层次创新理论
            theories = await self.generate_multi_level_theories_from_contradictions(
                contradictions_list, generation
            )
            
            if not theories:
                print(f"[WARN] 第 {generation} 代未能生成有效理论，停止演进")
                break
            
            # 批量评估理论（使用原有的评估逻辑）
            print(f"\n[EVAL] 批量评估 {len(theories)} 个理论...")
            self.performance_monitor.start_stage(f"evaluation_gen_{generation}")
            
            evaluations = await self._call_evaluation_batch(theories, gen_dir)
            
            self.performance_monitor.end_stage(f"evaluation_gen_{generation}")
            
            # 选择最佳理论
            best_theory, best_evaluation = self._select_best_theory(theories, evaluations)
            
            if best_theory and best_evaluation:
                composite_score = best_evaluation.get('combined_score', 0)
                
                # 记录演进历史
                evolution_record = {
                    'generation': generation,
                    'theory_name': best_theory.get('name', 'Unknown'),
                    'composite_score': composite_score,
                    'multi_level_config': best_theory.get('metadata', {}).get('multi_level_innovation', {}),
                    'evaluation_details': best_evaluation,
                    'timestamp': time.time()
                }
                evolution_history.append(evolution_record)
                
                print(f"\n✅ 第 {generation} 代最佳理论: {best_theory.get('name', 'Unknown')}")
                print(f"📊 综合评分: {composite_score:.3f}")
                
                # 检查是否达到目标分数
                if composite_score >= self.args.target_score:
                    print(f"🎉 达到目标分数 {self.args.target_score}，演进成功完成！")
                    break
                
                # 增强反馈循环（如果启用）
                if generation < self.args.max_generations - 1 and getattr(self.args, 'enable_feedback_loop', False):
                    print(f"\n🔄 启动增强反馈循环...")
                    enhanced_theory = await self._run_enhanced_feedback_loop(
                        best_theory, best_evaluation, gen_dir
                    )
                    
                    if enhanced_theory and "error" not in enhanced_theory:
                        print(f"✅ 反馈增强成功: {enhanced_theory.get('name', 'Unknown')}")
                        # 将增强理论加入下一代的候选池
                        contradictions_list.append(self._create_enhanced_contradiction(enhanced_theory, best_theory))
            else:
                print(f"❌ 第 {generation} 代未能找到有效的最佳理论")
        
        self.performance_monitor.end_stage("total_evolution")
        
        # 保存演进历史和性能报告
        await self._save_evolution_results(evolution_history)
        
        # 打印最终总结
        self._print_final_summary(evolution_history)
        self.performance_monitor.print_summary()

    async def _call_evaluation_batch(self, theories: List[Dict], gen_dir: Path) -> List[Optional[Dict]]:
        """并行评估多个理论 (保持原有评估逻辑)"""
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

    async def _call_evaluation(self, theory, gen_dir):
        """调用外部评估脚本 (保持原有评估逻辑，支持仪器修正)"""
        theory_hash = self._get_theory_hash(theory)
        
        # 检查缓存
        if theory_hash in self.evaluation_cache:
            print(f"[CACHE] 命中缓存: {theory.get('name', 'Unknown')}")
            self.performance_monitor.record_cache_hit()
            return self.evaluation_cache[theory_hash]
        
        self.performance_monitor.record_cache_miss()
        self.performance_monitor.start_stage(f"evaluation_{theory.get('name', 'Unknown')}")
        
        # 保存理论到文件
        theory_file = gen_dir / "generated_theory.json"
        with open(theory_file, 'w', encoding='utf-8') as f:
            json.dump(theory, f, ensure_ascii=False, indent=2)
        
        # 构建评估命令
        cmd = [
            "python", "demo/demo_1.py",
            "--theory_file", str(theory_file),
            "--experiment_dir", str(self.args.experiment_dir),
            "--output_dir", str(gen_dir)
        ]
        
        # 添加仪器修正参数
        if self.args.use_instrument_correction:
            cmd.append("--use_instrument_correction")
        
        print(f"[EVAL] 评估理论: {theory.get('name', 'Unknown')}")
        
        # 执行评估
        success = await self._run_command("理论评估", cmd)
        
        if success:
            # 加载评估结果
            results = self._load_evaluation_results(gen_dir)
            if results:
                # 缓存结果
                self.evaluation_cache[theory_hash] = results
                print(f"[CACHE] 缓存评估结果: {theory.get('name', 'Unknown')}")
                self.performance_monitor.end_stage(f"evaluation_{theory.get('name', 'Unknown')}")
                return results
        
        self.performance_monitor.end_stage(f"evaluation_{theory.get('name', 'Unknown')}")
        return None

    def _get_theory_hash(self, theory: Dict) -> str:
        """生成理论的唯一哈希值，用于缓存"""
        # 提取关键字段进行哈希
        key_content = {
            'name': theory.get('name', ''),
            'summary': theory.get('summary', ''),
            'formalism': theory.get('formalism', {}),
            'multi_level_targets': theory.get('metadata', {}).get('multi_level_innovation', {}).get('target_levels', [])
        }
        content_str = json.dumps(key_content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()

    def _load_evaluation_results(self, eval_output_dir):
        """加载评估结果 (保持原有逻辑)"""
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
        """异步运行命令 (保持原有逻辑)"""
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

    def _select_best_theory(self, theories: List[Dict], evaluations: List[Optional[Dict]]) -> tuple[Dict, Dict]:
        """从评估结果中选择最佳理论 (保持原有逻辑)"""
        best_theory = None
        best_evaluation = None
        best_score = -1
        
        for theory, evaluation in zip(theories, evaluations):
            if evaluation and evaluation.get('combined_score', 0) > best_score:
                best_score = evaluation.get('combined_score', 0)
                best_theory = theory
                best_evaluation = evaluation
        
        return best_theory, best_evaluation

    async def _run_enhanced_feedback_loop(self, theory: Dict, evaluation: Dict, gen_dir: Path) -> Optional[Dict]:
        """运行增强反馈循环 (可选功能)"""
        try:
            external_evaluator = ExternalEvaluator(self, gen_dir)
            enhanced_loop = EnhancedFeedbackLoop(
                self.llm,
                external_evaluator,
                target_score_threshold=0.8  # 使用固定阈值避免目标泄露
            )
            
            enhanced_theory = await enhanced_loop.improve_theory(theory, evaluation)
            return enhanced_theory
            
        except Exception as e:
            print(f"[WARN] 增强反馈循环失败: {e}")
            return None

    def _create_enhanced_contradiction(self, enhanced_theory: Dict, original_theory: Dict) -> Dict:
        """为增强理论创建新的矛盾对"""
        return {
            "theory1": original_theory.get('name', 'Original Theory'),
            "theory2": enhanced_theory.get('name', 'Enhanced Theory'),
            "contradictions": [
                {
                    "contradiction": "创新程度差异",
                    "theory1_position": "原始理论框架",
                    "theory2_position": "增强创新框架"
                }
            ]
        }

    async def _save_evolution_results(self, evolution_history: List[Dict]):
        """保存演进结果和性能报告"""
        
        # 演进历史
        history_file = self.run_dir / "evolution_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(evolution_history, f, ensure_ascii=False, indent=2)
        
        # 性能报告
        performance_report = self.performance_monitor.get_summary()
        performance_file = self.run_dir / "performance_report.json"
        with open(performance_file, 'w', encoding='utf-8') as f:
            json.dump(performance_report, f, ensure_ascii=False, indent=2)
        
        # 完整系统状态
        system_state = {
            'args': vars(self.args),
            'prior_theories_count': len(self.prior_theories),
            'evolution_history': evolution_history,
            'performance_report': performance_report,
            'cache_status': {
                'cached_evaluations': len(self.evaluation_cache),
                'cache_keys': list(self.evaluation_cache.keys())
            }
        }
        
        state_file = self.run_dir / "system_state.json"
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(system_state, f, ensure_ascii=False, indent=2)
        
        print(f"[SAVE] 演进结果已保存到: {self.run_dir}")

    def _print_final_summary(self, history):
        """打印最终的演进总结 (保持原有逻辑，增加多层次信息)"""
        if not history:
            print("未完成任何演进。")
            return
            
        print("\n🏆 多层次理论演进结果总结 🏆")
        print("="*80)
        print(f"{'代数':<8} {'理论名称':<40} {'综合评分':<12} {'创新层次':<25}")
        print("-"*80)
        
        for record in history:
            score = record.get('composite_score', 0)
            ml_config = record.get('multi_level_config', {})
            levels = ','.join(ml_config.get('target_levels', []))
            print(f"{record['generation']:<8} {record['theory_name']:<40} {score:<12.3f} {levels:<25}")
            
        best_record = max(history, key=lambda x: x.get('composite_score', 0))
        print("-"*80)
        print(f"\n⭐ 最佳理论: {best_record['theory_name']} (综合评分: {best_record.get('composite_score', 0):.3f})")
        
        # 多层次统计
        ml_configs = [r.get('multi_level_config', {}) for r in history]
        synthesis_modes = [cfg.get('synthesis_mode', 'unknown') for cfg in ml_configs]
        mode_counts = {mode: synthesis_modes.count(mode) for mode in set(synthesis_modes)}
        
        print("\n🔬 多层次创新统计:")
        for mode, count in mode_counts.items():
            print(f"  • {mode}: {count} 次")


async def main():
    parser = argparse.ArgumentParser(description="多层次增强型理论演进系统")
    
    # 演进参数
    evo_group = parser.add_argument_group('Evolution Parameters')
    evo_group.add_argument("--output_dir", type=str, default="theory_evolution_outputs", help="保存演进结果的总目录")
    evo_group.add_argument("--experiment_dir", type=str, default="demo/experiments", help="实验数据目录")
    evo_group.add_argument("--initial_theories_dir", type=str, default="data/theories_v2.1", help="先验理论目录")
    evo_group.add_argument("--max_generations", type=int, default=3, help="最大演进代数")
    evo_group.add_argument("--target_score", type=float, default=0.85, help="演进停止的目标综合评分")
    evo_group.add_argument("--theory_pairs", type=int, default=2, help="每代分析的理论矛盾对数量")
    
    # 多层次创新参数
    ml_group = parser.add_argument_group('Multi-Level Innovation Parameters')
    ml_group.add_argument("--multi_level_mode", type=str, default="fusion", choices=["parallel", "hierarchical", "fusion"], help="多层次综合模式")
    ml_group.add_argument("--innovation_intensity", type=float, default=0.7, help="创新强度 (0.0-1.0)")
    ml_group.add_argument("--target_levels", type=str, help="目标创新层次 (逗号分隔，如: interpretation,parameter_ext)")
    ml_group.add_argument("--enable_parallel_generation", action="store_true", help="启用并发理论生成")
    ml_group.add_argument("--enable_feedback_loop", action="store_true", help="启用增强反馈循环")

    # LLM 参数
    model_group = parser.add_argument_group('LLM Configuration')
    model_group.add_argument("--model_source", type=str, default="google", choices=["openai", "deepseek", "google"], help="LLM provider.")
    model_group.add_argument("--model_name", type=str, default="gemini-2.5-flash", help="Specific model name.")
    model_group.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for the LLM.")
    
    # 评估参数
    eval_group = parser.add_argument_group('Evaluation Parameters')
    eval_group.add_argument("--use_instrument_correction", action="store_true", default=True, help="启用仪器修正评估（默认开启）")
    eval_group.add_argument("--disable_instrument_correction", action="store_true", help="禁用仪器修正评估")

    args = parser.parse_args()
    
    if args.disable_instrument_correction:
        args.use_instrument_correction = False

    orchestrator = MultiLevelEvolutionOrchestrator(args)
    await orchestrator.run_evolution_cycle()

if __name__ == "__main__":
    asyncio.run(main())