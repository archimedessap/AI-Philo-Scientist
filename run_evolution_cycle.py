#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_evolution_cycle.py - 清晰的多代际理论演进调度器
=====================================================
基于阶段化设计的新一代调度器，彻底解决数据流混乱问题。

设计原则：
1. 阶段化：每个阶段有明确的输入/输出
2. 标准化：统一的数据格式和接口  
3. 可追溯：每个理论都有清晰的血缘关系
4. 容错性：任何阶段失败都不会影响整体状态
"""

import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import os
import glob
import re

# 导入我们的清单管理工具
import manifest_tools


class EvolutionOrchestrator:
    """多代际理论演进的核心调度器"""
    
    def __init__(self, config):
        self.config = config
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_root = Path(config['output_root']) / self.run_id
        self.manifest_path = self.run_root / "run_manifest.json"
        
        # 创建运行目录
        self.run_root.mkdir(parents=True, exist_ok=True)
        print(f"[INIT] 创建运行目录: {self.run_root}")
        
    def run_full_evolution(self):
        """运行完整的多代际演进流程"""
        print(f"\n{'='*80}")
        print(f"🚀 启动理论演进流程 - {self.run_id}")
        print(f"{'='*80}")
        
        try:
            # 1. 初始化 manifest
            self._initialize_manifest()
            
            # 2. Generation 0: 创生阶段
            success = self._run_generation_0()
            if not success:
                print("[FATAL] Generation 0 失败，流程终止")
                return False
                
            # 3. Generation 1+: 精炼循环
            for gen in range(1, self.config['max_generations']):
                success = self._run_generation_n(gen)
                if not success:
                    print(f"[INFO] Generation {gen} 未产生合格后代，演进自然结束")
                    break
                    
            print(f"\n🎉 演进流程完成！最终结果保存在: {self.run_root}")
            return True
            
        except Exception as e:
            print(f"[CRITICAL] 演进流程发生未捕获异常: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _initialize_manifest(self):
        """初始化运行清单"""
        print(f"\n[INIT] 初始化运行清单...")
        
        # 转换 config 为 manifest 需要的格式
        manifest_config = {
            'run_id': self.run_id,
            'created_at': datetime.now().isoformat(),
            'config': self.config.copy()
        }
        
        self.manifest = manifest_tools.initialize_manifest(self.run_id, type('Args', (), manifest_config)())
        manifest_tools.save_manifest(self.manifest, self.manifest_path)
        print(f"[INIT] 清单初始化完成: {self.manifest_path}")
    
    def _run_generation_0(self):
        """Generation 0: 理论创生 + 评估"""
        print(f"\n{'='*60}")
        print(f"🧬 Generation 0: 理论创生")
        print(f"{'='*60}")
        
        gen_dir = self.run_root / "generation_0"
        gen_dir.mkdir(exist_ok=True)
        
        # Stage 1: 创生新理论
        print(f"\n[STAGE 1] 理论创生...")
        synthesis_result = self._call_synthesis_stage(gen_dir)
        if not synthesis_result['success']:
            return False
            
        # Stage 2: 注册新理论到 manifest
        print(f"\n[STAGE 2] 注册新理论...")
        theory_files = self._find_generated_theories(synthesis_result['output_dir'])
        self._register_new_theories(theory_files, generation=0)
        
        # Stage 3: 评估所有理论
        print(f"\n[STAGE 3] 评估理论...")
        return self._evaluate_generation(0, gen_dir)
    
    def _run_generation_n(self, generation):
        """Generation N: 精炼现有理论 + 评估"""
        print(f"\n{'='*60}")
        print(f"🔄 Generation {generation}: 理论精炼")
        print(f"{'='*60}")
        
        gen_dir = self.run_root / f"generation_{generation}"
        gen_dir.mkdir(exist_ok=True)
        
        # Stage 1: 选择父代理论
        print(f"\n[STAGE 1] 选择父代理论...")
        parent_ids = self._select_parents(generation - 1)
        if not parent_ids:
            print(f"[INFO] 没有合格的父代理论，Generation {generation} 结束")
            return False
            
        print(f"[INFO] 选中 {len(parent_ids)} 个父代理论: {parent_ids}")
        
        # Stage 2: 精炼理论
        print(f"\n[STAGE 2] 精炼理论...")
        refinement_result = self._call_refinement_stage(gen_dir, parent_ids)
        if not refinement_result['success']:
            return False
            
        # Stage 3: 评估新变体
        print(f"\n[STAGE 3] 评估新变体...")
        return self._evaluate_generation(generation, gen_dir)
    
    def _call_synthesis_stage(self, gen_dir):
        """调用理论合成阶段"""
        synthesis_dir = gen_dir / "synthesis"
        synthesis_dir.mkdir(exist_ok=True)
        
        cmd = [
            "python", "run_direct_synthesis.py",
            "--theories_dir", self.config['initial_theories_dir'],
            "--max_pairs", str(self.config['max_pairs_to_analyze']),
            "--variants_per_contradiction", str(self.config['variants_per_contradiction']),
            "--model_source", self.config['synthesis_model_source'],
            "--model_name", self.config['synthesis_model_name'],
            "--output_dir", str(synthesis_dir)
        ]
        
        result = self._execute_command("理论合成", cmd)
        if result['success']:
            result['output_dir'] = synthesis_dir
        return result
    
    def _call_refinement_stage(self, gen_dir, parent_ids):
        """调用理论精炼阶段"""
        refinement_dir = gen_dir / "refinement"
        refinement_dir.mkdir(exist_ok=True)
        
        cmd = [
            "python", "run_m3_auto_refinement.py",
            "--manifest-path", str(self.manifest_path),
            "--theory-ids", ",".join(parent_ids),
            "--output-dir", str(refinement_dir),
            "--judge_model_source", self.config['evaluation_model_source'],
            "--judge_model_name", self.config['evaluation_model_name'],
            "--dialog_model_source", self.config.get('dialog_model_source', self.config['evaluation_model_source']),
            "--dialog_model_name", self.config.get('dialog_model_name', self.config['evaluation_model_name']),
            "--max-iters", str(self.config.get('max_refinement_iters', 3)),
            "--min-improve", str(self.config.get('min_improvement', 0.05))
        ]
        
        result = self._execute_command("理论精炼", cmd)
        if result['success']:
            # 重新加载 manifest，因为精炼阶段会向其中添加新变体
            self.manifest = manifest_tools.load_manifest(self.manifest_path)
        return result
    
    def _evaluate_generation(self, generation, gen_dir):
        """评估指定代际的所有未评估理论"""
        # 找到当前代际所有未评估的理论
        unevaluated_theories = self._get_unevaluated_theories(generation)
        
        if not unevaluated_theories:
            print(f"[INFO] Generation {generation} 没有需要评估的理论")
            return True
            
        print(f"[INFO] 发现 {len(unevaluated_theories)} 个待评估理论")
        for theory_id, theory_data in unevaluated_theories.items():
            print(f"  - {theory_id}: {theory_data['theory_name']} (状态: {theory_data.get('status', 'unknown')})")
        
        # 创建评估环境
        eval_dir = gen_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        
        temp_theories_dir = eval_dir / "theories"
        temp_theories_dir.mkdir(exist_ok=True)
        
        # 复制理论文件到临时目录
        for theory_id, theory_data in unevaluated_theories.items():
            src_path = Path(theory_data['file_path'])
            dest_path = temp_theories_dir / f"{theory_id}_{src_path.name}"
            shutil.copy(src_path, dest_path)
            print(f"[COPY] {src_path.name} -> {dest_path}")
        
        # 调用评估脚本
        eval_output_dir = eval_dir / "results"
        cmd = [
            "python", "demo/demo_1.py",
            "--theory_path", str(temp_theories_dir),
            "--experiment_dir", self.config['experiment_dir'],
            "--output_dir", str(eval_output_dir),
            "--model_source", self.config['evaluation_model_source'],
            "--model_name", self.config['evaluation_model_name'],
            "--run_role_evaluation"
        ]
        
        if self.config.get('use_instrument_correction', False):
            cmd.append("--use_instrument_correction")
            
        result = self._execute_command("理论评估", cmd)
        if not result['success']:
            return False
            
        # 更新 manifest 中的评分
        success = self._update_scores_from_evaluation(eval_output_dir)
        if not success:
            return False
            
        # 标记理论状态并选择优胜者
        self._mark_generation_complete(generation)
        
        return True
    
    def _find_generated_theories(self, synthesis_dir):
        """在合成输出目录中查找生成的理论文件"""
        # 查找 eval_ready_theories 目录
        pattern = str(synthesis_dir) + "/**/eval_ready_theories/*.json"
        theory_files = glob.glob(pattern, recursive=True)
        
        if not theory_files:
            # 尝试备用模式
            pattern = str(synthesis_dir) + "/**/*.json"
            all_files = glob.glob(pattern, recursive=True)
            theory_files = [f for f in all_files if 'eval_ready' in f or 'theory' in f.lower()]
            
        print(f"[FIND] 发现 {len(theory_files)} 个理论文件: {[Path(f).name for f in theory_files]}")
        return theory_files
    
    def _register_new_theories(self, theory_files, generation):
        """注册新理论到 manifest"""
        for theory_file in theory_files:
            try:
                with open(theory_file, 'r', encoding='utf-8') as f:
                    theory_data = json.load(f)
                
                theory_id = manifest_tools.register_theory_in_manifest(
                    self.manifest, theory_data, theory_file, generation=generation
                )
                print(f"[REGISTER] 注册理论: {theory_data.get('name', 'Unknown')} -> {theory_id}")
                
            except Exception as e:
                print(f"[WARN] 无法注册理论文件 {theory_file}: {e}")
                
        # 保存更新后的 manifest
        manifest_tools.save_manifest(self.manifest, self.manifest_path)
    
    def _get_unevaluated_theories(self, generation):
        """获取指定代际中所有未评估的理论"""
        unevaluated = {}
        for theory_id, theory_data in self.manifest['theories'].items():
            if (theory_data['generation'] == generation and 
                theory_data.get('status') in ['created', 'refined']):
                unevaluated[theory_id] = theory_data
        return unevaluated
    
    def _update_scores_from_evaluation(self, eval_output_dir):
        """从评估结果中更新 manifest 的分数"""
        try:
            # 查找 combined_rankings.json 文件
            pattern = str(eval_output_dir) + "/**/combined_rankings.json"
            ranking_files = glob.glob(pattern, recursive=True)
            
            if not ranking_files:
                print(f"[WARN] 在 {eval_output_dir} 中未找到 combined_rankings.json")
                return False
                
            ranking_file = ranking_files[0]  # 使用第一个找到的文件
            print(f"[UPDATE] 从评估结果更新分数: {ranking_file}")
            
            manifest_tools.update_manifest_with_evaluation(self.manifest, str(ranking_file))
            manifest_tools.save_manifest(self.manifest, self.manifest_path)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] 更新评估分数时出错: {e}")
            return False
    
    def _select_parents(self, generation):
        """选择指定代际的优胜理论作为下一代的父本"""
        promoted_ids = manifest_tools.select_best_theories_for_next_gen(
            self.manifest,
            current_gen=generation,
            top_n=self.config['top_n_survivors'],
            min_score=self.config['promotion_min_score']
        )
        
        return promoted_ids
    
    def _mark_generation_complete(self, generation):
        """标记代际完成，更新理论状态"""
        # 更新当前代际所有理论的状态
        for theory_id, theory_data in self.manifest['theories'].items():
            if theory_data['generation'] == generation:
                if theory_data.get('status') in ['created', 'refined']:
                    theory_data['status'] = 'evaluated'
                    
        # 选择并标记优胜者
        promoted_ids = self._select_parents(generation)
        for theory_id in promoted_ids:
            self.manifest['theories'][theory_id]['status'] = 'promoted'
            
        # 记录代际信息
        if 'generations' not in self.manifest:
            self.manifest['generations'] = {}
            
        self.manifest['generations'][str(generation)] = {
            'completed_at': datetime.now().isoformat(),
            'promoted_theories': promoted_ids,
            'promotion_min_score': self.config['promotion_min_score']
        }
        
        manifest_tools.save_manifest(self.manifest, self.manifest_path)
        
        print(f"[COMPLETE] Generation {generation} 完成，{len(promoted_ids)} 个理论晋级")
    
    def _execute_command(self, stage_name, cmd):
        """执行命令并返回结果"""
        print(f"[EXEC] {stage_name}: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                cwd=os.getcwd()
            )
            
            print(f"[SUCCESS] {stage_name} 执行成功")
            if result.stdout:
                print(f"[OUTPUT] {result.stdout[-500:]}")  # 只显示最后500字符
                
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {stage_name} 执行失败 (返回码: {e.returncode})")
            print(f"[STDERR] {e.stderr}")
            if e.stdout:
                print(f"[STDOUT] {e.stdout}")
                
            return {
                "success": False,
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr,
                "returncode": e.returncode
            }


def create_config_from_args(args):
    """从命令行参数创建配置字典"""
    return {
        'output_root': args.output_root,
        'initial_theories_dir': args.initial_theories_dir,
        'experiment_dir': args.experiment_dir,
        'max_generations': args.max_generations,
        'max_pairs_to_analyze': args.max_pairs_to_analyze,
        'variants_per_contradiction': args.variants_per_contradiction,
        'synthesis_model_source': args.synthesis_model_source,
        'synthesis_model_name': args.synthesis_model_name,
        'evaluation_model_source': args.evaluation_model_source,
        'evaluation_model_name': args.evaluation_model_name,
        'dialog_model_source': args.dialog_model_source,
        'dialog_model_name': args.dialog_model_name,
        'promotion_min_score': args.promotion_min_score,
        'top_n_survivors': args.top_n_survivors,
        'use_instrument_correction': args.use_instrument_correction,
        'max_refinement_iters': args.max_refinement_iters,
        'min_improvement': args.min_improvement
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="多代际理论演进调度器 - 新一代架构",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 基础参数
    parser.add_argument("--output_root", default="output_evolution_test", 
                       help="输出根目录")
    parser.add_argument("--initial_theories_dir", required=True,
                       help="初始理论目录")
    parser.add_argument("--experiment_dir", default="demo/experiments/",
                       help="实验数据目录")
    
    # 演进控制参数
    parser.add_argument("--max_generations", type=int, default=3,
                       help="最大演进代数")
    parser.add_argument("--promotion_min_score", type=float, default=0.6,
                       help="晋级最低分数阈值")
    parser.add_argument("--top_n_survivors", type=int, default=2,
                       help="每代保留的最优理论数量")
    
    # 理论生成参数
    parser.add_argument("--max_pairs_to_analyze", type=int, default=5,
                       help="理论合成时分析的最大理论对数")
    parser.add_argument("--variants_per_contradiction", type=int, default=2,
                       help="每个矛盾点生成的理论变体数")
    
    # 模型参数
    parser.add_argument("--synthesis_model_source", default="google",
                       choices=["openai", "deepseek", "google"],
                       help="理论合成模型来源")
    parser.add_argument("--synthesis_model_name", default="gemini-2.5-pro",
                       help="理论合成模型名称")
    parser.add_argument("--evaluation_model_source", default="google", 
                       choices=["openai", "deepseek", "google"],
                       help="理论评估模型来源")
    parser.add_argument("--evaluation_model_name", default="gemini-2.5-pro",
                       help="理论评估模型名称")
    parser.add_argument("--dialog_model_source", default="google",
                       choices=["openai", "deepseek", "google"],
                       help="对话精炼模型来源")
    parser.add_argument("--dialog_model_name", default="gemini-2.5-pro",
                       help="对话精炼模型名称")
    
    # 精炼参数
    parser.add_argument("--max_refinement_iters", type=int, default=3,
                       help="理论精炼最大迭代次数")
    parser.add_argument("--min_improvement", type=float, default=0.05,
                       help="精炼过程的最小改进阈值")
    
    # 评估选项
    parser.add_argument("--use_instrument_correction", action="store_true",
                       help="启用仪器修正评估")
    
    args = parser.parse_args()
    
    # 创建配置并启动演进
    config = create_config_from_args(args)
    orchestrator = EvolutionOrchestrator(config)
    
    success = orchestrator.run_full_evolution()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()