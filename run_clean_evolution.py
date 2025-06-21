#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_clean_evolution.py - 基于纯净数据流的理论演进系统
======================================================
完全重新设计的演进调度器，基于最初的清晰数据流概念。

数据流设计：
Generation 0: Initial Theories → [Synthesis] → New Theories → [Evaluation] → Scores
Generation N: Selected Theories → [Refinement] → Variants → [Evaluation] → Scores

关键原则：
1. 每个阶段只做一件事
2. 所有状态通过 manifest 管理
3. 简单、可预测的调用接口
4. 容错性和可观测性
"""

import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import os
import uuid

# 导入清单管理工具
import manifest_tools


class CleanEvolutionOrchestrator:
    """基于纯净数据流的演进调度器"""
    
    def __init__(self, config):
        self.config = config
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_root = Path(config['output_root']) / self.run_id
        self.manifest_path = self.run_root / "run_manifest.json"
        
        # 创建运行目录
        self.run_root.mkdir(parents=True, exist_ok=True)
        print(f"[🚀] 演进运行启动: {self.run_id}")
        print(f"[📁] 运行目录: {self.run_root}")
        
    def run(self):
        """运行完整的演进循环"""
        try:
            # 初始化
            self._init_manifest()
            
            # Generation 0: 创生
            success = self._run_generation_0()
            if not success:
                print("[❌] Generation 0 失败")
                return False
                
            # Generation 1+: 精炼循环  
            for gen in range(1, self.config['max_generations']):
                success = self._run_generation_n(gen)
                if not success:
                    print(f"[✅] Generation {gen} 自然结束")
                    break
                    
            print(f"[🎉] 演进完成: {self.run_root}")
            return True
            
        except Exception as e:
            print(f"[💥] 演进失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _init_manifest(self):
        """初始化运行清单"""
        print(f"\n[📋] 初始化清单...")
        
        # 创建简化的 args 对象
        args = type('Args', (), {
            'run_id': self.run_id,
            'config': self.config
        })()
        
        self.manifest = manifest_tools.initialize_manifest(self.run_id, args)
        manifest_tools.save_manifest(self.manifest, self.manifest_path)
        print(f"[✅] 清单已创建: {self.manifest_path}")
    
    def _run_generation_0(self):
        """Generation 0: 创生新理论"""
        print(f"\n{'='*60}")
        print(f"🧬 Generation 0: 理论创生")
        print(f"{'='*60}")
        
        gen_dir = self.run_root / "generation_0"
        gen_dir.mkdir(exist_ok=True)
        
        # 1. 调用合成
        print(f"\n[🔬] 调用理论合成...")
        synthesis_success = self._call_synthesis(gen_dir)
        if not synthesis_success:
            return False
            
        # 2. 注册新理论
        print(f"\n[📝] 注册新理论到清单...")
        registration_success = self._register_generated_theories(gen_dir, generation=0)
        if not registration_success:
            return False
            
        # 3. 评估理论
        print(f"\n[⚖️] 评估理论...")
        return self._evaluate_generation(0, gen_dir)
    
    def _run_generation_n(self, generation):
        """Generation N: 精炼理论"""
        print(f"\n{'='*60}")
        print(f"🔄 Generation {generation}: 理论精炼")
        print(f"{'='*60}")
        
        gen_dir = self.run_root / f"generation_{generation}"
        gen_dir.mkdir(exist_ok=True)
        
        # 1. 选择父代
        print(f"\n[🏆] 选择优胜者...")
        parents = self._select_parents(generation - 1)
        if not parents:
            print(f"[ℹ️] 没有合格的父代理论")
            return False
            
        print(f"[✅] 选中 {len(parents)} 个父代: {[p[:8] for p in parents]}")
        
        # 2. 调用精炼
        print(f"\n[⚙️] 调用理论精炼...")
        refinement_success = self._call_refinement(gen_dir, parents)
        if not refinement_success:
            return False
            
        # 3. 评估新变体
        print(f"\n[⚖️] 评估新变体...")
        return self._evaluate_generation(generation, gen_dir)
    
    def _call_synthesis(self, gen_dir):
        """调用理论合成模块"""
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
        
        return self._run_command("理论合成", cmd)
    
    def _call_refinement(self, gen_dir, parent_ids):
        """调用理论精炼模块 - 使用简化接口"""
        refinement_dir = gen_dir / "refinement"
        refinement_dir.mkdir(exist_ok=True)
        
        # 创建临时理论目录
        temp_theories_dir = refinement_dir / "input_theories"
        temp_theories_dir.mkdir(exist_ok=True)
        
        # 复制父代理论文件
        temp_summary = []
        for parent_id in parent_ids:
            parent_data = self.manifest['theories'][parent_id]
            src_path = Path(parent_data['file_path'])
            
            # 根据理论名称生成标准化的文件名
            theory_name = parent_data['theory_name']
            sanitized_name = theory_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('-', '_')
            dest_filename = f"{sanitized_name}.json"
            dest_path = temp_theories_dir / dest_filename
            
            # 复制文件
            shutil.copy(src_path, dest_path)
            print(f"    📂 复制: {src_path.name} -> {dest_filename}")
            
            temp_summary.append({
                "theory_name": parent_data['theory_name'],
                "file_path": str(dest_path),
                "success_rate": parent_data.get('score', 0.8),
                "composite_score": parent_data.get('score', 0.8)
            })
        
        # 保存临时summary文件
        temp_summary_path = refinement_dir / "temp_summary.json"
        with open(temp_summary_path, 'w', encoding='utf-8') as f:
            json.dump(temp_summary, f, indent=2)
        
        # 调用精炼脚本
        cmd = [
            "python", "run_refinement_loop.py",
            "--summary_file", str(temp_summary_path),
            "--theories_root", str(temp_theories_dir),
            "--output_root", str(refinement_dir / "output"),
            "--top_n", str(len(parent_ids)),
            "--max_iters", str(self.config.get('max_refinement_iters', 3)),
            "--min_improve", str(self.config.get('min_improvement', 0.02)),
            "--eval_mode", "real",
            "--judge_model_source", self.config['evaluation_model_source'],
            "--judge_model_name", self.config['evaluation_model_name'],
            "--dialog_model_source", self.config.get('dialog_model_source', self.config['evaluation_model_source']),
            "--dialog_model_name", self.config.get('dialog_model_name', self.config['evaluation_model_name'])
        ]
        
        success = self._run_command("理论精炼", cmd)
        if success:
            self._register_refined_theories(refinement_dir, parent_ids, generation=self._get_current_generation() + 1)
        
        return success
    
    def _call_evaluation(self, theories_dir, output_dir):
        """调用理论评估模块"""
        cmd = [
            "python", "demo/demo_1.py",
            "--theory_path", str(theories_dir),
            "--experiment_dir", self.config['experiment_dir'],
            "--output_dir", str(output_dir),
            "--model_source", self.config['evaluation_model_source'],
            "--model_name", self.config['evaluation_model_name'],
            "--run_role_evaluation",
            "--role_success_threshold", str(self.config.get('role_success_threshold', 0.3))  # 设置合理的角色评估阈值
        ]
        
        if self.config.get('use_instrument_correction', False):
            cmd.append("--use_instrument_correction")
            
        return self._run_command("理论评估", cmd)
    
    def _register_generated_theories(self, gen_dir, generation):
        """注册生成的新理论"""
        # 只查找eval_ready_theories目录下的最终理论文件，避免重复注册
        eval_ready_dirs = list(gen_dir.rglob("eval_ready_theories"))
        
        if not eval_ready_dirs:
            print(f"[⚠️] 未找到eval_ready_theories目录")
            return False
        
        theory_files = []
        for eval_dir in eval_ready_dirs:
            theory_files.extend(eval_dir.glob("*.json"))
        
        if not theory_files:
            print(f"[⚠️] 未找到生成的理论文件")
            return False
            
        print(f"[📝] 发现 {len(theory_files)} 个理论文件")
        
        for theory_file in theory_files:
            try:
                with open(theory_file, 'r', encoding='utf-8') as f:
                    theory_data = json.load(f)
                
                # 跳过非理论文件
                if 'name' not in theory_data and 'theory_name' not in theory_data:
                    continue
                    
                theory_id = manifest_tools.register_theory_in_manifest(
                    self.manifest, theory_data, theory_file, generation=generation
                )
                theory_name = theory_data.get('name', theory_data.get('theory_name', 'Unknown'))
                print(f"  ✅ {theory_name} -> {theory_id}")
                
            except Exception as e:
                print(f"  ❌ 无法注册 {theory_file}: {e}")
        
        manifest_tools.save_manifest(self.manifest, self.manifest_path)
        return True
    
    def _register_refined_theories(self, refinement_dir, parent_ids, generation):
        """注册精炼产生的理论变体"""
        output_dir = refinement_dir / "output" / "depth_runs"
        if not output_dir.exists():
            print(f"[⚠️] 精炼输出目录不存在: {output_dir}")
            return False
        
        registered_count = 0
        for parent_id in parent_ids:
            parent_data = self.manifest['theories'][parent_id]
            parent_name = parent_data['theory_name']
            
            # 生成标准化的理论名称进行匹配
            sanitized_parent_name = parent_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('-', '_')
            
            # 查找该理论的精炼结果 - 使用多种模式匹配
            theory_dirs = []
            # 首先尝试精确匹配标准化名称
            exact_match = output_dir / sanitized_parent_name
            if exact_match.exists() and exact_match.is_dir():
                theory_dirs.append(exact_match)
            else:
                # 然后尝试模糊匹配
                theory_dirs = list(output_dir.glob(f"*{sanitized_parent_name}*"))
                if not theory_dirs:
                    # 最后尝试原始名称的变体
                    theory_dirs = list(output_dir.glob(f"*{parent_name.replace(' ', '_')}*"))
                if not theory_dirs:
                    # 如果还是找不到，列出所有目录供调试
                    all_dirs = [d.name for d in output_dir.iterdir() if d.is_dir()]
                    print(f"  🔍 未找到 {parent_name} 的精炼目录，可用目录: {all_dirs}")
                    theory_dirs = list(output_dir.glob("*"))
            
            for theory_dir in theory_dirs:
                if not theory_dir.is_dir():
                    continue
                    
                # 查找精炼后的理论文件（扩展匹配模式）
                candidate_files = list(theory_dir.glob("candidate_*.json"))
                improved_files = list(theory_dir.rglob("improved_theory_*.json"))
                improved_files_alt = list(theory_dir.rglob("improved_*.json"))  # 匹配 improved_xxx.json 格式
                
                all_candidates = candidate_files + improved_files + improved_files_alt
                if not all_candidates:
                    print(f"  ⚠️ 在 {theory_dir} 中未找到精炼后的理论文件")
                    continue
                    
                latest_candidate = max(all_candidates, key=os.path.getmtime)
                
                try:
                    with open(latest_candidate, 'r', encoding='utf-8') as f:
                        refined_data = json.load(f)
                    
                    # 直接使用传入的generation参数，不依赖register_refined_variant的逻辑
                    refined_id = f"theory_{uuid.uuid4().hex[:8]}"
                    refined_name = refined_data.get('name', 'Unknown Variant')
                    
                    self.manifest["theories"][refined_id] = {
                        "theory_name": refined_name,
                        "file_path": str(latest_candidate.resolve()),
                        "generation": generation,  # 使用传入的正确generation
                        "status": "created",  # 新变体需要评估
                        "scores": {},
                        "refinement_run_dir": str(theory_dir),
                        "refinement_parent": parent_id
                    }
                    self.manifest["lineage"][refined_id] = parent_id
                    
                    print(f"  ✅ 精炼变体: {refined_name} -> {refined_id} (Gen {generation})")
                    registered_count += 1
                    
                except Exception as e:
                    print(f"  ❌ 无法注册精炼变体 {latest_candidate}: {e}")
        
        manifest_tools.save_manifest(self.manifest, self.manifest_path)
        print(f"[📝] 注册了 {registered_count} 个精炼变体")
        return registered_count > 0
    
    def _evaluate_generation(self, generation, gen_dir):
        """评估指定代际的所有理论"""
        unevaluated = self._get_unevaluated_theories(generation)
        if not unevaluated:
            print(f"[ℹ️] Generation {generation} 没有待评估的理论")
            return True
        
        print(f"[⚖️] 评估 {len(unevaluated)} 个理论:")
        for theory_id, theory_data in unevaluated.items():
            print(f"  📋 {theory_id[:8]}: {theory_data['theory_name']}")
        
        eval_dir = gen_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        
        temp_theories_dir = eval_dir / "theories"
        temp_theories_dir.mkdir(exist_ok=True)
        
        # 复制理论文件
        for theory_id, theory_data in unevaluated.items():
            src_path = Path(theory_data['file_path'])
            dest_path = temp_theories_dir / f"{theory_id}_{src_path.name}"
            shutil.copy(src_path, dest_path)
        
        eval_output_dir = eval_dir / "results"
        success = self._call_evaluation(temp_theories_dir, eval_output_dir)
        if not success:
            return False
        
        return self._update_scores(eval_output_dir, generation)
    
    def _get_unevaluated_theories(self, generation):
        """获取指定代际的未评估理论"""
        unevaluated = {}
        for theory_id, theory_data in self.manifest['theories'].items():
            if (theory_data['generation'] == generation and 
                theory_data.get('status') in ['created', 'refined']):
                unevaluated[theory_id] = theory_data
        return unevaluated
    
    def _update_scores(self, eval_output_dir, generation):
        """更新理论分数"""
        # 查找评估结果文件 - 使用更智能的路径搜索
        ranking_files = list(eval_output_dir.rglob("combined_rankings.json"))
        
        if not ranking_files:
            # 尝试查找其他可能的评估结果文件
            alt_files = list(eval_output_dir.rglob("final_evaluation_summary.json"))
            if alt_files:
                print(f"[📊] 找到实验评估结果，但缺少角色评估结果")
                # 使用实验评估结果更新分数
                self._update_scores_from_experimental_summary(alt_files[0], generation)
                self._mark_generation_complete(generation)
                return True
            
            print(f"[⚠️] 未找到任何评估结果文件在: {eval_output_dir}")
            # 列出实际存在的文件以便调试
            all_files = list(eval_output_dir.rglob("*.json"))
            print(f"[🔍] 实际找到的JSON文件: {[f.name for f in all_files[:10]]}")
            return False
        
        ranking_file = ranking_files[0]
        print(f"[📊] 从评估结果更新分数: {ranking_file}")
        
        manifest_tools.update_manifest_with_evaluation(self.manifest, str(ranking_file))
        manifest_tools.save_manifest(self.manifest, self.manifest_path)
        
        self._mark_generation_complete(generation)
        return True
    
    def _update_scores_from_experimental_summary(self, summary_file, generation):
        """从实验评估摘要文件更新分数"""
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            print(f"[📊] 从实验评估更新分数: {summary_file}")
            
            # 更新每个理论的分数
            for entry in summary_data:
                theory_name = entry.get('theory_name')
                success_rate = entry.get('success_rate', 0.0)
                
                # 查找对应的理论ID
                theory_id = None
                for tid, theory_data in self.manifest['theories'].items():
                    if (theory_data['theory_name'] == theory_name and 
                        theory_data['generation'] == generation):
                        theory_id = tid
                        break
                
                if theory_id:
                    # 更新分数（仅基于实验结果）
                    if 'scores' not in self.manifest['theories'][theory_id]:
                        self.manifest['theories'][theory_id]['scores'] = {}
                    
                    self.manifest['theories'][theory_id]['scores']['experimental_success_rate'] = success_rate
                    # 设置一个默认的综合分数（仅基于实验）
                    self.manifest['theories'][theory_id]['score'] = success_rate
                    # 重要：标记为已评估状态
                    self.manifest['theories'][theory_id]['status'] = 'evaluated'
                    
                    print(f"  ✅ 更新 {theory_id[:8]}: {theory_name} -> 分数: {success_rate:.3f}")
                else:
                    print(f"  ⚠️ 未找到理论: {theory_name}")
            
            # 保存更新后的清单
            manifest_tools.save_manifest(self.manifest, self.manifest_path)
                    
        except Exception as e:
            print(f"[❌] 更新实验分数时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _select_parents(self, generation):
        """选择优胜者作为下一代父本"""
        promoted_ids = manifest_tools.select_best_theories_for_next_gen(
            self.manifest,
            current_gen=generation,
            top_n=self.config['top_n_survivors'],
            min_score=self.config['promotion_min_score']
        )
        return promoted_ids
    
    def _mark_generation_complete(self, generation):
        """标记代际完成"""
        for theory_id, theory_data in self.manifest['theories'].items():
            if (theory_data['generation'] == generation and 
                theory_data.get('status') == 'created'):
                theory_data['status'] = 'evaluated'
        
        promoted_ids = self._select_parents(generation)
        for theory_id in promoted_ids:
            self.manifest['theories'][theory_id]['status'] = 'promoted'
        
        if 'generations' not in self.manifest:
            self.manifest['generations'] = {}
        
        self.manifest['generations'][str(generation)] = {
            'completed_at': datetime.now().isoformat(),
            'promoted_count': len(promoted_ids),
            'promoted_theories': promoted_ids
        }
        
        manifest_tools.save_manifest(self.manifest, self.manifest_path)
        print(f"[✅] Generation {generation} 完成，{len(promoted_ids)} 个理论晋级")
    
    def _get_current_generation(self):
        """获取当前最大代际数"""
        if not self.manifest['theories']:
            return -1
        return max(t['generation'] for t in self.manifest['theories'].values())
    
    def _run_command(self, stage_name, cmd):
        """执行系统命令"""
        print(f"[🔧] {stage_name}: {' '.join(cmd)}")
        
        # 根据阶段设置不同的超时时间
        timeout_settings = {
            "理论合成": 7200,      # 2小时 - 合成相对较快
            "理论评估": 14400,     # 4小时 - 评估需要较长时间
            "理论精炼": 21600,     # 6小时 - 精炼是最耗时的阶段
        }
        
        # 默认超时时间
        timeout = 14400  # 4小时
        
        # 根据阶段名称选择合适的超时时间
        for stage_key, stage_timeout in timeout_settings.items():
            if stage_key in stage_name:
                timeout = stage_timeout
                break
        
        print(f"[⏰] 超时设置: {timeout//60} 分钟")
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=timeout
            )
            print(f"[✅] {stage_name} 成功")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"[❌] {stage_name} 失败 (返回码: {e.returncode})")
            print(f"[stderr] {e.stderr[-500:]}")
            return False
            
        except subprocess.TimeoutExpired:
            print(f"[⏰] {stage_name} 超时 ({timeout//60} 分钟)")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="基于纯净数据流的理论演进系统",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 核心参数
    parser.add_argument("--initial_theories_dir", required=True,
                       help="初始理论目录")
    parser.add_argument("--output_root", default="output_clean_evolution",
                       help="输出根目录")
    parser.add_argument("--experiment_dir", default="demo/experiments/",
                       help="实验数据目录")
    
    # 演进控制
    parser.add_argument("--max_generations", type=int, default=3,
                       help="最大演进代数")
    parser.add_argument("--promotion_min_score", type=float, default=0.3,
                       help="晋级最低分数")
    parser.add_argument("--top_n_survivors", type=int, default=2,
                       help="每代保留的理论数量")
    
    # 生成参数  
    parser.add_argument("--max_pairs_to_analyze", type=int, default=3,
                       help="合成时分析的理论对数")
    parser.add_argument("--variants_per_contradiction", type=int, default=1,
                       help="每个矛盾生成的变体数")
    
    # 模型参数
    parser.add_argument("--synthesis_model_source", default="google",
                       choices=["openai", "deepseek", "google"])
    parser.add_argument("--synthesis_model_name", default="gemini-2.5-pro")
    parser.add_argument("--evaluation_model_source", default="google",
                       choices=["openai", "deepseek", "google"])
    parser.add_argument("--evaluation_model_name", default="gemini-2.5-pro")
    parser.add_argument("--dialog_model_source", default="google",
                       choices=["openai", "deepseek", "google"])
    parser.add_argument("--dialog_model_name", default="gemini-2.5-pro")
    
    # 精炼参数
    parser.add_argument("--max_refinement_iters", type=int, default=3,
                       help="精炼最大迭代次数")
    parser.add_argument("--min_improvement", type=float, default=0.02,
                       help="精炼最小改进阈值")
    
    # 评估选项
    parser.add_argument("--role_success_threshold", type=float, default=0.1,
                       help="角色评估的最低成功率阈值")
    parser.add_argument("--use_instrument_correction", action="store_true", default=True,
                       help="启用仪器修正评估（默认开启）")
    parser.add_argument("--disable_instrument_correction", action="store_true",
                       help="禁用仪器修正评估")
    
    args = parser.parse_args()
    
    # 处理仪器修正设置
    if args.disable_instrument_correction:
        args.use_instrument_correction = False
    
    # 创建配置
    config = {
        'initial_theories_dir': args.initial_theories_dir,
        'output_root': args.output_root,
        'experiment_dir': args.experiment_dir,
        'max_generations': args.max_generations,
        'promotion_min_score': args.promotion_min_score,
        'top_n_survivors': args.top_n_survivors,
        'max_pairs_to_analyze': args.max_pairs_to_analyze,
        'variants_per_contradiction': args.variants_per_contradiction,
        'synthesis_model_source': args.synthesis_model_source,
        'synthesis_model_name': args.synthesis_model_name,
        'evaluation_model_source': args.evaluation_model_source,
        'evaluation_model_name': args.evaluation_model_name,
        'dialog_model_source': args.dialog_model_source,
        'dialog_model_name': args.dialog_model_name,
        'max_refinement_iters': args.max_refinement_iters,
        'min_improvement': args.min_improvement,
        'role_success_threshold': args.role_success_threshold,
        'use_instrument_correction': args.use_instrument_correction
    }
    
    # 启动演进
    orchestrator = CleanEvolutionOrchestrator(config)
    success = orchestrator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 