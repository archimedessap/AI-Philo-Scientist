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

# 导入断点续跑管理器
from utils.checkpoint_manager import CheckpointManager, ResumableRunner


class CleanEvolutionOrchestrator:
    """基于纯净数据流的演进调度器"""
    
    def __init__(self, config, resume_run_id=None):
        self.config = config
        
        # 支持续跑
        if resume_run_id:
            self.run_id = resume_run_id
            self.run_root = Path(config['output_root']) / self.run_id
            self.is_resume = True
            print(f"[🔄] 续跑模式: {self.run_id}")
        else:
            self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.run_root = Path(config['output_root']) / self.run_id
            self.is_resume = False
            # 创建运行目录
            self.run_root.mkdir(parents=True, exist_ok=True)
            print(f"[🚀] 演进运行启动: {self.run_id}")
        
        self.manifest_path = self.run_root / "run_manifest.json"
        print(f"[📁] 运行目录: {self.run_root}")
        
        # 初始化断点管理器
        self.checkpoint_manager = CheckpointManager(self.run_id, 
                                                   checkpoint_dir=str(self.run_root / "checkpoints"))
        
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
            
            # 生成与先验理论的基准对比报告
            self._generate_benchmark_comparison()
                    
            print(f"[🎉] 演进完成: {self.run_root}")
            
            # 演进后处理
            self._post_evolution_processing()
            
            return True
            
        except Exception as e:
            print(f"演进流程意外中断: {e}")
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
        """调用理论合成模块 - 支持多种生成方法，不回退"""
        synthesis_dir = gen_dir / "synthesis"
        synthesis_dir.mkdir(exist_ok=True)

        # 获取生成方法配置（默认为 direct_synthesis 保持向后兼容）
        synthesis_method = self.config.get('synthesis_method', 'direct_synthesis')

        print(f"[🎯] 使用生成方法: {synthesis_method}")

        # 使用新的理论生成中心
        try:
            from theory_generation.generation_hub import get_generation_hub
            
            hub = get_generation_hub()
            
            # 检查方法是否可用
            available_methods = hub.list_methods()
            print(f"[📋] 可用生成方法: {', '.join(available_methods)}")
            
            if synthesis_method not in available_methods:
                print(f"[❌] 错误: 方法 {synthesis_method} 不可用")
                print(f"[💡] 请使用以下方法之一: {', '.join(available_methods)}")
                raise ValueError(f"未知的生成方法: {synthesis_method}")
            
            # 调用理论生成
            print(f"[🚀] 开始使用 {synthesis_method} 方法生成理论...")
            
            # 构建生成参数
            generation_params = {
                'method': synthesis_method,
                'theories_dir': self.config['initial_theories_dir'],
                'output_dir': str(synthesis_dir),
                'max_pairs': self.config['max_pairs_to_analyze'],
                'variants_per_contradiction': self.config['variants_per_contradiction'],
                'model_source': self.config['synthesis_model_source'],
                'model_name': self.config['synthesis_model_name']
            }
            
            # 添加文献概念相关参数（如果方法支持）
            if synthesis_method in ['unified', 'unified_generator']:
                if self.config.get('use_raw_literature', False):
                    generation_params['use_raw_literature'] = True
                    generation_params['force_load_literature'] = True  # 强制加载文献概念
                    generation_params['literature_concepts_dir'] = self.config.get('literature_concepts_dir', 'data/enhanced_concepts')
                    print(f"[📚] 启用原始文献概念增强")
            
            result = hub.generate_theories(**generation_params)
            
            # 检查生成结果
            if result.get('success', False):
                theories_count = len(result.get('theories', []))
                print(f"[✅] 理论合成成功: 生成 {theories_count} 个理论")
                
                # 显示额外的元数据信息
                if 'metadata' in result:
                    metadata = result['metadata']
                    if 'space_analysis' in metadata:
                        space_info = metadata['space_analysis']
                        print(f"[📊] 概念空间: {space_info.get('total_concepts', 0)} 个概念")
                        print(f"[📊] 理论空间: {space_info.get('total_theories', 0)} 个理论") 
                        print(f"[📊] 空白区域: {space_info.get('conceptual_gaps', 0)} 个")
                
                return True
            else:
                error_msg = result.get('error_message', '未知错误')
                print(f"[❌] 理论合成失败: {error_msg}")
                raise RuntimeError(f"理论生成失败: {error_msg}")
                
        except ImportError as e:
            print(f"[❌] 无法导入理论生成中心: {e}")
            print(f"[💡] 请检查理论生成模块是否正确安装")
            raise ImportError(f"理论生成中心导入失败: {e}")
        
        except Exception as e:
            print(f"[❌] 理论生成过程出错: {e}")
            print(f"[🔍] 错误类型: {type(e).__name__}")
            print(f"[💡] 请检查生成方法 '{synthesis_method}' 的配置和依赖")
            raise RuntimeError(f"理论生成失败: {e}")
    
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
            
            # 使用简化的文件名格式，去掉theory_id前缀
            # 直接使用原始文件名，这样精炼脚本的文件名匹配逻辑就能正常工作
            dest_path = temp_theories_dir / src_path.name
            shutil.copy(src_path, dest_path)
            
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
            
            # 查找该理论的精炼结果
            theory_dirs = list(output_dir.glob(f"*{parent_name.replace(' ', '_')}*"))
            if not theory_dirs:
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
        
        # 复制并转换理论文件格式
        from utils.theory_format_converter import convert_theory_file
        
        for theory_id, theory_data in unevaluated.items():
            src_path = Path(theory_data['file_path'])
            dest_path = temp_theories_dir / f"{theory_id}_{src_path.name}"
            
            # 复制文件
            shutil.copy(src_path, dest_path)
            
            # 转换格式（原地修改）
            try:
                convert_theory_file(dest_path, dest_path)
                print(f"  ✅ 转换理论格式: {theory_data['theory_name']}")
            except Exception as e:
                print(f"  ⚠️  转换失败: {theory_data['theory_name']} - {e}")
        
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
        
        # 根据不同阶段设置不同的超时时间（加倍后的时间）
        timeout_settings = {
            "理论合成": 14400,     # 4小时（原2小时 × 2）
            "理论评估": 28800,     # 8小时（原4小时 × 2）
            "理论精炼": 57600,     # 16小时（原8小时 × 2）
        }
        
        # 默认超时时间：1小时（原30分钟 × 2）
        timeout = 3600
        
        # 根据阶段名称选择合适的超时时间
        if "合成" in stage_name or "synthesis" in " ".join(cmd).lower():
            timeout = timeout_settings["理论合成"]
        elif "评估" in stage_name or "demo_1.py" in " ".join(cmd):
            timeout = timeout_settings["理论评估"]
        elif "精炼" in stage_name or "refinement" in " ".join(cmd).lower():
            timeout = timeout_settings["理论精炼"]
        
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
    
    def _generate_benchmark_comparison(self):
        """生成与先验理论基准的对比报告"""
        try:
            from utils.theory_comparison_visualizer import TheoryComparisonVisualizer
            
            print(f"\n{'='*60}")
            print(f"🏆 生成与先验理论基准的对比报告...")
            print(f"{'='*60}")
            
            # 创建可视化工具
            visualizer = TheoryComparisonVisualizer()
            
            # 加载演进理论的评估结果（传入整个输出根目录）
            evolved_results = visualizer.load_evolved_theories_results(str(self.config['output_root']))
            
            # 检查是否有演进理论结果
            if not evolved_results.get('role_evaluation') and not evolved_results.get('experimental'):
                print("[⚠️] 没有找到演进理论的评估结果，跳过对比报告生成")
                return
            
            # 生成对比报告
            comparison_output_dir = self.run_root / "benchmark_comparison"
            visualizer.create_comprehensive_comparison_report(
                evolved_results, 
                str(comparison_output_dir)
            )
            
            print(f"[✅] 基准对比报告已生成到: {comparison_output_dir}")
            
            # 显示简要对比结果
            self._display_comparison_summary(evolved_results, visualizer)
            
        except ImportError:
            print("[⚠️] 可视化工具未找到，跳过对比报告生成")
        except Exception as e:
            print(f"[⚠️] 生成对比报告时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _display_comparison_summary(self, evolved_results, visualizer):
        """显示对比结果摘要"""
        print(f"\n{'='*50}")
        print(f"📊 对比结果摘要")
        print(f"{'='*50}")
        
        # 先验理论基准
        benchmark_scores = list(visualizer.prior_theories_benchmark["role_evaluation_benchmark"].values())
        avg_prior_composite = sum(s["composite_score"] for s in benchmark_scores) / len(benchmark_scores)
        best_prior = max(visualizer.prior_theories_benchmark["role_evaluation_benchmark"].items(),
                        key=lambda x: x[1]["composite_score"])
        
        print(f"🏛️  先验理论基准:")
        print(f"   平均角色评估综合分: {avg_prior_composite:.3f}")
        print(f"   最佳理论: {best_prior[0]} (综合分: {best_prior[1]['composite_score']:.3f})")
        
        # 演进理论结果
        if 'role_evaluation' in evolved_results and evolved_results['role_evaluation']:
            evolved_role = evolved_results['role_evaluation']
            avg_evolved_composite = sum(t.get('role_composite_score', 0) for t in evolved_role) / len(evolved_role)
            best_evolved = max(evolved_role, key=lambda x: x.get('role_composite_score', 0))
            
            print(f"\n🚀 演进理论结果:")
            print(f"   理论数量: {len(evolved_role)}个")
            print(f"   平均角色评估综合分: {avg_evolved_composite:.3f}")
            print(f"   最佳理论: {best_evolved['theory_name']} (综合分: {best_evolved.get('role_composite_score', 0):.3f})")
            
            # 对比分析
            print(f"\n🔍 对比分析:")
            improvement = avg_evolved_composite - avg_prior_composite
            if improvement > 0:
                print(f"✅ 演进理论平均分超越先验基准 (+{improvement:.3f})")
            else:
                print(f"❌ 演进理论平均分低于先验基准 ({improvement:.3f})")
            
            best_evolved_score = best_evolved.get('role_composite_score', 0)
            if best_evolved_score > best_prior[1]["composite_score"]:
                breakthrough = best_evolved_score - best_prior[1]["composite_score"]
                print(f"🏆 发现突破性理论！")
                print(f"   {best_evolved['theory_name']} 超越 {best_prior[0]} (+{breakthrough:.3f})")
            else:
                print(f"📈 最佳演进理论尚未超越先验基准")
        else:
            print(f"\n⚠️  演进理论缺少角色评估结果")
        
        print(f"{'='*50}")

    def _post_evolution_processing(self):
        """演进流程结束后的处理，包括注册理论和生成报告"""
        print(f"\n{'='*60}")
        print("🏁 演进流程完成，开始进行后处理...")
        print(f"{'='*60}")
        
        try:
            # 1. 注册演进理论到全局理论库
            print("\n[步骤1/2] 注册晋级理论到全局理论库...")
            from utils.global_theory_registry import GlobalTheoryRegistry
            registry = GlobalTheoryRegistry()
            
            # 确保先验理论已注册
            stats = registry.get_statistics()
            if stats.get("prior_theories", 0) == 0:
                print("库中无先验理论，首先注册...")
                registry.register_prior_theories(self.config['initial_theories_dir'])
            
            # 注册本次运行的晋级理论
            registered_count = registry.register_evolved_theories_from_run(str(self.run_root))
            print(f"✅ 成功注册 {registered_count} 个演进理论")
            
            # 2. 从全局库生成可视化报告
            print("\n[步骤2/2] 从全局理论库生成可视化对比报告...")
            from utils.global_theory_visualizer import GlobalTheoryVisualizer
            visualizer = GlobalTheoryVisualizer(output_dir=self.run_root / "global_comparison")
            visualizer.generate_comprehensive_report()
            
            # 3. 打印注册库摘要
            print("\n[步骤3/3] 全局理论库统计...")
            registry.print_summary()
            
            print("\n[✅] 后处理全部完成！")

        except ImportError as e:
            print(f"[❌] 后处理失败: 无法导入模块, {e}")
            print("请确保 'utils.global_theory_registry' 和 'utils.global_theory_visualizer' 存在且路径正确。")
        except Exception as e:
            print(f"[❌] 后处理失败: {e}")
            import traceback
            traceback.print_exc()


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
    parser.add_argument("--synthesis_method", default="direct_synthesis",
                       help="理论生成方法 (direct_synthesis, unified)")
    parser.add_argument("--max_pairs_to_analyze", type=int, default=3,
                       help="合成时分析的理论对数")
    parser.add_argument("--variants_per_contradiction", type=int, default=1,
                       help="每个矛盾生成的变体数")
    
    # 文献概念控制参数
    parser.add_argument("--use_raw_literature", action="store_true",
                       help="是否使用原始文献概念（需要先运行prepare_enhanced_concepts.py）")
    parser.add_argument("--literature_concepts_dir", default="data/enhanced_concepts",
                       help="文献概念目录")
    parser.add_argument("--auto_extract_concepts", action="store_true",
                       help="如果没有预提取的概念，是否自动提取（会显著增加运行时间）")
    
    # 模型参数
    parser.add_argument("--synthesis_model_source", default="google",
                       choices=["openai", "deepseek", "xai", "google"])
    parser.add_argument("--synthesis_model_name", default="gemini-2.5-pro")
    parser.add_argument("--evaluation_model_source", default="google",
                       choices=["openai", "deepseek", "xai", "google"])
    parser.add_argument("--evaluation_model_name", default="gemini-2.5-pro")
    parser.add_argument("--dialog_model_source", default="google",
                       choices=["openai", "deepseek", "xai", "google"])
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
    
    # 断点续跑参数
    parser.add_argument("--resume", type=str, default=None,
                       help="续跑的运行ID (例如: run_20250724_123456)")
    
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
        'synthesis_method': args.synthesis_method,  # 新增：理论生成方法
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
        'use_instrument_correction': args.use_instrument_correction,
        # 文献概念相关配置
        'use_raw_literature': args.use_raw_literature,
        'literature_concepts_dir': args.literature_concepts_dir,
        'auto_extract_concepts': args.auto_extract_concepts
    }
    
    # 启动演进（支持续跑）
    orchestrator = CleanEvolutionOrchestrator(config, resume_run_id=args.resume)
    success = orchestrator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 