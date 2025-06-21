#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robust_evolution_runner.py
===========================
健壮的多代理论演进运行器，具有完整的断点续跑和容错能力

特性:
1. 自动检测中断状态并恢复
2. 分阶段执行，每阶段可独立重试
3. 网络中断容错
4. 完整的状态追踪
5. 智能缓存机制

用法:
python robust_evolution_runner.py --config config.yaml
python robust_evolution_runner.py --resume output_clean_evolution/run_20250620_160034
"""

import json
import argparse
import subprocess
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('robust_runner.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class RobustEvolutionRunner:
    """健壮的演进运行器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.run_dir = None
        self.manifest = None
        
    def detect_state(self, run_dir: str) -> Dict:
        """检测运行状态"""
        run_path = Path(run_dir)
        state = {
            "run_dir": run_dir,
            "exists": run_path.exists(),
            "phases_completed": [],
            "current_generation": 0,
            "total_theories": 0,
            "evaluated_theories": 0,
            "can_resume": False,
            "max_generations": self.config.get("max_generations", 3)
        }
        
        if not run_path.exists():
            return state
            
        # 检查清单文件
        manifest_path = run_path / "run_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                state["total_theories"] = len(manifest.get("theories", {}))
                state["can_resume"] = True
                self.manifest = manifest
            except Exception as e:
                self.logger.warning(f"无法读取清单文件: {e}")
                
        # 检查各代的完成状态
        for gen_dir in sorted(run_path.glob("generation_*")):
            gen_num = int(gen_dir.name.split("_")[1])
            state["current_generation"] = max(state["current_generation"], gen_num)
            
            # 检查合成阶段
            if (gen_dir / "synthesis").exists():
                synthesis_theories = list((gen_dir / "synthesis").glob("**/eval_ready_theories/*.json"))
                if synthesis_theories:
                    state["phases_completed"].append(f"gen{gen_num}_synthesis")
                    
            # 检查评估阶段
            eval_dir = gen_dir / "evaluation"
            if eval_dir.exists():
                # 查找评估结果文件
                results_files = list(eval_dir.glob("**/combined_rankings.json"))
                if results_files:
                    state["phases_completed"].append(f"gen{gen_num}_evaluation")
                    
            # 检查精炼阶段
            refine_dir = gen_dir / "refinement"
            if refine_dir.exists():
                refinement_outputs = list(refine_dir.glob("**/improved_*.json"))
                if refinement_outputs:
                    state["phases_completed"].append(f"gen{gen_num}_refinement")
                    
        return state
        
    def resume_from_state(self, run_dir: str) -> bool:
        """从检测到的状态恢复运行"""
        state = self.detect_state(run_dir)
        self.logger.info(f"检测到运行状态: {state}")
        
        if not state["can_resume"]:
            self.logger.error("无法恢复：清单文件不存在或损坏")
            return False
        
        self.run_dir = run_dir
        completed_phases = state["phases_completed"]
        current_gen = state["current_generation"]
        max_gen = state["max_generations"]
        
        # 按代际顺序执行各个阶段
        for generation in range(0, max_gen + 1):
            
            # Generation 0: 合成 -> 评估
            if generation == 0:
                if f"gen{generation}_synthesis" not in completed_phases:
                    self.logger.info(f"🧬 继续第{generation}代：理论合成")
                    if not self._run_synthesis(run_dir, generation):
                        return False
                    completed_phases.append(f"gen{generation}_synthesis")
                        
                if f"gen{generation}_evaluation" not in completed_phases:
                    self.logger.info(f"📊 继续第{generation}代：理论评估")
                    if not self._run_evaluation(run_dir, generation):
                        return False
                    completed_phases.append(f"gen{generation}_evaluation")
                    
            # Generation 1+: 精炼 -> 评估
            else:
                # 检查是否还有理论可以晋级
                if not self._should_continue_to_generation(run_dir, generation):
                    self.logger.info(f"✅ 没有理论符合第{generation}代晋级条件，演进完成")
                    break
                    
                if f"gen{generation}_refinement" not in completed_phases:
                    self.logger.info(f"🔧 继续第{generation}代：理论精炼")
                    if not self._run_refinement(run_dir, generation):
                        return False
                    completed_phases.append(f"gen{generation}_refinement")
                        
                if f"gen{generation}_evaluation" not in completed_phases:
                    self.logger.info(f"📊 继续第{generation}代：理论评估")
                    if not self._run_evaluation(run_dir, generation):
                        return False
                    completed_phases.append(f"gen{generation}_evaluation")
                    
        self.logger.info("🎉 所有代际演进完成！")
        return True
        
    def _should_continue_to_generation(self, run_dir: str, target_generation: int) -> bool:
        """检查是否应该继续到指定代"""
        if target_generation == 0:
            return True
            
        # 加载清单并检查前一代的晋级理论
        import manifest_tools
        manifest_path = Path(run_dir) / "run_manifest.json"
        manifest = manifest_tools.load_manifest(manifest_path)
        
        # 获取前一代有分数的理论
        prev_gen_theories = manifest_tools.get_theories_by_generation(manifest, target_generation - 1)
        promotion_threshold = self.config.get("promotion_min_score", 0.3)
        
        promotable_count = 0
        for theory_id, theory_info in prev_gen_theories.items():
            score = theory_info.get("scores", {}).get("combined_score")
            if score is not None and score >= promotion_threshold:
                promotable_count += 1
                
        self.logger.info(f"第{target_generation-1}代有 {promotable_count} 个理论达到晋级阈值 {promotion_threshold}")
        return promotable_count > 0
        
    def _run_synthesis(self, run_dir: str, generation: int) -> bool:
        """运行理论合成阶段"""
        self.logger.info(f"🧬 运行第{generation}代理论合成...")
        
        try:
            cmd = [
                "python", "run_direct_synthesis.py",
                "--initial_theories_dir", self.config["initial_theories_dir"],
                "--output_dir", f"{run_dir}/generation_{generation}/synthesis",
                "--max_pairs_to_analyze", str(self.config.get("max_pairs_to_analyze", 8)),
                "--variants_per_contradiction", str(self.config.get("variants_per_contradiction", 1)),
                "--model_source", self.config.get("synthesis_model_source", "google"),
                "--model_name", self.config.get("synthesis_model_name", "gemini-2.5-pro")
            ]
            
            result = self._run_with_retry(cmd, max_retries=3, stage_name="理论合成")
            if result:
                self._register_synthesized_theories(run_dir, generation)
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"合成阶段失败: {e}")
            return False
            
    def _run_evaluation(self, run_dir: str, generation: int) -> bool:
        """运行理论评估阶段"""
        self.logger.info(f"📊 运行第{generation}代理论评估...")
        
        try:
            theory_dir = f"{run_dir}/generation_{generation}/evaluation/theories"
            output_dir = f"{run_dir}/generation_{generation}/evaluation/results"
            
            # 确保理论目录存在并准备理论文件
            if not self._prepare_theories_for_evaluation(run_dir, generation):
                self.logger.error("准备评估理论失败")
                return False
                
            cmd = [
                "python", "demo/demo_1.py",
                "--theory_path", theory_dir,
                "--experiment_dir", self.config["experiment_dir"],
                "--output_dir", output_dir,
                "--model_source", self.config.get("evaluation_model_source", "google"),
                "--model_name", self.config.get("evaluation_model_name", "gemini-1.5-pro"),
                "--run_role_evaluation",
                "--role_success_threshold", str(self.config.get("role_success_threshold", 0.1)),
                "--use_instrument_correction"
            ]
            
            result = self._run_with_retry(cmd, max_retries=2)
            if result:
                self._update_manifest_with_evaluation(run_dir, generation)
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"评估阶段失败: {e}")
            return False
            
    def _run_refinement(self, run_dir: str, generation: int) -> bool:
        """运行理论精炼阶段"""
        self.logger.info(f"🔧 运行第{generation}代理论精炼...")
        
        try:
            cmd = [
                "python", "run_m3_auto_refinement.py",
                "--run_dir", run_dir,
                "--target_generation", str(generation),
                "--model_source", self.config.get("dialog_model_source", "google"),
                "--model_name", self.config.get("dialog_model_name", "gemini-2.5-pro"),
                "--max_iters", str(self.config.get("max_refinement_iters", 3)),
                "--min_improvement", str(self.config.get("min_improvement", 0.02))
            ]
            
            result = self._run_with_retry(cmd, max_retries=2)
            if result:
                self._register_refined_theories(run_dir, generation)
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"精炼阶段失败: {e}")
            return False
            
    def _run_with_retry(self, cmd: List[str], max_retries: int = 3, stage_name: str = "") -> bool:
        """带重试机制的命令执行"""
        
        # 根据阶段设置不同的超时时间
        timeout_settings = {
            "synthesis": 7200,      # 2小时 - 合成
            "evaluation": 14400,    # 4小时 - 评估  
            "refinement": 28800,    # 8小时 - 精炼（最耗时）
        }
        
        # 默认超时时间
        timeout = 14400  # 4小时
        
        # 根据命令或阶段名称选择合适的超时时间
        cmd_str = " ".join(cmd).lower()
        if "synthesis" in cmd_str or "合成" in stage_name:
            timeout = timeout_settings["synthesis"]
        elif "evaluation" in cmd_str or "demo_1.py" in cmd_str or "评估" in stage_name:
            timeout = timeout_settings["evaluation"] 
        elif "refinement" in cmd_str or "run_m3_auto_refinement" in cmd_str or "精炼" in stage_name:
            timeout = timeout_settings["refinement"]
        
        for attempt in range(1, max_retries + 1):
            self.logger.info(f"尝试 {attempt}/{max_retries}: {' '.join(cmd)}")
            self.logger.info(f"⏰ 超时设置: {timeout//60} 分钟")
            
            try:
                result = subprocess.run(
                    cmd, 
                    check=True, 
                    capture_output=True, 
                    text=True,
                    timeout=timeout
                )
                self.logger.info(f"✅ 命令执行成功")
                return True
                
            except subprocess.TimeoutExpired:
                self.logger.warning(f"⏰ 命令超时 ({timeout//60} 分钟)，尝试 {attempt}/{max_retries}")
                if attempt < max_retries:
                    time.sleep(60)  # 等待1分钟后重试
                    
            except subprocess.CalledProcessError as e:
                self.logger.error(f"❌ 命令失败，退出代码: {e.returncode}")
                self.logger.error(f"输出: {e.stdout}")
                self.logger.error(f"错误: {e.stderr}")
                if attempt < max_retries:
                    time.sleep(30)  # 等待30秒后重试
                    
            except Exception as e:
                self.logger.error(f"💥 未预期的错误: {e}")
                if attempt < max_retries:
                    time.sleep(30)
                    
        return False
        
    def _register_synthesized_theories(self, run_dir: str, generation: int):
        """注册合成的理论到清单"""
        import manifest_tools
        manifest_path = Path(run_dir) / "run_manifest.json"
        synthesis_dir = Path(run_dir) / f"generation_{generation}" / "synthesis"
        
        # 查找合成的理论文件
        theory_files = list(synthesis_dir.glob("**/eval_ready_theories/*.json"))
        
        manifest = manifest_tools.load_manifest(manifest_path)
        for theory_file in theory_files:
            manifest_tools.register_theory_from_file(manifest, str(theory_file), generation)
        manifest_tools.save_manifest(manifest, manifest_path)
        
        self.logger.info(f"✅ 已注册 {len(theory_files)} 个第{generation}代理论")
        
    def _register_refined_theories(self, run_dir: str, generation: int):
        """注册精炼的理论到清单"""
        import manifest_tools
        manifest_path = Path(run_dir) / "run_manifest.json"
        refinement_dir = Path(run_dir) / f"generation_{generation}" / "refinement"
        
        # 查找精炼的理论文件
        theory_files = list(refinement_dir.glob("**/improved_*.json"))
        
        manifest = manifest_tools.load_manifest(manifest_path)
        for theory_file in theory_files:
            manifest_tools.register_theory_from_file(manifest, str(theory_file), generation)
        manifest_tools.save_manifest(manifest, manifest_path)
        
        self.logger.info(f"✅ 已注册 {len(theory_files)} 个第{generation}代精炼理论")
        
    def _prepare_theories_for_evaluation(self, run_dir: str, generation: int) -> bool:
        """为评估准备理论文件"""
        import manifest_tools
        manifest_path = Path(run_dir) / "run_manifest.json"
        theory_dir = Path(run_dir) / f"generation_{generation}" / "evaluation" / "theories"
        theory_dir.mkdir(parents=True, exist_ok=True)
        
        manifest = manifest_tools.load_manifest(manifest_path)
        theories = manifest_tools.get_theories_by_generation(manifest, generation)
        
        if not theories:
            self.logger.error(f"第{generation}代没有找到理论")
            return False
        
        copied_count = 0
        for theory_id, theory_info in theories.items():
            src_path = Path(theory_info["file_path"])
            if src_path.exists():
                # 复制到评估目录
                theory_name = theory_info['theory_name'].replace(' ', '_').replace('(', '').replace(')', '').lower()
                dst_path = theory_dir / f"{theory_name}.json"
                import shutil
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            else:
                self.logger.warning(f"理论文件不存在: {src_path}")
                
        self.logger.info(f"✅ 已准备 {copied_count} 个理论用于第{generation}代评估")
        return copied_count > 0
        
    def _update_manifest_with_evaluation(self, run_dir: str, generation: int):
        """用评估结果更新清单"""
        import manifest_tools
        manifest_path = Path(run_dir) / "run_manifest.json"
        results_dir = Path(run_dir) / f"generation_{generation}" / "evaluation" / "results"
        
        # 查找最新的评估结果
        ranking_files = list(results_dir.glob("**/combined_rankings.json"))
        if ranking_files:
            latest_file = max(ranking_files, key=lambda p: p.stat().st_mtime)
            manifest = manifest_tools.load_manifest(manifest_path)
            manifest_tools.update_manifest_with_evaluation(manifest, str(latest_file))
            manifest_tools.save_manifest(manifest, manifest_path)
            self.logger.info(f"✅ 已更新清单，评估结果来自: {latest_file}")
        else:
            self.logger.warning(f"未找到第{generation}代的评估结果文件")

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    # 默认配置
    return {
        "initial_theories_dir": "data/theories_v2.1",
        "experiment_dir": "demo/experiments",
        "max_generations": 3,
        "promotion_min_score": 0.3,
        "max_pairs_to_analyze": 8,
        "variants_per_contradiction": 1,
        "synthesis_model_source": "google",
        "synthesis_model_name": "gemini-2.5-pro",
        "evaluation_model_source": "google", 
        "evaluation_model_name": "gemini-1.5-pro",
        "dialog_model_source": "google",
        "dialog_model_name": "gemini-2.5-pro",
        "max_refinement_iters": 3,
        "min_improvement": 0.02,
        "role_success_threshold": 0.1
    }

def main():
    parser = argparse.ArgumentParser(description="健壮的演进运行器")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--resume", type=str, help="恢复指定运行目录")
    parser.add_argument("--new_run", action="store_true", help="开始新的运行")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    runner = RobustEvolutionRunner(config)
    
    try:
        if args.resume:
            # 恢复模式
            success = runner.resume_from_state(args.resume)
            if success:
                print("🎉 恢复运行成功完成！")
            else:
                print("❌ 恢复运行失败")
                sys.exit(1)
                
        elif args.new_run:
            # 新运行模式
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_dir = f"output_clean_evolution/run_{timestamp}"
            
            # 使用 run_clean_evolution.py 开始新运行
            cmd_args = []
            for k, v in config.items():
                if k in ["initial_theories_dir", "experiment_dir", "max_generations", 
                        "promotion_min_score", "max_pairs_to_analyze", "variants_per_contradiction"]:
                    arg_name = k.replace('_', '-') if not k.endswith('_dir') else k
                    cmd_args.extend([f"--{arg_name}", str(v)])
            
            cmd = ["python", "run_clean_evolution.py"] + cmd_args
            
            success = runner._run_with_retry(cmd, max_retries=1)
            if success:
                print(f"🎉 新运行启动成功: {run_dir}")
            else:
                print("❌ 新运行启动失败")
                sys.exit(1)
        else:
            print("请指定 --resume <run_dir> 或 --new_run")
            
    except KeyboardInterrupt:
        print("\n⚠️ 运行被用户中断")
        sys.exit(130)
    except Exception as e:
        logging.error(f"💥 未预期的错误: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 