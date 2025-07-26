#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
断点续跑管理器

提供运行状态的保存和恢复功能，支持从中断处继续执行
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List, Callable
from dataclasses import dataclass, asdict
import traceback

from utils.logging_config import get_logger
from utils.manifest_manager import ManifestManager


@dataclass
class CheckpointData:
    """检查点数据"""
    checkpoint_id: str
    stage: str
    step: int
    total_steps: int
    data: Dict[str, Any]
    created_at: str
    error_info: Optional[Dict[str, str]] = None


class CheckpointManager:
    """断点续跑管理器"""
    
    def __init__(self, run_id: str, checkpoint_dir: str = "checkpoints"):
        """
        初始化断点管理器
        
        Args:
            run_id: 运行ID
            checkpoint_dir: 检查点目录
        """
        self.logger = get_logger('checkpoint_manager')
        self.run_id = run_id
        self.checkpoint_dir = Path(checkpoint_dir) / run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查点文件路径
        self.state_file = self.checkpoint_dir / "run_state.json"
        self.data_file = self.checkpoint_dir / "checkpoint_data.pkl"
        
        # 运行状态
        self.current_stage = None
        self.current_step = 0
        self.total_steps = 0
        self.checkpoints: List[CheckpointData] = []
        
        self.logger.info(f"初始化断点管理器: {run_id}")
    
    def save_checkpoint(
        self,
        stage: str,
        step: int,
        total_steps: int,
        data: Dict[str, Any],
        error_info: Optional[Dict[str, str]] = None
    ) -> str:
        """
        保存检查点
        
        Args:
            stage: 当前阶段
            step: 当前步骤
            total_steps: 总步骤数
            data: 要保存的数据
            error_info: 错误信息（如果有）
            
        Returns:
            检查点ID
        """
        checkpoint_id = f"{stage}_{step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        checkpoint = CheckpointData(
            checkpoint_id=checkpoint_id,
            stage=stage,
            step=step,
            total_steps=total_steps,
            data=data,
            created_at=datetime.now().isoformat(),
            error_info=error_info
        )
        
        self.checkpoints.append(checkpoint)
        self.current_stage = stage
        self.current_step = step
        self.total_steps = total_steps
        
        # 保存状态
        self._save_state()
        
        self.logger.info(f"保存检查点: {checkpoint_id} (阶段: {stage}, 步骤: {step}/{total_steps})")
        
        return checkpoint_id
    
    def _save_state(self):
        """保存运行状态"""
        state = {
            'run_id': self.run_id,
            'current_stage': self.current_stage,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'checkpoints': [asdict(cp) for cp in self.checkpoints],
            'last_updated': datetime.now().isoformat()
        }
        
        # 保存JSON状态
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        # 保存二进制数据
        with open(self.data_file, 'wb') as f:
            pickle.dump(self.checkpoints, f)
    
    def load_checkpoint(self, checkpoint_id: Optional[str] = None) -> Optional[CheckpointData]:
        """
        加载检查点
        
        Args:
            checkpoint_id: 检查点ID，如果为None则加载最新的
            
        Returns:
            检查点数据
        """
        if not self.state_file.exists():
            self.logger.warning("未找到检查点状态文件")
            return None
        
        try:
            # 加载状态
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # 加载检查点数据
            if self.data_file.exists():
                with open(self.data_file, 'rb') as f:
                    self.checkpoints = pickle.load(f)
            else:
                # 从JSON恢复
                self.checkpoints = [
                    CheckpointData(**cp_data) 
                    for cp_data in state.get('checkpoints', [])
                ]
            
            self.current_stage = state.get('current_stage')
            self.current_step = state.get('current_step', 0)
            self.total_steps = state.get('total_steps', 0)
            
            # 查找指定的检查点
            if checkpoint_id:
                for cp in self.checkpoints:
                    if cp.checkpoint_id == checkpoint_id:
                        self.logger.info(f"加载检查点: {checkpoint_id}")
                        return cp
                self.logger.warning(f"未找到检查点: {checkpoint_id}")
                return None
            
            # 返回最新的检查点
            if self.checkpoints:
                latest = self.checkpoints[-1]
                self.logger.info(f"加载最新检查点: {latest.checkpoint_id}")
                return latest
            
            return None
            
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            return None
    
    def can_resume(self) -> bool:
        """检查是否可以续跑"""
        return self.state_file.exists() and len(self.checkpoints) > 0
    
    def get_resume_info(self) -> Optional[Dict[str, Any]]:
        """获取续跑信息"""
        if not self.can_resume():
            return None
        
        latest = self.load_checkpoint()
        if not latest:
            return None
        
        return {
            'checkpoint_id': latest.checkpoint_id,
            'stage': latest.stage,
            'step': latest.step,
            'total_steps': latest.total_steps,
            'created_at': latest.created_at,
            'has_error': latest.error_info is not None,
            'error_message': latest.error_info.get('message') if latest.error_info else None
        }
    
    def clear_checkpoints(self):
        """清除所有检查点"""
        import shutil
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints = []
        self.current_stage = None
        self.current_step = 0
        self.total_steps = 0
        
        self.logger.info("已清除所有检查点")


class ResumableRunner:
    """可续跑的任务运行器"""
    
    def __init__(self, run_id: str, manifest_path: Optional[str] = None):
        """
        初始化可续跑运行器
        
        Args:
            run_id: 运行ID
            manifest_path: 清单文件路径
        """
        self.logger = get_logger('resumable_runner')
        self.run_id = run_id
        self.checkpoint_manager = CheckpointManager(run_id)
        self.manifest_manager = ManifestManager(manifest_path) if manifest_path else None
        
        # 任务定义
        self.stages: Dict[str, List[Callable]] = {}
        self.stage_names: List[str] = []
        
    def add_stage(self, stage_name: str, tasks: List[Callable]):
        """
        添加执行阶段
        
        Args:
            stage_name: 阶段名称
            tasks: 任务函数列表
        """
        self.stages[stage_name] = tasks
        self.stage_names.append(stage_name)
        self.logger.debug(f"添加阶段: {stage_name} (包含 {len(tasks)} 个任务)")
    
    async def run(self, resume: bool = True) -> Dict[str, Any]:
        """
        运行任务
        
        Args:
            resume: 是否从断点续跑
            
        Returns:
            运行结果
        """
        results = {}
        start_stage_idx = 0
        start_task_idx = 0
        
        # 检查是否需要续跑
        if resume and self.checkpoint_manager.can_resume():
            resume_info = self.checkpoint_manager.get_resume_info()
            if resume_info:
                self.logger.info(f"从断点续跑: {resume_info['checkpoint_id']}")
                print(f"\n🔄 从断点续跑:")
                print(f"  阶段: {resume_info['stage']}")
                print(f"  进度: {resume_info['step']}/{resume_info['total_steps']}")
                
                # 加载检查点数据
                checkpoint = self.checkpoint_manager.load_checkpoint()
                if checkpoint:
                    results = checkpoint.data.get('results', {})
                    
                    # 找到续跑位置
                    if resume_info['stage'] in self.stage_names:
                        start_stage_idx = self.stage_names.index(resume_info['stage'])
                        start_task_idx = resume_info['step']
                        
                        # 如果是错误恢复，从当前任务重试
                        if resume_info['has_error']:
                            self.logger.info(f"从错误处重试: {resume_info['error_message']}")
                        else:
                            # 否则从下一个任务开始
                            start_task_idx += 1
                            
                            # 如果当前阶段已完成，移到下一阶段
                            if start_task_idx >= len(self.stages[resume_info['stage']]):
                                start_stage_idx += 1
                                start_task_idx = 0
        else:
            self.logger.info("开始全新运行")
            # 清除旧的检查点
            self.checkpoint_manager.clear_checkpoints()
        
        # 执行任务
        try:
            for stage_idx in range(start_stage_idx, len(self.stage_names)):
                stage_name = self.stage_names[stage_idx]
                tasks = self.stages[stage_name]
                
                # 确定起始任务
                task_start = start_task_idx if stage_idx == start_stage_idx else 0
                
                print(f"\n📌 执行阶段: {stage_name}")
                self.logger.info(f"开始执行阶段: {stage_name}")
                
                stage_results = results.get(stage_name, {})
                
                for task_idx in range(task_start, len(tasks)):
                    task = tasks[task_idx]
                    task_name = task.__name__
                    
                    try:
                        print(f"  ▶️ 任务 {task_idx + 1}/{len(tasks)}: {task_name}")
                        self.logger.info(f"执行任务: {task_name}")
                        
                        # 执行任务
                        if asyncio.iscoroutinefunction(task):
                            result = await task()
                        else:
                            result = task()
                        
                        # 保存结果
                        stage_results[task_name] = result
                        results[stage_name] = stage_results
                        
                        # 保存检查点
                        self.checkpoint_manager.save_checkpoint(
                            stage=stage_name,
                            step=task_idx,
                            total_steps=len(tasks),
                            data={'results': results}
                        )
                        
                        print(f"    ✅ 完成")
                        
                    except Exception as e:
                        error_msg = f"任务 {task_name} 失败: {str(e)}"
                        self.logger.error(error_msg)
                        
                        # 保存错误检查点
                        self.checkpoint_manager.save_checkpoint(
                            stage=stage_name,
                            step=task_idx,
                            total_steps=len(tasks),
                            data={'results': results},
                            error_info={
                                'message': error_msg,
                                'traceback': traceback.format_exc()
                            }
                        )
                        
                        print(f"    ❌ 失败: {str(e)}")
                        raise
                
                # 更新清单（如果有）
                if self.manifest_manager:
                    self.manifest_manager.update_generation_info(
                        generation=0,  # 需要根据实际情况调整
                        info={
                            f"{stage_name}_completed": True,
                            f"{stage_name}_completed_at": datetime.now().isoformat()
                        }
                    )
            
            # 运行完成
            self.logger.info("所有任务执行完成")
            print("\n✅ 所有任务执行完成！")
            
            # 清除检查点（可选）
            # self.checkpoint_manager.clear_checkpoints()
            
            return results
            
        except Exception as e:
            self.logger.error(f"运行失败: {e}")
            print(f"\n❌ 运行失败: {e}")
            print("\n💡 提示: 可以使用 --resume 参数从断点继续运行")
            raise


# 使用示例
if __name__ == "__main__":
    async def example_task_1():
        print("执行示例任务1")
        return {"status": "completed", "data": [1, 2, 3]}
    
    async def example_task_2():
        print("执行示例任务2")
        # 模拟可能失败的任务
        import random
        if random.random() < 0.3:
            raise Exception("模拟的随机错误")
        return {"status": "completed", "data": [4, 5, 6]}
    
    async def example_task_3():
        print("执行示例任务3")
        return {"status": "completed", "data": [7, 8, 9]}
    
    # 创建运行器
    runner = ResumableRunner("example_run_001")
    
    # 添加阶段
    runner.add_stage("initialization", [example_task_1])
    runner.add_stage("processing", [example_task_2, example_task_3])
    
    # 运行
    import asyncio
    results = asyncio.run(runner.run(resume=True))