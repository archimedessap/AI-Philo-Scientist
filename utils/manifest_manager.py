#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行清单管理器

管理run_manifest.json的读写和更新，特别是评估分数的自动回写
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading

from utils.logging_config import get_logger, log_error_with_context


class ManifestManager:
    """运行清单管理器"""
    
    def __init__(self, manifest_path: str):
        """
        初始化清单管理器
        
        Args:
            manifest_path: run_manifest.json文件路径
        """
        self.logger = get_logger('manifest_manager')
        self.manifest_path = Path(manifest_path)
        self.lock = threading.Lock()  # 文件锁，防止并发写入
        
        # 确保清单文件存在
        if not self.manifest_path.exists():
            self.logger.warning(f"清单文件不存在，创建新文件: {manifest_path}")
            self._create_default_manifest()
        
        self.logger.info(f"初始化清单管理器: {manifest_path}")
    
    def _create_default_manifest(self):
        """创建默认清单文件"""
        default_manifest = {
            "run_id": f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "config": {},
            "theories": {},
            "lineage": {},
            "generations": {}
        }
        
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(default_manifest, f, ensure_ascii=False, indent=2)
    
    def read_manifest(self) -> Dict[str, Any]:
        """读取清单文件"""
        try:
            with self.lock:
                with open(self.manifest_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"读取清单文件失败: {e}")
            log_error_with_context(e, {
                'manifest_path': str(self.manifest_path)
            })
            return {}
    
    def write_manifest(self, data: Dict[str, Any]):
        """写入清单文件"""
        try:
            with self.lock:
                # 先备份现有文件
                if self.manifest_path.exists():
                    backup_path = self.manifest_path.with_suffix('.json.bak')
                    self.manifest_path.rename(backup_path)
                
                # 写入新数据
                with open(self.manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                # 删除备份
                if backup_path.exists():
                    backup_path.unlink()
                    
                self.logger.debug(f"清单文件已更新: {self.manifest_path}")
                
        except Exception as e:
            self.logger.error(f"写入清单文件失败: {e}")
            # 恢复备份
            if backup_path.exists():
                backup_path.rename(self.manifest_path)
            raise
    
    def update_theory_scores(self, theory_id: str, scores: Dict[str, float], 
                           final_score: Optional[float] = None,
                           eval_summary_path: Optional[str] = None):
        """
        更新理论的评估分数
        
        Args:
            theory_id: 理论ID
            scores: 各项评分
            final_score: 最终综合分数
            eval_summary_path: 评估摘要文件路径
        """
        manifest = self.read_manifest()
        
        if theory_id not in manifest.get('theories', {}):
            self.logger.warning(f"理论 {theory_id} 不在清单中")
            return
        
        # 更新分数
        manifest['theories'][theory_id]['scores'] = scores
        
        if final_score is not None:
            manifest['theories'][theory_id]['score'] = final_score
        
        if eval_summary_path:
            manifest['theories'][theory_id]['eval_summary_path'] = eval_summary_path
        
        # 添加更新时间戳
        manifest['theories'][theory_id]['scores_updated_at'] = datetime.now().isoformat()
        
        self.write_manifest(manifest)
        self.logger.info(f"已更新理论 {theory_id} 的评估分数")
    
    def batch_update_scores(self, score_updates: List[Dict[str, Any]]):
        """
        批量更新多个理论的分数
        
        Args:
            score_updates: 分数更新列表，每项包含:
                - theory_id: 理论ID
                - scores: 各项评分
                - final_score: 最终分数（可选）
                - eval_summary_path: 评估摘要路径（可选）
        """
        manifest = self.read_manifest()
        updated_count = 0
        
        for update in score_updates:
            theory_id = update.get('theory_id')
            if not theory_id or theory_id not in manifest.get('theories', {}):
                self.logger.warning(f"跳过无效理论ID: {theory_id}")
                continue
            
            # 更新分数
            theory_data = manifest['theories'][theory_id]
            
            if 'scores' in update:
                theory_data['scores'] = update['scores']
            
            if 'final_score' in update:
                theory_data['score'] = update['final_score']
            
            if 'eval_summary_path' in update:
                theory_data['eval_summary_path'] = update['eval_summary_path']
            
            theory_data['scores_updated_at'] = datetime.now().isoformat()
            updated_count += 1
        
        if updated_count > 0:
            self.write_manifest(manifest)
            self.logger.info(f"批量更新了 {updated_count} 个理论的评估分数")
    
    def add_theory(self, theory_id: str, theory_data: Dict[str, Any]):
        """添加新理论到清单"""
        manifest = self.read_manifest()
        
        if 'theories' not in manifest:
            manifest['theories'] = {}
        
        manifest['theories'][theory_id] = {
            'theory_name': theory_data.get('name', 'Unknown'),
            'file_path': theory_data.get('file_path', ''),
            'generation': theory_data.get('generation', 0),
            'status': theory_data.get('status', 'created'),
            'scores': {},
            'created_at': datetime.now().isoformat()
        }
        
        self.write_manifest(manifest)
        self.logger.info(f"已添加新理论到清单: {theory_id}")
    
    def update_generation_info(self, generation: int, info: Dict[str, Any]):
        """更新代信息"""
        manifest = self.read_manifest()
        
        if 'generations' not in manifest:
            manifest['generations'] = {}
        
        if str(generation) not in manifest['generations']:
            manifest['generations'][str(generation)] = {}
        
        manifest['generations'][str(generation)].update(info)
        manifest['generations'][str(generation)]['updated_at'] = datetime.now().isoformat()
        
        self.write_manifest(manifest)
        self.logger.debug(f"已更新第 {generation} 代信息")
    
    def get_theory_info(self, theory_id: str) -> Optional[Dict[str, Any]]:
        """获取特定理论的信息"""
        manifest = self.read_manifest()
        return manifest.get('theories', {}).get(theory_id)
    
    def get_unscored_theories(self) -> List[str]:
        """获取尚未评分的理论ID列表"""
        manifest = self.read_manifest()
        unscored = []
        
        for theory_id, theory_data in manifest.get('theories', {}).items():
            if not theory_data.get('scores') or not theory_data.get('score'):
                unscored.append(theory_id)
        
        return unscored
    
    def mark_theory_status(self, theory_id: str, status: str):
        """
        标记理论状态
        
        Args:
            theory_id: 理论ID
            status: 状态 (created, evaluating, evaluated, promoted, rejected, error)
        """
        manifest = self.read_manifest()
        
        if theory_id in manifest.get('theories', {}):
            manifest['theories'][theory_id]['status'] = status
            manifest['theories'][theory_id]['status_updated_at'] = datetime.now().isoformat()
            self.write_manifest(manifest)
            self.logger.debug(f"理论 {theory_id} 状态更新为: {status}")


# 便捷函数
def update_scores_from_evaluation(manifest_path: str, evaluation_results: Dict[str, Any]):
    """
    从评估结果更新清单中的分数
    
    Args:
        manifest_path: 清单文件路径
        evaluation_results: 评估结果，格式：
            {
                'theory_id': {
                    'scores': {...},
                    'final_score': float,
                    'eval_summary_path': str
                },
                ...
            }
    """
    manager = ManifestManager(manifest_path)
    
    score_updates = []
    for theory_id, results in evaluation_results.items():
        score_updates.append({
            'theory_id': theory_id,
            'scores': results.get('scores', {}),
            'final_score': results.get('final_score'),
            'eval_summary_path': results.get('eval_summary_path')
        })
    
    manager.batch_update_scores(score_updates)