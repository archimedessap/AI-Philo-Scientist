#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
反馈循环模块
============
该模块实现了理论生成和评估的迭代优化机制，通过分析评估结果来指导下一轮的理论生成。

主要功能:
1. 分析评估结果，提取关键反馈信息
2. 根据反馈调整理论生成参数
3. 管理多轮迭代过程
4. 记录和追踪优化过程
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class FeedbackMetrics:
    """反馈指标数据类"""
    experiment_success_rate: float
    role_evaluation_score: float
    theory_coherence: float
    innovation_score: float
    overall_score: float

class FeedbackAnalyzer:
    """反馈分析器，用于分析评估结果并提取关键反馈"""
    
    def __init__(self, evaluation_dir: str):
        self.evaluation_dir = evaluation_dir
        self.logger = logging.getLogger(__name__)
        
    def analyze_evaluation_results(self) -> Dict[str, FeedbackMetrics]:
        """分析评估结果，返回每个理论的反馈指标"""
        results = {}
        
        # 读取实验评估结果
        experiment_results = self._read_experiment_results()
        # 读取角色评估结果
        role_results = self._read_role_evaluation_results()
        
        # 合并结果并计算综合指标
        for theory_name in set(experiment_results.keys()) | set(role_results.keys()):
            metrics = FeedbackMetrics(
                experiment_success_rate=experiment_results.get(theory_name, {}).get('success_rate', 0.0),
                role_evaluation_score=role_results.get(theory_name, {}).get('role_score', 0.0),
                theory_coherence=role_results.get(theory_name, {}).get('coherence', 0.0),
                innovation_score=role_results.get(theory_name, {}).get('innovation', 0.0),
                overall_score=0.0  # 将在下面计算
            )
            
            # 计算综合得分
            metrics.overall_score = self._calculate_overall_score(metrics)
            results[theory_name] = metrics
            
        return results
    
    def _read_experiment_results(self) -> Dict[str, Dict]:
        """读取实验评估结果"""
        results = {}
        experiment_file = os.path.join(self.evaluation_dir, "experiment_results.json")
        
        if os.path.exists(experiment_file):
            with open(experiment_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for theory_name, metrics in data.items():
                    results[theory_name] = {
                        'success_rate': metrics.get('success_rate', 0.0)
                    }
        
        return results
    
    def _read_role_evaluation_results(self) -> Dict[str, Dict]:
        """读取角色评估结果"""
        results = {}
        role_file = os.path.join(self.evaluation_dir, "role_evaluation_results.json")
        
        if os.path.exists(role_file):
            with open(role_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for theory_name, metrics in data.items():
                    results[theory_name] = {
                        'role_score': metrics.get('role_score', 0.0),
                        'coherence': metrics.get('coherence', 0.0),
                        'innovation': metrics.get('innovation', 0.0)
                    }
        
        return results
    
    def _calculate_overall_score(self, metrics: FeedbackMetrics) -> float:
        """计算综合得分"""
        weights = {
            'experiment_success_rate': 0.4,
            'role_evaluation_score': 0.3,
            'theory_coherence': 0.2,
            'innovation_score': 0.1
        }
        
        return (
            metrics.experiment_success_rate * weights['experiment_success_rate'] +
            metrics.role_evaluation_score * weights['role_evaluation_score'] +
            metrics.theory_coherence * weights['theory_coherence'] +
            metrics.innovation_score * weights['innovation_score']
        )

class FeedbackLoopManager:
    """反馈循环管理器，用于管理多轮迭代过程"""
    
    def __init__(self, base_dir: str, max_iterations: int = 3):
        self.base_dir = base_dir
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.logger = logging.getLogger(__name__)
        
        # 创建迭代历史记录目录
        self.history_dir = os.path.join(base_dir, "feedback_history")
        os.makedirs(self.history_dir, exist_ok=True)
        
    def start_new_iteration(self) -> Tuple[str, Dict]:
        """开始新的迭代，返回迭代目录和参数"""
        if self.current_iteration >= self.max_iterations:
            raise ValueError(f"已达到最大迭代次数 ({self.max_iterations})")
            
        self.current_iteration += 1
        iteration_dir = os.path.join(self.base_dir, f"iteration_{self.current_iteration}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        # 获取上一轮的反馈（如果有）
        feedback_params = self._get_feedback_parameters()
        
        return iteration_dir, feedback_params
    
    def record_iteration_results(self, iteration_dir: str, feedback_metrics: Dict[str, FeedbackMetrics]):
        """记录当前迭代的结果"""
        # 保存反馈指标
        metrics_file = os.path.join(self.history_dir, f"iteration_{self.current_iteration}_metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump({
                theory_name: {
                    'experiment_success_rate': metrics.experiment_success_rate,
                    'role_evaluation_score': metrics.role_evaluation_score,
                    'theory_coherence': metrics.theory_coherence,
                    'innovation_score': metrics.innovation_score,
                    'overall_score': metrics.overall_score
                }
                for theory_name, metrics in feedback_metrics.items()
            }, f, indent=2, ensure_ascii=False)
        
        # 更新历史记录
        self._update_history(iteration_dir, feedback_metrics)
    
    def _get_feedback_parameters(self) -> Dict:
        """根据历史反馈生成新的参数"""
        if self.current_iteration == 1:
            return {
                'max_pairs_to_analyze': 20,
                'variants_per_contradiction': 3,
                'synthesis_model_name': "gemini-1.5-pro-latest",
                'evaluation_model_name': "deepseek-reasoner"
            }
            
        # 读取上一轮的反馈
        prev_metrics_file = os.path.join(self.history_dir, f"iteration_{self.current_iteration-1}_metrics.json")
        if not os.path.exists(prev_metrics_file):
            return self._get_feedback_parameters()
            
        with open(prev_metrics_file, 'r', encoding='utf-8') as f:
            prev_metrics = json.load(f)
            
        # 分析上一轮的结果并调整参数
        return self._adjust_parameters(prev_metrics)
    
    def _adjust_parameters(self, prev_metrics: Dict) -> Dict:
        """根据上一轮的结果调整参数"""
        # 计算平均得分
        avg_scores = {
            'experiment_success_rate': np.mean([m['experiment_success_rate'] for m in prev_metrics.values()]),
            'role_evaluation_score': np.mean([m['role_evaluation_score'] for m in prev_metrics.values()]),
            'theory_coherence': np.mean([m['theory_coherence'] for m in prev_metrics.values()]),
            'innovation_score': np.mean([m['innovation_score'] for m in prev_metrics.values()])
        }
        
        # 根据得分调整参数
        params = {
            'max_pairs_to_analyze': 20,
            'variants_per_contradiction': 3,
            'synthesis_model_name': "gemini-1.5-pro-latest",
            'evaluation_model_name': "deepseek-reasoner"
        }
        
        # 如果实验成功率低，增加变体数量
        if avg_scores['experiment_success_rate'] < 0.6:
            params['variants_per_contradiction'] = 5
            
        # 如果创新性得分低，增加分析的理论对数量
        if avg_scores['innovation_score'] < 0.5:
            params['max_pairs_to_analyze'] = 30
            
        return params
    
    def _update_history(self, iteration_dir: str, feedback_metrics: Dict[str, FeedbackMetrics]):
        """更新历史记录"""
        history_file = os.path.join(self.history_dir, "iteration_history.json")
        
        # 读取现有历史记录
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
            
        # 添加新的迭代记录
        history.append({
            'iteration': self.current_iteration,
            'directory': iteration_dir,
            'metrics': {
                theory_name: {
                    'experiment_success_rate': metrics.experiment_success_rate,
                    'role_evaluation_score': metrics.role_evaluation_score,
                    'theory_coherence': metrics.theory_coherence,
                    'innovation_score': metrics.innovation_score,
                    'overall_score': metrics.overall_score
                }
                for theory_name, metrics in feedback_metrics.items()
            }
        })
        
        # 保存更新后的历史记录
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False) 