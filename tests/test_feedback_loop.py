#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pytest
from tools.feedback_loop import FeedbackAnalyzer, FeedbackLoopManager, FeedbackMetrics

@pytest.fixture
def test_dir(tmp_path):
    """创建测试目录结构"""
    # 创建评估结果目录
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    
    # 创建实验评估结果
    experiment_results = {
        "theory_1": {"success_rate": 0.8},
        "theory_2": {"success_rate": 0.6}
    }
    with open(eval_dir / "experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(experiment_results, f)
    
    # 创建角色评估结果
    role_results = {
        "theory_1": {
            "role_score": 0.9,
            "coherence": 0.85,
            "innovation": 0.75
        },
        "theory_2": {
            "role_score": 0.7,
            "coherence": 0.65,
            "innovation": 0.8
        }
    }
    with open(eval_dir / "role_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(role_results, f)
    
    return eval_dir

def test_feedback_analyzer(test_dir):
    """测试反馈分析器"""
    analyzer = FeedbackAnalyzer(str(test_dir))
    results = analyzer.analyze_evaluation_results()
    
    # 验证结果
    assert "theory_1" in results
    assert "theory_2" in results
    
    # 验证指标计算
    theory1_metrics = results["theory_1"]
    assert theory1_metrics.experiment_success_rate == 0.8
    assert theory1_metrics.role_evaluation_score == 0.9
    assert theory1_metrics.theory_coherence == 0.85
    assert theory1_metrics.innovation_score == 0.75
    assert 0 <= theory1_metrics.overall_score <= 1

def test_feedback_loop_manager(tmp_path):
    """测试反馈循环管理器"""
    manager = FeedbackLoopManager(str(tmp_path), max_iterations=2)
    
    # 测试第一轮迭代
    iteration_dir, params = manager.start_new_iteration()
    assert os.path.exists(iteration_dir)
    assert params["max_pairs_to_analyze"] == 20
    assert params["variants_per_contradiction"] == 3
    
    # 模拟第一轮结果
    feedback_metrics = {
        "theory_1": FeedbackMetrics(
            experiment_success_rate=0.5,
            role_evaluation_score=0.6,
            theory_coherence=0.7,
            innovation_score=0.8,
            overall_score=0.65
        )
    }
    manager.record_iteration_results(iteration_dir, feedback_metrics)
    
    # 测试第二轮迭代
    iteration_dir, params = manager.start_new_iteration()
    assert os.path.exists(iteration_dir)
    # 由于第一轮实验成功率低，应该增加变体数量
    assert params["variants_per_contradiction"] == 5
    
    # 测试达到最大迭代次数
    with pytest.raises(ValueError):
        manager.start_new_iteration() 