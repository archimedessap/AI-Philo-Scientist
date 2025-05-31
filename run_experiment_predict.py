#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行实验预测评估
python run_experiment_predict.py --theory_file data/high_quality_theories
为生成的量子力学诠释理论运行实验预测，与关键实验进行比对。
"""

import os
import argparse
import json
from theory_validation.experimetal_validation.experiment_evaluator import ExperimentEvaluator
import glob

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] 创建目录: {directory}")

def main():
    parser = argparse.ArgumentParser(description="运行理论实验预测")
    
    # 输入参数
    parser.add_argument("--theory_file", type=str, required=False,
                      help="要评估的理论JSON文件路径")
    parser.add_argument("--experiments_file", type=str, 
                      default="theory_experiment/data/experiments.jsonl",
                      help="实验数据文件路径")
    parser.add_argument("--predictor_module", type=str, default=None,
                      help="自定义预测器模块路径 (可选)")
    parser.add_argument("--theory_dir", type=str, 
                      help="包含多个理论文件的目录路径，用于批量评估")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="data/experiment_evaluations",
                      help="输出目录")
    
    args = parser.parse_args()
    
    # 验证必要参数：至少需要提供theory_file或theory_dir中的一个
    if not args.theory_file and not args.theory_dir:
        parser.error("必须提供--theory_file或--theory_dir参数中的一个")
    
    # 确保输出目录存在
    ensure_directory_exists(args.output_dir)
    
    # 使用try/except包装导入，提供更好的错误信息
    try:
        from theory_validation.experimetal_validation.experiment_evaluator import ExperimentEvaluator
        # 尝试初始化评估器验证它是否正常工作
        evaluator = ExperimentEvaluator()
        print("[INFO] 成功导入实验评估器")
    except ImportError as e:
        print(f"[ERROR] 导入实验评估器失败: {str(e)}")
        print(f"[ERROR] 请检查文件路径和拼写: theory_validation/experimetal_validation/experiment_evaluator.py")
        # 提供友好的修复建议
        print("[INFO] 建议: 1. 检查拼写 (experimetal → experimental?)")
        print("[INFO]       2. 检查文件是否存在")
    
    # 单个文件处理
    if args.theory_file:
        try:
            with open(args.theory_file, 'r', encoding='utf-8') as f:
                theory = json.load(f)
            print(f"[INFO] 已加载理论: {theory.get('name', '未命名')}")

            # 评估理论
            print(f"\n[步骤1] 运行理论与实验比对")
            results = evaluator.evaluate_theory(theory, args.predictor_module)
            
            # 输出结果
            theory_name = results.get('theory_name', '未命名理论')
            theory_id = results.get('theory_id', '未知ID')
            avg_chi2 = results.get('avg_chi2', 0.0)
            
            output_file = os.path.join(args.output_dir, f"{theory_name.replace(' ', '_').lower()}_experiment_results.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 打印简要结果
            print(f"\n评估结果摘要:")
            print(f"理论: {theory_name} (ID: {theory_id})")
            print(f"平均χ²/n值: {avg_chi2:.4f}")
            
            conflicts = results.get('conflicts', [])
            if conflicts:
                print(f"与以下实验存在冲突:")
                for exp_id in conflicts:
                    print(f"  - {exp_id}")
            else:
                print("该理论与所有关键实验兼容")
            
            print(f"\n[完成] 评估结果已保存至: {output_file}")
        except Exception as e:
            print(f"[ERROR] 加载理论失败: {str(e)}")
            return
    elif args.theory_dir:
        # 批量处理目录下所有JSON文件
        if os.path.exists(args.theory_dir):
            theory_files = glob.glob(os.path.join(args.theory_dir, "*.json"))
            print(f"[INFO] 在 {args.theory_dir} 中找到 {len(theory_files)} 个理论文件")
            
            for theory_file in theory_files:
                print(f"\n[处理] {os.path.basename(theory_file)}")
                try:
                    with open(theory_file, 'r', encoding='utf-8') as f:
                        theory = json.load(f)
                    
                    # 评估理论
                    results = evaluator.evaluate_theory(theory, args.predictor_module)
                    
                    # 保存结果
                    theory_name = results.get('theory_name', '未命名').replace(' ', '_').lower()
                    output_file = os.path.join(args.output_dir, f"{theory_name}_experiment_results.json")
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    
                    print(f"[保存] 评估结果已保存到: {output_file}")
                except Exception as e:
                    print(f"[ERROR] 处理 {theory_file} 失败: {str(e)}")
        else:
            print(f"[ERROR] 目录不存在: {args.theory_dir}")

if __name__ == "__main__":
    main()
