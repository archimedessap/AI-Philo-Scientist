#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-evaluate synthesized theories against the experiment library.

示例:
python run_experiment_evaluation.py \
    --theory_file data/synthesized_theories/synthesis_20250516_174112/all_synthesized_theories.json \
    --experiments_path theory_experiment/data/experiments.json \
    --predictor_module theory_experiment.predictors.auto_predictor \
    --model_source deepseek \
    --model_name deepseek-chat \
    --output_file data/evaluations/deepseek_chat_eval.json
"""
import argparse, asyncio, glob, json, os, importlib
from pathlib import Path

# ----------------------------------------------------------------------
def load_theories(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        # 处理可能的单个理论或理论列表情况
        return [data] if isinstance(data, dict) else data
    if p.suffix == ".jsonl":
        with p.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    raise ValueError("only .json / .jsonl supported")

# ----------------------------------------------------------------------
async def evaluate_one(evaluator, theory, predictor_mod, llm_name, model_source="openai"):
    # 不再依赖模块级全局变量传递模型名称
    # 创建预测器实例并传入模型信息
    pred = predictor_mod.Predictor(theory, model_name=llm_name, model_source=model_source)
    # 执行评估
    res = await evaluator.evaluate_theory(theory, pred)
    print(f"[√] {theory.get('name')}  score={res.get('final_score',0):.2f}")
    return res

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Run experiment evaluation")
    ap.add_argument("--theory_file", help=".json / .jsonl file with theories")
    ap.add_argument("--theory_dir",  help="dir containing *.json theories")
    ap.add_argument("--experiments_path",
                    default="theory_experiment/data/experiments.json")
    ap.add_argument("--predictor_module",
                    default="theory_experiment.predictors.auto_predictor")
    ap.add_argument("--model_source", 
                    default="deepseek",
                    choices=["openai", "deepseek"],
                    help="model source (openai or deepseek)")
    ap.add_argument("--model_name",
                    default="deepseek-chat",
                    help="model name for LLMInterface (e.g. gpt-4o, deepseek-chat)")
    ap.add_argument("--output_file",
                    default="experiment_eval_results.json",
                    help="where to store evaluation report")
    args = ap.parse_args()

    # ---------------- load evaluator ----------------
    try:
        # 确保使用正确的导入路径
        from theory_experiment.experiment_evaluator import ExperimentEvaluator
        evaluator = ExperimentEvaluator(args.experiments_path)
        print(f"[INFO] 已加载实验评估器，实验数据：{len(evaluator.experiments)}条")
    except ImportError as e:
        print(f"[ERROR] 导入实验评估器失败: {str(e)}")
        print("[HINT] 请确认评估器位于 theory_experiment/experiment_evaluator.py")
        return

    # 加载预测器模块
    try:
        predictor_mod = importlib.import_module(args.predictor_module)
        print(f"[INFO] 已加载预测器模块: {args.predictor_module}")
    except ImportError as e:
        print(f"[ERROR] 导入预测器模块失败: {str(e)}")
        return

    # ---------------- load theories -----------------
    theories = []
    if args.theory_file:
        try:
            loaded = load_theories(args.theory_file)
            theories.extend(loaded)
            print(f"[INFO] 从{args.theory_file}加载了{len(loaded)}个理论")
        except Exception as e:
            print(f"[ERROR] 加载理论文件失败: {str(e)}")
    
    if args.theory_dir:
        count = 0
        for f in glob.glob(os.path.join(args.theory_dir, "*.json")):
            try:
                loaded = load_theories(f)
                theories.extend(loaded)
                count += len(loaded)
            except Exception as e:
                print(f"[ERROR] 加载{f}失败: {str(e)}")
        print(f"[INFO] 从目录{args.theory_dir}加载了{count}个理论")
    
    if not theories:
        print("[ERROR] 未加载到任何理论，请检查文件路径")
        return
    
    print(f"[INFO] 总共加载了{len(theories)}个理论，使用{args.model_source}/{args.model_name}模型进行评估")

    # ---------------- async run ---------------------
    print("[INFO] 开始异步评估...")
    
    # 定义主异步函数
    async def run_all_evaluations():
        return await asyncio.gather(*[evaluate_one(evaluator, th, predictor_mod, 
                                                 args.model_name, args.model_source) 
                                    for th in theories])
    
    # 使用asyncio.run执行主函数
    results = asyncio.run(run_all_evaluations())

    # 创建输出目录（如有必要）
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入结果
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n[INFO] 评估结果已写入: {args.output_file}")

if __name__ == "__main__":
    main()
