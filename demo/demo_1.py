#!/usr/bin/env python3
# coding: utf-8
"""
量子理论评估工具 - 多理论多实验评估器
可以指定LLM模型，评估多个量子理论对多个实验的预测能力

用法示例:
# 评估单个理论对单个实验
python demo/demo_1.py --model_source deepseek --model_name deepseek-chat --theory_file demo/theories/theories_more/copenhagen.json --setup_file demo/experiments/c60_double_slit_setup.json --measured_file demo/experiments/c60_double_slit_measured.json --output_file demo/outputs/deepseek_chat/theory_evaluation.json --raw_output_file demo/outputs/deepseek_chat/llm_response_raw.txt

# 评估目录中的所有理论对单个实验
python demo/demo_1.py --model_source deepseek --model_name deepseek-chat --theory_dir demo/theories/theories_more --setup_file demo/experiments/c60_double_slit_setup.json --measured_file demo/experiments/c60_double_slit_measured.json --output_dir demo/outputs/deepseek_chat/theories

# 评估包含多个理论的JSON文件中的所有理论对单个实验
python demo/demo_1.py --model_source deepseek --model_name deepseek-chat --theories_json_file data/synthesized_theories/synthesis_20250520_152448/all_synthesized_theories.json --setup_file demo/experiments/c60_double_slit_setup.json --measured_file demo/experiments/c60_double_slit_measured.json --output_dir demo/outputs/deepseek_chat/theories_batch

# 评估单个理论对目录中的所有实验
python demo/demo_1.py --model_source deepseek --model_name deepseek-chat --theory_file demo/theories/theories_more/copenhagen.json --experiment_dir demo/experiments --output_dir demo/outputs/deepseek_chat/experiments

# 评估目录中的所有理论对所有实验
python demo/demo_1.py --model_source deepseek --model_name deepseek-chat --theory_dir demo/theories/theories_more --experiment_dir demo/experiments --output_dir demo/outputs/deepseek_chat/all_evaluations

# 评估包含多个理论的JSON文件中的所有理论对所有实验
python demo/demo_1.py --model_source deepseek --model_name deepseek-chat --theories_json_file data/synthesized_theories/synthesis_20250520_152448/all_synthesized_theories.json --experiment_dir demo/experiments --output_dir demo/outputs/deepseek_chat/all_batch_evaluations
"""
import sys, os, json, asyncio, argparse, glob, re
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from theory_generation.llm_interface import LLMInterface

async def evaluate_theory_experiment(theory, setup_exp, measured_data, llm, args, output_prefix=None):
    """评估单个理论对单个实验的预测能力"""
    # 获取实验ID
    exp_id = setup_exp["id"]
    
    # 检查是否有对应的测量数据
    if measured_data and exp_id in measured_data:
        complete_exp = setup_exp.copy()
        complete_exp["measured"] = {
            "value": measured_data[exp_id]["value"],
            "sigma": measured_data[exp_id]["sigma"]
        }
    else:
        print(f"[ERROR] 在测量数据中找不到实验ID: {exp_id}")
        return None
    
    # 提取实验目标
    exp_type = setup_exp.get("type", setup_exp.get("category", "未知类型"))
    exp_target = setup_exp.get("target_value", setup_exp.get("observable", "未定义目标"))
    
    # 修改提示，更清晰地展示理论的各个部分，同时使其适用于各种实验类型
    # 重要：只使用setup_exp（不包含测量值）
    prompt = f"""
    You are a quantum-mechanics assistant. Analyze the following experiment using the given theory.
    
    ## Experiment
    {json.dumps(setup_exp, indent=2)}
    
    ## Theory: {theory.get("name", "未知理论")}
    
    ### Summary
    {theory.get("summary", "No summary provided.")}
    
    ### Philosophy
    {json.dumps(theory.get("philosophy", {}), indent=2)}
    
    ### Parameters
    {json.dumps(theory.get("parameters", {}), indent=2)}
    
    ### Formalism
    {json.dumps(theory.get("formalism", {}), indent=2)}
    
    ### Semantics
    {json.dumps(theory.get("semantics", {}), indent=2)}
    
    ### Additional Information
    {json.dumps({k: v for k, v in theory.items() if k not in ["name", "summary", "philosophy", "parameters", "formalism", "semantics"]}, indent=2)}
    
    ## Task
    1. Use the theory's principles, formalisms, and concepts to analyze the given experiment.
    2. Derive the predicted value for this experiment ('{exp_target}' for {exp_type} experiment) step by step.
    3. Include detailed mathematical formulas and equations at each step of your derivation.
    4. Make sure to use the semantics and philosophical framework of the theory in your derivation.
    5. Calculate the final numerical value as precisely as possible.
    
    ## Output Format
    Return your answer as TWO JSON objects (one per line):
    {{"derivation": "Step 1: [Formula 1] ... Step 2: [Formula 2] ... etc."}}
    {{"value": 0.XX}}
    
    In your derivation:
    - Use LaTeX notation for all mathematical formulas
    - Show each step of calculation explicitly
    - Explain the physical meaning of each term in your equations
    - Make your mathematical reasoning clear and detailed
    """
    
    # 调用LLM
    theory_name = theory.get("name", "未知理论")
    print(f"[INFO] 正在评估理论'{theory_name}'对实验'{exp_id}'，使用{args.model_source}/{args.model_name}模型...")
    response = await llm.query_async([{"role": "user", "content": prompt}], temperature=args.temperature)
    
    # 保存和打印原始响应
    print("\n原始LLM响应:\n" + "="*50)
    print(response)
    print("="*50)
    
    # 确定输出文件路径
    if args.output_dir:
        # 创建一个有意义的文件名前缀
        if output_prefix:
            filename_prefix = output_prefix
        else:
            theory_filename = theory_name.replace(" ", "_").lower()
            exp_filename = exp_id.replace(" ", "_").lower()
            filename_prefix = f"{theory_filename}_vs_{exp_filename}"
        
        raw_output_file = os.path.join(args.output_dir, f"{filename_prefix}_response_raw.txt")
        output_file = os.path.join(args.output_dir, f"{filename_prefix}_evaluation.json")
    else:
        raw_output_file = args.raw_output_file
        output_file = args.output_file
    
    # 确保目录存在
    output_dir = os.path.dirname(raw_output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] 已创建目录: {output_dir}")

    with open(raw_output_file, "w", encoding="utf-8") as f:
        f.write(response)
    print(f"[INFO] 原始响应已保存到: {raw_output_file}")
    
    # 提取derivation和value
    derivation = "未找到推导过程"
    value = None
    
    # 解析响应中的JSON对象
    for line in response.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if "derivation" in obj:
                    derivation = obj["derivation"]
                    print("\n推导过程:\n" + "-"*50)
                    print(derivation)
                    print("-"*50)
                if "value" in obj:
                    value = float(obj["value"])
                    print(f"\n预测值: {value}\n")
            except json.JSONDecodeError:
                continue
    
    # 计算与实验值的偏差（使用合并后的完整实验数据）
    chi2 = None
    success = None  # 新增：预测成功标志
    chi2_threshold = args.chi2_threshold if hasattr(args, 'chi2_threshold') else 4.0  # 默认χ²阈值为4
    
    if value is not None:
        measured = complete_exp.get("measured", {}).get("value")
        sigma = complete_exp.get("measured", {}).get("sigma")
        if measured is not None and sigma is not None:
            chi2 = ((value - measured) / sigma) ** 2
            # 判断预测是否成功（χ²小于阈值）
            success = chi2 < chi2_threshold
            
            print(f"χ²值: {chi2:.4f}")
            print(f"与实验值 {measured} 的偏差: {abs(value - measured):.4f}")
            print(f"预测结果: {'成功' if success else '失败'} (χ²阈值={chi2_threshold})")
    
    # 创建包含推导和结果的结构化输出
    structured_output = {
        "theory_name": theory_name,
        "experiment_id": exp_id,
        "derivation": derivation,
        "predicted_value": value,
        "measured_value": measured_data[exp_id]["value"],
        "sigma": measured_data[exp_id]["sigma"],
        "chi2": chi2,
        "success": success,  # 新增：添加预测成功标志
        "chi2_threshold": chi2_threshold,  # 新增：记录使用的χ²阈值
        "model_info": {
            "source": args.model_source,
            "name": args.model_name,
            "temperature": args.temperature
        }
    }
    
    # 保存结构化输出
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] 已创建目录: {output_dir}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(structured_output, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 评估结果已保存到: {output_file}")
    
    return structured_output

def load_experiments_from_directory(experiment_dir):
    """从目录中加载所有实验文件，匹配设置和测量数据"""
    print(f"[INFO] 正在从目录 {experiment_dir} 加载实验...")
    
    # 查找所有JSON文件
    experiment_files = glob.glob(os.path.join(experiment_dir, "*.json"))
    
    # 分离设置和测量文件
    setup_files = {}
    measured_files = {}
    
    for file_path in experiment_files:
        filename = os.path.basename(file_path)
        # 通过文件名判断类型
        if "setup" in filename:
            # 提取实验ID（假设格式为 {id}_setup.json）
            match = re.match(r"(.+?)_setup\.json", filename)
            if match:
                exp_id = match.group(1)
                setup_files[exp_id] = file_path
        elif "measured" in filename:
            # 提取实验ID（假设格式为 {id}_measured.json）
            match = re.match(r"(.+?)_measured\.json", filename)
            if match:
                exp_id = match.group(1)
                measured_files[exp_id] = file_path
    
    # 找出既有设置又有测量数据的实验ID
    common_exp_ids = set(setup_files.keys()) & set(measured_files.keys())
    print(f"[INFO] 找到 {len(common_exp_ids)} 个完整实验数据集")
    
    # 加载实验数据
    experiments = {}
    measured_db = {}
    
    for exp_id in common_exp_ids:
        try:
            # 加载设置
            with open(setup_files[exp_id], 'r') as f:
                setup_data = json.load(f)
            
            # 加载测量数据
            with open(measured_files[exp_id], 'r') as f:
                measured_data = json.load(f)
            
            # 存储数据
            experiments[exp_id] = setup_data
            measured_db[exp_id] = measured_data
            
            print(f"[INFO] 已加载实验: {exp_id}")
        except Exception as e:
            print(f"[ERROR] 加载实验 {exp_id} 时出错: {str(e)}")
    
    return experiments, measured_db

async def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="量子理论评估工具")
    
    # 理论相关参数
    theory_group = parser.add_mutually_exclusive_group(required=True)
    theory_group.add_argument("--theory_file", 
                      help="单个理论JSON文件路径")
    theory_group.add_argument("--theory_dir", 
                      help="包含多个理论JSON文件的目录路径")
    theory_group.add_argument("--theories_json_file",
                      help="包含多个理论的单个JSON文件路径（理论数组）")
    
    # 实验相关参数
    experiment_group = parser.add_mutually_exclusive_group(required=True)
    experiment_group.add_argument("--setup_file",
                      help="实验设置JSON文件路径")
    experiment_group.add_argument("--experiment_dir",
                      help="包含多个实验的目录路径（含setup和measured文件）")
    
    parser.add_argument("--measured_file",
                      help="实验测量值JSON文件路径（与--setup_file一起使用）")
    
    parser.add_argument("--model_source", default="deepseek",
                      choices=["openai", "deepseek"],
                      help="LLM模型来源")
    parser.add_argument("--model_name", default="deepseek-chat",
                      help="LLM模型名称")
    parser.add_argument("--temperature", type=float, default=0.2,
                      help="LLM温度参数")
    parser.add_argument("--chi2_threshold", type=float, default=10.0,
                      help="理论预测成功的χ²阈值")
    
    # 输出相关参数
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output_file", 
                      help="单个评估结果的JSON文件路径")
    output_group.add_argument("--output_dir", 
                      help="多个评估结果的输出目录")
    
    parser.add_argument("--raw_output_file", 
                      help="原始LLM响应文本文件路径（仅用于单个评估）")
    
    args = parser.parse_args()
    
    # 校验参数组合
    if args.setup_file and not args.measured_file:
        parser.error("使用--setup_file时必须指定--measured_file")
    if (args.theory_file and args.setup_file) and not (args.output_file and args.raw_output_file):
        parser.error("评估单个理论对单个实验时，必须指定--output_file和--raw_output_file")
    if (args.theory_dir or args.theories_json_file or args.experiment_dir) and not args.output_dir:
        parser.error("在批量评估模式下必须指定--output_dir")
    
    # 创建LLM
    llm = LLMInterface(model_name=args.model_name, model_source=args.model_source)
    
    # 加载实验数据
    if args.experiment_dir:
        # 从目录加载多个实验
        experiments, measured_db = load_experiments_from_directory(args.experiment_dir)
        if not experiments:
            parser.error(f"在目录 {args.experiment_dir} 中未找到有效的实验数据")
    else:
        # 加载单个实验
        with open(args.setup_file) as f:
            setup_exp = json.load(f)
        
        with open(args.measured_file) as f:
            measured_data = json.load(f)
        
        # 将测量值转换为ID索引的字典
        if isinstance(measured_data, list):
            measured_db = {d["id"]: d for d in measured_data}
        else:
            measured_db = {measured_data["id"]: measured_data}
        
        experiments = {setup_exp["id"]: setup_exp}
    
    # 加载理论
    if args.theory_dir:
        # 从目录加载多个理论
        theory_files = glob.glob(os.path.join(args.theory_dir, "*.json"))
        print(f"[INFO] 在目录 {args.theory_dir} 中找到 {len(theory_files)} 个理论文件")
        theories = {}
        
        for theory_file in theory_files:
            try:
                with open(theory_file) as f:
                    theory = json.load(f)
                theory_name = theory.get("name", os.path.basename(theory_file))
                theories[theory_name] = theory
            except Exception as e:
                print(f"[ERROR] 加载理论文件 {theory_file} 时出错: {str(e)}")
    elif args.theories_json_file:
        # 从单个JSON文件加载多个理论（理论数组）
        print(f"[INFO] 从 {args.theories_json_file} 加载多个理论")
        theories = {}
        try:
            with open(args.theories_json_file) as f:
                theories_list = json.load(f)
            
            if not isinstance(theories_list, list):
                parser.error(f"文件 {args.theories_json_file} 不包含理论数组")
            
            print(f"[INFO] 在文件中找到 {len(theories_list)} 个理论")
            for i, theory in enumerate(theories_list):
                theory_name = theory.get("name", f"理论_{i+1}")
                theories[theory_name] = theory
        except Exception as e:
            parser.error(f"加载理论文件 {args.theories_json_file} 时出错: {str(e)}")
    else:
        # 加载单个理论
        with open(args.theory_file) as f:
            theory = json.load(f)
        theory_name = theory.get("name", "未知理论")
        theories = {theory_name: theory}
    
    # 评估所有理论对所有实验
    all_results = []
    
    for theory_name, theory in theories.items():
        theory_results = []
        for exp_id, setup_exp in experiments.items():
            try:
                # 创建一个有意义的输出前缀
                theory_filename = theory_name.replace(" ", "_").lower()
                exp_filename = exp_id.replace(" ", "_").lower()
                output_prefix = f"{theory_filename}_vs_{exp_filename}"
                
                # 评估单个理论对单个实验
                result = await evaluate_theory_experiment(
                    theory, setup_exp, measured_db, llm, args, output_prefix
                )
                
                if result:
                    theory_results.append(result)
                    all_results.append(result)
            except Exception as e:
                print(f"[ERROR] 评估理论 '{theory_name}' 对实验 '{exp_id}' 时出错: {str(e)}")
        
        # 保存每个理论的汇总结果
        if args.output_dir and len(theory_results) > 0:
            theory_summary_file = os.path.join(
                args.output_dir, 
                f"{theory_name.replace(' ', '_').lower()}_summary.json"
            )
            
            with open(theory_summary_file, "w", encoding="utf-8") as f:
                json.dump(theory_results, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 理论 '{theory_name}' 的汇总评估结果已保存到: {theory_summary_file}")
    
    # 保存总体汇总结果
    if args.output_dir and len(all_results) > 0:
        # 创建实验汇总结果
        experiment_results = {}
        for result in all_results:
            exp_id = result["experiment_id"]
            if exp_id not in experiment_results:
                experiment_results[exp_id] = []
            experiment_results[exp_id].append({
                "theory_name": result["theory_name"],
                "predicted_value": result["predicted_value"],
                "chi2": result["chi2"],
                "success": result["success"]  # 添加成功标志
            })
        
        # 保存每个实验的汇总结果
        for exp_id, results in experiment_results.items():
            exp_summary_file = os.path.join(
                args.output_dir,
                f"{exp_id.replace(' ', '_').lower()}_comparison.json"
            )
            with open(exp_summary_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 实验 '{exp_id}' 的理论比较结果已保存到: {exp_summary_file}")
        
        # 保存总体汇总
        summary_file = os.path.join(args.output_dir, "evaluation_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 所有评估结果的汇总已保存到: {summary_file}")
        
        # 生成理论排名比较
        theory_scores = {}
        theory_success_rates = {}  # 新增：记录每个理论的成功率
        
        for result in all_results:
            theory_name = result["theory_name"]
            chi2 = result.get("chi2")
            success = result.get("success")
            
            # 记录χ²得分
            if chi2 is not None:
                if theory_name not in theory_scores:
                    theory_scores[theory_name] = []
                theory_scores[theory_name].append(chi2)
            
            # 记录成功/失败
            if success is not None:
                if theory_name not in theory_success_rates:
                    theory_success_rates[theory_name] = {"success": 0, "total": 0}
                theory_success_rates[theory_name]["total"] += 1
                if success:
                    theory_success_rates[theory_name]["success"] += 1
        
        # 计算平均χ²得分和成功率
        theory_rankings = []
        for theory_name, scores in theory_scores.items():
            avg_chi2 = sum(scores) / len(scores) if scores else float('inf')
            
            # 计算成功率
            success_rate = 0
            if theory_name in theory_success_rates and theory_success_rates[theory_name]["total"] > 0:
                success_rate = theory_success_rates[theory_name]["success"] / theory_success_rates[theory_name]["total"]
            
            theory_rankings.append({
                "theory_name": theory_name,
                "experiments_count": len(scores),
                "average_chi2": avg_chi2,
                "total_chi2": sum(scores) if scores else float('inf'),
                "success_count": theory_success_rates.get(theory_name, {}).get("success", 0),
                "success_rate": success_rate,
                "chi2_threshold": args.chi2_threshold
            })
        
        # 按成功率排序（从高到低），然后按平均χ²排序（从低到高）
        theory_rankings.sort(key=lambda x: (-x["success_rate"], x["average_chi2"]))
        
        # 保存理论排名
        ranking_file = os.path.join(args.output_dir, "theory_rankings.json")
        with open(ranking_file, "w", encoding="utf-8") as f:
            json.dump(theory_rankings, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 理论排名结果已保存到: {ranking_file}")
        
        # 打印排名总结
        print("\n理论排名总结:")
        print("-" * 80)
        print(f"{'理论名称':<30} {'成功率':<10} {'平均χ²':<10} {'实验数':<10}")
        print("-" * 80)
        for rank in theory_rankings:
            print(f"{rank['theory_name']:<30} {rank['success_rate']*100:.1f}% {rank['average_chi2']:<10.4f} {rank['experiments_count']:<10}")
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(main())