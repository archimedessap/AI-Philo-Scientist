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
python demo/demo_1.py --model_source openai --model_name gpt-4o-mini --theory_file demo/theories/theories_more/copenhagen.json --experiment_dir demo/experiments --output_dir demo/outputs/gpt_4o_mini/experiments --use_instrument_correction

# 评估目录中的所有理论对所有实验
python demo/demo_1.py --model_source deepseek --model_name deepseek-reasoner --theory_dir demo/theories/theories_more --experiment_dir demo/experiments --output_dir demo/outputs/deepseek_reasoner/all_evaluations --use_instrument_correction

# 评估包含多个理论的JSON文件中的所有理论对所有实验
python demo/demo_1.py --model_source deepseek --model_name deepseek-reasoner --theories_json_file data/synthesized_theories/synthesis_20250603_151008/all_synthesized_theories.json --experiment_dir demo/experiments --output_dir demo/outputs/deepseek_reasoner/all_batch_evaluations --use_instrument_correction
"""
import sys, os, json, asyncio, argparse, glob, re
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from theory_generation.llm_interface import LLMInterface
from demo.instrument_correction import InstrumentCorrector

async def evaluate_theory_experiment(theory, setup_exp, measured_data, llm, args, output_prefix=None, corrected_setup_exp=None):
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
    chi2_corrected = None
    success = None  # 新增：预测成功标志
    success_corrected = None  # 仪器修正后的成功标志
    chi2_threshold = args.chi2_threshold if hasattr(args, 'chi2_threshold') else 4.0  # 默认χ²阈值为4
    
    # 初始化仪器修正器（如果启用）
    corrector = None
    correction_result = None
    if getattr(args, 'use_instrument_correction', False):
        corrector = InstrumentCorrector()
    
    if value is not None:
        measured = complete_exp.get("measured", {}).get("value")
        sigma = complete_exp.get("measured", {}).get("sigma")
        if measured is not None and sigma is not None:
            # 原始χ²计算（无仪器修正）
            chi2 = ((value - measured) / sigma) ** 2
            success = chi2 < chi2_threshold
            
            # 仪器修正评估（如果启用）
            if corrector is not None and corrected_setup_exp is not None:
                # 使用corrected_setup_exp中的仪器参数进行修正
                correction_result = corrector.evaluate_with_correction(
                    value, corrected_setup_exp, {"value": measured, "sigma": sigma}
                )
                chi2_corrected = correction_result["chi2_corrected"]
                success_corrected = chi2_corrected < chi2_threshold
                
                print(f"原始χ²值: {chi2:.4f}")
                print(f"修正后χ²值: {chi2_corrected:.4f}")
                print(f"理论预测值: {value:.4f}")
                print(f"仪器修正后预测值: {correction_result['corrected_prediction']:.4f}")
                print(f"实验测量值: {measured:.4f}")
                print(f"原始偏差: {abs(value - measured):.4f}")
                print(f"修正后偏差: {abs(correction_result['corrected_prediction'] - measured):.4f}")
                print(f"原始预测结果: {'成功' if success else '失败'} (χ²阈值={chi2_threshold})")
                print(f"修正后预测结果: {'成功' if success_corrected else '失败'} (χ²阈值={chi2_threshold})")
                
                # 显示仪器参数
                if "instrument_corrections" in corrected_setup_exp:
                    inst_params = correction_result["instrument_params"]
                    print(f"仪器参数: η={inst_params['detection_efficiency']:.3f}, B={inst_params['background_noise']:.3f}, sys={inst_params['systematic_bias']:.3f}")
            else:
                # 传统模式输出
                print(f"χ²值: {chi2:.4f}")
                print(f"与实验值 {measured} 的偏差: {abs(value - measured):.4f}")
                print(f"预测结果: {'成功' if success else '失败'} (χ²阈值={chi2_threshold})")
    
    # 创建包含推导和结果的结构化输出
    structured_output = {
        "theory_name": theory_name,
        "experiment_id": exp_id,
        "derivation": derivation,
        "predicted_value": float(value),  # 确保是Python原生float
        "measured_value": float(measured_data[exp_id]["value"]),
        "sigma": float(measured_data[exp_id]["sigma"]),
        "chi2": float(chi2),
        "success": bool(success),  # 确保是Python原生bool
        "chi2_threshold": float(chi2_threshold),
        "model_info": {
            "source": args.model_source,
            "name": args.model_name,
            "temperature": float(args.temperature)
        }
    }
    
    # 如果使用了仪器修正，添加相关信息
    if correction_result is not None:
        structured_output.update({
            "corrected_predicted_value": float(correction_result["corrected_prediction"]),
            "total_sigma": float(correction_result["total_sigma"]),
            "chi2_corrected": float(chi2_corrected),
            "success_corrected": bool(success_corrected),  # 确保是Python原生bool
            "instrument_correction": correction_result["instrument_params"]
        })
    
    # 保存结构化输出
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] 已创建目录: {output_dir}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(structured_output, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 评估结果已保存到: {output_file}")
    
    return structured_output

def load_experiments_from_directory(experiment_dir, use_instrument_correction=False):
    """从目录中加载所有实验文件，匹配设置和测量数据"""
    print(f"[INFO] 正在从目录 {experiment_dir} 加载实验...")
    
    # 查找所有JSON文件
    experiment_files = glob.glob(os.path.join(experiment_dir, "*.json"))
    
    # 分离设置和测量文件
    setup_files = {}
    setup_corrected_files = {}  # 新增：存储修正版本的文件
    measured_files = {}
    
    for file_path in experiment_files:
        filename = os.path.basename(file_path)
        # 通过文件名判断类型
        if "setup_corrected" in filename:
            # setup_corrected.json 文件
            match = re.match(r"(.+?)_setup_corrected\.json", filename)
            if match:
                exp_id = match.group(1)
                setup_corrected_files[exp_id] = file_path
        elif "setup" in filename and filename.endswith("_setup.json"):
            # 原始 setup.json 文件
            match = re.match(r"(.+?)_setup\.json", filename)
            if match:
                exp_id = match.group(1)
                setup_files[exp_id] = file_path
        elif "measured" in filename:
            # 测量数据文件
            match = re.match(r"(.+?)_measured\.json", filename)
            if match:
                exp_id = match.group(1)
                measured_files[exp_id] = file_path
    
    # 根据是否使用仪器修正选择加载策略
    if use_instrument_correction:
        # 仪器修正模式：需要同时有原始setup文件和corrected文件
        common_exp_ids = set(setup_files.keys()) & set(setup_corrected_files.keys()) & set(measured_files.keys())
        print(f"[INFO] 仪器修正模式：找到 {len(common_exp_ids)} 个完整实验数据集")
    else:
        # 非修正模式：优先使用corrected文件（让LLM自己处理），回退到原始文件
        available_setup_ids = set(setup_corrected_files.keys()) | set(setup_files.keys())
        common_exp_ids = available_setup_ids & set(measured_files.keys())
        print(f"[INFO] 标准模式：找到 {len(common_exp_ids)} 个完整实验数据集")
    
    # 加载实验数据
    experiments = {}
    experiments_corrected = {}  # 新增：存储修正版本的实验设置
    measured_db = {}
    
    for exp_id in common_exp_ids:
        try:
            # 加载测量数据
            with open(measured_files[exp_id], 'r') as f:
                measured_data = json.load(f)
            measured_db[exp_id] = measured_data
            
            if use_instrument_correction:
                # 仪器修正模式：分别加载原始和修正版本
                with open(setup_files[exp_id], 'r') as f:
                    setup_data = json.load(f)
                with open(setup_corrected_files[exp_id], 'r') as f:
                    setup_corrected_data = json.load(f)
                
                experiments[exp_id] = setup_data  # 用于LLM纯理论预测
                experiments_corrected[exp_id] = setup_corrected_data  # 用于仪器修正
                
                print(f"[INFO] 已加载实验: {exp_id} (原始+修正设置)")
            else:
                # 非修正模式：优先使用corrected，回退到原始
                if exp_id in setup_corrected_files:
                    with open(setup_corrected_files[exp_id], 'r') as f:
                        setup_data = json.load(f)
                    file_type = "corrected"
                else:
                    with open(setup_files[exp_id], 'r') as f:
                        setup_data = json.load(f)
                    file_type = "original"
                
                experiments[exp_id] = setup_data
                print(f"[INFO] 已加载实验: {exp_id} (使用{file_type}设置)")
                
        except Exception as e:
            print(f"[ERROR] 加载实验 {exp_id} 时出错: {str(e)}")
    
    return experiments, experiments_corrected, measured_db

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
    parser.add_argument("--use_instrument_correction", action="store_true",
                      help="是否使用仪器修正（默认关闭，兼容旧版本）")
    parser.add_argument("--correction_mode", choices=["raw", "corrected", "both"], default="both",
                      help="评估模式：raw(仅原始)，corrected(仅修正)，both(两者)")
    
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
        experiments, experiments_corrected, measured_db = load_experiments_from_directory(args.experiment_dir, args.use_instrument_correction)
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
        experiments_corrected = {}  # 单个实验模式下不支持仪器修正
    
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
                    theory, setup_exp, measured_db, llm, args, output_prefix, experiments_corrected.get(exp_id)
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
            
            # 当使用仪器修正时，优先使用修正后的结果
            if args.use_instrument_correction and "success_corrected" in result:
                chi2 = result.get("chi2_corrected", chi2)
                success = result.get("success_corrected", success)
            
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
        
        # 自动角色评估：对成功率>=80%的理论进行多角色评估
        high_success_theories = [
            rank for rank in theory_rankings 
            if rank['success_rate'] >= 0.8  # 80%成功率阈值
        ]
        
        if high_success_theories:
            print(f"\n[INFO] 发现 {len(high_success_theories)} 个成功率≥80%的理论，开始角色评估...")
            
            # 导入角色评估模块
            try:
                import sys
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from theory_validation.agent_validation.theory_evaluator import TheoryEvaluator
                
                # 初始化角色评估器
                role_evaluator = TheoryEvaluator(llm)
                
                # 创建角色评估输出目录
                role_output_dir = os.path.join(args.output_dir, "role_evaluations")
                os.makedirs(role_output_dir, exist_ok=True)
                
                role_results = []
                
                for rank in high_success_theories:
                    theory_name = rank['theory_name']
                    theory_data = theories[theory_name]
                    
                    print(f"[INFO] 正在对理论 '{theory_name}' 进行角色评估...")
                    
                    try:
                        # 执行角色评估
                        role_result = await role_evaluator.evaluate_theory(theory_data, predictor_module=None)
                        role_result['experiment_success_rate'] = rank['success_rate']
                        role_result['average_chi2'] = rank['average_chi2']
                        role_results.append(role_result)
                        
                        # 保存单个理论的角色评估结果
                        theory_role_file = os.path.join(
                            role_output_dir,
                            f"{theory_name.replace(' ', '_').lower()}_role_evaluation.json"
                        )
                        with open(theory_role_file, "w", encoding="utf-8") as f:
                            json.dump(role_result, f, ensure_ascii=False, indent=2)
                        
                        print(f"[INFO] 理论 '{theory_name}' 的角色评估结果已保存到: {theory_role_file}")
                        
                    except Exception as e:
                        print(f"[ERROR] 评估理论 '{theory_name}' 的角色时出错: {str(e)}")
                
                # 保存综合角色评估结果
                if role_results:
                    role_summary_file = os.path.join(role_output_dir, "role_evaluation_summary.json")
                    with open(role_summary_file, "w", encoding="utf-8") as f:
                        json.dump(role_results, f, ensure_ascii=False, indent=2)
                    
                    # 计算综合排名（实验成功率 + 角色评估）
                    combined_rankings = []
                    for role_result in role_results:
                        theory_name = role_result['theory']['name']
                        
                        # 计算角色评估平均分
                        role_scores = [
                            eval_result['overall_score'] 
                            for eval_result in role_result['evaluations']
                            if 'overall_score' in eval_result and eval_result['overall_score'] is not None
                        ]
                        avg_role_score = sum(role_scores) / len(role_scores) if role_scores else 0
                        
                        # 综合评分 = 实验成功率 * 0.6 + 角色评估分 * 0.4
                        combined_score = (
                            role_result['experiment_success_rate'] * 0.6 + 
                            avg_role_score / 10.0 * 0.4  # 角色评估分通常是1-10分，归一化到0-1
                        )
                        
                        combined_rankings.append({
                            'theory_name': theory_name,
                            'experiment_success_rate': role_result['experiment_success_rate'],
                            'average_chi2': role_result['average_chi2'],
                            'average_role_score': avg_role_score,
                            'combined_score': combined_score,
                            'role_details': {
                                eval_result['role']: eval_result.get('overall_score', 0)
                                for eval_result in role_result['evaluations']
                            }
                        })
                    
                    # 按综合评分排序
                    combined_rankings.sort(key=lambda x: -x['combined_score'])
                    
                    # 保存综合排名
                    combined_ranking_file = os.path.join(role_output_dir, "combined_rankings.json")
                    with open(combined_ranking_file, "w", encoding="utf-8") as f:
                        json.dump(combined_rankings, f, ensure_ascii=False, indent=2)
                    
                    print(f"[INFO] 角色评估汇总已保存到: {role_summary_file}")
                    print(f"[INFO] 综合排名已保存到: {combined_ranking_file}")
                    
                    # 打印综合排名
                    print("\n🏆 综合排名（实验 60% + 角色评估 40%）:")
                    print("-" * 100)
                    print(f"{'理论名称':<30} {'实验成功率':<12} {'角色平均分':<12} {'综合评分':<12} {'详细评分'}")
                    print("-" * 100)
                    for rank in combined_rankings:
                        role_detail = ", ".join([f"{role}:{score:.1f}" for role, score in rank['role_details'].items()])
                        print(f"{rank['theory_name']:<30} {rank['experiment_success_rate']*100:>8.1f}% "
                              f"{rank['average_role_score']:>10.1f} {rank['combined_score']:>10.3f} {role_detail}")
                    print("-" * 100)
                    
                else:
                    print("[WARNING] 没有成功完成的角色评估结果")
                    
            except ImportError as e:
                print(f"[ERROR] 无法导入角色评估模块: {str(e)}")
                print("[INFO] 跳过角色评估步骤")
            except Exception as e:
                print(f"[ERROR] 角色评估过程中出错: {str(e)}")
        else:
            print("\n[INFO] 没有理论达到80%成功率阈值，跳过角色评估")

if __name__ == "__main__":
    asyncio.run(main())