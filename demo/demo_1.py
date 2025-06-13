#!/usr/bin/env python3
# coding: utf-8
"""
量子理论评估工具 - 多理论多实验评估器
可以指定LLM模型，评估多个量子理论对多个实验的预测能力

"""
import sys, os, json, asyncio, argparse, glob, re, time
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from theory_generation.llm_interface import LLMInterface
from demo.instrument_correction import InstrumentCorrector
# 导入新的角色评估模块
from demo.auto_role_evaluation import run_role_evaluation_for_theories

def load_theories_from_sources(theories_path: str, schema_version: str = "2.1") -> dict:
    """
    从目录或单个文件加载理论数据
    
    Args:
        theories_path: 理论目录或单个理论文件的路径
        schema_version: 要求加载的理论schema版本 ('any' to load all)
        
    Returns:
        dict: 理论名称到理论数据的映射
    """
    theories = {}
    
    # 确定是目录还是文件
    if os.path.isdir(theories_path):
        theory_files = glob.glob(os.path.join(theories_path, "*.json"))
        print(f"[INFO] 在目录 {theories_path} 中找到 {len(theory_files)} 个理论文件")
    elif os.path.isfile(theories_path) and theories_path.endswith('.json'):
        theory_files = [theories_path]
        print(f"[INFO] 正在加载单个理论文件: {theories_path}")
    else:
        print(f"[ERROR] 无效的理论路径: {theories_path}")
        return {}
        
    for theory_file in theory_files:
        try:
            with open(theory_file, 'r', encoding='utf-8') as f:
                theory = json.load(f)
            
            # 兼容包含理论列表的JSON文件
            if isinstance(theory, list):
                theory_list = theory
            else:
                theory_list = [theory]
            
            for t in theory_list:
                # 检查schema版本
                load_any_schema = schema_version is None or schema_version.lower() == 'any'
                file_schema_version = t.get("metadata", {}).get("schema_version")
                
                if not load_any_schema and file_schema_version != schema_version:
                    print(f"[WARN] 跳过理论 '{t.get('name', '未命名')}': schema版本不匹配 (需要 {schema_version}, 文件为 {file_schema_version})")
                    continue

                theory_name = t.get("name", os.path.basename(theory_file))
                if theory_name in theories:
                    print(f"[WARN] 发现重复的理论名称 '{theory_name}'，将覆盖旧版本。")
                theories[theory_name] = t

        except Exception as e:
            print(f"[ERROR] 加载理论文件 {theory_file} 时出错: {str(e)}")
            
    return theories

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
    
    # 使用Schema v2.1更新Prompt
    prompt = f"""
    You are a quantum physics expert tasked with evaluating a theoretical model against experimental data.
    
    ## Theory: {theory.get("name", "Unknown Theory")} (Schema Version: {theory.get("metadata", {}).get("schema_version", "N/A")})
    
    ### Core Identity
    - **UID:** {theory.get("metadata", {}).get("uid", "N/A")}
    - **Lineage:** {json.dumps(theory.get("metadata", {}).get("lineage", {}), indent=2)}
    - **Relation to SQM:** {theory.get("mathematical_relation_to_sqm", "Not specified")}

    ### Core Principles
    {json.dumps(theory.get("core_principles", "No principles provided."), indent=2)}
    
    ### Formalism
    {json.dumps(theory.get("formalism", "No formalism provided."), indent=2)}
    
    ### Predictions and Verifiability
    {json.dumps(theory.get("predictions_and_verifiability", "No predictions provided."), indent=2)}
    
    ## Experiment
    {json.dumps(setup_exp, indent=2)}
    
    ## Task
    1.  **Analyze**: Based on the theory's **Core Principles** and **Formalism**, analyze the provided experiment.
    2.  **Derive**: Step-by-step, derive the predicted value for the experiment's target: **'{exp_target}'**. 
        - If the theory is an **interpretation** of SQM, use standard quantum formalism and explain how the theory's principles interpret the result.
        - If the theory is a **modification** or **extension** of SQM, explicitly use the modified equations from the **Formalism** section. Highlight how the `deviations_from_sqm` lead to a different prediction.
    3.  **Calculate**: Compute the final numerical value for the prediction.
    4.  **Justify**: Explain how the derivation is a logical consequence of the theory's specific axioms and mathematical structure.

    ## Output Format
    Return your answer as TWO JSON objects (one per line):
    {{"derivation": "Step-by-step derivation using LaTeX for math, explaining each step logically from the theory's perspective."}}
    {{"value": <your_calculated_numerical_value>}}
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
            except (json.JSONDecodeError, TypeError, ValueError):
                print(f"[WARN] 无法解析或转换行，或值无效: {line}")
                continue
    
    # 如果LLM未能提供有效数值，则无法继续评估
    if value is None:
        print(f"[WARN] 未能从LLM响应中提取有效的预测数值。跳过对实验 '{exp_id}' 的评估。")
        return None

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
    setup_corrected_files = {}
    measured_files = {}
    
    # 新增: 直接加载所有测量数据到一个字典
    measured_db = {}

    for file_path in experiment_files:
        filename = os.path.basename(file_path)
        
        # 统一加载所有测量数据
        if "_measured.json" in filename:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                exp_id = data.get("id")
                if exp_id:
                    measured_db[exp_id] = data
            except Exception as e:
                print(f"[WARN] 无法加载测量文件 {filename}: {e}")
            continue

        # 分离不同类型的设置文件
        if "setup_corrected" in filename:
            match = re.match(r"(.+?)_setup_corrected\.json", filename)
            if match:
                setup_corrected_files[match.group(1)] = file_path
        elif "setup" in filename and filename.endswith("_setup.json"):
            match = re.match(r"(.+?)_setup\.json", filename)
            if match:
                setup_files[match.group(1)] = file_path

    # 现在，根据模式确定要加载哪些实验
    experiments_to_load = {}
    instrument_setups = {}

    if use_instrument_correction:
        # 仪器修正模式: 需要 setup, setup_corrected, 和 measured 数据
        valid_ids = set(setup_files.keys()) & set(setup_corrected_files.keys()) & set(measured_db.keys())
        print(f"[INFO] 仪器修正模式：找到 {len(valid_ids)} 个完整实验数据集")
        for exp_id in valid_ids:
            try:
                with open(setup_files[exp_id], 'r') as f:
                    experiments_to_load[exp_id] = json.load(f)
                with open(setup_corrected_files[exp_id], 'r') as f:
                    instrument_setups[exp_id] = json.load(f)
                print(f"[INFO] 已加载实验: {exp_id} (原始+修正设置)")
            except Exception as e:
                print(f"[ERROR] 加载仪器修正实验 {exp_id} 时出错: {e}")

    else:
        # 标准模式: 需要 setup (或 setup_corrected) 和 measured 数据
        available_setup_ids = set(setup_files.keys()) | set(setup_corrected_files.keys())
        valid_ids = available_setup_ids & set(measured_db.keys())
        print(f"[INFO] 标准模式：找到 {len(valid_ids)} 个完整实验数据集")
        for exp_id in valid_ids:
            try:
                # 优先使用 corrected 文件
                file_to_load = setup_corrected_files.get(exp_id, setup_files.get(exp_id))
                file_type = "corrected" if exp_id in setup_corrected_files else "original"
                with open(file_to_load, 'r') as f:
                    experiments_to_load[exp_id] = json.load(f)
                print(f"[INFO] 已加载实验: {exp_id} (使用{file_type}设置)")
            except Exception as e:
                print(f"[ERROR] 加载标准实验 {exp_id} 时出错: {e}")
                
    return experiments_to_load, instrument_setups, measured_db

def get_file_list(path_spec):
    """根据路径规格（单个文件、目录、通配符）获取文件列表"""
    if '*' in path_spec or '?' in path_spec:
        return glob.glob(path_spec)
    elif os.path.isdir(path_spec):
        return [os.path.join(path_spec, f) for f in os.listdir(path_spec)]
    elif os.path.isfile(path_spec):
        return [path_spec]
    return []

async def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(
        description="Quantum Theory Evaluator - A tool to evaluate quantum theories against experiments using LLMs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- 主要运行参数 ---
    main_group = parser.add_argument_group('Main Parameters')
    main_group.add_argument("--theory_path", type=str, required=True, help="Path to the JSON file or directory containing theory definitions.")
    main_group.add_argument("--experiment_dir", type=str, required=True, help="Directory containing experiment setup and data files.")
    main_group.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save evaluation results.")
    main_group.add_argument("--schema_version", type=str, default="any", help="The schema version of the theories to be loaded ('any' to load all).")
    main_group.add_argument("--max_theories", type=int, default=None, help="Maximum number of theories to evaluate.")
    main_group.add_argument("--max_experiments", type=int, default=None, help="Maximum number of experiments to run for each theory.")
    main_group.add_argument("--chi2_threshold", type=float, default=4.0, help="Chi-squared threshold for determining prediction success.")
    main_group.add_argument("--use_instrument_correction", action='store_true', help="Enable instrument correction model.")
    main_group.add_argument("--start-at-index", type=int, default=0, help="0-based index of the theory to start evaluation from.")

    # --- LLM 相关参数 ---
    model_group = parser.add_argument_group('LLM Configuration')
    model_group.add_argument("--model_source", type=str, default="deepseek", choices=["openai", "deepseek", "google"], help="LLM provider.")
    model_group.add_argument("--model_name", type=str, default="deepseek-reasoner", help="Specific model name.")
    model_group.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature for the LLM (0.0 to 1.0).")

    # --- 角色评估参数 ---
    role_eval_group = parser.add_argument_group('Role-playing Evaluation (Optional)')
    role_eval_group.add_argument("--run_role_evaluation", action='store_true', help="Run role-playing evaluation after experimental evaluation.")
    role_eval_group.add_argument("--role_success_threshold", type=float, default=0.75, help="Success rate threshold for a theory to be passed to role evaluation.")
    role_eval_group.add_argument("--role_model_source", type=str, default="openai", choices=["openai", "deepseek", "google"], help="LLM provider for role evaluation.")
    role_eval_group.add_argument("--role_model_name", type=str, default="gpt-4o-mini", help="Specific model name for role evaluation.")

    args = parser.parse_args()

    # --- 1. 设置 ---
    # 创建唯一的输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, f"run_{timestamp}_{args.model_name.replace('/', '_')}")
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"[SETUP] Results will be saved in: {run_output_dir}")

    # 初始化LLM
    llm = LLMInterface(model_source=args.model_source, model_name=args.model_name)
    print(f"[SETUP] Initialized LLM: {args.model_source}/{args.model_name}")

    # --- 2. 加载数据 ---
    print(f"\n[LOAD] Loading theories from: {args.theory_path}")
    schema_to_load = None if args.schema_version.lower() == 'any' else args.schema_version
    theories = load_theories_from_sources(args.theory_path, schema_version=schema_to_load)
    if not theories:
        print("[ERROR] No theories loaded. Exiting.")
        return
    
    # 限制理论数量
    if args.max_theories is not None and args.max_theories > 0:
        theories = dict(list(theories.items())[:args.max_theories])
    print(f"[LOAD] Loaded {len(theories)} theories for evaluation: {list(theories.keys())}")

    # 加载实验
    print(f"\n[LOAD] Loading experiments from: {args.experiment_dir}")
    
    experiments, instrument_setups, measured_data = load_experiments_from_directory(
        args.experiment_dir, args.use_instrument_correction
    )
    if not experiments:
        print("[ERROR] No experiments loaded. Exiting.")
        return
        
    # 限制实验数量
    if args.max_experiments is not None and args.max_experiments > 0:
        experiments = dict(list(experiments.items())[:args.max_experiments])
    print(f"[LOAD] Loaded {len(experiments)} experiments for evaluation: {list(experiments.keys())}")


    # --- 3. 运行评估 ---
    print(f"\n[EVAL] Starting evaluation...")
    all_results = []
    
    for i, (theory_name, theory) in enumerate(theories.items()):
        if i < args.start_at_index:
            continue
            
        print(f"\n--- Evaluating Theory {i+1}/{len(theories)}: {theory_name} ---")
        theory_results = []
        
        # 为每个理论创建一个子目录
        theory_output_dir = os.path.join(run_output_dir, theory_name.replace(" ", "_").lower())
        os.makedirs(theory_output_dir, exist_ok=True)

        for j, (exp_id, setup_exp) in enumerate(experiments.items()):
            print(f"--- Running Experiment {j+1}/{len(experiments)}: {exp_id} ---")

            # 确定仪器修正的设置（如果启用）
            corrected_setup = instrument_setups.get(exp_id) if args.use_instrument_correction else None
            
            result = await evaluate_theory_experiment(
                theory=theory,
                setup_exp=setup_exp,
                measured_data=measured_data,
                llm=llm,
                args=args,
                # 将所有子输出定向到理论的目录
                output_prefix=f"{theory_name.replace(' ', '_').lower()}_vs_{exp_id.replace(' ', '_').lower()}",
                corrected_setup_exp=corrected_setup
            )
            
            if result:
                theory_results.append(result)
        
        # 保存该理论的所有评估结果
        if theory_results:
            theory_summary_file = os.path.join(theory_output_dir, "_summary.json")
            with open(theory_summary_file, "w", encoding="utf-8") as f:
                json.dump(theory_results, f, ensure_ascii=False, indent=2)
            print(f"[SUMMARY] Theory '{theory_name}' summary saved to: {theory_summary_file}")
            all_results.extend(theory_results)

    # --- 4. 生成最终报告 ---
    if all_results:
        # 对最终结果进行排名
        # 使用修正后的chi2（如果可用），否则使用原始chi2
        use_corrected = args.use_instrument_correction and all(
            'chi2_corrected' in r for r in all_results
        )
        
        # 汇总每个理论的表现
        theory_performance = {}
        for result in all_results:
            t_name = result['theory_name']
            if t_name not in theory_performance:
                theory_performance[t_name] = {
                    'success_count': 0,
                    'total_count': 0,
                    'chi2_sum': 0,
                    'chi2_list': []
                }
            
            theory_performance[t_name]['total_count'] += 1
            
            # 判断成功与否
            is_success = result.get('success_corrected', result['success']) if use_corrected else result['success']
            if is_success:
                theory_performance[t_name]['success_count'] += 1

            # 累加chi2
            chi2_to_add = result.get('chi2_corrected', result['chi2']) if use_corrected else result['chi2']
            if chi2_to_add is not None:
                theory_performance[t_name]['chi2_sum'] += chi2_to_add
                theory_performance[t_name]['chi2_list'].append(chi2_to_add)

        # 计算成功率和平均chi2
        ranked_theories = []
        for t_name, perf_data in theory_performance.items():
            success_rate = perf_data['success_count'] / perf_data['total_count'] if perf_data['total_count'] > 0 else 0
            average_chi2 = perf_data['chi2_sum'] / len(perf_data['chi2_list']) if perf_data['chi2_list'] else float('inf')
            
            ranked_theories.append({
                "theory_name": t_name,
                "success_rate": success_rate,
                "average_chi2": average_chi2,
                "experiments_count": perf_data['total_count']
            })
            
        # 按成功率降序，平均chi2升序排序
        ranked_theories.sort(key=lambda x: (-x['success_rate'], x['average_chi2']))

        final_summary_file = os.path.join(run_output_dir, "final_evaluation_summary.json")
        with open(final_summary_file, "w", encoding="utf-8") as f:
            json.dump(ranked_theories, f, ensure_ascii=False, indent=2)
        print(f"\n[FINAL] Overall experimental evaluation summary saved to: {final_summary_file}")

        # 打印实验评估排名
        print("\n" + "="*100)
        print("📊 实验评估排名")
        print("="*100)
        print(f"{'排名':<5} {'理论名称':<40} {'成功率':<15} {'平均χ²':<15} {'实验数':<10}")
        print("-"*100)
        for i, rank in enumerate(ranked_theories, 1):
            print(f"{i:<5} {rank['theory_name']:<40} {rank['success_rate']*100:14.1f}% {rank['average_chi2']:<15.4f} {rank['experiments_count']:<10}")
        print("-"*100)
        
        # --- 5. 自动运行角色评估 (如果启用) ---
        if args.run_role_evaluation:
            # 筛选高成功率理论
            high_success_theories = [
                r for r in ranked_theories 
                if r['success_rate'] >= args.role_success_threshold
            ]
            
            if high_success_theories:
                # 调用角色评估模块
                await run_role_evaluation_for_theories(
                    high_success_theories=high_success_theories,
                    all_theories_definitions=theories, # 使用已加载的理论定义
                    output_dir=run_output_dir, # 在同一运行目录下输出
                    model_source=args.role_model_source,
                    model_name=args.role_model_name
                )
            else:
                print(f"\n[INFO] 没有理论达到 {args.role_success_threshold*100:.0f}% 的成功率阈值，跳过角色评估。")

    else:
        print("\n[FINAL] No evaluations were successfully completed.")

    print("\nEvaluation run finished.")


if __name__ == "__main__":
    # Windows平台兼容性设置
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INFO] 运行被用户中断。")
    except Exception as e:
        print(f"[CRITICAL ERROR] 发生未处理的异常: {e}")
        import traceback
        traceback.print_exc()