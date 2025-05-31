#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
理论创新程序

读取分割后的多个理论文件，进行理论比较找出主要矛盾点，
在高维空间放松矛盾点，并生成新的理论变体。
"""

import os
import json
import pickle
import random
import argparse
import asyncio
import glob
import numpy as np
from theory_generation.llm_interface import LLMInterface

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] 创建目录: {directory}")

async def load_theories_from_directory(directory):
    """从目录加载所有理论文件"""
    theories = []
    
    # 查找所有JSON文件
    json_files = glob.glob(os.path.join(directory, "*.json"))
    # 排除all_quantum_interpretations.json
    json_files = [f for f in json_files if not os.path.basename(f).startswith("all_")]
    
    print(f"[INFO] 在 {directory} 中找到 {len(json_files)} 个理论文件")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                theory = json.load(f)
            theories.append(theory)
            print(f"[INFO] 加载理论: {theory.get('name', os.path.basename(file_path))}")
        except Exception as e:
            print(f"[ERROR] 加载理论文件 {file_path} 失败: {str(e)}")
    
    print(f"[INFO] 共加载 {len(theories)} 个理论")
    return theories

def load_embeddings(embeddings_path):
    """加载统一嵌入空间"""
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        if isinstance(embeddings, dict) and "theories" in embeddings:
            # 统一嵌入格式
            theories_embeddings = embeddings["theories"]
            print(f"[INFO] 已加载理论嵌入: {len(theories_embeddings)} 个理论")
            return embeddings
        else:
            # 旧格式，只有理论嵌入
            print(f"[INFO] 已加载旧格式嵌入: {len(embeddings)} 个理论")
            return {"theories": embeddings, "concepts": {}, "formulas": {}}
    except Exception as e:
        print(f"[ERROR] 加载嵌入失败: {str(e)}")
        return {"theories": {}, "concepts": {}, "formulas": {}}

def get_theory_by_name(theories, name):
    """通过名称获取理论详情"""
    for theory in theories:
        if theory.get("name") == name:
            return theory
    return None

def get_theory_by_id(theories, theory_id):
    """通过ID获取理论详情"""
    for theory in theories:
        if theory.get("id") == theory_id:
            return theory
    return None

def get_theory(theories, id_or_name):
    """通过ID或名称获取理论详情"""
    # 先尝试通过ID查找
    theory = get_theory_by_id(theories, id_or_name)
    if theory:
        return theory
    
    # 如果没找到，再尝试通过名称查找
    return get_theory_by_name(theories, id_or_name)

async def compare_theories(llm, theory1, theory2):
    """比较两个理论，找出主要矛盾点"""
    # 构建提示
    prompt = f"""请分析以下两个量子力学诠释理论之间的主要矛盾点:

理论1: {theory1.get('name')}
{theory1.get('core_principles', '')}

理论2: {theory2.get('name')}
{theory2.get('core_principles', '')}

请根据重要性排序，找出这两个理论之间的3-5个关键矛盾点。
不需要刻意对应特定概念或维度，只需列出他们真正的主要分歧。

输出格式为JSON数组，每个矛盾点包含:
- "contradiction": 矛盾点名称
- "theory1_position": 理论1的立场
- "theory2_position": 理论2的立场
- "importance": 重要性评分（1-10）
- "explanation": 简短解释

严格按照以上JSON格式输出，不要包含其他信息。"""

    # 调用LLM
    response = await llm.query_async(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    # 提取JSON
    try:
        # 尝试直接解析
        contradictions = json.loads(response)
        return contradictions
    except:
        # 尝试从文本中提取JSON部分
        try:
            import re
            json_match = re.search(r'(\[.*\])', response, re.DOTALL)
            if json_match:
                contradictions = json.loads(json_match.group(1))
                return contradictions
            else:
                print(f"[ERROR] 无法从响应中提取JSON: {response[:100]}...")
                return []
        except Exception as e:
            print(f"[ERROR] 解析矛盾点失败: {str(e)}")
            return []

def compute_vector_relaxation(vector1, vector2, num_variants=10, relaxation_strengths=None):
    """在高维空间中放松两个向量之间的矛盾点，生成多个变体"""
    # 默认放松强度
    if relaxation_strengths is None:
        relaxation_strengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # 确保生成正确数量的变体
    if len(relaxation_strengths) < num_variants:
        # 扩展放松强度
        relaxation_strengths = np.linspace(min(relaxation_strengths), 
                                          max(relaxation_strengths), 
                                          num_variants).tolist()
    elif len(relaxation_strengths) > num_variants:
        # 随机选择指定数量的放松强度
        relaxation_strengths = random.sample(relaxation_strengths, num_variants)
    
    # 计算差向量
    if len(vector1) != len(vector2):
        print(f"[ERROR] 向量维度不匹配: {len(vector1)} vs {len(vector2)}")
        return []
    
    diff_vector = np.array(vector2) - np.array(vector1)
    
    # 生成变体向量
    variants = []
    for strength in relaxation_strengths:
        # 放松方向改变
        variant_vector = np.array(vector1) + strength * diff_vector
        # 归一化
        variant_vector = variant_vector / np.linalg.norm(variant_vector)
        variants.append(variant_vector.tolist())
    
    return variants

def find_nearest_theories(theory_embeddings, variant_vector, k=5):
    """找到最接近变体向量的K个理论"""
    neighbors = []
    
    for theory_name, embedding in theory_embeddings.items():
        # 计算余弦相似度
        similarity = np.dot(variant_vector, embedding) / (
            np.linalg.norm(variant_vector) * np.linalg.norm(embedding)
        )
        neighbors.append((theory_name, similarity))
    
    # 按相似度排序并返回前K个
    neighbors.sort(key=lambda x: x[1], reverse=True)
    return neighbors[:k]

async def generate_theory_from_contradiction(llm, theory1, theory2, contradiction, variant_vector, 
                                            nearest_theories, all_theories):
    """基于矛盾点变体和最近邻理论生成新理论"""
    # 提取原理论信息
    theory1_name = theory1.get("name")
    theory2_name = theory2.get("name")
    
    # 提取矛盾信息
    contradiction_name = contradiction.get("contradiction")
    pos1 = contradiction.get("theory1_position")
    pos2 = contradiction.get("theory2_position")
    
    # 构建最近邻理论信息
    neighbors_info = []
    for name, similarity in nearest_theories:
        theory = get_theory(all_theories, name)
        if theory:
            core = theory.get("core_principles", "")
            # 截取核心原则的前200个字符
            if len(core) > 200:
                core = core[:200] + "..."
            neighbors_info.append(f"- {name} (相似度: {similarity:.2f}): {core}")
    
    neighbors_text = "\n".join(neighbors_info)
    
    # 构建提示 - 新的提示词模板
    prompt = f"""请生成一个新的量子力学诠释理论，这个理论处理的主要问题是解决以下两个现有理论的矛盾点:

理论1: {theory1_name}
立场: {pos1}

理论2: {theory2_name}
立场: {pos2}

矛盾点: {contradiction_name}

我已经在高维向量空间中探索了放松这个矛盾的可能性，并找到了以下最相似的理论:
{neighbors_text}

请创建一个新的量子力学诠释理论，该理论:
1. 尝试以新颖方式调和或超越上述矛盾
2. 吸收最近邻理论的某些元素
3. 具有内部一致性和物理可行性
4. 写出≤3条核心公设（postulates）
5. 给出关键数学方程：
   • 若继续沿用薛定谔方程，请照列
   • 若需改动，请提供LaTeX形式的完整新方程，并标注各符号含义

请使用以下JSON格式返回新理论:
{{
  "name": "新理论名称",
  "id": "T_NEW_XXX",
  "core_principles": "新理论的核心原则的详细描述",
  "postulates": [
    "第一条核心公设",
    "第二条核心公设",
    "第三条核心公设"
  ],
  "philosophical_assumptions": "哲学假设",
  "historical_development": "可能的发展路径",
  "quantum_phenomena_explanation": {{
    "wave_function_collapse": "如何解释波函数坍缩",
    "quantum_entanglement": "如何解释量子纠缠",
    "quantum_measurement": "如何解释量子测量"
  }},
  "comparison_with_other_interpretations": "与其他诠释的比较",
  "mathematical_formulation": {{
    "equations": [
      {{
        "description": "方程描述",
        "latex": "数学方程的LaTeX表示",
        "symbols_explanation": "各符号含义解释"
      }}
    ],
    "additional_details": "数学表述的其他细节"
  }},
  "testability": "可测试性和实验预测"
}}

请确保理论在数学上严格，并具有能够区分于现有理论的清晰解释力。输出必须是有效的JSON格式，不要包含任何其他文本。"""

    # 调用LLM生成新理论
    response = await llm.query_async(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    
    # 提取JSON
    try:
        # 尝试直接解析
        new_theory = json.loads(response)
        return new_theory
    except:
        # 尝试从文本中提取JSON部分
        try:
            import re
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                new_theory = json.loads(json_match.group(1))
                return new_theory
            else:
                print(f"[ERROR] 无法从响应中提取JSON: {response[:100]}...")
                return None
        except Exception as e:
            print(f"[ERROR] 解析新理论失败: {str(e)}")
            return None

async def main():
    parser = argparse.ArgumentParser(description="理论创新程序")
    
    # LLM接口参数
    parser.add_argument("--model_source", type=str, default="openai",
                        choices=["openai", "deepseek"],
                        help="模型来源")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini",
                        help="模型名称")
    
    # 输入参数
    parser.add_argument("--theories_dir", type=str, 
                        default="data/generated_theories",
                        help="理论文件目录")
    parser.add_argument("--embeddings_file", type=str, 
                        default="data/embeddings/unified_embeddings.pkl",
                        help="统一嵌入文件")
    parser.add_argument("--specific_pair", type=str, default=None,
                        help="特定理论对，格式为'理论1,理论2'，可以是理论名称或ID")
    parser.add_argument("--max_pairs", type=int, default=3,
                        help="最大比较对数，默认为3")
    
    # 矛盾放松参数
    parser.add_argument("--variants_per_contradiction", type=int, default=10,
                        help="每个矛盾点的变体数量")
    parser.add_argument("--neighbors_count", type=int, default=5,
                        help="每个变体的最近邻理论数量")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="data/innovated_theories",
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    ensure_directory_exists(args.output_dir)
    
    # 初始化LLM接口
    llm = LLMInterface(
        model_source=args.model_source,
        model_name=args.model_name,
        request_interval=1.0
    )
    
    # 显示当前使用的模型信息
    model_info = llm.get_current_model_info()
    print(f"[INFO] 当前使用的模型: {model_info['source']} - {model_info['name']}")
    
    # 1. 加载理论数据
    print(f"\n[步骤1] 从 {args.theories_dir} 加载理论数据")
    all_theories = await load_theories_from_directory(args.theories_dir)
    if not all_theories:
        print("[ERROR] 未加载到理论数据，程序终止")
        return
    
    # 2. 加载嵌入向量
    print(f"\n[步骤2] 从 {args.embeddings_file} 加载嵌入向量")
    embeddings = load_embeddings(args.embeddings_file)
    theory_embeddings = embeddings["theories"]
    if not theory_embeddings:
        print("[ERROR] 未加载到理论嵌入向量，程序终止")
        return
    
    # 3. 确定要比较的理论对
    theory_pairs = []
    
    if args.specific_pair:
        # 使用指定的理论对
        theory_identifiers = args.specific_pair.split(',')
        if len(theory_identifiers) != 2:
            print(f"[ERROR] 理论对格式错误: {args.specific_pair}，应为'理论1,理论2'")
            return
            
        # 验证理论存在（可以通过ID或名称）
        theory1 = get_theory(all_theories, theory_identifiers[0])
        theory2 = get_theory(all_theories, theory_identifiers[1])
        
        if not theory1:
            print(f"[ERROR] 找不到理论: {theory_identifiers[0]}")
            return
        if not theory2:
            print(f"[ERROR] 找不到理论: {theory_identifiers[1]}")
            return
            
        print(f"[INFO] 选择比较理论: {theory1.get('name')} (ID: {theory1.get('id', 'N/A')}) vs {theory2.get('name')} (ID: {theory2.get('id', 'N/A')})")
        theory_pairs.append((theory1, theory2))
    else:
        # 生成理论对组合
        available_theories = [t for t in all_theories if t.get("name") in theory_embeddings]
        
        # 随机选择最大对数的理论对
        if len(available_theories) >= 2:
            import random
            from itertools import combinations
            
            all_combinations = list(combinations(available_theories, 2))
            random.shuffle(all_combinations)
            
            theory_pairs = all_combinations[:args.max_pairs]
        else:
            print("[ERROR] 可用理论数量不足，需要至少2个理论")
            return
    
    print(f"[INFO] 将比较 {len(theory_pairs)} 对理论")
    
    # 4. 对每对理论，找出矛盾点并生成新理论
    all_new_theories = []
    
    for i, (theory1, theory2) in enumerate(theory_pairs, 1):
        theory1_name = theory1.get("name")
        theory2_name = theory2.get("name")
        
        print(f"\n[处理理论对 {i}/{len(theory_pairs)}] {theory1_name} vs {theory2_name}")
        
        # 为理论对创建单独目录（提前创建）
        pair_dir = os.path.join(
            args.output_dir, 
            f"{theory1_name}_vs_{theory2_name}".replace(" ", "_")
        )
        ensure_directory_exists(pair_dir)
        
        # 创建一个空列表，用于后续汇总
        pair_new_theories = []
        # 创建汇总文件，但暂时为空
        output_file = os.path.join(pair_dir, "all_variants.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        
        # 4.1 找出矛盾点
        print(f"[步骤3.{i}.1] 比较理论，找出主要矛盾点")
        contradictions = await compare_theories(llm, theory1, theory2)
        
        if not contradictions:
            print(f"[WARN] 未找到矛盾点，跳过该理论对")
            continue
            
        print(f"[INFO] 找到 {len(contradictions)} 个矛盾点")
        
        # 4.2 对每个矛盾点，在向量空间中放松，并生成新理论
        theories_count = 0
        
        for j, contradiction in enumerate(contradictions, 1):
            contradiction_name = contradiction.get("contradiction", f"矛盾点{j}")
            print(f"[步骤3.{i}.2] 处理矛盾点: {contradiction_name}")
            
            # 获取理论的嵌入向量
            vector1 = theory_embeddings.get(theory1_name)
            vector2 = theory_embeddings.get(theory2_name)
            
            if not vector1 or not vector2:
                print(f"[WARN] 缺少理论嵌入向量，跳过该矛盾点")
                continue
            
            # 生成变体向量
            relaxed_variants = compute_vector_relaxation(
                vector1, vector2, 
                num_variants=args.variants_per_contradiction
            )
            
            # 为每个变体生成新理论
            for k, variant_vector in enumerate(relaxed_variants, 1):
                print(f"[步骤3.{i}.3] 处理矛盾点 {j} 的变体 {k}/{len(relaxed_variants)}")
                
                # 找到最近邻理论
                nearest_theories = find_nearest_theories(
                    theory_embeddings, variant_vector,
                    k=args.neighbors_count
                )
                
                # 生成新理论
                new_theory = await generate_theory_from_contradiction(
                    llm, theory1, theory2, contradiction, 
                    variant_vector, nearest_theories, all_theories
                )
                
                if new_theory:
                    # 添加元数据
                    new_theory["generated_from"] = {
                        "theory1": theory1_name,
                        "theory2": theory2_name,
                        "contradiction": contradiction_name,
                        "variant_index": k,
                        "nearest_theories": [name for name, _ in nearest_theories]
                    }
                    
                    # 即时单独保存该理论
                    theory_name = new_theory.get("name", f"新理论_{theories_count+1}")
                    theory_id = new_theory.get("id", f"T_NEW_{theories_count+1}")
                    
                    # 生成文件名
                    theory_filename = theory_name.replace(" ", "_").lower()
                    single_file = os.path.join(pair_dir, f"{theory_id}_{theory_filename}.json")
                    
                    with open(single_file, 'w', encoding='utf-8') as f:
                        json.dump(new_theory, f, ensure_ascii=False, indent=2)
                    
                    print(f"[INFO] 立即保存新理论: {theory_name} → {single_file}")
                    
                    # 同时更新汇总文件
                    pair_new_theories.append(new_theory)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(pair_new_theories, f, ensure_ascii=False, indent=2)
                    
                    # 添加到全局理论列表
                    all_new_theories.append(new_theory)
                    theories_count += 1
        
        # 该理论对处理完毕，输出汇总信息
        if theories_count > 0:
            print(f"[INFO] 理论对 {theory1_name} vs {theory2_name} 已生成 {theories_count} 个新理论，保存到: {pair_dir}")
        else:
            print(f"[WARN] 理论对 {theory1_name} vs {theory2_name} 未生成任何新理论")
    
    # 5. 保存所有新理论
    if all_new_theories:
        all_theories_file = os.path.join(args.output_dir, "all_innovated_theories.json")
        with open(all_theories_file, 'w', encoding='utf-8') as f:
            json.dump(all_new_theories, f, ensure_ascii=False, indent=2)
            
        print(f"\n[完成] 总共生成了 {len(all_new_theories)} 个新理论，已保存到: {all_theories_file}")
    else:
        print("\n[完成] 未生成任何新理论")

if __name__ == "__main__":
    asyncio.run(main())