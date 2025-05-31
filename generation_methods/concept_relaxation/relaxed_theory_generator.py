#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于概念矛盾点放松法的理论生成器

通过识别两个理论之间的矛盾点，并在这些点上进行概念放松，
生成新的理论解释。
"""

from typing import Dict, List, Tuple, Any
import numpy as np

from .contradiction_detector import identify_contradictions
from .concept_relaxation import relax_concept
from theory_generation.llm_interface import LLMInterface

def generate_relaxed_theory(theory_a: Dict, theory_b: Dict, embedding_model: Any = None, **kwargs) -> Dict:
    """
    使用概念矛盾点放松法生成新理论
    
    Args:
        theory_a: 第一个理论的描述
        theory_b: 第二个理论的描述
        embedding_model: 用于嵌入的模型
        **kwargs: 其他参数
    
    Returns:
        Dict: 生成的新理论
    """
    print(f"[INFO] 开始基于 {theory_a.get('name', '理论A')} 和 {theory_b.get('name', '理论B')} 生成新理论...")
    
    # 步骤1: 识别两个理论之间的矛盾点
    contradictions = identify_contradictions(theory_a, theory_b, embedding_model)
    
    # 步骤2: 对每个矛盾点进行概念放松
    relaxed_concepts = []
    for contradiction in contradictions:
        relaxed_concept = relax_concept(
            contradiction, 
            theory_a, 
            theory_b,
            **kwargs
        )
        relaxed_concepts.append(relaxed_concept)
    
    # 步骤3: 使用LLM整合放松后的概念，生成连贯的新理论
    llm = LLMInterface(
        model_source=kwargs.get("model_source", "openai"),
        model_name=kwargs.get("model_name", "gpt-4o-mini"),
        api_key=kwargs.get("api_key")
    )
    
    # 构建提示词
    prompt = f"""
为我生成一个连贯的新量子理论，整合以下放松后的概念:

理论A: {theory_a.get('name', '理论A')}
理论B: {theory_b.get('name', '理论B')}

放松后的概念:
{relaxed_concepts}

请确保新理论:
1. 整合了上述放松后的概念
2. 解决了两个原始理论之间的矛盾
3. 提供了新的哲学视角
4. 具有内部一致性
5. 能够解释量子现象

以JSON格式返回，包含以下字段:
- theory_name: 新理论名称
- description: 理论描述
- key_concepts: 关键概念列表
- philosophical_stance: 哲学立场
- mathematical_formalism: 数学形式化描述
- empirical_predictions: 经验预测
- relation_to_original_theories: 与原始理论的关系

仅返回JSON，不要添加任何其他文本。
"""
    
    # 调用LLM生成新理论
    messages = [{"role": "user", "content": prompt}]
    response = llm.query(messages, temperature=0.7)
    new_theory = llm.extract_json(response)
    
    print(f"[INFO] 新理论 '{new_theory.get('theory_name', '未命名理论')}' 生成完成")
    
    return new_theory
