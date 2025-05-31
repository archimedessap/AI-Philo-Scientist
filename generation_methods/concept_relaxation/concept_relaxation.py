#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
概念放松模块

基于识别的矛盾点，进行概念放松，生成新的理论概念。
"""

from typing import Dict, List, Any
from theory_generation.llm_interface import LLMInterface

def relax_concept(contradiction: Dict, theory_a: Dict, theory_b: Dict, **kwargs) -> Dict:
    """
    对矛盾概念进行放松，生成新的整合概念
    
    Args:
        contradiction: 矛盾点描述
        theory_a: 第一个理论
        theory_b: 第二个理论
        **kwargs: 其他参数
    
    Returns:
        Dict: 放松后的新概念
    """
    # 初始化LLM
    llm = LLMInterface(
        model_source=kwargs.get("model_source", "openai"),
        model_name=kwargs.get("model_name", "gpt-4o-mini"),
        api_key=kwargs.get("api_key")
    )
    
    # 准备提示词
    concept_a = contradiction.get("concept_a", {})
    concept_b = contradiction.get("concept_b", {})
    concept_name = concept_a.get("name", "未知概念")
    
    prompt = f"""
我需要你帮助放松两个矛盾的量子理论概念，创造一个新的整合概念。

理论A ({theory_a.get('name', '理论A')}) 中的概念:
名称: {concept_a.get('name', '未指定')}
描述: {concept_a.get('description', '未提供描述')}

理论B ({theory_b.get('name', '理论B')}) 中的概念:
名称: {concept_b.get('name', '未指定')}
描述: {concept_b.get('description', '未提供描述')}

矛盾描述: {contradiction.get('description', '这两个概念存在矛盾')}

请创造一个新的整合概念，它应该:
1. 放松或拓展现有概念的限制
2. 保留两个原始概念的核心洞见
3. 解决或缓解两者之间的矛盾
4. 提供新的思考角度
5. 在物理和哲学上都有意义

以JSON格式返回，包含以下字段:
- name: 新概念的名称
- description: 详细描述
- relation_to_original: 与原始概念的关系
- philosophical_implications: 哲学含义
- mathematical_formulation: 可能的数学表达（如适用）

仅返回JSON，不要添加任何其他文本。
"""
    
    # 调用LLM创建放松概念
    messages = [{"role": "user", "content": prompt}]
    response = llm.query(messages, temperature=0.7)
    relaxed_concept = llm.extract_json(response)
    
    # 确保结果包含基本字段
    if not relaxed_concept:
        relaxed_concept = {
            "name": f"整合的{concept_name}",
            "description": "未能成功生成整合概念",
            "relation_to_original": "未知",
            "philosophical_implications": "未知",
            "mathematical_formulation": "未知"
        }
    
    return relaxed_concept
