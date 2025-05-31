#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
理论生成方法选择器

负责根据指定的方法名称，调用相应的理论生成方法实现。
"""

from generation_methods.concept_relaxation.relaxed_theory_generator import generate_relaxed_theory

def select_generation_method(method_name, **kwargs):
    """
    根据方法名称选择并调用对应的理论生成方法
    
    Args:
        method_name (str): 理论生成方法名称
        **kwargs: 传递给具体方法的参数
    
    Returns:
        生成的理论
    
    Raises:
        NotImplementedError: 如果指定的方法未实现
    """
    # 当前支持的方法映射
    method_map = {
        "concept_relaxation": generate_relaxed_theory
    }
    
    if method_name in method_map:
        print(f"[INFO] 使用 {method_name} 方法生成理论")
        return method_map[method_name](**kwargs)
    else:
        raise NotImplementedError(f"方法 '{method_name}' 暂未实现，请选择其他方法。")
