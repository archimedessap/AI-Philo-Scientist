#!/usr/bin/env python3
"""
theory_format_converter.py - 理论格式转换器
============================================

将unified生成器生成的理论格式转换为demo_1.py期望的格式。

Author: UniversalTheoryGen Team
Date: 2025-01-24
"""

import json
from pathlib import Path
from typing import Dict, Any


def convert_theory_format(theory_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将unified生成器的理论格式转换为demo_1.py期望的格式
    
    Args:
        theory_data: 原始理论数据
        
    Returns:
        转换后的理论数据
    """
    # 复制原始数据
    converted = theory_data.copy()
    
    # 映射字段名
    field_mappings = {
        'core_assumptions': 'core_principles',
        'mathematical_formalism': 'formalism',
        'empirical_predictions': 'predictions_and_verifiability'
    }
    
    for old_field, new_field in field_mappings.items():
        if old_field in converted and new_field not in converted:
            converted[new_field] = converted[old_field]
    
    # 确保必需字段存在
    if 'core_principles' not in converted:
        # 尝试从content中提取
        if 'content' in converted:
            content = converted['content']
            # 简单提取Core Assumptions部分
            if '**Core Assumptions:**' in content:
                start = content.find('**Core Assumptions:**')
                end = content.find('**', start + 20)
                if end == -1:
                    end = len(content)
                principles = content[start:end].strip()
                converted['core_principles'] = principles
            else:
                converted['core_principles'] = converted.get('description', 'No principles provided.')
        else:
            converted['core_principles'] = 'No principles provided.'
    
    # 确保formalism字段
    if 'formalism' not in converted and 'mathematical_formalism' in converted:
        converted['formalism'] = converted['mathematical_formalism']
    elif 'formalism' not in converted:
        converted['formalism'] = 'No formalism provided.'
    
    # 确保predictions字段
    if 'predictions_and_verifiability' not in converted:
        if 'empirical_predictions' in converted and converted['empirical_predictions']:
            converted['predictions_and_verifiability'] = converted['empirical_predictions']
        else:
            converted['predictions_and_verifiability'] = 'No predictions provided.'
    
    # 确保mathematical_relation_to_sqm字段
    if 'mathematical_relation_to_sqm' not in converted:
        # 尝试从differences_from_existing中推断
        if 'differences_from_existing' in converted:
            diff = converted['differences_from_existing'].lower()
            if 'interpretation' in diff:
                converted['mathematical_relation_to_sqm'] = 'interpretation'
            elif 'modification' in diff or 'extension' in diff:
                converted['mathematical_relation_to_sqm'] = 'modification'
            else:
                converted['mathematical_relation_to_sqm'] = 'not_specified'
        else:
            converted['mathematical_relation_to_sqm'] = 'not_specified'
    
    return converted


def convert_theory_file(input_path: Path, output_path: Path = None) -> Path:
    """
    转换理论文件格式
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径（可选，默认覆盖原文件）
        
    Returns:
        输出文件路径
    """
    # 读取原始数据
    with open(input_path, 'r', encoding='utf-8') as f:
        theory_data = json.load(f)
    
    # 转换格式
    converted_data = convert_theory_format(theory_data)
    
    # 确定输出路径
    if output_path is None:
        output_path = input_path
    
    # 保存转换后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    return output_path


def convert_directory(input_dir: Path, output_dir: Path = None, in_place: bool = False):
    """
    转换目录中的所有理论文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录（可选）
        in_place: 是否原地修改文件
    """
    input_dir = Path(input_dir)
    
    if not in_place and output_dir is None:
        raise ValueError("必须指定output_dir或设置in_place=True")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有JSON文件
    theory_files = list(input_dir.glob("*.json"))
    print(f"找到 {len(theory_files)} 个理论文件")
    
    for theory_file in theory_files:
        try:
            if in_place:
                output_path = theory_file
            else:
                output_path = output_dir / theory_file.name
            
            convert_theory_file(theory_file, output_path)
            print(f"✅ 转换成功: {theory_file.name}")
            
        except Exception as e:
            print(f"❌ 转换失败: {theory_file.name} - {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="转换理论文件格式")
    parser.add_argument("input", help="输入文件或目录")
    parser.add_argument("-o", "--output", help="输出文件或目录")
    parser.add_argument("--in-place", action="store_true", 
                       help="原地修改文件（仅用于目录模式）")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 单文件模式
        output_path = Path(args.output) if args.output else None
        result = convert_theory_file(input_path, output_path)
        print(f"转换完成: {result}")
        
    elif input_path.is_dir():
        # 目录模式
        output_dir = Path(args.output) if args.output else None
        convert_directory(input_path, output_dir, args.in_place)
        print("批量转换完成")
        
    else:
        print(f"错误: {input_path} 不是有效的文件或目录")