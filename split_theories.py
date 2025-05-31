#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分割理论文件

将单一的理论集合JSON文件分割成多个单独的JSON文件，每个文件包含一个量子诠释理论。
"""

import os
import json
import argparse
import re

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] 创建目录: {directory}")

def sanitize_filename(name):
    """将理论名称转换为有效的文件名"""
    # 删除非法字符和空格
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    # 空格转下划线
    name = name.replace(" ", "_")
    # 确保文件名不太长
    if len(name) > 100:
        name = name[:100]
    return name.lower()

def main():
    parser = argparse.ArgumentParser(description="将理论集合文件分割为单独的JSON文件")
    parser.add_argument("--input_file", type=str, 
                      default="data/generated_theories/all_quantum_interpretations.json",
                      help="包含所有理论的JSON文件")
    parser.add_argument("--output_dir", type=str, 
                      default="data/generated_theories",
                      help="输出目录，默认与输入文件相同")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    ensure_directory_exists(args.output_dir)
    
    # 1. 加载理论数据
    print(f"[步骤1] 加载理论数据: {args.input_file}")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            theories = json.load(f)
        
        if not isinstance(theories, list):
            print(f"[ERROR] 文件格式错误: {args.input_file} 不是理论数组")
            return
            
        print(f"[INFO] 加载了 {len(theories)} 个理论")
    except Exception as e:
        print(f"[ERROR] 加载理论文件失败: {str(e)}")
        return
    
    # 2. 为每个理论创建单独的文件
    print(f"[步骤2] 开始分割理论文件")
    
    created_count = 0
    for theory in theories:
        # 获取理论ID和名称
        theory_id = theory.get("id", "")
        theory_name = theory.get("name", "")
        
        if not theory_name:
            print(f"[WARN] 跳过无名称的理论: {theory_id}")
            continue
        
        # 生成文件名
        if theory_id:
            # 使用ID+名称作为文件名
            filename = f"{theory_id}_{sanitize_filename(theory_name)}.json"
        else:
            # 仅使用名称作为文件名
            filename = f"{sanitize_filename(theory_name)}.json"
        
        # 保存理论到单独文件
        output_path = os.path.join(args.output_dir, filename)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(theory, f, ensure_ascii=False, indent=2)
            
            created_count += 1
            print(f"[INFO] 保存理论: {theory_name} -> {filename}")
        except Exception as e:
            print(f"[ERROR] 保存理论 {theory_name} 失败: {str(e)}")
    
    # 3. 总结
    print(f"\n[完成] 已将 {len(theories)} 个理论分割成 {created_count} 个单独的文件")
    print(f"文件保存在: {args.output_dir}")
    print("现在可以直接使用这些单独的理论文件进行后续处理")

if __name__ == "__main__":
    main()
