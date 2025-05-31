#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
概念提取器

从文献资料中提取量子理论相关的概念和公式。
支持从多种格式（文本、PDF等）中提取信息。
直接使用LLM接口进行概念提取。
"""

import os
import csv
import json
from typing import List, Dict, Tuple, Optional, Any
import glob
import re

from theory_generation.llm_interface import LLMInterface

class ConceptExtractor:
    """从文献资料中提取概念和公式的类"""
    
    def __init__(self, llm_interface: LLMInterface):
        """
        初始化概念提取器
        
        Args:
            llm_interface: LLM接口实例
        """
        self.llm = llm_interface
        self.extracted_concepts = []
        self.extracted_formulas = []
        
    def extract_from_text(self, text: str, source: str = "unknown") -> Tuple[List[Dict], List[Dict]]:
        """
        从文本中提取概念和公式
        
        Args:
            text: 要提取的文本内容
            source: 文本来源标识
            
        Returns:
            Tuple[List[Dict], List[Dict]]: 提取的概念和公式列表
        """
        print(f"[INFO] 从文本中提取概念和公式 (来源: {source})")
        
        # 使用LLM接口的概念提取方法
        concepts, formulas = self.llm.extract_concepts_from_text(text, source)
        
        # 添加到提取结果中
        self.extracted_concepts.extend(concepts)
        self.extracted_formulas.extend(formulas)
        
        print(f"[INFO] 提取了 {len(concepts)} 个概念和 {len(formulas)} 个公式")
        return concepts, formulas

    def extract_from_file(self, file_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        从文件中提取概念和公式
        
        Args:
            file_path: 要处理的文件路径
            
        Returns:
            Tuple[List[Dict], List[Dict]]: 提取的概念和公式列表
        """
        if not os.path.exists(file_path):
            print(f"[ERROR] 文件不存在: {file_path}")
            return [], []
        
        # 获取文件名作为来源标识
        file_name = os.path.basename(file_path)
        
        # 根据文件扩展名处理不同类型的文件
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        try:
            if ext in ['.txt', '.md', '.rst', '.tex']:
                # 处理纯文本文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                return self.extract_from_text(text, source=file_name)
                
            elif ext == '.pdf':
                # PDF处理需要额外的库
                try:
                    from pypdf import PdfReader
                    
                    reader = PdfReader(file_path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n\n"
                    
                    return self.extract_from_text(text, source=file_name)
                except ImportError:
                    print("[WARN] 处理PDF需要pypdf库，请使用pip install pypdf安装")
                    return [], []
                    
            elif ext in ['.json', '.jsonl']:
                # 处理JSON文件，假设其中已包含概念和公式
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                concepts = data.get('concepts', [])
                formulas = data.get('formulas', [])
                
                # 添加来源信息
                for concept in concepts:
                    concept["source"] = file_name
                for formula in formulas:
                    formula["source"] = file_name
                
                self.extracted_concepts.extend(concepts)
                self.extracted_formulas.extend(formulas)
                
                return concepts, formulas
                
            else:
                print(f"[WARN] 不支持的文件类型: {ext}")
                return [], []
                
        except Exception as e:
            print(f"[ERROR] 处理文件 {file_path} 时出错: {str(e)}")
            return [], []
    
    def extract_from_directory(self, directory_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        从目录中提取概念和公式
        
        Args:
            directory_path: 目录路径
            
        Returns:
            Tuple[List[Dict], List[Dict]]: 提取的概念和公式列表
        """
        if not os.path.isdir(directory_path):
            print(f"[ERROR] 目录不存在: {directory_path}")
            return [], []
        
        # 支持的文件类型
        supported_extensions = ['.txt', '.md', '.rst', '.tex', '.pdf', '.json', '.jsonl']
        
        # 查找所有支持的文件
        all_files = []
        for ext in supported_extensions:
            all_files.extend(glob.glob(os.path.join(directory_path, f"*{ext}")))
        
        if not all_files:
            print(f"[WARN] 在 {directory_path} 中未找到支持的文件")
            return [], []
        
        print(f"[INFO] 在 {directory_path} 中找到了 {len(all_files)} 个文件")
        
        # 处理每个文件
        for file_path in all_files:
            print(f"[INFO] 处理文件: {os.path.basename(file_path)}")
            self.extract_from_file(file_path)
        
        # 去重，基于概念和公式的名称
        self.extracted_concepts = self._deduplicate_items(self.extracted_concepts, key='name')
        self.extracted_formulas = self._deduplicate_items(self.extracted_formulas, key='name')
        
        print(f"[INFO] 去重后共有 {len(self.extracted_concepts)} 个概念和 {len(self.extracted_formulas)} 个公式")
        
        return self.extracted_concepts, self.extracted_formulas
    
    def _deduplicate_items(self, items: List[Dict], key: str = 'name') -> List[Dict]:
        """
        基于指定键去除重复项
        
        Args:
            items: 要去重的项目列表
            key: 用于去重的键
            
        Returns:
            List[Dict]: 去重后的列表
        """
        seen = {}
        unique_items = []
        
        for item in items:
            item_key = item.get(key, '').strip().lower()
            if not item_key:
                continue  # 跳过没有有效键的项目
                
            if item_key not in seen:
                seen[item_key] = True
                unique_items.append(item)
        
        return unique_items
    
    def save_to_csv(self, output_dir: str) -> None:
        """
        将提取的概念和公式保存到CSV文件
        
        Args:
            output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存概念
        concepts_path = os.path.join(output_dir, 'extracted_concepts.csv')
        with open(concepts_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['name', 'description', 'domain', 'source']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for concept in self.extracted_concepts:
                writer.writerow({
                    'name': concept.get('name', ''),
                    'description': concept.get('description', ''),
                    'domain': concept.get('domain', ''),
                    'source': concept.get('source', '')
                })
        
        # 保存公式
        formulas_path = os.path.join(output_dir, 'extracted_formulas.csv')
        with open(formulas_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['name', 'expression', 'description', 'variables', 'source']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for formula in self.extracted_formulas:
                # 处理变量列表，将其转换为字符串表示
                variables = formula.get('variables', [])
                var_text = '; '.join([f"{v.get('symbol', '')}: {v.get('meaning', '')}" for v in variables])
                
                writer.writerow({
                    'name': formula.get('name', ''),
                    'expression': formula.get('expression', ''),
                    'description': formula.get('description', ''),
                    'variables': var_text,
                    'source': formula.get('source', '')
                })
        
        # 也保存为JSON以保留完整数据结构
        json_path = os.path.join(output_dir, 'extracted_data.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'concepts': self.extracted_concepts,
                'formulas': self.extracted_formulas
            }, f, ensure_ascii=False, indent=2)
        
        print(f"[INFO] 概念保存到: {concepts_path}")
        print(f"[INFO] 公式保存到: {formulas_path}")
        print(f"[INFO] 完整数据保存到: {json_path}")
    
    def extract_from_preprocessed(self, json_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        从预处理的JSON文件中提取概念和公式
        
        Args:
            json_path: 预处理JSON文件路径
            
        Returns:
            Tuple[List[Dict], List[Dict]]: 提取的概念和公式列表
        """
        if not os.path.exists(json_path):
            print(f"[ERROR] 预处理文件不存在: {json_path}")
            return [], []
        
        try:
            # 读取预处理文件
            with open(json_path, 'r', encoding='utf-8') as f:
                preprocessed_data = json.load(f)
            
            # 获取文件名作为来源标识
            file_name = os.path.basename(json_path)
            
            # 提取原始文本
            text = preprocessed_data.get('text', '')
            metadata = preprocessed_data.get('metadata', {})
            original_filename = metadata.get('filename', file_name)
            
            if not text:
                print(f"[WARN] 预处理文件 {file_name} 不包含文本内容")
                return [], []
            
            # 从文本中提取概念和公式
            source = f"预处理: {original_filename}"
            return self.extract_from_text(text, source=source)
            
        except Exception as e:
            print(f"[ERROR] 处理预处理文件 {json_path} 时出错: {str(e)}")
            return [], [] 