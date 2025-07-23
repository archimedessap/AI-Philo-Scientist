#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_adapter.py - 理论生成方法适配器基类
====================================

定义所有理论生成方法必须实现的统一接口。
确保与 run_clean_evolution.py 的完全兼容性。
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime


class TheoryGenerationMethod(ABC):
    """理论生成方法基类
    
    所有理论生成方法都必须继承此类并实现 generate() 方法。
    此设计确保与现有 run_clean_evolution.py 系统的完全兼容性。
    """
    
    def __init__(self, 
                 theories_dir: Union[str, Path],
                 output_dir: Union[str, Path],
                 max_pairs: int = 20,
                 variants_per_contradiction: int = 3,
                 model_source: str = "google",
                 model_name: str = "gemini-2.5-flash",
                 **kwargs):
        """初始化方法适配器
        
        Args:
            theories_dir: 先验理论目录
            output_dir: 输出目录
            max_pairs: 最大分析对数
            variants_per_contradiction: 每个矛盾生成的变体数
            model_source: 模型来源
            model_name: 模型名称
            **kwargs: 其他参数
        """
        self.theories_dir = Path(theories_dir)
        self.output_dir = Path(output_dir)
        self.max_pairs = max_pairs
        self.variants_per_contradiction = variants_per_contradiction
        self.model_source = model_source
        self.model_name = model_name
        self.kwargs = kwargs
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 记录初始化信息
        self._log_info(f"初始化 {self.__class__.__name__}")
        self._log_info(f"理论目录: {self.theories_dir}")
        self._log_info(f"输出目录: {self.output_dir}")
        self._log_info(f"模型: {self.model_source}/{self.model_name}")
    
    @abstractmethod
    def generate(self) -> Dict[str, Any]:
        """生成理论 - 子类必须实现
        
        Returns:
            生成结果字典，必须包含以下字段：
            - success: bool, 是否成功
            - theories: List[Dict], 生成的理论列表
            - metadata: Dict, 生成元数据
            - output_dir: str, 实际输出目录
        """
        pass
    
    def _ensure_output_format(self, result: Any) -> Dict[str, Any]:
        """确保输出格式标准化
        
        Args:
            result: 原始生成结果
            
        Returns:
            标准化的结果字典
        """
        if isinstance(result, dict):
            # 确保必要字段存在
            standard_result = {
                'success': result.get('success', True),
                'theories': result.get('theories', []),
                'metadata': result.get('metadata', {}),
                'output_dir': str(self.output_dir),
                'timestamp': datetime.now().isoformat(),
                'method_class': self.__class__.__name__
            }
            
            # 合并其他字段
            for key, value in result.items():
                if key not in standard_result:
                    standard_result[key] = value
                    
            return standard_result
        else:
            # 处理非字典结果
            return {
                'success': True,
                'theories': result if isinstance(result, list) else [result],
                'metadata': {'raw_result_type': type(result).__name__},
                'output_dir': str(self.output_dir),
                'timestamp': datetime.now().isoformat(),
                'method_class': self.__class__.__name__
            }
    
    def _load_theories(self) -> List[Dict[str, Any]]:
        """加载先验理论
        
        Returns:
            理论列表
        """
        theories = []
        
        if not self.theories_dir.exists():
            self._log_warning(f"理论目录不存在: {self.theories_dir}")
            return theories
        
        # 查找理论文件 - 支持 .txt 和 .json 格式
        txt_files = list(self.theories_dir.glob("*.txt"))
        json_files = list(self.theories_dir.glob("*.json"))
        
        self._log_info(f"找到 {len(txt_files)} 个TXT理论文件和 {len(json_files)} 个JSON理论文件")
        
        # 加载TXT格式理论文件
        for theory_file in txt_files:
            try:
                with open(theory_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                if content:
                    theories.append({
                        'name': theory_file.stem,
                        'content': content,
                        'file_path': str(theory_file),
                        'format': 'txt'
                    })
            except Exception as e:
                self._log_error(f"加载TXT理论文件失败 {theory_file}: {e}")
        
        # 加载JSON格式理论文件
        for theory_file in json_files:
            try:
                with open(theory_file, 'r', encoding='utf-8') as f:
                    theory_data = json.load(f)
                
                # 提取理论内容
                name = theory_data.get('name', theory_file.stem)
                
                # 构建内容字符串 - 整合理论的各个部分
                content_parts = []
                
                # 添加基本信息
                if 'summary' in theory_data:
                    content_parts.append(f"概述: {theory_data['summary']}")
                elif 'description' in theory_data:
                    content_parts.append(f"描述: {theory_data['description']}")
                
                # 添加哲学假设
                if 'philosophical_assumptions' in theory_data:
                    assumptions = theory_data['philosophical_assumptions']
                    content_parts.append(f"哲学假设: {self._format_complex_field(assumptions)}")
                
                # 添加核心原理
                if 'core_principles' in theory_data:
                    principles = theory_data['core_principles']
                    content_parts.append(f"核心原理: {self._format_complex_field(principles)}")
                
                # 添加物理机制
                if 'physical_mechanisms' in theory_data:
                    mechanisms = theory_data['physical_mechanisms']
                    content_parts.append(f"物理机制: {self._format_complex_field(mechanisms)}")
                
                # 添加数学形式主义
                if 'mathematical_formalism' in theory_data:
                    math_form = theory_data['mathematical_formalism']
                    content_parts.append(f"数学形式主义: {self._format_complex_field(math_form)}")
                
                # 添加预测
                if 'predictions' in theory_data:
                    predictions = theory_data['predictions']
                    content_parts.append(f"预测: {self._format_complex_field(predictions)}")
                
                # 添加哲学观点
                if 'philosophy' in theory_data:
                    philosophy = theory_data['philosophy']
                    content_parts.append(f"哲学观点: {self._format_complex_field(philosophy)}")
                
                # 如果没有找到标准字段，使用整个JSON作为内容
                if not content_parts:
                    content_parts.append(json.dumps(theory_data, ensure_ascii=False, indent=2))
                
                content = '\n\n'.join(content_parts)
                
                if content:
                    theories.append({
                        'name': name,
                        'content': content,
                        'file_path': str(theory_file),
                        'format': 'json',
                        'raw_data': theory_data  # 保留原始JSON数据
                    })
            except Exception as e:
                self._log_error(f"加载JSON理论文件失败 {theory_file}: {e}")
        
        self._log_info(f"成功加载 {len(theories)} 个理论")
        return theories
    
    def _format_complex_field(self, field_value) -> str:
        """格式化复杂的JSON字段为可读字符串
        
        Args:
            field_value: 要格式化的字段值（可能是字典、列表或字符串）
            
        Returns:
            格式化后的字符串
        """
        if isinstance(field_value, str):
            return field_value
        elif isinstance(field_value, list):
            formatted_items = []
            for item in field_value:
                if isinstance(item, dict):
                    # 处理字典项
                    if 'principle' in item and 'statement' in item:
                        formatted_items.append(f"{item['principle']}: {item['statement']}")
                    elif 'mechanism' in item and 'description' in item:
                        formatted_items.append(f"{item['mechanism']}: {item['description']}")
                    else:
                        # 通用字典格式化
                        dict_parts = []
                        for k, v in item.items():
                            if isinstance(v, (str, int, float)):
                                dict_parts.append(f"{k}: {v}")
                        if dict_parts:
                            formatted_items.append(f"({', '.join(dict_parts)})")
                        else:
                            formatted_items.append(str(item))
                elif isinstance(item, str):
                    formatted_items.append(item)
                else:
                    formatted_items.append(str(item))
            return '; '.join(formatted_items)
        elif isinstance(field_value, dict):
            # 处理嵌套字典
            formatted_parts = []
            for key, value in field_value.items():
                if isinstance(value, dict):
                    # 递归处理嵌套字典
                    nested_parts = []
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, list):
                            nested_parts.append(f"{nested_key}: {self._format_complex_field(nested_value)}")
                        elif isinstance(nested_value, str):
                            nested_parts.append(f"{nested_key}: {nested_value}")
                        else:
                            nested_parts.append(f"{nested_key}: {str(nested_value)}")
                    formatted_parts.append(f"{key}: ({'; '.join(nested_parts)})")
                elif isinstance(value, list):
                    formatted_parts.append(f"{key}: {self._format_complex_field(value)}")
                elif isinstance(value, str):
                    formatted_parts.append(f"{key}: {value}")
                else:
                    formatted_parts.append(f"{key}: {str(value)}")
            return '; '.join(formatted_parts)
        else:
            return str(field_value)
    
    def _save_result(self, result: Dict[str, Any], filename: str = "generation_result.json"):
        """保存生成结果
        
        Args:
            result: 生成结果
            filename: 文件名
        """
        result_path = self.output_dir / filename
        
        try:
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            self._log_info(f"保存结果到: {result_path}")
        except Exception as e:
            self._log_error(f"保存结果失败: {e}")
    
    def _log_info(self, message: str):
        """记录信息日志"""
        print(f"[📝] {self.__class__.__name__}: {message}")
    
    def _log_warning(self, message: str):
        """记录警告日志"""
        print(f"[⚠️] {self.__class__.__name__}: {message}")
    
    def _log_error(self, message: str):
        """记录错误日志"""
        print(f"[❌] {self.__class__.__name__}: {message}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取适配器配置
        
        Returns:
            配置字典
        """
        return {
            'theories_dir': str(self.theories_dir),
            'output_dir': str(self.output_dir),
            'max_pairs': self.max_pairs,
            'variants_per_contradiction': self.variants_per_contradiction,
            'model_source': self.model_source,
            'model_name': self.model_name,
            **self.kwargs
        }


class GenerationResult:
    """生成结果的标准化包装类"""
    
    def __init__(self, 
                 success: bool = True,
                 theories: List[Dict] = None,
                 metadata: Dict = None,
                 output_dir: str = "",
                 error_message: str = ""):
        self.success = success
        self.theories = theories or []
        self.metadata = metadata or {}
        self.output_dir = output_dir
        self.error_message = error_message
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'success': self.success,
            'theories': self.theories,
            'metadata': self.metadata,
            'output_dir': self.output_dir,
            'error_message': self.error_message,
            'timestamp': self.timestamp
        }
    
    def __str__(self) -> str:
        status = "✅" if self.success else "❌"
        return f"{status} 生成{len(self.theories)}个理论 (输出: {self.output_dir})" 