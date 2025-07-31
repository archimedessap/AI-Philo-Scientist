#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generation_hub.py - AI-Philo1.0 理论生成中心
============================================

统一管理和调度所有理论生成方法的中心模块。
保持与 run_clean_evolution.py 完全兼容的接口。

设计原则：
1. 保持与现有系统的完全兼容性
2. 支持动态方法注册和选择
3. 统一的输入输出格式
4. 最小化对现有代码的影响
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Type
from datetime import datetime
import importlib.util

# 导入基础类型
try:
    from .methods.base_adapter import TheoryGenerationMethod
except ImportError:
    # 当作为模块直接运行时的备用导入
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from methods.base_adapter import TheoryGenerationMethod


class TheoryGenerationHub:
    """理论生成中心 - 统一管理所有生成方法"""
    
    def __init__(self):
        self.methods: Dict[str, Type[TheoryGenerationMethod]] = {}
        self.method_descriptions: Dict[str, str] = {}
        self._register_default_methods()
    
    def register_method(self, name: str, method_class: Type[TheoryGenerationMethod], description: str = ""):
        """注册生成方法
        
        Args:
            name: 方法名称
            method_class: 方法适配器类
            description: 方法描述
        """
        self.methods[name] = method_class
        self.method_descriptions[name] = description
        print(f"[🔧] 注册生成方法: {name} - {description}")
    
    def generate_theories(self, method: str = "direct_synthesis", **kwargs) -> Dict[str, Any]:
        """调用指定方法生成理论
        
        Args:
            method: 生成方法名称
            **kwargs: 传递给生成方法的参数
            
        Returns:
            生成结果字典，包含生成的理论和元数据
            
        Raises:
            ValueError: 如果指定的方法不存在
        """
        if method not in self.methods:
            available_methods = ", ".join(self.methods.keys())
            raise ValueError(f"未知生成方法: {method}. 可用方法: {available_methods}")
        
        print(f"[🎯] 使用 {method} 方法生成理论")
        print(f"[📝] 方法描述: {self.method_descriptions.get(method, '无描述')}")
        
        # 创建方法实例并生成理论
        generator = self.methods[method](**kwargs)
        result = generator.generate()
        
        # 添加生成元数据
        if isinstance(result, dict):
            result['generation_metadata'] = {
                'method': method,
                'timestamp': datetime.now().isoformat(),
                'hub_version': '1.0.0'
            }
        
        return result
    
    def list_methods(self) -> List[str]:
        """列出所有可用方法"""
        return list(self.methods.keys())
    
    def get_method_info(self, method: str = None) -> Dict[str, str]:
        """获取方法信息
        
        Args:
            method: 方法名称，None则返回所有方法信息
            
        Returns:
            方法信息字典
        """
        if method:
            if method not in self.methods:
                raise ValueError(f"未知方法: {method}")
            return {method: self.method_descriptions.get(method, "无描述")}
        else:
            return self.method_descriptions.copy()
    
    def _register_default_methods(self):
        """注册默认的生成方法"""
        try:
            # 导入并注册 direct_synthesis 适配器
            try:
                from .methods.direct_synthesis_adapter import DirectSynthesisAdapter
            except ImportError:
                from methods.direct_synthesis_adapter import DirectSynthesisAdapter
            self.register_method(
                "direct_synthesis", 
                DirectSynthesisAdapter,
                "基于矛盾分析的直接合成方法 (AI-Philo1.0核心方法)"
            )
        except ImportError as e:
            print(f"[⚠️] 无法导入 direct_synthesis 适配器: {e}")
        
        try:
            # 导入并注册 multi_level 适配器
            try:
                from .methods.multi_level_adapter import MultiLevelAdapter
            except ImportError:
                from methods.multi_level_adapter import MultiLevelAdapter
            self.register_method(
                "multi_level", 
                MultiLevelAdapter,
                "多层级创新生成方法"
            )
        except ImportError as e:
            print(f"[⚠️] 无法导入 multi_level 适配器: {e}")
        
        try:
            # 导入并注册 unified_generator 适配器
            try:
                from .methods.unified_generator_adapter import UnifiedGeneratorAdapter
            except ImportError:
                from methods.unified_generator_adapter import UnifiedGeneratorAdapter
            self.register_method(
                "unified_generator", 
                UnifiedGeneratorAdapter,
                "统一生成器方法"
            )
        except ImportError as e:
            print(f"[⚠️] 无法导入 unified_generator 适配器: {e}")
        
        try:
            # 导入并注册增强的 unified 适配器（基于高维概念空间）
            try:
                from .methods.unified_generator_adapter import UnifiedGeneratorAdapter
            except ImportError:
                from methods.unified_generator_adapter import UnifiedGeneratorAdapter
            self.register_method(
                "unified", 
                UnifiedGeneratorAdapter,
                "基于高维概念空间的增强统一生成方法（包含概念提取、知识图谱和物理嵌入）"
            )
        except ImportError as e:
            print(f"[⚠️] 无法导入增强的 unified 适配器: {e}")
        
        try:
            # 导入并注册优化版 unified 适配器
            try:
                from .methods.unified_generator_adapter_optimized import UnifiedGeneratorAdapterOptimized
            except ImportError:
                from methods.unified_generator_adapter_optimized import UnifiedGeneratorAdapterOptimized
            self.register_method(
                "unified_optimized", 
                UnifiedGeneratorAdapterOptimized,
                "优化版基于高维概念空间的统一生成方法（并行加载、缓存支持、批处理优化）"
            )
        except ImportError as e:
            print(f"[⚠️] 无法导入优化版 unified 适配器: {e}")
        
        try:
            # 导入并注册 concept_relaxation 适配器
            try:
                from .methods.concept_relaxation_adapter import ConceptRelaxationAdapter
            except ImportError:
                from methods.concept_relaxation_adapter import ConceptRelaxationAdapter
            self.register_method(
                "concept_relaxation", 
                ConceptRelaxationAdapter,
                "概念放松生成方法"
            )
        except ImportError as e:
            print(f"[⚠️] 无法导入 concept_relaxation 适配器: {e}")
        
        try:
            # 导入并注册 feedback_aware 适配器
            try:
                from .methods.feedback_aware_generator import FeedbackAwareGenerator
            except ImportError:
                from methods.feedback_aware_generator import FeedbackAwareGenerator
            self.register_method(
                "feedback_aware", 
                FeedbackAwareGenerator,
                "基于评估反馈的智能生成方法（根据评估结果自动调整生成策略）"
            )
        except ImportError as e:
            print(f"[⚠️] 无法导入 feedback_aware 适配器: {e}")


# 创建全局实例
_hub_instance = None

def get_generation_hub() -> TheoryGenerationHub:
    """获取理论生成中心的全局实例"""
    global _hub_instance
    if _hub_instance is None:
        _hub_instance = TheoryGenerationHub()
    return _hub_instance


# 便捷函数，保持向后兼容
def generate_theories(method: str = "direct_synthesis", **kwargs) -> Dict[str, Any]:
    """便捷函数：调用指定方法生成理论"""
    hub = get_generation_hub()
    return hub.generate_theories(method=method, **kwargs)


def list_available_methods() -> List[str]:
    """便捷函数：列出所有可用的生成方法"""
    hub = get_generation_hub()
    return hub.list_methods()


if __name__ == "__main__":
    # 测试和演示代码
    hub = get_generation_hub()
    
    print("=== AI-Philo1.0 理论生成中心 ===")
    print(f"可用方法数量: {len(hub.list_methods())}")
    
    for method, description in hub.get_method_info().items():
        print(f"  • {method}: {description}")
    
    print("\n[✅] 理论生成中心初始化完成") 