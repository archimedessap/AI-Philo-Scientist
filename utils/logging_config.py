#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志配置模块

提供项目全局的日志配置和管理功能
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    
    # ANSI 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[35m',  # 紫色
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # 添加颜色
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        
        # 为不同类型的日志添加emoji
        emoji_map = {
            'DEBUG': '🔍',
            'INFO': '✅',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨'
        }
        
        # 获取原始级别名称（去除颜色代码）
        original_levelname = record.levelname.split(self.RESET)[0].split('\033[')[1].split('m')[1] if '\033[' in record.levelname else record.levelname
        
        if original_levelname in emoji_map:
            record.msg = f"{emoji_map[original_levelname]} {record.msg}"
        
        return super().format(record)


class UnifiedLogger:
    """统一日志管理器"""
    
    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建运行日志目录
        self.run_log_dir = self.log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_log_dir.mkdir(exist_ok=True)
        
        # 设置根日志器
        self._setup_root_logger()
    
    def _setup_root_logger(self):
        """设置根日志器"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # 清除现有处理器
        root_logger.handlers.clear()
        
        # 控制台处理器（带颜色）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # 文件处理器（详细日志）
        file_handler = logging.handlers.RotatingFileHandler(
            self.run_log_dir / 'unified_theory_gen.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # 错误日志单独文件
        error_handler = logging.FileHandler(
            self.run_log_dir / 'errors.log',
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
    
    def get_logger(self, name: str, module_specific: bool = False) -> logging.Logger:
        """
        获取日志器
        
        Args:
            name: 日志器名称
            module_specific: 是否创建模块特定的日志文件
            
        Returns:
            配置好的日志器
        """
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        
        # 如果需要模块特定的日志文件
        if module_specific:
            module_log_file = self.run_log_dir / f"{name.replace('.', '_')}.log"
            module_handler = logging.FileHandler(module_log_file, encoding='utf-8')
            module_handler.setLevel(logging.DEBUG)
            module_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            module_handler.setFormatter(module_formatter)
            logger.addHandler(module_handler)
        
        self._loggers[name] = logger
        return logger
    
    def log_config(self, config: Dict[str, Any], logger_name: str = 'config'):
        """记录配置信息"""
        logger = self.get_logger(logger_name)
        logger.info("=" * 60)
        logger.info("配置信息:")
        for key, value in config.items():
            if isinstance(value, (dict, list)) and len(str(value)) > 100:
                logger.info(f"  {key}: [复杂对象，已省略]")
            else:
                logger.info(f"  {key}: {value}")
        logger.info("=" * 60)
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any], 
                              logger_name: str = 'error'):
        """记录带上下文的错误"""
        logger = self.get_logger(logger_name)
        logger.error(f"发生错误: {type(error).__name__}: {str(error)}")
        logger.error("错误上下文:")
        for key, value in context.items():
            logger.error(f"  {key}: {value}")
        logger.error("堆栈跟踪:", exc_info=True)
    
    def create_run_summary(self, summary: Dict[str, Any]):
        """创建运行摘要"""
        summary_file = self.run_log_dir / 'run_summary.json'
        import json
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 同时记录到日志
        logger = self.get_logger('summary')
        logger.info("运行摘要已保存到: %s", summary_file)


# 全局日志管理器实例
unified_logger = UnifiedLogger()


# 便捷函数
def get_logger(name: str, module_specific: bool = False) -> logging.Logger:
    """获取日志器的便捷函数"""
    return unified_logger.get_logger(name, module_specific)


def log_config(config: Dict[str, Any], logger_name: str = 'config'):
    """记录配置的便捷函数"""
    unified_logger.log_config(config, logger_name)


def log_error_with_context(error: Exception, context: Dict[str, Any], 
                          logger_name: str = 'error'):
    """记录错误的便捷函数"""
    unified_logger.log_error_with_context(error, context, logger_name)


# 装饰器：自动记录函数执行
def log_execution(logger_name: Optional[str] = None):
    """
    装饰器：自动记录函数执行情况
    
    Args:
        logger_name: 日志器名称，默认使用函数所在模块名
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 获取日志器
            nonlocal logger_name
            if logger_name is None:
                logger_name = func.__module__
            logger = get_logger(logger_name)
            
            # 记录函数开始
            func_name = func.__name__
            logger.debug(f"开始执行函数: {func_name}")
            start_time = datetime.now()
            
            try:
                # 执行函数
                result = func(*args, **kwargs)
                
                # 记录成功
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.debug(f"函数 {func_name} 执行成功，耗时: {elapsed:.2f}秒")
                
                return result
                
            except Exception as e:
                # 记录错误
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.error(f"函数 {func_name} 执行失败，耗时: {elapsed:.2f}秒")
                logger.error(f"错误类型: {type(e).__name__}")
                logger.error(f"错误信息: {str(e)}")
                logger.debug("详细错误:", exc_info=True)
                raise
        
        return wrapper
    return decorator


# 初始化消息
if __name__ != "__main__":
    logger = get_logger('logging_config')
    logger.info("统一日志系统已初始化")
    logger.info(f"日志目录: {unified_logger.run_log_dir}")