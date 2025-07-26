#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重试装饰器

提供带有指数退避的重试机制
"""

import time
import asyncio
import functools
from typing import TypeVar, Callable, Optional, Union, Tuple, Type
from utils.logging_config import get_logger

T = TypeVar('T')


def retry_with_exponential_backoff(
    retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger_name: Optional[str] = None
):
    """
    重试装饰器，支持指数退避
    
    Args:
        retries: 最大重试次数
        initial_delay: 初始延迟时间（秒）
        backoff_factor: 退避因子
        max_delay: 最大延迟时间（秒）
        exceptions: 需要重试的异常类型元组
        logger_name: 日志器名称
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            logger = get_logger(logger_name or func.__module__)
            last_exception = None
            
            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < retries:
                        delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
                        logger.warning(
                            f"函数 {func.__name__} 执行失败 (尝试 {attempt + 1}/{retries + 1}): {str(e)}"
                        )
                        logger.info(f"等待 {delay:.1f} 秒后重试...")
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"函数 {func.__name__} 在 {retries + 1} 次尝试后仍然失败"
                        )
            
            if last_exception:
                raise last_exception
            
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            logger = get_logger(logger_name or func.__module__)
            last_exception = None
            
            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < retries:
                        delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
                        logger.warning(
                            f"异步函数 {func.__name__} 执行失败 (尝试 {attempt + 1}/{retries + 1}): {str(e)}"
                        )
                        logger.info(f"等待 {delay:.1f} 秒后重试...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"异步函数 {func.__name__} 在 {retries + 1} 次尝试后仍然失败"
                        )
            
            if last_exception:
                raise last_exception
        
        # 根据函数类型返回相应的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def retry_on_api_error(
    retries: int = 3,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    max_delay: float = 30.0
):
    """
    专门用于API调用的重试装饰器
    
    会捕获常见的API错误并进行重试
    """
    # 定义API相关的异常
    api_exceptions = (
        ConnectionError,
        TimeoutError,
        IOError,
    )
    
    # 尝试导入常见的API异常
    try:
        import requests
        api_exceptions = api_exceptions + (requests.exceptions.RequestException,)
    except ImportError:
        pass
    
    try:
        import aiohttp
        api_exceptions = api_exceptions + (aiohttp.ClientError,)
    except ImportError:
        pass
    
    return retry_with_exponential_backoff(
        retries=retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
        max_delay=max_delay,
        exceptions=api_exceptions,
        logger_name='api_retry'
    )


class RetryContext:
    """重试上下文管理器"""
    
    def __init__(
        self,
        retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        logger_name: Optional[str] = None
    ):
        self.retries = retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.exceptions = exceptions
        self.logger = get_logger(logger_name or 'retry_context')
        self.attempt = 0
        self.last_exception = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, self.exceptions):
            self.last_exception = exc_val
            
            if self.attempt < self.retries:
                self.attempt += 1
                delay = min(
                    self.initial_delay * (self.backoff_factor ** (self.attempt - 1)),
                    self.max_delay
                )
                self.logger.warning(
                    f"操作失败 (尝试 {self.attempt}/{self.retries}): {str(exc_val)}"
                )
                self.logger.info(f"等待 {delay:.1f} 秒后重试...")
                time.sleep(delay)
                return True  # 抑制异常
            else:
                self.logger.error(
                    f"操作在 {self.retries} 次尝试后仍然失败"
                )
                return False  # 传播异常
        
        return False
    
    def should_retry(self) -> bool:
        """检查是否应该继续重试"""
        return self.attempt <= self.retries


# 使用示例
if __name__ == "__main__":
    # 示例1: 使用装饰器
    @retry_with_exponential_backoff(retries=3, initial_delay=1.0)
    def unreliable_function():
        import random
        if random.random() < 0.7:
            raise ConnectionError("模拟连接错误")
        return "成功!"
    
    # 示例2: 使用上下文管理器
    retry_ctx = RetryContext(retries=3, initial_delay=1.0)
    while retry_ctx.should_retry():
        with retry_ctx:
            # 执行可能失败的操作
            import random
            if random.random() < 0.7:
                raise ConnectionError("模拟连接错误")
            print("操作成功!")
            break