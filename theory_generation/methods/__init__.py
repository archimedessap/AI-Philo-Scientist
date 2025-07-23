#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theory_generation.methods - 理论生成方法适配器包
============================================

本包包含所有理论生成方法的标准化适配器。
每个适配器都实现统一的接口，确保与AI-Philo1.0系统的兼容性。
"""

from .base_adapter import TheoryGenerationMethod

__all__ = [
    'TheoryGenerationMethod',
] 