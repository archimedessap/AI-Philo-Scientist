#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility helpers for parsing model configuration strings."""

from typing import List, Dict


def parse_model_config_string(config_string: str) -> List[Dict[str, str]]:
    """Parse a comma-separated list of ``source:model`` pairs.

    Args:
        config_string: Input such as ``"openai:gpt-4o-mini,deepseek:deepseek-chat"``.

    Returns:
        A list of dicts with ``model_source`` and ``model_name`` keys.

    Raises:
        ValueError: If the string cannot be parsed into valid pairs.
    """
    if not config_string:
        return []

    parsed: List[Dict[str, str]] = []
    for raw_item in config_string.split(','):
        item = raw_item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f"模型配置 '{item}' 缺少 ':' 分隔符")
        source, name = item.split(':', 1)
        source = source.strip()
        name = name.strip()
        if not source or not name:
            raise ValueError(f"模型配置 '{item}' 缺少模型来源或模型名称")
        parsed.append({
            "model_source": source,
            "model_name": name
        })
    if not parsed:
        raise ValueError("未解析到任何有效的模型配置")
    return parsed
