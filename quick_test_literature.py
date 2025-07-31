#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_test_literature.py - 快速测试文献概念加载
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from theory_generation.methods.unified_generator_adapter import UnifiedSpaceBasedGenerator

# 创建测试实例
config = {
    'theories_dir': 'data/theories_test',
    'output_dir': 'output_quick_test',
    'model_source': 'google',
    'model_name': 'gemini-2.5-flash',
    'force_load_literature': True  # 强制加载文献概念
}

print("🚀 创建生成器实例...")
generator = UnifiedSpaceBasedGenerator(**config)

print("\n📚 测试文献概念加载...")
concepts = generator._load_literature_concepts()

print(f"\n✅ 加载了 {len(concepts)} 个概念")
print("\n📝 示例概念:")
for i, concept in enumerate(concepts[:5]):
    print(f"{i+1}. {concept.get('name', 'Unknown')} - {concept.get('domain', 'N/A')}")
    print(f"   {concept.get('description', 'No description')[:60]}...")

print("\n✨ 测试完成！")