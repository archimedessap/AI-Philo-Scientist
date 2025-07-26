#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_concept_loading.py - 测试文献概念加载
"""

import csv
from pathlib import Path

def test_concept_loading():
    """测试文献概念加载"""
    
    print("🔍 检查文献概念文件...")
    
    # 检查extracted_concepts
    csv_file = Path('data/extracted_concepts/concepts.csv')
    if csv_file.exists():
        print(f"✅ 找到概念文件: {csv_file}")
        
        concepts = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 10:  # 只显示前10个
                    break
                if row.get('Name') and row.get('Description'):
                    concepts.append({
                        'name': row['Name'].strip(),
                        'description': row['Description'].strip()[:50] + '...',  # 截断描述
                        'domain': row.get('Domain', '').strip(),
                        'source': row.get('Source', '')[:30] + '...'  # 截断来源
                    })
        
        print(f"\n📊 共找到概念数量: {len(concepts)}")
        print("\n📝 示例概念:")
        for concept in concepts:
            print(f"  - {concept['name']} ({concept['domain']})")
            print(f"    描述: {concept['description']}")
            print(f"    来源: {concept['source']}")
            print()
    else:
        print(f"❌ 未找到概念文件: {csv_file}")
    
    # 检查enhanced_concepts
    enhanced_dir = Path('data/enhanced_concepts')
    if enhanced_dir.exists():
        print("\n🔍 检查增强概念文件...")
        concept_files = list(enhanced_dir.glob('*enhanced_concepts*.json'))
        for cf in concept_files:
            print(f"  - {cf.name}")

if __name__ == "__main__":
    test_concept_loading()