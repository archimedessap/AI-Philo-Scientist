#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_unified_with_literature.py - 测试基于原始文献概念的unified方法
"""

import asyncio
import json
from pathlib import Path
from theory_generation.methods.unified_generator_adapter import UnifiedGeneratorAdapter

async def test_unified_with_literature():
    """测试基于原始文献概念的unified方法"""
    
    # 配置参数
    config = {
        'theories_dir': 'data/theories_test',  # 使用测试数据集
        'output_dir': 'output_test_literature_unified',
        'model_source': 'google',
        'model_name': 'gemini-2.5-flash',
        'use_structured_format': True,
        'test_mode': True,  # 启用测试模式
        'force_load_literature': True  # 强制加载文献概念
    }
    
    print("🚀 开始测试基于原始文献概念的unified方法")
    print(f"📁 理论目录: {config['theories_dir']}")
    print(f"📁 输出目录: {config['output_dir']}")
    
    # 创建生成器实例
    generator = UnifiedGeneratorAdapter(**config)
    
    # 临时修改_load_literature_concepts方法以强制加载
    original_method = generator._load_literature_concepts
    
    def force_load_literature_concepts():
        """强制加载文献概念（即使在测试模式下）"""
        # 直接调用父类方法，跳过测试模式检查
        concepts = []
        
        # 从extracted_concepts加载
        csv_file = Path('data/extracted_concepts/concepts.csv')
        if csv_file.exists():
            try:
                import csv
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    count = 0
                    for row in reader:
                        if count >= 20:  # 限制数量
                            break
                        if row.get('Name') and row.get('Description'):
                            concepts.append({
                                'name': row['Name'].strip(),
                                'description': row['Description'].strip(),
                                'domain': row.get('Domain', '').strip(),
                                'source': row.get('Source', '').strip()
                            })
                            count += 1
                
                print(f"✅ 加载了 {len(concepts)} 个文献概念")
            except Exception as e:
                print(f"❌ 加载文献概念失败: {e}")
        
        return concepts
    
    # 替换方法
    generator._load_literature_concepts = force_load_literature_concepts
    
    # 执行生成
    result = await generator._async_generate()
    
    # 保存结果
    output_path = Path(config['output_dir']) / 'test_result.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 测试完成！结果保存到: {output_path}")
    
    # 显示统计信息
    if result.get('success'):
        metadata = result.get('metadata', {})
        print("\n📊 生成统计:")
        print(f"  - 概念空间大小: {metadata.get('concept_space_size', 0)}")
        print(f"  - 理论空间大小: {metadata.get('theory_space_size', 0)}")
        print(f"  - 发现的概念空白: {metadata.get('conceptual_gaps_found', 0)}")
        print(f"  - 生成的理论数量: {len(result.get('theories', []))}")
        
        # 检查是否包含文献概念
        space_data_path = Path(config['output_dir']) / 'concept_space_data.json'
        if space_data_path.exists():
            with open(space_data_path, 'r') as f:
                space_data = json.load(f)
                concept_names = list(space_data.get('concept_space', {}).keys())
                lit_concepts = [name for name in concept_names if 'Collapse' in name or 'Measurement' in name]
                if lit_concepts:
                    print(f"\n✅ 成功加载文献概念，示例: {lit_concepts[:3]}")
                else:
                    print("\n⚠️ 未检测到文献概念")
    else:
        print(f"\n❌ 生成失败: {result.get('error_message', '未知错误')}")

if __name__ == "__main__":
    asyncio.run(test_unified_with_literature())