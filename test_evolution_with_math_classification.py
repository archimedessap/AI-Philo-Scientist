#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试带有数学分类的演进系统
"""
import asyncio
import json
import os
import subprocess
from pathlib import Path

async def test_generation_0():
    """测试第0代理论合成"""
    print("🧪 测试第0代理论合成...")
    
    # 创建测试输出目录
    test_dir = Path("test_results/math_classification_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行理论合成
    cmd = [
        "python", "run_direct_synthesis.py",
        "--theories_dir", "data/theories_v2.1",
        "--output_dir", str(test_dir / "generation_0"),
        "--num_theories", "2",
        "--model_source", "deepseek",
        "--model_name", "deepseek-reasoner"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ 理论合成失败:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    
    # 检查生成的理论是否有数学分类标注
    eval_ready_dir = test_dir / "generation_0" / "eval_ready_theories"
    if eval_ready_dir.exists():
        theory_files = list(eval_ready_dir.glob("*.json"))
        print(f"\n📋 检查生成的理论文件:")
        
        for theory_file in theory_files:
            with open(theory_file, 'r', encoding='utf-8') as f:
                theory_data = json.load(f)
            
            theory_name = theory_data.get("name", "未命名")
            math_classification = theory_data.get("metadata", {}).get("mathematical_classification", {})
            
            if math_classification:
                math_type = math_classification.get("type", "unknown")
                uses_standard_qm = math_classification.get("uses_standard_qm_math", False)
                
                print(f"✅ {theory_name}")
                print(f"   数学分类: {math_type}")
                print(f"   使用标准QM数学: {'是' if uses_standard_qm else '否'}")
            else:
                print(f"❌ {theory_name} - 缺少数学分类标注")
    
    return True

async def main():
    """主测试函数"""
    print("🚀 开始测试带有数学分类的演进系统")
    
    # 测试第0代理论合成
    success1 = await test_generation_0()
    if not success1:
        print("❌ 第0代测试失败")
        return
    
    print("\n✅ 测试通过！")
    print("🎉 带有数学分类的演进系统工作正常")

if __name__ == "__main__":
    asyncio.run(main())
