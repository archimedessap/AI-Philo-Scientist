from __future__ import annotations

"""
schema_validator.py
===================
提供理论 JSON 的 Schema 校验功能。当前实现采用 `jsonschema` 进行轻量验证，
仅对常用字段做类型约束，以保持向后兼容。
"""

import json
from typing import Dict
from jsonschema import validate, ValidationError

# ------------- Schema 定义 -------------

THEORY_SCHEMA_V21: Dict = {
    "type": "object",
    "required": ["name"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "summary": {"type": "string"},
        "philosophy": {"type": "object"},
        "formalism": {"type": "object"},
        "parameters": {"type": "object"},
        "semantics": {"type": "object"},
    },
    "additionalProperties": True,
}


# ------------- 校验工具函数 -------------

def is_valid_theory_json(data: Dict) -> bool:
    """检查给定 dict 是否符合理论 Schema v2.1。"""
    try:
        validate(instance=data, schema=THEORY_SCHEMA_V21)
        return True
    except ValidationError:
        return False


def assert_valid(data: Dict):
    """若 dict 不符合 Schema，则抛出带提示的 ValidationError。"""
    validate(instance=data, schema=THEORY_SCHEMA_V21)


# ------------- CLI 方便调试 -------------

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="校验理论 JSON 是否符合 Schema v2.1")
    parser.add_argument("json_file", help="理论 JSON 文件路径")
    args = parser.parse_args()

    try:
        with open(args.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert_valid(data)
        print("✅ 通过校验！")
    except Exception as e:
        print(f"❌ 校验失败: {e}")
        sys.exit(1) 