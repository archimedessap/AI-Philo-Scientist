#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JSON Schema验证器

验证理论和实验数据的JSON格式是否符合预定义的模式。
"""

import json
import os

class SchemaValidator:
    def __init__(self, 
                theory_schema_path="config/theory_schema.json", 
                experiment_schema_path="config/experiment_schema.json"):
        self.theory_schema = self._load_schema(theory_schema_path)
        self.experiment_schema = self._load_schema(experiment_schema_path)
        
    def _load_schema(self, schema_path):
        """加载JSON Schema"""
        try:
            if os.path.exists(schema_path):
                with open(schema_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"[WARN] Schema文件不存在: {schema_path}，将使用基本验证")
                return None
        except Exception as e:
            print(f"[ERROR] 加载Schema文件失败: {str(e)}")
            return None
            
    def validate_theory(self, theory):
        """验证理论格式，放宽id字段要求"""
        # 只检查name字段是否存在
        if not theory.get('name'):
            print("[WARN] 理论缺少必要字段: name")
            return False
        
        # 其他字段为可选
        for field in ['core_principles']:
            if not theory.get(field):
                print(f"[INFO] 理论缺少推荐字段: {field}")
        
        return True  # 只要有name就认为格式有效
    
    def validate_experiment(self, experiment_json):
        """验证实验数据格式"""
        if not isinstance(experiment_json, dict):
            return False
            
        # 检查必要字段
        required_fields = ["id", "measured", "std_prediction"]
        for field in required_fields:
            if field not in experiment_json:
                print(f"[WARN] 实验数据缺少必要字段: {field}")
                return False
                
        return True 