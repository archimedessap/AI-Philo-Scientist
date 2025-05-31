#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
理论适配器

将不同格式的理论JSON适配为统一格式。
"""

import re

class TheoryAdapter:
    """理论格式适配器"""
    
    @staticmethod
    def adapt(theory_json):
        """将理论JSON适配为标准格式
        
        Args:
            theory_json: 原始理论JSON
            
        Returns:
            适配后的理论JSON
        """
        if not theory_json:
            # 处理空输入
            return {"dynamics": {"type": "linear", "parameters": {}}}
            
        adapted = theory_json.copy()
        
        # 确保有dynamics字段
        if "dynamics" not in adapted:
            adapted["dynamics"] = {}
            
            # 从名称和描述推断类型
            theory_name = adapted.get("name", "").lower()
            description = adapted.get("description", "").lower()
            
            # 推断dynamics.type
            if any(x in theory_name or x in description for x in ["grw", "ghirardi"]):
                adapted["dynamics"]["type"] = "GRW"
            elif any(x in theory_name or x in description for x in ["csl", "continuous spontaneous"]):
                adapted["dynamics"]["type"] = "linear+CSL"
            elif "thermal" in theory_name and "pilot" in theory_name:
                adapted["dynamics"]["type"] = "thermal" 
            elif any(x in theory_name or x in description for x in ["bohm", "pilot", "guide"]):
                adapted["dynamics"]["type"] = "bohmian"
            elif any(x in theory_name or x in description for x in ["nonlinear", "非线性"]):
                adapted["dynamics"]["type"] = "nonlinear_Schr"
            else:
                adapted["dynamics"]["type"] = "linear"
                
        # 确保有parameters字段
        if "parameters" not in adapted["dynamics"]:
            adapted["dynamics"]["parameters"] = {}
            
            # 从其他地方提取参数
            if "math_core" in adapted and "params" in adapted["math_core"]:
                adapted["dynamics"]["parameters"].update(adapted["math_core"]["params"])
                
            elif "mathematical_formulation" in adapted:
                math_form = adapted["mathematical_formulation"]
                if "parameters" in math_form:
                    adapted["dynamics"]["parameters"].update(math_form["parameters"])
                elif "params" in math_form:
                    adapted["dynamics"]["parameters"].update(math_form["params"])
                    
        # 处理None值参数
        params = adapted["dynamics"]["parameters"]
        for key, value in list(params.items()):
            if value is None:
                params[key] = 0
        
        return adapted
