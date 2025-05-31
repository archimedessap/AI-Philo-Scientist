def _get_dynamics_type(self):
    """获取理论动力学类型"""
    # 1. 首先检查是否直接指定了dynamics.type
    if "dynamics" in self.theory and "type" in self.theory["dynamics"]:
        return self.theory["dynamics"]["type"]
        
    # 2. 尝试从理论描述推断类型
    theory_name = self.theory.get("name", "").lower()
    description = self.theory.get("description", "").lower()
    
    # 标准量子力学变体
    if any(x in theory_name or x in description for x in ["copenhagen", "哥本哈根", "标准量子", "many-worlds", "多世界"]):
        return "linear"
        
    # GRW/CSL类模型
    if any(x in theory_name or x in description for x in ["grw", "ghirardi"]):
        return "GRW"
    elif any(x in theory_name or x in description for x in ["csl", "continuous spontaneous", "连续自发"]):
        return "linear+CSL"
        
    # 区分不同类型的导航波理论
    if "thermal" in theory_name and ("pilot" in theory_name or "导航波" in theory_name):
        return "thermal"  # 热力学导航波
    elif any(x in theory_name or x in description for x in ["bohm", "pilot", "guide", "导航波", "玻姆"]):
        return "bohmian"  # 玻姆理论
        
    # 非线性模型
    if any(x in theory_name or x in description for x in ["nonlinear", "非线性"]):
        return "nonlinear_Schr"
        
    # 默认为标准量子力学
    return "linear"
