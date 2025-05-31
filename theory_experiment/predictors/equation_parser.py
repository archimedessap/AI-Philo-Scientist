import sympy as sp
import re

class EquationParser:
    def __init__(self):
        self.symbols_cache = {}
        
    def parse_latex(self, latex_str):
        """将LaTeX方程转换为sympy表达式"""
        # 预处理：清理LaTeX特殊标记
        cleaned = self._preprocess_latex(latex_str)
        
        try:
            # 使用sympy解析表达式
            expr = sp.sympify(cleaned)
            return expr
        except Exception as e:
            print(f"无法解析LaTeX表达式: {e}")
            return None
            
    def extract_variables(self, latex_str):
        """从LaTeX方程中提取变量"""
        # 使用正则表达式寻找变量
        var_pattern = r'\\([a-zA-Z]+)|(?<![a-zA-Z\\])([a-zA-Z]+)(?![a-zA-Z])'
        matches = re.finditer(var_pattern, latex_str)
        
        variables = set()
        for match in matches:
            var = match.group(1) or match.group(2)
            if var not in ['sin', 'cos', 'tan', 'exp', 'log', 'lim']:
                variables.add(var)
                
        return list(variables)
        
    def compute_observable(self, expr, params, observable_name):
        """计算可观测量"""
        # 替换参数
        for param, value in params.items():
            if param in self.symbols_cache:
                symbol = self.symbols_cache[param]
            else:
                symbol = sp.symbols(param)
                self.symbols_cache[param] = symbol
            
            expr = expr.subs(symbol, value)
            
        # 根据可观测量类型计算值
        # ...具体实现取决于观测量类型
        
        return expr
