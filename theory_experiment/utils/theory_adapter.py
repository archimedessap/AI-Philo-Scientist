#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
理论适配器

将不同格式的理论JSON适配为统一格式。
"""

import re
from typing import Any, Dict, Iterable, List, Optional

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

    @staticmethod
    def _coerce_numeric(value: Any) -> Any:
        """将常见的 {"value": x} 结构或字符串数值转换为浮点数。"""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            if "value" in value:
                try:
                    return float(value["value"])
                except (TypeError, ValueError):
                    return None
            if "mean" in value:
                try:
                    return float(value["mean"])
                except (TypeError, ValueError):
                    return None
        if isinstance(value, str):
            stripped = value.strip()
            try:
                return float(stripped)
            except ValueError:
                return None
        return None

    @staticmethod
    def collect_parameters(theory_json: Dict[str, Any]) -> Dict[str, float]:
        """聚合理论中分散的参数并返回展平后的数值字典。"""
        if not isinstance(theory_json, dict):
            return {}

        aggregated: Dict[str, float] = {}

        def merge_params(container: Any):
            if isinstance(container, dict):
                for key, raw in container.items():
                    numeric = TheoryAdapter._coerce_numeric(raw)
                    if numeric is not None and key not in aggregated:
                        aggregated[key] = numeric
                return

            if isinstance(container, list):
                for item in container:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name") or item.get("id")
                    for field in ("value", "default", "default_value", "mean"):
                        if field in item:
                            numeric = TheoryAdapter._coerce_numeric(item[field])
                            if numeric is not None:
                                aggregated[name or field] = numeric
                                break

        for src in TheoryAdapter._parameter_sources(theory_json):
            merge_params(src)

        return aggregated

    @staticmethod
    def _collect_math_relations(theory_json: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        relations = []
        direct = theory_json.get('math_relation_to_SQM') or theory_json.get('math_relation_to_sqm')
        if isinstance(direct, dict):
            relations.append(direct)

        machine_summary = theory_json.get('machine_summary')
        if isinstance(machine_summary, dict):
            rel = TheoryAdapter._normalize_math_relation(machine_summary)
            if isinstance(rel, dict):
                relations.append(rel)

        card_workflow = theory_json.get('card_workflow')
        if isinstance(card_workflow, dict):
            machine = card_workflow.get('machine_summary')
            if isinstance(machine, dict):
                rel = TheoryAdapter._normalize_math_relation(machine)
                if isinstance(rel, dict):
                    relations.append(rel)

        return relations

    @staticmethod
    def _normalize_math_relation(machine_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        relation = machine_summary.get('math_relation_to_SQM') or machine_summary.get('math_relation_to_sqm')
        if isinstance(relation, dict):
            return relation

        meta = machine_summary.get('meta', {}) if isinstance(machine_summary, dict) else {}
        dynamics = machine_summary.get('dynamics', {}) if isinstance(machine_summary, dict) else {}
        if isinstance(meta, dict):
            relation_type = meta.get('relation_to_SQM')
            if relation_type == 'minimal_change':
                relation_type = 'no_change'
            if relation_type:
                math_change = relation_type != 'no_change'
                return {
                    'type': relation_type,
                    'math_change': math_change,
                    'equations_summary': dynamics.get('equation') or dynamics.get('commentary') or ''
                }
        return None

    @staticmethod
    def has_math_modification(theory_json: Dict[str, Any]) -> bool:
        for relation in TheoryAdapter._collect_math_relations(theory_json):
            math_change = relation.get('math_change')
            if math_change is not None:
                return bool(math_change)
            relation_type = relation.get('type')
            if relation_type == 'minimal_change':
                relation_type = 'no_change'
            if relation_type in {'modified_dynamics', 'modified_logic', 'modified_measurement', 'modified_parameters', 'retrocausal', 'extension', 'modification'}:
                return True

        metadata = theory_json.get('metadata', {})
        if isinstance(metadata, dict):
            classification = metadata.get('mathematical_classification', {})
            if isinstance(classification, dict):
                if classification.get('math_change') is True:
                    return True
                uses_standard = classification.get('uses_standard_qm_math')
                if uses_standard is False:
                    return True

        legacy = theory_json.get('mathematical_relation_to_sqm') or theory_json.get('mathematical_relation_to_SQM')
        if isinstance(legacy, dict):
            legacy_type = legacy.get('type')
            if legacy_type in {'extension', 'modification', 'modified_dynamics', 'modified_logic'}:
                return True
            if legacy.get('math_change') is True:
                return True

        return False

    @staticmethod
    def _contains_range(container: Any) -> bool:
        if isinstance(container, dict):
            if any(key in container for key in ('range', 'bounds', 'interval')):
                return True
            keys = container.keys()
            if {'min', 'max'}.issubset(keys) or {'lower', 'upper'}.issubset(keys):
                return True
            for value in container.values():
                if TheoryAdapter._contains_range(value):
                    return True
        if isinstance(container, list):
            if len(container) == 2 and all(isinstance(v, (int, float)) for v in container):
                return True
            return any(TheoryAdapter._contains_range(item) for item in container)
        return False

    @staticmethod
    @staticmethod
    def _parameter_sources(theory_json: Dict[str, Any]) -> List[Any]:
        sources: List[Any] = []
        if not isinstance(theory_json, dict):
            return sources

        if theory_json.get('parameters') is not None:
            sources.append(theory_json.get('parameters'))

        math_core = theory_json.get('math_core')
        if isinstance(math_core, dict):
            for key in ('params', 'parameters'):
                if math_core.get(key) is not None:
                    sources.append(math_core.get(key))

        math_form = theory_json.get('mathematical_formulation')
        if isinstance(math_form, dict):
            for key in ('parameters', 'params'):
                if math_form.get(key) is not None:
                    sources.append(math_form.get(key))

        adapted = TheoryAdapter.adapt(theory_json)
        if isinstance(adapted, dict):
            dynamics = adapted.get('dynamics', {})
            if isinstance(dynamics, dict) and dynamics.get('parameters') is not None:
                sources.append(dynamics.get('parameters'))

        return sources

    @staticmethod
    def has_parameter_support(theory_json: Dict[str, Any]) -> bool:
        if TheoryAdapter.collect_parameters(theory_json):
            return True

        for src in TheoryAdapter._parameter_sources(theory_json):
            if src and TheoryAdapter._contains_range(src):
                return True
        return False

    @staticmethod
    def needs_parameter_support(theory_json: Dict[str, Any]) -> bool:
        for src in TheoryAdapter._parameter_sources(theory_json):
            if not src:
                continue
            if isinstance(src, dict) and src:
                return True
            if isinstance(src, list) and len(src) > 0:
                return True
        return False
