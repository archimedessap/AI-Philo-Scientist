#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
量子理论评估器

从物理学家、哲学家和数学家多个角度评估生成的新量子诠释理论。
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional
import sys
from pathlib import Path

# 添加项目根目录到sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.retry_decorator import retry_on_api_error
from utils.model_config_parser import parse_model_config_string

def safe_get_nested(obj, path, subpath=None, default=''):
    """安全获取嵌套字段，无论是字典还是字符串"""
    value = obj.get(path, default) if isinstance(obj, dict) else default
    
    # 如果没有子路径，或者值不是字典，直接返回值
    if subpath is None or not isinstance(value, dict):
        return value
        
    # 如果值是字典且有子路径，继续获取
    return value.get(subpath, default)

class TheoryEvaluator:
    """多角色量子理论评估器"""
    
    def __init__(self, llm_interface, multi_model_configs: Optional[List[Dict[str, str]]] = None):
        """
        初始化理论评估器
        
        Args:
            llm_interface: LLM接口实例
            multi_model_configs: 角色评估时要轮询的模型配置列表
        """
        self.llm = llm_interface
        self.evaluation_results = []  # 评估结果
        self.experiments = []  # 添加空实验列表

        if multi_model_configs is None:
            env_models = os.environ.get("ROLE_EVAL_MODELS")
            if env_models:
                try:
                    multi_model_configs = parse_model_config_string(env_models)
                except ValueError as exc:
                    print(f"[WARN] 环境变量 ROLE_EVAL_MODELS 解析失败，将使用默认模型: {exc}")

        self.multi_model_configs = multi_model_configs or []

        self.consistent_model_defaults = [
            {"model_source": self.llm.model_source, "model_name": self.llm.model_name},
            {"model_source": "openai", "model_name": "gpt-4o-mini"},
            {"model_source": "google", "model_name": "gemini-2.5-flash"}
        ]
        self.inconsistent_model_defaults = [
            {"model_source": "openai", "model_name": "gpt-4o-mini"},
            {"model_source": "google", "model_name": "gemini-2.5-flash"},
            {"model_source": "deepseek", "model_name": "deepseek-chat"},
            {"model_source": self.llm.model_source, "model_name": self.llm.model_name}
        ]

        # 安全导入验证器
        try:
            from theory_experiment.experimetal_validation.schema_validator import SchemaValidator
            self.schema_validator = SchemaValidator()
        except ImportError:
            # 如果导入失败，创建一个简单的替代验证器
            class SimpleValidator:
                def validate_theory(self, theory):
                    return True  # 总是返回有效
            self.schema_validator = SimpleValidator()
            print("[WARN] 无法导入SchemaValidator，使用简单验证器替代")
        
        # 定义评估角色及其关注点
        self.evaluation_roles = {
            "physicist": {
                "name": "物理学家",
                "focus": [
                    "与已知物理实验的兼容性", 
                    "可检验的预测", 
                    "物理直觉的合理性"
                ]
            },
            "philosopher": {
                "name": "哲学家",
                "focus": [
                    "逻辑一致性", 
                    "本体论清晰度", 
                    "认识论立场", 
                    "与哲学传统的关系"
                ]
            },
            "mathematician": {
                "name": "数学家",
                "focus": [
                    "数学形式化的严谨性", 
                    "数学结构的优雅性",
                    "与现有数学框架的兼容性"
                ]
            }
        }

    def _select_model_configs(self, theory: Dict[str, Any]) -> List[Dict[str, str]]:
        if self.multi_model_configs:
            return self.multi_model_configs
        if self._is_qm_consistent(theory):
            return self.consistent_model_defaults
        return self.inconsistent_model_defaults

    def _is_qm_consistent(self, theory: Dict[str, Any]) -> bool:
        for relation in self._collect_math_relations(theory):
            math_change = relation.get('math_change')
            if math_change is not None:
                return not bool(math_change)
            relation_type = relation.get('type')
            if relation_type == 'minimal_change':
                relation_type = 'no_change'
            if relation_type in {'modified_dynamics', 'modified_logic', 'modified_measurement', 'modified_parameters', 'retrocausal'}:
                return False
            if relation_type in {'interpretation', 'no_change'}:
                return True
        meta = theory.get('metadata', {}).get('mathematical_classification', {})
        if isinstance(meta, dict) and 'uses_standard_qm_math' in meta:
            return bool(meta['uses_standard_qm_math'])
        legacy = theory.get('mathematical_relation_to_sqm', {})
        if isinstance(legacy, dict):
            legacy_type = legacy.get('type')
            if legacy_type in {'extension', 'modification', 'modified_dynamics', 'modified_logic'}:
                return False
        return True

    def has_math_modification(self, theory: Dict[str, Any]) -> bool:
        """Return True when the theory introduces changes to standard QM mathematics."""
        return not self._is_qm_consistent(theory)

    def _collect_math_relations(self, theory: Dict[str, Any]) -> List[Dict[str, Any]]:
        relations: List[Dict[str, Any]] = []
        direct = theory.get('math_relation_to_SQM')
        if isinstance(direct, dict):
            relations.append(direct)
        machine_summary = theory.get('machine_summary')
        if isinstance(machine_summary, dict):
            relation = self._normalize_math_relation(machine_summary)
            if relation:
                relations.append(relation)
        card_workflow = theory.get('card_workflow', {})
        if isinstance(card_workflow, dict):
            machine = card_workflow.get('machine_summary')
            if isinstance(machine, dict):
                relation = self._normalize_math_relation(machine)
                if relation:
                    relations.append(relation)
        return relations

    def _normalize_math_relation(self, machine_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

    def _stringify(self, value: Any) -> str:
        if value is None:
            return ''
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple, set)):
            return '; '.join(self._stringify(item) for item in value if item)
        if isinstance(value, dict):
            parts = []
            for key, item in value.items():
                rendered = self._stringify(item)
                if rendered:
                    parts.append(f"{key}: {rendered}")
            return '; '.join(parts)
        return str(value)

    def _extract_theory_context(self, theory: Dict[str, Any]) -> Dict[str, str]:
        context: Dict[str, str] = {}
        machine = theory.get('machine_summary') or {}
        if not machine and isinstance(theory.get('card_workflow'), dict):
            machine = theory['card_workflow'].get('machine_summary', {})

        context['core'] = self._stringify(
            theory.get('core_principles')
            or theory.get('core_assumptions')
            or machine.get('key_claims')
            or theory.get('writeup')
            or theory.get('summary')
        )
        relation = (self._normalize_math_relation(machine)
                    or theory.get('math_relation_to_SQM')
                    or theory.get('math_relation_to_sqm')
                    or {})
        context['math'] = self._stringify(
            relation.get('equations_summary')
            or theory.get('mathematical_formulation')
        )
        measurement_info = machine.get('measurement') if isinstance(machine, dict) else None
        if not measurement_info:
            measurement_info = theory.get('measurement_strategy') or theory.get('measurement_postulate')
        context['measurement'] = self._stringify(
            (measurement_info.get('model') if isinstance(measurement_info, dict) else measurement_info)
        )
        context['born'] = self._stringify(
            (measurement_info.get('probability_rule') if isinstance(measurement_info, dict) else None)
            or theory.get('born_rule_explanation')
        )
        context['locality'] = self._stringify(
            machine.get('locality_note')
            or theory.get('non_locality_analysis')
        )
        predictions = machine.get('testable_predictions') or theory.get('predictions') or []
        context['predictions'] = self._stringify(predictions)
        contradictions = theory.get('contradictions')
        if not contradictions and isinstance(theory.get('card_workflow'), dict):
            contradictions = theory['card_workflow'].get('contradictions')
        context['contradictions'] = self._stringify(contradictions)
        context['extra'] = self._stringify(theory.get('additional_notes') or theory.get('philosophical_stance'))
        return context

    async def evaluate_theory(self, theory_json, predictor_module=None):
        """评估理论"""
        # 初始化基本结果
        base_result = {
            'theory_name': theory_json.get('name', '未命名理论'),
            'theory_id': theory_json.get('id', 'AUTO_' + str(hash(theory_json.get('name', '')))[0:8]),
            'evaluations': {},  # 确保初始化评估字段
            'avg_chi2': 0,
            'conflicts': [],
            'detailed_results': []
        }
        
        # 进行格式验证但不阻止评估流程
        valid_format = self.schema_validator.validate_theory(theory_json)
        if not valid_format:
            print(f"[INFO] 理论缺少id字段，已自动生成临时ID")
            base_result['status'] = 'info'  # 改为info而非warning
        
        # 若缺少实验数据，记录警告但继续评估
        if not self.experiments:
            base_result.setdefault('warnings', []).append('没有实验数据可供评估')

        model_cfgs = self._select_model_configs(theory_json)

        # 直接进行角色评估
        role_results = {}
        role_scores: List[float] = []

        for role_id, role_info in self.evaluation_roles.items():
            print(f"[INFO] 开始{role_info['name']}评估")
            eval_result = await self._evaluate_as_role(theory_json, role_id, role_info, model_cfgs)
            role_results[role_id] = eval_result
            score = eval_result.get('score')
            if isinstance(score, (int, float)):
                role_scores.append(float(score))
            print(f"[INFO] {role_info['name']}评估完成，得分: {eval_result.get('score', '未知')}")

        base_result['evaluations'] = role_results
        overall_score = sum(role_scores) / len(role_scores) if role_scores else 0.0
        base_result['overall_score'] = overall_score
        base_result['role_score_breakdown'] = {
            role_id: result.get('score', 0.0) for role_id, result in role_results.items()
        }

        instrumentation_data = None
        if not self._is_qm_consistent(theory_json):
            instrumentation_targets = [cfg for cfg in model_cfgs if cfg.get('model_source') in {'openai', 'google'}]
            if instrumentation_targets:
                instrumentation_data = await self._run_instrumentation_review(theory_json, instrumentation_targets)
                base_result['instrumentation_review'] = instrumentation_data
                average_instrument_score = instrumentation_data.get('average_score')
                if isinstance(average_instrument_score, (int, float)) and average_instrument_score > 0:
                    overall_score = (overall_score + average_instrument_score) / 2
                    base_result['overall_score'] = overall_score

        # 生成评估总结
        if base_result['evaluations']:
            summary_payload = {
                'evaluations': base_result['evaluations'],
                'overall_score': overall_score,
                'instrumentation_review': instrumentation_data
            }
            summary = await self._generate_evaluation_summary(theory_json, summary_payload)
            base_result['summary'] = summary

        return base_result
    
    @retry_on_api_error(retries=3, initial_delay=2.0)
    async def _evaluate_as_role(
        self,
        theory: Dict,
        role_id: str,
        role_info: Dict,
        model_cfgs: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        """
        从特定角色视角评估理论（带重试机制）
        
        Args:
            theory: 要评估的理论
            role_id: 角色ID
            role_info: 角色信息
            
        Returns:
            Dict: 角色评估结果
        """
        role_name = role_info["name"]
        breakdown = []
        valid_evaluations = []

        configs = model_cfgs or self._select_model_configs(theory)
        if not configs:
            configs = [
                {
                    "model_source": self.llm.model_source,
                    "model_name": self.llm.model_name
                }
            ]

        for cfg in configs:
            evaluation = await self._evaluate_role_with_model(theory, role_id, role_info, cfg)
            breakdown.append({
                "model_source": cfg.get("model_source"),
                "model_name": cfg.get("model_name"),
                "result": evaluation
            })

            score = evaluation.get("score")
            if isinstance(score, str):
                try:
                    score = float(score)
                    evaluation["score"] = score
                except ValueError:
                    score = None
            if isinstance(score, (int, float)) and "error" not in evaluation:
                valid_evaluations.append(evaluation)

        aggregated = self._aggregate_role_evaluations(role_name, valid_evaluations, breakdown)
        return aggregated

    async def _evaluate_role_with_model(self, theory: Dict, role_id: str, role_info: Dict, model_cfg: Dict) -> Dict:
        """使用指定模型评估角色视角。"""
        role_name = role_info["name"]
        focus_points = role_info["focus"]

        focus_text = "\n".join([f"- {point}" for point in focus_points])
        context = self._extract_theory_context(theory)

        # 构建提示
        prompt = f"""
        你是一位资深的量子物理学{role_name}，需要评估一个新提出的量子诠释理论。
        
        作为{role_name}，你特别关注:
        {focus_text}
        
        要评估的理论:
        理论名称: {theory.get('name', '未命名理论')}
        核心主张: {context.get('core', '')}
        SQM数学关系: {context.get('math', '')}
        测量与Born处理: {context.get('measurement', '')} | Born规则: {context.get('born', '')}
        非局域性/局域性说明: {context.get('locality', '')}
        可检验预测: {context.get('predictions', '')}
        现有矛盾摘要: {context.get('contradictions', '')}
        其它补充: {context.get('extra', '')}
        
        请从{role_name}的视角评估这个理论，考虑上述关注点，以JSON格式返回评估结果:
        {{
          "strengths": [
            "理论优势1",
            "理论优势2"
          ],
          "weaknesses": [
            "理论弱点1",
            "理论弱点2"
          ],
          "questions": [
            "有待解决的问题1",
            "有待解决的问题2"
          ],
          "score": 评分(0-10),
          "detailed_comments": "详细评价...",
          "improvement_suggestions": "改进建议..."
        }}
        
        请基于理论的科学和哲学价值进行客观评估，给出合理的评分和具体的分析。
        """
        
        # 调用LLM进行评估
        try:
            response = await self.llm.query_async(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # 使用较低的温度以获得确定性结果
                model_source=model_cfg.get("model_source"),
                model_name=model_cfg.get("model_name")
            )

            # 解析结果
            evaluation = self.llm.extract_json(response)
            if evaluation:
                evaluation["role"] = role_name
                evaluation["model_source"] = model_cfg.get("model_source") or self.llm.model_source
                evaluation["model_name"] = model_cfg.get("model_name") or self.llm.model_name
                return evaluation

            return {
                "role": role_name,
                "error": "评估结果解析失败",
                "score": 0,
                "model_source": model_cfg.get("model_source") or self.llm.model_source,
                "model_name": model_cfg.get("model_name") or self.llm.model_name
            }
        except Exception as e:
            print(f"[ERROR] {role_name}评估失败: {str(e)}")
            return {
                "role": role_name,
                "error": str(e),
                "score": 0,
                "model_source": model_cfg.get("model_source") or self.llm.model_source,
                "model_name": model_cfg.get("model_name") or self.llm.model_name
            }

    def _aggregate_role_evaluations(self, role_name: str, evaluations: List[Dict], breakdown: List[Dict]) -> Dict:
        """聚合多个模型的角色评估结果。"""
        strengths: List[str] = []
        weaknesses: List[str] = []
        questions: List[str] = []
        suggestions: List[str] = []
        comments: List[str] = []
        scores: List[float] = []
        breakdown_entries: List[Dict[str, Any]] = []

        for item in breakdown:
            result = item['result']
            model_source = item.get('model_source')
            model_name = item.get('model_name')
            entry = {
                "model_source": model_source,
                "model_name": model_name,
                "status": "ok",
                "score": result.get('score', 0)
            }
            if 'error' in result:
                entry['status'] = 'error'
                entry['detail'] = result['error']
            breakdown_entries.append(entry)

        for evaluation in evaluations:
            score = evaluation.get('score')
            if isinstance(score, (int, float)):
                scores.append(float(score))

            for key, target in (
                ('strengths', strengths),
                ('weaknesses', weaknesses),
                ('questions', questions),
                ('improvement_suggestions', suggestions)
            ):
                values = evaluation.get(key, []) or []
                if isinstance(values, str):
                    values = [values]
                for value in values:
                    if value and value not in target:
                        target.append(value)

            detail = evaluation.get('detailed_comments')
            if detail and detail not in comments:
                comments.append(detail)

        average_score = sum(scores) / len(scores) if scores else 0.0

        aggregated = {
            "role": role_name,
            "score": average_score,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "questions": questions,
            "detailed_comments": "\n\n".join(comments) if comments else "",
            "improvement_suggestions": suggestions,
            "model_breakdown": breakdown_entries
        }

        if not evaluations and breakdown:
            aggregated["warnings"] = [
                "所有模型评估均失败，已将得分置为0"]

        return aggregated

    async def _run_instrumentation_review(self, theory: Dict[str, Any], model_cfgs: List[Dict[str, str]]) -> Dict[str, Any]:
        context = self._extract_theory_context(theory)
        schema = {
            "type": "object",
            "required": ["score", "instrument_readiness", "experimental_path"],
            "properties": {
                "score": {"type": "number"},
                "instrument_readiness": {"type": "string"},
                "experimental_path": {"type": "string"},
                "risks": {"type": "string"}
            }
        }
        system_prompt = "你是一位量子实验设计专家，负责评估该诠释的实验可行性与仪器要求。"
        user_prompt = f"""
        理论名称: {theory.get('name', '未命名理论')}
        核心主张: {context.get('core', '')}
        SQM数学关系: {context.get('math', '')}
        关键预测: {context.get('predictions', '')}
        非局域性/测量说明: {context.get('locality', '')} | {context.get('measurement', '')}
        已知矛盾: {context.get('contradictions', '')}

        请评估以下内容:
        1. 实验或仪器上是否可行，需额外什么条件？
        2. 给出可能的实验路线或观测方案。
        3. 标出主要风险或未解决的问题。

        使用JSON返回，字段为 score(0-10)、instrument_readiness、experimental_path、risks。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        reviews: List[Dict[str, Any]] = []
        scores: List[float] = []
        for cfg in model_cfgs:
            try:
                review = await self.llm.query_structured_json(
                    messages=messages,
                    schema=schema,
                    schema_name="instrument_review",
                    temperature=0.2,
                    model_source=cfg.get("model_source"),
                    model_name=cfg.get("model_name")
                )
            except Exception as exc:
                review = {"error": str(exc)}
            review = review or {"error": "无有效响应"}
            review["model_source"] = cfg.get("model_source")
            review["model_name"] = cfg.get("model_name")
            score = review.get("score")
            if isinstance(score, str):
                try:
                    score = float(score)
                    review["score"] = score
                except ValueError:
                    score = None
            if isinstance(score, (int, float)):
                scores.append(float(score))
            reviews.append(review)
        average = sum(scores) / len(scores) if scores else 0.0
        return {
            "average_score": average,
            "reviews": reviews
        }

    async def _generate_evaluation_summary(self, theory: Dict, evaluations: Dict) -> Dict:
        """
        生成评估总结
        
        Args:
            theory: 被评估的理论
            evaluations: 所有角色的评估结果
            
        Returns:
            Dict: 评估总结
        """
        # 提取各角色评估的要点
        evaluation_summary = ""
        overall_score = evaluations.get("overall_score", 0)
        
        for role_id, eval_data in evaluations.get("evaluations", {}).items():
            role_name = eval_data.get("role", role_id)
            score = eval_data.get("score", 0)
            strengths = "\n".join([f"- {s}" for s in eval_data.get("strengths", [])])
            weaknesses = "\n".join([f"- {w}" for w in eval_data.get("weaknesses", [])])
            
            evaluation_summary += f"""
            {role_name}评分: {score}/10
            优势:
            {strengths}
            
            弱点:
            {weaknesses}
            
            """
        
        # 构建提示
        instrumentation = evaluations.get("instrumentation_review") or {}
        instrumentation_summary = ""
        if instrumentation:
            avg = instrumentation.get('average_score', 0)
            instrumentation_summary = f"仪器/实验可行性平均得分: {avg:.2f}/10\n"

        core_overview = self._stringify(
            theory.get('core_principles')
            or theory.get('summary')
            or theory.get('writeup')
        )

        prompt = f"""
        作为量子物理学理论评审委员会主席，你需要对一个新提出的量子诠释理论做出总体评价。
        
        理论名称: {theory.get('name', '未命名理论')}
        核心原理: {core_overview}
        
        各专家评价摘要:
        {evaluation_summary}

        {instrumentation_summary}
        
        总体评分: {overall_score:.2f}/10
        
        请提供总体评价，包括:
        1. 这个理论的潜在价值
        2. 是否推荐进一步发展这个理论
        3. 建议的改进方向
        
        以JSON格式返回:
        {{
          "potential_value": "对这个理论潜在价值的评价...",
          "recommendation": "推荐意见...",
          "improvement_directions": "改进方向..."
        }}
        
        请根据专家评价和总体评分给出平衡、客观的建议。
        """
        
        # 调用LLM生成总结
        try:
            response = await self.llm.query_async(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            # 解析结果
            summary = self.llm.extract_json(response)
            return summary if summary else {
                "potential_value": "无法生成评价",
                "recommendation": "无法提供推荐意见",
                "improvement_directions": []
            }
        except Exception as e:
            print(f"[ERROR] 生成评估总结失败: {str(e)}")
            return {
                "potential_value": f"评估过程出错: {str(e)}",
                "recommendation": "无法提供推荐意见",
                "improvement_directions": []
            }
    
    async def evaluate_theories(self, theories: List[Dict]) -> List[Dict]:
        """
        评估多个理论
        
        Args:
            theories: 理论列表
            
        Returns:
            List[Dict]: 评估结果列表
        """
        results = []
        for i, theory in enumerate(theories):
            print(f"[INFO] 评估理论 {i+1}/{len(theories)}")
            result = await self.evaluate_theory(theory)
            results.append(result)
        
        self.evaluation_results = results
        return results
    
    def save_evaluation_results(self, output_path: str) -> None:
        """
        保存评估结果
        
        Args:
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_results, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 评估结果已保存到: {output_path}")
        except Exception as e:
            print(f"[ERROR] 保存评估结果失败: {str(e)}") 
