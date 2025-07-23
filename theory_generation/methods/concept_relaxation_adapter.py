#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
concept_relaxation_adapter.py - 概念放松方法适配器
=============================================

将概念放松生成方法封装为标准化适配器。
通过放松理论间矛盾概念来生成新理论。
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

try:
    from .base_adapter import TheoryGenerationMethod, GenerationResult
    from ..llm_interface import LLMInterface
    from ..direct_synthesis.contradiction_analyzer import ContradictionAnalyzer
except ImportError:
    # 备用导入路径
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    
    from methods.base_adapter import TheoryGenerationMethod, GenerationResult
    from llm_interface import LLMInterface
    from direct_synthesis.contradiction_analyzer import ContradictionAnalyzer


class ConceptRelaxationAdapter(TheoryGenerationMethod):
    """概念放松方法适配器"""
    
    def __init__(self, 
                 theories_dir: str,
                 output_dir: str,
                 max_pairs: int = 20,
                 variants_per_contradiction: int = 3,
                 model_source: str = "google",
                 model_name: str = "gemini-2.5-flash",
                 # 概念放松特有参数
                 relaxation_intensity: float = 0.6,
                 concept_flexibility: float = 0.7,
                 integration_mode: str = "adaptive",
                 **kwargs):
        """初始化概念放松适配器
        
        Args:
            theories_dir: 理论文件目录
            output_dir: 输出目录
            max_pairs: 最大比较对数
            variants_per_contradiction: 每个矛盾的假说变体数量
            model_source: 模型来源
            model_name: 模型名称
            relaxation_intensity: 概念放松强度 (0.0-1.0)
            concept_flexibility: 概念灵活性 (0.0-1.0)
            integration_mode: 整合模式 (adaptive, conservative, radical)
            **kwargs: 其他参数
        """
        super().__init__(
            theories_dir=theories_dir,
            output_dir=output_dir,
            max_pairs=max_pairs,
            variants_per_contradiction=variants_per_contradiction,
            model_source=model_source,
            model_name=model_name,
            **kwargs
        )
        
        # 概念放松特有配置
        self.relaxation_intensity = relaxation_intensity
        self.concept_flexibility = concept_flexibility
        self.integration_mode = integration_mode
        
        self._log_info(f"概念放松配置:")
        self._log_info(f"  放松强度: {self.relaxation_intensity}")
        self._log_info(f"  概念灵活性: {self.concept_flexibility}")
        self._log_info(f"  整合模式: {self.integration_mode}")
    
    def generate(self) -> Dict[str, Any]:
        """生成理论 - 异步包装器"""
        try:
            # 运行异步生成过程
            result = asyncio.run(self._async_generate())
            return self._ensure_output_format(result)
        except Exception as e:
            self._log_error(f"概念放松生成过程出错: {e}")
            return GenerationResult(
                success=False,
                error_message=str(e),
                output_dir=str(self.output_dir)
            ).to_dict()
    
    async def _async_generate(self) -> Dict[str, Any]:
        """异步生成理论"""
        self._log_info("开始概念放松理论生成")
        
        # 1. 初始化LLM接口
        llm = LLMInterface(
            model_source=self.model_source,
            model_name=self.model_name,
            request_interval=1.0
        )
        
        model_info = llm.get_current_model_info()
        self._log_info(f"使用模型: {model_info['source']} - {model_info['name']}")
        
        # 2. 加载理论数据并进行矛盾分析
        analyzer = ContradictionAnalyzer(llm)
        analyzer.load_theories(str(self.theories_dir))
        
        if not analyzer.theories:
            raise ValueError("未加载到理论数据")
        
        self._log_info(f"成功加载 {len(analyzer.theories)} 个理论")
        
        # 3. 选择理论对进行分析
        theory_pairs = self._select_theory_pairs(analyzer.theories)
        self._log_info(f"将分析 {len(theory_pairs)} 对理论的概念放松")
        
        # 4. 进行矛盾分析
        all_analyses = []
        for theory1, theory2 in theory_pairs:
            analysis = await analyzer.find_contradictions(theory1, theory2)
            if "error" not in analysis:
                all_analyses.append(analysis)
        
        self._log_info(f"完成 {len(all_analyses)} 个矛盾分析")
        
        # 5. 生成基于概念放松的理论
        all_theories = []
        for analysis in all_analyses:
            self._log_info(f"为矛盾进行概念放松: {analysis.get('theory1')} vs {analysis.get('theory2')}")
            
            # 为每个矛盾生成多个变体
            for variant_idx in range(self.variants_per_contradiction):
                try:
                    # 使用概念放松方法生成理论
                    theory = await self._generate_relaxed_theory(llm, analysis, variant_idx)
                    
                    if theory and isinstance(theory, dict):
                        # 添加概念放松特性标记
                        theory['variant_info'] = {
                            'source_contradiction': f"{analysis.get('theory1')} vs {analysis.get('theory2')}",
                            'variant_index': variant_idx + 1,
                            'generation_method': 'concept_relaxation',
                            'relaxation_intensity': self.relaxation_intensity,
                            'concept_flexibility': self.concept_flexibility,
                            'integration_mode': self.integration_mode
                        }
                        
                        # 添加概念放松元数据
                        if 'metadata' not in theory:
                            theory['metadata'] = {}
                        theory['metadata']['relaxation_config'] = {
                            'relaxation_intensity': self.relaxation_intensity,
                            'concept_flexibility': self.concept_flexibility,
                            'integration_mode': self.integration_mode,
                            'contradictions_relaxed': len(analysis.get('contradictions', []))
                        }
                        
                        # 添加放松概念标记
                        if 'concepts' not in theory['metadata']:
                            theory['metadata']['concepts'] = []
                        theory['metadata']['concepts'].extend([
                            'relaxed_conceptual_framework',
                            'adaptive_philosophical_stance',
                            'integrated_mathematical_structure'
                        ])
                        
                        all_theories.append(theory)
                        self._log_info(f"成功生成概念放松理论变体 {variant_idx + 1}")
                    
                except Exception as e:
                    self._log_warning(f"生成变体 {variant_idx + 1} 失败: {e}")
                    continue
        
        # 6. 保存结果
        return self._process_results(all_theories)
    
    async def _generate_relaxed_theory(self, llm: LLMInterface, analysis: Dict, variant_idx: int) -> Dict[str, Any]:
        """基于概念放松生成理论"""
        
        # 提取矛盾信息
        contradictions = analysis.get('contradictions', [])
        theory1 = analysis.get('theory1', 'Theory A')
        theory2 = analysis.get('theory2', 'Theory B')
        
        # 构建概念放松的提示词
        relaxation_prompt = self._build_relaxation_prompt(theory1, theory2, contradictions, variant_idx)
        
        # 调用LLM生成放松理论
        messages = [{"role": "user", "content": relaxation_prompt}]
        response = await llm.query_async(
            messages, 
            temperature=0.6 + self.relaxation_intensity * 0.3  # 动态调整temperature
        )
        
        # 解析响应
        theory = llm.extract_json(response)
        
        if not theory:
            # 如果JSON解析失败，创建基础理论结构
            theory = {
                "name": f"Relaxed Theory {variant_idx + 1}",
                "description": "基于概念放松生成的理论",
                "key_concepts": ["conceptual_flexibility", "adaptive_framework"],
                "philosophical_stance": "conceptual_pluralism",
                "mathematical_formalism": "relaxed_formalism",
                "empirical_predictions": ["adaptive_predictions"],
                "relation_to_original_theories": f"Relaxation of {theory1} and {theory2}"
            }
        
        return theory
    
    def _build_relaxation_prompt(self, theory1: str, theory2: str, contradictions: List[Dict], variant_idx: int) -> str:
        """构建概念放松的提示词"""
        
        # 根据整合模式调整提示
        mode_instructions = {
            "adaptive": "采用适应性方法，灵活整合两个理论的优势",
            "conservative": "保持理论的核心结构，只进行最小必要的概念调整",
            "radical": "大胆创新，彻底重构概念框架"
        }
        
        mode_instruction = mode_instructions.get(self.integration_mode, mode_instructions["adaptive"])
        
        # 构建矛盾描述
        contradiction_text = ""
        for i, contradiction in enumerate(contradictions[:3]):  # 限制数量
            contradiction_text += f"""
矛盾点 {i+1}: {contradiction.get('aspect', '未知方面')}
- 描述: {contradiction.get('description', '无描述')}
- {theory1}的立场: {contradiction.get('theory1_position', '未知立场')}
- {theory2}的立场: {contradiction.get('theory2_position', '未知立场')}
"""
        
        prompt = f"""
你是一个量子理论创新专家，擅长通过概念放松来解决理论矛盾。

任务：基于以下两个理论之间的矛盾，通过概念放松生成一个新的量子理论。

原始理论：
理论A: {theory1}
理论B: {theory2}

矛盾点：{contradiction_text}

概念放松指导：
- 放松强度: {self.relaxation_intensity} (0-1, 越高越激进)
- 概念灵活性: {self.concept_flexibility} (0-1, 越高越灵活)
- 整合策略: {mode_instruction}

请生成一个新的量子理论，要求：

1. **概念放松**: 对矛盾概念进行适当放松，允许更灵活的解释
2. **矛盾调和**: 通过放松来调和两个理论的根本分歧
3. **创新整合**: 创造性地整合两个理论的洞见
4. **内在一致性**: 确保新理论的内部逻辑一致性
5. **变体特色**: 第{variant_idx + 1}个变体，应有独特的放松角度

以JSON格式返回，包含：
{{
    "name": "新理论名称",
    "description": "理论的详细描述",
    "key_concepts": ["关键概念列表"],
    "philosophical_stance": "哲学立场",
    "mathematical_formalism": "数学形式化描述",
    "empirical_predictions": ["经验预测"],
    "relaxed_concepts": ["被放松的概念"],
    "integration_strategy": "整合策略说明",
    "relation_to_original_theories": "与原始理论的关系"
}}

只返回JSON，不要添加任何解释文字。
"""
        
        return prompt
    
    def _select_theory_pairs(self, theories: Dict) -> List[tuple]:
        """选择要分析的理论对"""
        theory_names = list(theories.keys())
        
        if len(theory_names) < 2:
            raise ValueError("至少需要2个理论才能进行比较")
        
        import random
        from itertools import combinations
        
        # 生成所有可能的理论对并随机选择
        all_pairs = list(combinations(theory_names, 2))
        random.shuffle(all_pairs)
        return all_pairs[:self.max_pairs]
    
    def _process_results(self, all_theories: List[Dict]) -> Dict[str, Any]:
        """处理和保存结果"""
        if not all_theories:
            self._log_warning("未生成任何概念放松理论")
            return GenerationResult(
                success=True,
                theories=[],
                metadata={'message': '未生成任何概念放松理论'},
                output_dir=str(self.output_dir)
            ).to_dict()
        
        # 保存理论到文件
        output_file = self.output_dir / "relaxed_theories.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_theories, f, ensure_ascii=False, indent=2)
        
        # 创建评估就绪的理论文件
        eval_theories_dir = self.output_dir / "eval_ready_theories"
        eval_theories_dir.mkdir(exist_ok=True)
        
        for i, theory in enumerate(all_theories):
            theory_name = theory.get("name", f"relaxed_theory_{i+1}")
            safe_name = theory_name.replace(" ", "_").replace("/", "_").lower()
            
            eval_file = eval_theories_dir / f"{safe_name}.json"
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(theory, f, ensure_ascii=False, indent=2)
        
        # 统计信息
        integration_modes = {}
        relaxed_concepts_count = 0
        
        for theory in all_theories:
            mode = theory.get('variant_info', {}).get('integration_mode', 'unknown')
            integration_modes[mode] = integration_modes.get(mode, 0) + 1
            
            relaxed_concepts = theory.get('relaxed_concepts', [])
            relaxed_concepts_count += len(relaxed_concepts)
        
        avg_relaxed_concepts = relaxed_concepts_count / max(len(all_theories), 1)
        
        self._log_info(f"总共生成 {len(all_theories)} 个概念放松理论")
        self._log_info(f"整合模式统计: {integration_modes}")
        self._log_info(f"平均放松概念数: {avg_relaxed_concepts:.1f}")
        self._log_info(f"评估就绪文件保存到: {eval_theories_dir}")
        
        # 返回标准化结果
        return GenerationResult(
            success=True,
            theories=all_theories,
            metadata={
                'generation_method': 'concept_relaxation',
                'relaxation_intensity': self.relaxation_intensity,
                'concept_flexibility': self.concept_flexibility,
                'integration_mode': self.integration_mode,
                'integration_mode_statistics': integration_modes,
                'avg_relaxed_concepts': avg_relaxed_concepts,
                'eval_theories_dir': str(eval_theories_dir)
            },
            output_dir=str(self.output_dir)
        ).to_dict() 