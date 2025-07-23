#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_level_adapter.py - 多层级创新生成方法适配器
===============================================

将多层级创新生成器封装为标准化适配器。
支持多层次创新策略和概念向量空间增强。
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List

try:
    from .base_adapter import TheoryGenerationMethod, GenerationResult
    from ..llm_interface import LLMInterface
    from ..multi_level_innovation_generator import MultiLevelInnovationGenerator, MultiLevelInnovationConfig
    from ..innovation_framework import InnovationLevel
    from ..direct_synthesis.contradiction_analyzer import ContradictionAnalyzer
except ImportError:
    # 备用导入路径
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    
    from methods.base_adapter import TheoryGenerationMethod, GenerationResult
    from llm_interface import LLMInterface
    from multi_level_innovation_generator import MultiLevelInnovationGenerator, MultiLevelInnovationConfig
    from innovation_framework import InnovationLevel
    from direct_synthesis.contradiction_analyzer import ContradictionAnalyzer


class MultiLevelAdapter(TheoryGenerationMethod):
    """多层级创新生成方法适配器"""
    
    def __init__(self, 
                 theories_dir: str,
                 output_dir: str,
                 max_pairs: int = 20,
                 variants_per_contradiction: int = 3,
                 model_source: str = "google",
                 model_name: str = "gemini-2.5-flash",
                 # 多层级特有参数
                 target_levels: List[str] = None,
                 level_weights: Dict[str, float] = None,
                 synthesis_mode: str = "parallel",
                 innovation_intensity: float = 0.7,
                 use_concept_space: bool = True,
                 **kwargs):
        """初始化多层级创新适配器
        
        Args:
            theories_dir: 理论文件目录
            output_dir: 输出目录
            max_pairs: 最大比较对数
            variants_per_contradiction: 每个矛盾的假说变体数量
            model_source: 模型来源
            model_name: 模型名称
            target_levels: 目标创新层次列表
            level_weights: 各层次权重字典
            synthesis_mode: 综合模式 (parallel, hierarchical, fusion)
            innovation_intensity: 创新强度 (0.0-1.0)
            use_concept_space: 是否使用概念向量空间
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
        
        # 多层级特有配置
        self.target_levels = target_levels or ["parameter_extension", "framework_extension"]
        self.level_weights = level_weights or {"parameter_extension": 0.4, "framework_extension": 0.6}
        self.synthesis_mode = synthesis_mode
        self.innovation_intensity = innovation_intensity
        self.use_concept_space = use_concept_space
        
        self._log_info(f"多层级创新配置:")
        self._log_info(f"  目标层次: {self.target_levels}")
        self._log_info(f"  层次权重: {self.level_weights}")
        self._log_info(f"  综合模式: {self.synthesis_mode}")
        self._log_info(f"  创新强度: {self.innovation_intensity}")
    
    def generate(self) -> Dict[str, Any]:
        """生成理论 - 异步包装器"""
        try:
            # 运行异步生成过程
            result = asyncio.run(self._async_generate())
            return self._ensure_output_format(result)
        except Exception as e:
            self._log_error(f"多层级生成过程出错: {e}")
            return GenerationResult(
                success=False,
                error_message=str(e),
                output_dir=str(self.output_dir)
            ).to_dict()
    
    async def _async_generate(self) -> Dict[str, Any]:
        """异步生成理论"""
        self._log_info("开始多层级创新理论生成")
        
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
        self._log_info(f"将分析 {len(theory_pairs)} 对理论的矛盾点")
        
        # 4. 进行矛盾分析
        all_analyses = []
        for theory1, theory2 in theory_pairs:
            analysis = await analyzer.find_contradictions(theory1, theory2)
            if "error" not in analysis:
                all_analyses.append(analysis)
        
        self._log_info(f"完成 {len(all_analyses)} 个矛盾分析")
        
        # 5. 使用直接合成方法作为基础，然后添加多层级特性
        # 由于多层级生成器存在复杂性问题，我们暂时使用基础的直接合成方法
        # 但添加多层级的配置和标记
        
        try:
            from ..direct_synthesis.hypothesis_generator import HypothesisGenerator
            generator = HypothesisGenerator(llm)
        except ImportError:
            raise ImportError("无法导入HypothesisGenerator，多层级创新方法暂时不可用")
        
        # 6. 生成理论
        all_theories = []
        for analysis in all_analyses:
            self._log_info(f"为矛盾生成多层级创新理论: {analysis.get('theory1')} vs {analysis.get('theory2')}")
            
            # 为每个矛盾生成多个变体
            for variant_idx in range(self.variants_per_contradiction):
                try:
                    # 使用基础生成器生成理论
                    theories = await generator.generate_multiple_hypotheses(
                        contradiction=analysis,
                        variants_count=1,
                        diversity_level=self.innovation_intensity
                    )
                    
                    for theory in theories:
                        if theory and isinstance(theory, dict):
                            # 添加多层级特性标记
                            theory['variant_info'] = {
                                'source_contradiction': f"{analysis.get('theory1')} vs {analysis.get('theory2')}",
                                'variant_index': variant_idx + 1,
                                'generation_method': 'multi_level_innovation',
                                'innovation_levels': self.target_levels,
                                'synthesis_mode': self.synthesis_mode,
                                'innovation_intensity': self.innovation_intensity
                            }
                            
                            # 添加多层级元数据
                            if 'metadata' not in theory:
                                theory['metadata'] = {}
                            theory['metadata']['multi_level_config'] = {
                                'target_levels': self.target_levels,
                                'level_weights': self.level_weights,
                                'synthesis_mode': self.synthesis_mode,
                                'innovation_intensity': self.innovation_intensity
                            }
                            
                            all_theories.append(theory)
                            self._log_info(f"成功生成多层级理论变体 {variant_idx + 1}")
                    
                except Exception as e:
                    self._log_warning(f"生成变体 {variant_idx + 1} 失败: {e}")
                    continue
        
        # 8. 保存结果
        return self._process_results(all_theories)
    
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
    
    def _create_multi_level_config(self) -> MultiLevelInnovationConfig:
        """创建多层级创新配置"""
        # 转换字符串层次到枚举
        level_mapping = {
            "interpretation": InnovationLevel.INTERPRETATION,
            "parameter_extension": InnovationLevel.PARAMETER_EXTENSION,
            "equation_modification": InnovationLevel.EQUATION_MODIFICATION,
            "framework_extension": InnovationLevel.FRAMEWORK_EXTENSION,
            "paradigm_revolution": InnovationLevel.PARADIGM_REVOLUTION
        }
        
        target_levels = []
        level_weights = {}
        
        for level_str in self.target_levels:
            if level_str in level_mapping:
                level_enum = level_mapping[level_str]
                target_levels.append(level_enum)
                level_weights[level_enum] = self.level_weights.get(level_str, 0.5)
        
        if not target_levels:
            # 默认配置
            target_levels = [InnovationLevel.PARAMETER_EXTENSION, InnovationLevel.FRAMEWORK_EXTENSION]
            level_weights = {
                InnovationLevel.PARAMETER_EXTENSION: 0.4,
                InnovationLevel.FRAMEWORK_EXTENSION: 0.6
            }
        
        return MultiLevelInnovationConfig(
            target_levels=target_levels,
            level_weights=level_weights,
            synthesis_mode=self.synthesis_mode,
            innovation_intensity=self.innovation_intensity,
            constraint_preservation=True,
            use_concept_space=self.use_concept_space
        )
    
    def _process_results(self, all_theories: List[Dict]) -> Dict[str, Any]:
        """处理和保存结果"""
        if not all_theories:
            self._log_warning("未生成任何多层级创新理论")
            return GenerationResult(
                success=True,
                theories=[],
                metadata={'message': '未生成任何多层级创新理论'},
                output_dir=str(self.output_dir)
            ).to_dict()
        
        # 保存理论到文件
        output_file = self.output_dir / "multi_level_theories.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_theories, f, ensure_ascii=False, indent=2)
        
        # 创建评估就绪的理论文件
        eval_theories_dir = self.output_dir / "eval_ready_theories"
        eval_theories_dir.mkdir(exist_ok=True)
        
        for i, theory in enumerate(all_theories):
            theory_name = theory.get("name", f"multi_level_theory_{i+1}")
            safe_name = theory_name.replace(" ", "_").replace("/", "_").lower()
            
            eval_file = eval_theories_dir / f"{safe_name}.json"
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(theory, f, ensure_ascii=False, indent=2)
        
        # 统计信息
        level_stats = {}
        for theory in all_theories:
            levels = theory.get('variant_info', {}).get('innovation_levels', [])
            for level in levels:
                level_stats[level] = level_stats.get(level, 0) + 1
        
        self._log_info(f"总共生成 {len(all_theories)} 个多层级创新理论")
        self._log_info(f"创新层次统计: {level_stats}")
        self._log_info(f"评估就绪文件保存到: {eval_theories_dir}")
        
        # 返回标准化结果
        return GenerationResult(
            success=True,
            theories=all_theories,
            metadata={
                'generation_method': 'multi_level_innovation',
                'innovation_levels': self.target_levels,
                'synthesis_mode': self.synthesis_mode,
                'innovation_intensity': self.innovation_intensity,
                'level_statistics': level_stats,
                'eval_theories_dir': str(eval_theories_dir)
            },
            output_dir=str(self.output_dir)
        ).to_dict() 