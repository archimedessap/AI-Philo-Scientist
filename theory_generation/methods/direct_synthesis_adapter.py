#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
direct_synthesis_adapter.py - 直接合成方法适配器
===========================================

将现有的 run_direct_synthesis.py 功能封装为标准化适配器。
保持与 run_clean_evolution.py 的完全兼容性。
"""

import os
import json
import asyncio
import time
import glob
from pathlib import Path
from typing import Dict, Any, List

try:
    from .base_adapter import TheoryGenerationMethod, GenerationResult
    from ..llm_interface import LLMInterface
    from ..direct_synthesis.contradiction_analyzer import ContradictionAnalyzer
    from ..direct_synthesis.hypothesis_generator import HypothesisGenerator
except ImportError:
    # 备用导入路径
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    
    from methods.base_adapter import TheoryGenerationMethod, GenerationResult
    from llm_interface import LLMInterface
    from direct_synthesis.contradiction_analyzer import ContradictionAnalyzer
    from direct_synthesis.hypothesis_generator import HypothesisGenerator


class DirectSynthesisAdapter(TheoryGenerationMethod):
    """直接合成方法适配器
    
    基于矛盾分析的直接合成方法 - AI-Philo1.0的核心方法
    """
    
    def __init__(self, 
                 theories_dir: str,
                 output_dir: str,
                 max_pairs: int = 20,
                 variants_per_contradiction: int = 3,
                 model_source: str = "google",
                 model_name: str = "gemini-2.5-flash",
                 specific_pair: str = None,
                 schema_version: str = "2.1",
                 diversity_level: float = 0.7,
                 **kwargs):
        """初始化直接合成适配器
        
        Args:
            theories_dir: 理论文件目录
            output_dir: 输出目录
            max_pairs: 最大比较对数
            variants_per_contradiction: 每个矛盾的假说变体数量
            model_source: 模型来源
            model_name: 模型名称
            specific_pair: 特定理论对，格式为'理论1,理论2'
            schema_version: 理论schema版本
            diversity_level: 变体多样性等级(0.0-1.0)
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
        
        self.specific_pair = specific_pair
        self.schema_version = schema_version
        self.diversity_level = diversity_level
        
        # 创建时间戳子目录
        self.synthesis_dir = self.output_dir / f"synthesis_{time.strftime('%Y%m%d_%H%M%S')}"
        self.synthesis_dir.mkdir(parents=True, exist_ok=True)
        
        self._log_info(f"创建合成输出目录: {self.synthesis_dir}")
    
    def generate(self) -> Dict[str, Any]:
        """生成理论 - 异步包装器"""
        try:
            # 运行异步生成过程
            result = asyncio.run(self._async_generate())
            return self._ensure_output_format(result)
        except Exception as e:
            self._log_error(f"生成过程出错: {e}")
            return GenerationResult(
                success=False,
                error_message=str(e),
                output_dir=str(self.synthesis_dir)
            ).to_dict()
    
    async def _async_generate(self) -> Dict[str, Any]:
        """异步生成理论"""
        self._log_info("开始基于矛盾分析的直接合成")
        
        # 1. 初始化LLM接口
        llm = LLMInterface(
            model_source=self.model_source,
            model_name=self.model_name,
            request_interval=1.0
        )
        
        model_info = llm.get_current_model_info()
        self._log_info(f"使用模型: {model_info['source']} - {model_info['name']}")
        
        # 2. 加载理论数据
        self._log_info(f"从 {self.theories_dir} 加载理论数据")
        analyzer = ContradictionAnalyzer(llm)
        
        load_schema_version = None if self.schema_version.lower() == 'any' else self.schema_version
        analyzer.load_theories(str(self.theories_dir), schema_version=load_schema_version)
        
        if not analyzer.theories:
            raise ValueError("未加载到理论数据")
        
        self._log_info(f"成功加载 {len(analyzer.theories)} 个理论")
        
        # 3. 确定要比较的理论对
        theory_pairs = self._select_theory_pairs(analyzer.theories)
        self._log_info(f"将分析 {len(theory_pairs)} 对理论的矛盾点")
        
        # 4. 矛盾分析
        all_analyses = await self._analyze_contradictions(analyzer, theory_pairs)
        
        # 5. 生成新理论
        all_hypotheses = await self._generate_hypotheses(llm, all_analyses)
        
        # 6. 保存和返回结果
        return self._process_results(all_hypotheses)
    
    def _select_theory_pairs(self, theories: Dict) -> List[tuple]:
        """选择要分析的理论对"""
        theory_pairs = []
        theory_names = list(theories.keys())
        
        if self.specific_pair:
            # 使用指定的理论对
            theory_names_pair = self.specific_pair.split(',')
            if len(theory_names_pair) != 2:
                raise ValueError(f"理论对格式错误: {self.specific_pair}，应为'理论1,理论2'")
            theory_pairs.append((theory_names_pair[0].strip(), theory_names_pair[1].strip()))
        else:
            # 自动选择理论对
            if len(theory_names) < 2:
                raise ValueError("至少需要2个理论才能进行比较")
                
            import random
            from itertools import combinations
            
            # 生成所有可能的理论对并随机选择
            all_pairs = list(combinations(theory_names, 2))
            random.shuffle(all_pairs)
            theory_pairs = all_pairs[:self.max_pairs]
        
        return theory_pairs
    
    async def _analyze_contradictions(self, analyzer: ContradictionAnalyzer, theory_pairs: List[tuple]) -> List[Dict]:
        """分析理论矛盾"""
        self._log_info("开始矛盾分析")
        all_analyses = []
        
        for theory1, theory2 in theory_pairs:
            self._log_info(f"分析矛盾: {theory1} vs {theory2}")
            
            analysis = await analyzer.find_contradictions(theory1, theory2)
            if "error" not in analysis:
                all_analyses.append(analysis)
                
                # 保存单个分析结果
                pair_name = f"{theory1}_vs_{theory2}".replace(" ", "_")
                analysis_dir = self.synthesis_dir / pair_name
                analysis_dir.mkdir(exist_ok=True)
                
                analysis_file = analysis_dir / "contradiction_analysis.json"
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, ensure_ascii=False, indent=2)
            else:
                self._log_warning(f"分析失败: {theory1} vs {theory2} - {analysis.get('error', '未知错误')}")
        
        # 保存所有分析结果
        analyses_file = self.synthesis_dir / "all_contradiction_analyses.json"
        with open(analyses_file, 'w', encoding='utf-8') as f:
            json.dump(all_analyses, f, ensure_ascii=False, indent=2)
        
        self._log_info(f"完成 {len(all_analyses)} 个矛盾分析")
        return all_analyses
    
    async def _generate_hypotheses(self, llm: LLMInterface, analyses: List[Dict]) -> List[Dict]:
        """基于矛盾生成新假说"""
        self._log_info("开始基于矛盾点合成新理论")
        generator = HypothesisGenerator(llm)
        
        # 导入数学分类器
        try:
            import sys
            if '.' not in sys.path:
                sys.path.append('.')
            from utils.mathematical_classifier import MathematicalClassifier
            classifier = MathematicalClassifier()
        except ImportError as e:
            self._log_warning(f"无法导入数学分类器: {e}")
            classifier = None
        
        for analysis in analyses:
            theory1 = analysis.get("theory1")
            theory2 = analysis.get("theory2")
            pair_name = f"{theory1}_vs_{theory2}".replace(" ", "_")
            
            self._log_info(f"处理矛盾: {theory1} vs {theory2}")
            
            # 为该理论对创建输出目录
            pair_dir = self.synthesis_dir / pair_name
            hypotheses_dir = pair_dir / "hypotheses"
            hypotheses_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成多个假说变体
            hypotheses = await generator.generate_multiple_hypotheses(
                contradiction=analysis,
                variants_count=self.variants_per_contradiction,
                diversity_level=self.diversity_level
            )
            
            # 保存生成的假说
            for i, hypothesis in enumerate(hypotheses):
                # 添加数学分类标注
                if classifier:
                    hypothesis = classifier.annotate_theory_with_classification(hypothesis)
                
                hypothesis_name = hypothesis.get("name", f"新理论_{i+1}")
                safe_name = hypothesis_name.replace(" ", "_").replace("/", "_").lower()
                
                # 保存到文件
                hypothesis_file = hypotheses_dir / f"{safe_name}.json"
                with open(hypothesis_file, 'w', encoding='utf-8') as f:
                    json.dump(hypothesis, f, ensure_ascii=False, indent=2)
            
            # 保存所有假说到一个文件
            all_file = hypotheses_dir / "all_variants.json"
            with open(all_file, 'w', encoding='utf-8') as f:
                json.dump(hypotheses, f, ensure_ascii=False, indent=2)
                
            self._log_info(f"为 {theory1} vs {theory2} 生成了 {len(hypotheses)} 个理论假说")
        
        return generator.generated_hypotheses
    
    def _process_results(self, all_hypotheses: List[Dict]) -> Dict[str, Any]:
        """处理和保存结果"""
        if not all_hypotheses:
            self._log_warning("未生成任何新理论")
            return GenerationResult(
                success=True,
                theories=[],
                metadata={'message': '未生成任何新理论'},
                output_dir=str(self.synthesis_dir)
            ).to_dict()
        
        # 保存汇总文件
        summary_file = self.synthesis_dir / "all_synthesized_theories.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_hypotheses, f, ensure_ascii=False, indent=2)
        
        # 创建标准格式的理论文件，用于评估
        eval_theories_dir = self.synthesis_dir / "eval_ready_theories"
        eval_theories_dir.mkdir(exist_ok=True)
        
        # 导入数学分类器（如果需要）
        try:
            import sys
            if '.' not in sys.path:
                sys.path.append('.')
            from utils.mathematical_classifier import MathematicalClassifier
            classifier = MathematicalClassifier()
        except ImportError:
            classifier = None
        
        for hypothesis in all_hypotheses:
            # 确保理论有数学分类标注
            if classifier and "mathematical_classification" not in hypothesis.get("metadata", {}):
                hypothesis = classifier.annotate_theory_with_classification(hypothesis)
            
            # 获取理论名
            theory_name = hypothesis.get("name", "未命名理论")
            safe_name = theory_name.replace(" ", "_").replace("/", "_").lower()
            
            # 保存标准格式的理论文件
            eval_file = eval_theories_dir / f"{safe_name}.json"
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(hypothesis, f, ensure_ascii=False, indent=2)
        
        # 生成数学分类统计
        classification_stats = {"standard_qm": 0, "modified_qm": 0, "extended_qm": 0}
        for hypothesis in all_hypotheses:
            math_classification = hypothesis.get("metadata", {}).get("mathematical_classification", {})
            math_type = math_classification.get("type", "unknown")
            if math_type in classification_stats:
                classification_stats[math_type] += 1
        
        self._log_info(f"总共合成了 {len(all_hypotheses)} 个新理论")
        self._log_info(f"标准格式的评估理论文件已保存到: {eval_theories_dir}")
        self._log_info(f"数学分类统计: {classification_stats}")
        
        # 返回标准化结果
        return GenerationResult(
            success=True,
            theories=all_hypotheses,
            metadata={
                'synthesis_dir': str(self.synthesis_dir),
                'eval_theories_dir': str(eval_theories_dir),
                'classification_stats': classification_stats,
                'method': 'direct_synthesis',
                'contradictions_analyzed': len(all_hypotheses) // self.variants_per_contradiction
            },
            output_dir=str(self.synthesis_dir)
        ).to_dict() 