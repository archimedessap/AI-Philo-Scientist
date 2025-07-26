#!/usr/bin/env python3
"""
Formula Extractor - 公式提取和分类器

该模块从科学文献中提取数学公式，进行分类、格式化和关系分析。
支持LaTeX格式统一、公式链推导、适用条件提取。

Author: UniversalTheoryGen Team
Date: 2025-01-24
"""

import json
import os
import sys
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import asyncio
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from theory_generation.llm_interface import LLMInterface
from utils.logging_config import get_logger
from utils.retry_decorator import retry_with_exponential_backoff

# 初始化日志
logger = get_logger('formula_extractor')

@dataclass
class Formula:
    """公式数据结构"""
    id: str  # 唯一标识符
    expression: str  # LaTeX格式的公式
    name: str  # 公式名称（如"薛定谔方程"）
    category: str  # definition, physical_law, derivation, boundary_condition
    description: str  # 公式描述
    variables: Dict[str, str]  # 变量说明 {"ψ": "wave function", "H": "Hamiltonian"}
    assumptions: List[str]  # 假设条件
    domain: str  # 适用领域
    source: str  # 来源文献
    related_concepts: List[str]  # 相关概念
    derivation_from: Optional[str]  # 从哪个公式推导而来
    derivation_steps: List[str]  # 推导步骤（如果有）
    confidence: float  # 置信度

@dataclass
class FormulaRelation:
    """公式关系"""
    formula1_id: str
    relation_type: str  # derives_from, equivalent_to, special_case_of, generalizes
    formula2_id: str
    conditions: List[str]  # 关系成立的条件
    proof_sketch: str  # 简要证明或说明

class FormulaExtractor:
    """公式提取器"""
    
    # 公式类别定义
    FORMULA_CATEGORIES = {
        'definition': 'Mathematical definitions and fundamental equations',
        'physical_law': 'Physical laws and principles',
        'derivation': 'Derived results and theorems',
        'boundary_condition': 'Boundary and initial conditions'
    }
    
    # 公式模式
    FORMULA_PATTERNS = [
        # LaTeX格式
        r'\\begin\{equation\}(.*?)\\end\{equation\}',
        r'\\begin\{align\}(.*?)\\end\{align\}',
        r'\\\[(.*?)\\\]',
        r'\$\$(.*?)\$\$',
        
        # 内联公式
        r'\$([^\$]+)\$',
        
        # Unicode和文本格式
        r'([A-Za-zΨψΦφ]\w*\s*=\s*[^=\n]{3,})',
        r'(∂[^/]+/∂[^=\n]+\s*=\s*[^=\n]{3,})',
        r'(d[A-Za-z]/dt\s*=\s*[^=\n]{3,})',
        r'(\|[^|]+\|²\s*=\s*[^=\n]{3,})',
        r'(⟨[^⟩]+⟩\s*=\s*[^=\n]{3,})',
        r'(H\s*[Ψψ]\s*=\s*E\s*[Ψψ])',
        r'(i[ℏh]\s*∂[Ψψ]/∂t\s*=\s*[^=\n]{3,})'
    ]
    
    def __init__(self,
                 model_source: str = "google",
                 model_name: str = "gemini-2.0-flash-exp",
                 output_dir: str = "data/extracted_formulas"):
        """初始化公式提取器"""
        self.llm = LLMInterface(model_source=model_source, model_name=model_name)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.formula_counter = 0
        self.formulas_db = {}  # ID -> Formula mapping
        self.stats = defaultdict(int)
    
    def extract_formulas_from_text(self, text: str) -> List[str]:
        """从文本中提取所有公式"""
        formulas = []
        
        for pattern in self.FORMULA_PATTERNS:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            formulas.extend(matches)
        
        # 清理和去重
        cleaned_formulas = []
        seen = set()
        
        for formula in formulas:
            cleaned = self._clean_formula(formula)
            if cleaned and len(cleaned) > 5 and cleaned not in seen:
                seen.add(cleaned)
                cleaned_formulas.append(cleaned)
        
        return cleaned_formulas
    
    async def extract_formulas_from_text_async(self, text: str, source: str) -> List[Formula]:
        """异步从文本中提取并分析公式"""
        formula_texts = self.extract_formulas_from_text(text)
        formulas = []
        
        # 分析每个公式
        for formula_text in formula_texts[:10]:  # 限制数量
            formula = await self.analyze_formula(formula_text, text, source)
            if formula:
                formulas.append(formula)
        
        return formulas
    
    def _clean_formula(self, formula: str) -> str:
        """清理公式格式"""
        # 移除多余的空白
        formula = ' '.join(formula.split())
        
        # 移除外层的美元符号
        formula = formula.strip('$')
        
        # 标准化一些符号
        replacements = {
            'Psi': 'ψ',
            'PSI': 'ψ',
            'phi': 'φ',
            'PHI': 'φ',
            'hbar': 'ℏ',
            'partial': '∂',
            '\\\\': '\\'
        }
        
        for old, new in replacements.items():
            formula = formula.replace(old, new)
        
        return formula.strip()
    
    @retry_with_exponential_backoff(retries=3)
    async def analyze_formula(self, formula_text: str, context: str, source: str) -> Optional[Formula]:
        """分析单个公式的详细信息"""
        prompt = f"""
        Analyze this physics/mathematics formula in detail.
        
        Formula: {formula_text}
        
        Context: {context[:1000]}
        
        Provide analysis in JSON format:
        {{
            "name": "Common name of the formula (e.g., 'Schrödinger Equation')",
            "category": "One of: definition, physical_law, derivation, boundary_condition",
            "description": "What this formula represents",
            "variables": {{
                "variable_symbol": "meaning",
                ...
            }},
            "assumptions": ["list of assumptions or conditions"],
            "domain": "Physics domain (e.g., quantum_mechanics, statistical_mechanics)",
            "related_concepts": ["list of related physics concepts"],
            "is_derivation": true/false,
            "derivation_from": "parent formula if this is derived",
            "confidence": 0.0-1.0
        }}
        """
        
        try:
            response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
            data = json.loads(self._extract_json(response))
            
            # 生成唯一ID
            self.formula_counter += 1
            formula_id = f"F{self.formula_counter:04d}"
            
            formula = Formula(
                id=formula_id,
                expression=self._convert_to_latex(formula_text),
                name=data.get('name', 'Unnamed Formula'),
                category=data.get('category', 'derivation'),
                description=data.get('description', ''),
                variables=data.get('variables', {}),
                assumptions=data.get('assumptions', []),
                domain=data.get('domain', 'physics'),
                source=source,
                related_concepts=data.get('related_concepts', []),
                derivation_from=data.get('derivation_from'),
                derivation_steps=[],  # 后续填充
                confidence=float(data.get('confidence', 0.7))
            )
            
            self.formulas_db[formula_id] = formula
            self.stats[formula.category] += 1
            
            return formula
            
        except Exception as e:
            logger.warning(f"Failed to analyze formula: {e}")
            return None
    
    def _convert_to_latex(self, formula: str) -> str:
        """将公式转换为标准LaTeX格式"""
        # 如果已经是LaTeX格式，直接返回
        if '\\' in formula or formula.startswith('$'):
            return formula
        
        # Unicode到LaTeX的映射
        unicode_to_latex = {
            'ψ': r'\psi',
            'Ψ': r'\Psi',
            'φ': r'\phi',
            'Φ': r'\Phi',
            'ℏ': r'\hbar',
            '∂': r'\partial',
            '∫': r'\int',
            '∑': r'\sum',
            '∏': r'\prod',
            '√': r'\sqrt',
            '∞': r'\infty',
            '≈': r'\approx',
            '≠': r'\neq',
            '≤': r'\leq',
            '≥': r'\geq',
            '⟨': r'\langle',
            '⟩': r'\rangle',
            '|': r'\vert'
        }
        
        latex_formula = formula
        for unicode_char, latex_cmd in unicode_to_latex.items():
            latex_formula = latex_formula.replace(unicode_char, latex_cmd)
        
        # 处理分数
        latex_formula = re.sub(r'(\w+)/(\w+)', r'\\frac{\1}{\2}', latex_formula)
        
        # 处理上下标
        latex_formula = re.sub(r'(\w+)_(\w+)', r'\1_{\2}', latex_formula)
        latex_formula = re.sub(r'(\w+)\^(\w+)', r'\1^{\2}', latex_formula)
        
        return latex_formula
    
    async def extract_formula_relations(self, formulas: List[Formula]) -> List[FormulaRelation]:
        """提取公式之间的关系"""
        relations = []
        
        # 按领域分组公式
        domain_groups = defaultdict(list)
        for formula in formulas:
            domain_groups[formula.domain].append(formula)
        
        # 在同一领域内分析关系
        for domain, domain_formulas in domain_groups.items():
            if len(domain_formulas) < 2:
                continue
            
            # 分析公式对
            for i in range(len(domain_formulas)):
                for j in range(i+1, min(i+5, len(domain_formulas))):
                    relation = await self._analyze_formula_pair(
                        domain_formulas[i], 
                        domain_formulas[j]
                    )
                    if relation:
                        relations.append(relation)
        
        return relations
    
    @retry_with_exponential_backoff(retries=2)
    async def _analyze_formula_pair(self, f1: Formula, f2: Formula) -> Optional[FormulaRelation]:
        """分析两个公式之间的关系"""
        prompt = f"""
        Analyze the mathematical relationship between these two physics formulas:
        
        Formula 1: {f1.name}
        Expression: {f1.expression}
        Description: {f1.description}
        
        Formula 2: {f2.name}
        Expression: {f2.expression}
        Description: {f2.description}
        
        Determine their relationship:
        - derives_from: One can be derived from the other
        - equivalent_to: Different forms of the same equation
        - special_case_of: One is a special case of the other
        - generalizes: One generalizes the other
        - none: No direct relationship
        
        Response in JSON:
        {{
            "relation_type": "type or none",
            "direction": "1to2 or 2to1",
            "conditions": ["conditions for the relationship"],
            "proof_sketch": "brief explanation"
        }}
        """
        
        try:
            response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
            data = json.loads(self._extract_json(response))
            
            if data.get('relation_type') == 'none':
                return None
            
            # 确定方向
            if data.get('direction') == '2to1':
                formula1_id, formula2_id = f2.id, f1.id
            else:
                formula1_id, formula2_id = f1.id, f2.id
            
            return FormulaRelation(
                formula1_id=formula1_id,
                relation_type=data.get('relation_type', 'derives_from'),
                formula2_id=formula2_id,
                conditions=data.get('conditions', []),
                proof_sketch=data.get('proof_sketch', '')
            )
            
        except Exception as e:
            logger.warning(f"Failed to analyze formula relation: {e}")
            return None
    
    async def extract_derivation_chain(self, formulas: List[Formula]) -> Dict[str, List[str]]:
        """提取公式推导链"""
        derivation_chains = defaultdict(list)
        
        # 构建推导图
        for formula in formulas:
            if formula.derivation_from:
                derivation_chains[formula.derivation_from].append(formula.id)
        
        # 对于有推导关系的公式，尝试提取推导步骤
        for parent_id, derived_ids in derivation_chains.items():
            parent = self.formulas_db.get(parent_id)
            if not parent:
                continue
            
            for derived_id in derived_ids:
                derived = self.formulas_db.get(derived_id)
                if derived:
                    steps = await self._extract_derivation_steps(parent, derived)
                    derived.derivation_steps = steps
        
        return dict(derivation_chains)
    
    async def _extract_derivation_steps(self, parent: Formula, derived: Formula) -> List[str]:
        """提取推导步骤"""
        prompt = f"""
        Provide the key steps to derive Formula 2 from Formula 1:
        
        Formula 1: {parent.expression}
        Formula 2: {derived.expression}
        
        List the main mathematical steps (max 5 steps):
        """
        
        try:
            response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
            steps = [line.strip() for line in response.strip().split('\n') 
                    if line.strip() and not line.startswith('#')]
            return steps[:5]
        except:
            return []
    
    def _extract_json(self, text: str) -> str:
        """从响应中提取JSON"""
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        json_match = re.search(r'(\{.*?\})', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        return text
    
    async def process_document(self, doc_path: str) -> Tuple[List[Formula], List[FormulaRelation]]:
        """处理单个文档提取公式"""
        logger.info(f"Processing document for formulas: {doc_path}")
        
        # 读取预处理的文档
        with open(doc_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_formulas = []
        
        # 判断文档格式
        if isinstance(data, list):
            # 新格式：列表形式
            for i, doc in enumerate(data):
                # 合并title和content
                text = doc.get('title', '') + '\n\n' + doc.get('content', '')
                if len(text) < 100:
                    continue
                
                formulas = await self.extract_formulas_from_text_async(
                    text[:5000],  # 限制长度避免超时
                    f"{os.path.basename(doc_path)}#doc_{i}"
                )
                
                all_formulas.extend(formulas)
                break  # 只处理第一个文档以加快测试
        else:
            # 旧格式：字典形式
            for i, segment in enumerate(data.get('segments', [])):
                text = segment.get('text', '')
                if len(text) < 100:
                    continue
                
                formulas = await self.extract_formulas_from_text_async(
                    text,
                    f"{os.path.basename(doc_path)}#segment_{i}"
                )
                
                all_formulas.extend(formulas)
        
        # 提取公式关系
        relations = await self.extract_formula_relations(all_formulas)
        
        # 提取推导链
        await self.extract_derivation_chain(all_formulas)
        
        return all_formulas, relations
    
    def save_results(self, formulas: List[Formula], relations: List[FormulaRelation],
                     output_prefix: str = "formulas"):
        """保存提取结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存公式
        formulas_file = os.path.join(self.output_dir, f"{output_prefix}_{timestamp}.json")
        formulas_data = {
            'formulas': [asdict(f) for f in formulas],
            'total_count': len(formulas),
            'categories': dict(self.stats),
            'timestamp': timestamp
        }
        
        with open(formulas_file, 'w', encoding='utf-8') as f:
            json.dump(formulas_data, f, ensure_ascii=False, indent=2)
        
        # 保存关系
        if relations:
            relations_file = os.path.join(self.output_dir, f"{output_prefix}_relations_{timestamp}.json")
            with open(relations_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(r) for r in relations], f, ensure_ascii=False, indent=2)
            logger.info(f"Relations saved to: {relations_file}")
        
        # 保存LaTeX格式的公式列表（方便查看）
        latex_file = os.path.join(self.output_dir, f"{output_prefix}_latex_{timestamp}.txt")
        with open(latex_file, 'w', encoding='utf-8') as f:
            for formula in formulas:
                f.write(f"% {formula.name} ({formula.category})\n")
                f.write(f"% {formula.description}\n")
                f.write(f"$${formula.expression}$$\n\n")
        
        logger.info(f"Formulas saved to: {formulas_file}")
        logger.info(f"LaTeX formulas saved to: {latex_file}")
        
        return formulas_file


async def main():
    """主函数 - 测试公式提取器"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Formula Extractor')
    parser.add_argument('--input_dir', type=str, default='data/preprocessed_documents',
                       help='Directory containing preprocessed documents')
    parser.add_argument('--output_dir', type=str, default='data/extracted_formulas',
                       help='Output directory for extracted formulas')
    parser.add_argument('--model_source', type=str, default='google',
                       help='LLM model source')
    parser.add_argument('--model_name', type=str, default='gemini-2.0-flash-exp',
                       help='LLM model name')
    parser.add_argument('--max_docs', type=int, default=3,
                       help='Maximum number of documents to process')
    
    args = parser.parse_args()
    
    # 初始化提取器
    extractor = FormulaExtractor(
        model_source=args.model_source,
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    # 获取文档
    doc_files = []
    if os.path.exists(args.input_dir):
        for file in os.listdir(args.input_dir):
            if file.endswith('.json'):
                doc_files.append(os.path.join(args.input_dir, file))
    
    if not doc_files:
        logger.warning(f"No preprocessed documents found in {args.input_dir}")
        return
    
    # 处理文档
    all_formulas = []
    all_relations = []
    
    for i, doc_path in enumerate(doc_files[:args.max_docs]):
        logger.info(f"Processing document {i+1}/{min(len(doc_files), args.max_docs)}")
        formulas, relations = await extractor.process_document(doc_path)
        all_formulas.extend(formulas)
        all_relations.extend(relations)
    
    # 保存结果
    extractor.save_results(all_formulas, all_relations)
    
    # 打印统计
    print("\n=== Formula Extraction Statistics ===")
    print(f"Total formulas extracted: {len(all_formulas)}")
    print(f"Total relations found: {len(all_relations)}")
    print("\nFormula categories:")
    for cat, count in extractor.stats.items():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    asyncio.run(main())