#!/usr/bin/env python3
"""
prepare_enhanced_concepts.py - 自动化概念预处理脚本
=====================================================

从原始文献中提取概念、公式并构建知识图谱，
为unified理论生成方法准备增强的概念空间。

使用方法：
    python prepare_enhanced_concepts.py [选项]

Author: UniversalTheoryGen Team
Date: 2025-01-24
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
import shutil

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logging_config import get_logger
from utils.checkpoint_manager import CheckpointManager

# 初始化日志
logger = get_logger('prepare_enhanced_concepts')


class ConceptPreprocessor:
    """概念预处理器 - 协调概念提取、公式提取和知识图谱构建"""
    
    def __init__(self, config):
        self.config = config
        self.checkpoint_manager = CheckpointManager(
            run_id=f"concept_prep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
    async def run(self):
        """运行完整的预处理流程"""
        try:
            logger.info("=== 开始概念预处理流程 ===")
            
            # 步骤1：提取概念
            if not self.config.get('skip_concept_extraction', False):
                await self._extract_concepts()
            else:
                logger.info("跳过概念提取（使用现有数据）")
            
            # 步骤2：提取公式
            if not self.config.get('skip_formula_extraction', False):
                await self._extract_formulas()
            else:
                logger.info("跳过公式提取（使用现有数据）")
            
            # 步骤3：构建知识图谱
            if not self.config.get('skip_graph_building', False):
                self._build_knowledge_graph()
            else:
                logger.info("跳过知识图谱构建（使用现有数据）")
            
            # 步骤4：生成统计报告
            self._generate_report()
            
            logger.info("=== 概念预处理完成 ===")
            return True
            
        except Exception as e:
            logger.error(f"预处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _extract_concepts(self):
        """从文献中提取概念"""
        logger.info("\n[步骤1/4] 提取概念...")
        
        try:
            from enhanced_concept_extractor import EnhancedConceptExtractor
            
            extractor = EnhancedConceptExtractor(
                model_source=self.config['model_source'],
                model_name=self.config['model_name']
            )
            
            # 获取所有预处理文档
            preprocessed_dir = Path(self.config['preprocessed_dir'])
            if not preprocessed_dir.exists():
                logger.warning(f"预处理目录不存在: {preprocessed_dir}")
                return
            
            doc_files = list(preprocessed_dir.glob('*.json'))
            logger.info(f"找到 {len(doc_files)} 个预处理文档")
            
            # 限制处理数量（可配置）
            max_docs = self.config.get('max_documents', None)
            if max_docs:
                doc_files = doc_files[:max_docs]
                logger.info(f"限制处理前 {max_docs} 个文档")
            
            all_concepts = []
            all_relations = []
            
            # 批量处理文档
            batch_size = self.config.get('batch_size', 5)
            for i in range(0, len(doc_files), batch_size):
                batch = doc_files[i:i+batch_size]
                logger.info(f"处理批次 {i//batch_size + 1}/{(len(doc_files) + batch_size - 1)//batch_size}")
                
                for doc_file in batch:
                    try:
                        logger.info(f"处理文档: {doc_file.name}")
                        concepts, relations = await extractor.process_document(str(doc_file))
                        all_concepts.extend(concepts)
                        all_relations.extend(relations)
                    except Exception as e:
                        logger.error(f"处理文档 {doc_file.name} 失败: {e}")
                        continue
            
            # 保存结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(self.config['output_dir']) / 'enhanced_concepts'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存概念
            concepts_file = output_dir / f'enhanced_concepts_{timestamp}.json'
            with open(concepts_file, 'w', encoding='utf-8') as f:
                json.dump([c.__dict__ if hasattr(c, '__dict__') else c for c in all_concepts], 
                         f, ensure_ascii=False, indent=2)
            
            # 保存关系
            relations_file = output_dir / f'enhanced_relations_{timestamp}.json'
            with open(relations_file, 'w', encoding='utf-8') as f:
                json.dump([r.__dict__ if hasattr(r, '__dict__') else r for r in all_relations], 
                         f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 提取了 {len(all_concepts)} 个概念和 {len(all_relations)} 个关系")
            logger.info(f"概念保存到: {concepts_file}")
            logger.info(f"关系保存到: {relations_file}")
            
            # 更新checkpoint
            self.checkpoint_manager.save_checkpoint({
                'concepts_extracted': True,
                'concepts_file': str(concepts_file),
                'relations_file': str(relations_file),
                'concepts_count': len(all_concepts),
                'relations_count': len(all_relations)
            })
            
        except ImportError as e:
            logger.error(f"无法导入概念提取器: {e}")
            raise
    
    async def _extract_formulas(self):
        """从文献中提取公式"""
        logger.info("\n[步骤2/4] 提取公式...")
        
        try:
            from formula_extractor import FormulaExtractor
            
            extractor = FormulaExtractor(
                model_source=self.config['model_source'],
                model_name=self.config['model_name']
            )
            
            # 获取预处理文档
            preprocessed_dir = Path(self.config['preprocessed_dir'])
            doc_files = list(preprocessed_dir.glob('*.json'))
            
            # 限制处理数量
            max_docs = self.config.get('max_documents', None)
            if max_docs:
                doc_files = doc_files[:max_docs]
            
            all_formulas = []
            all_relations = []
            
            # 处理文档
            for doc_file in doc_files:
                try:
                    logger.info(f"处理文档公式: {doc_file.name}")
                    formulas, relations = await extractor.process_document(str(doc_file))
                    all_formulas.extend(formulas)
                    all_relations.extend(relations)
                except Exception as e:
                    logger.error(f"处理文档 {doc_file.name} 的公式失败: {e}")
                    continue
            
            # 保存结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(self.config['output_dir']) / 'extracted_formulas'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            formulas_file = output_dir / f'enhanced_formulas_{timestamp}.json'
            with open(formulas_file, 'w', encoding='utf-8') as f:
                json.dump([f.__dict__ if hasattr(f, '__dict__') else f for f in all_formulas], 
                         f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 提取了 {len(all_formulas)} 个公式")
            logger.info(f"公式保存到: {formulas_file}")
            
            # 更新checkpoint
            checkpoint_data = self.checkpoint_manager.load_checkpoint() or {}
            checkpoint_data.update({
                'formulas_extracted': True,
                'formulas_file': str(formulas_file),
                'formulas_count': len(all_formulas)
            })
            self.checkpoint_manager.save_checkpoint(checkpoint_data)
            
        except ImportError as e:
            logger.error(f"无法导入公式提取器: {e}")
            raise
    
    def _build_knowledge_graph(self):
        """构建知识图谱"""
        logger.info("\n[步骤3/4] 构建知识图谱...")
        
        try:
            from knowledge_graph_builder import KnowledgeGraphBuilder
            
            builder = KnowledgeGraphBuilder()
            
            # 加载最新的概念和公式
            checkpoint_data = self.checkpoint_manager.load_checkpoint() or {}
            
            # 加载概念
            if 'concepts_file' in checkpoint_data:
                builder.load_concepts(checkpoint_data['concepts_file'])
                logger.info(f"加载概念文件: {checkpoint_data['concepts_file']}")
            
            # 加载关系
            if 'relations_file' in checkpoint_data:
                builder.load_concept_relations(checkpoint_data['relations_file'])
                logger.info(f"加载关系文件: {checkpoint_data['relations_file']}")
            
            # 加载公式
            if 'formulas_file' in checkpoint_data:
                builder.load_formulas(checkpoint_data['formulas_file'])
                logger.info(f"加载公式文件: {checkpoint_data['formulas_file']}")
            
            # 加载理论
            theories_dir = Path(self.config.get('theories_dir', 'data/theories_v2.1'))
            if theories_dir.exists():
                builder.load_theories(str(theories_dir))
                logger.info(f"加载理论目录: {theories_dir}")
            
            # 推理隐式关系
            builder.infer_implicit_relations()
            
            # 保存知识图谱
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            kg_file = builder.save_graph(f"enhanced_kg_{timestamp}")
            
            # 生成统计
            stats = builder.kg.get_statistics()
            logger.info("\n知识图谱统计:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
            
            # 更新checkpoint
            checkpoint_data.update({
                'knowledge_graph_built': True,
                'knowledge_graph_file': kg_file,
                'kg_stats': stats
            })
            self.checkpoint_manager.save_checkpoint(checkpoint_data)
            
            logger.info(f"✅ 知识图谱保存到: {kg_file}")
            
        except ImportError as e:
            logger.error(f"无法导入知识图谱构建器: {e}")
            raise
    
    def _generate_report(self):
        """生成预处理报告"""
        logger.info("\n[步骤4/4] 生成报告...")
        
        checkpoint_data = self.checkpoint_manager.load_checkpoint() or {}
        
        # 创建报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'results': {
                'concepts': {
                    'extracted': checkpoint_data.get('concepts_extracted', False),
                    'count': checkpoint_data.get('concepts_count', 0),
                    'file': checkpoint_data.get('concepts_file', '')
                },
                'relations': {
                    'count': checkpoint_data.get('relations_count', 0),
                    'file': checkpoint_data.get('relations_file', '')
                },
                'formulas': {
                    'extracted': checkpoint_data.get('formulas_extracted', False),
                    'count': checkpoint_data.get('formulas_count', 0),
                    'file': checkpoint_data.get('formulas_file', '')
                },
                'knowledge_graph': {
                    'built': checkpoint_data.get('knowledge_graph_built', False),
                    'file': checkpoint_data.get('knowledge_graph_file', ''),
                    'statistics': checkpoint_data.get('kg_stats', {})
                }
            }
        }
        
        # 保存报告
        report_file = Path(self.config['output_dir']) / f'preprocessing_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 报告保存到: {report_file}")
        
        # 打印摘要
        logger.info("\n=== 预处理摘要 ===")
        logger.info(f"概念: {report['results']['concepts']['count']} 个")
        logger.info(f"关系: {report['results']['relations']['count']} 个")
        logger.info(f"公式: {report['results']['formulas']['count']} 个")
        if 'kg_stats' in checkpoint_data:
            logger.info(f"知识图谱节点: {checkpoint_data['kg_stats'].get('num_nodes', 0)} 个")
            logger.info(f"知识图谱边: {checkpoint_data['kg_stats'].get('num_edges', 0)} 条")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="从原始文献中提取概念、公式并构建知识图谱",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 输入输出参数
    parser.add_argument('--preprocessed_dir', default='data/preprocessed',
                       help='预处理文档目录')
    parser.add_argument('--theories_dir', default='data/theories_v2.1',
                       help='理论目录（用于知识图谱）')
    parser.add_argument('--output_dir', default='data',
                       help='输出根目录')
    
    # 模型参数
    parser.add_argument('--model_source', default='google',
                       choices=['openai', 'google', 'deepseek'],
                       help='LLM模型源')
    parser.add_argument('--model_name', default='gemini-2.0-flash-exp',
                       help='LLM模型名称')
    
    # 处理控制参数
    parser.add_argument('--max_documents', type=int, default=None,
                       help='最大处理文档数（默认处理所有）')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='批处理大小')
    
    # 跳过选项
    parser.add_argument('--skip_concept_extraction', action='store_true',
                       help='跳过概念提取')
    parser.add_argument('--skip_formula_extraction', action='store_true',
                       help='跳过公式提取')
    parser.add_argument('--skip_graph_building', action='store_true',
                       help='跳过知识图谱构建')
    
    # 快速测试模式
    parser.add_argument('--test_mode', action='store_true',
                       help='测试模式（只处理少量文档）')
    
    args = parser.parse_args()
    
    # 测试模式配置
    if args.test_mode:
        args.max_documents = 2
        logger.info("🧪 测试模式：只处理2个文档")
    
    # 创建配置
    config = vars(args)
    
    # 运行预处理
    processor = ConceptPreprocessor(config)
    success = asyncio.run(processor.run())
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()