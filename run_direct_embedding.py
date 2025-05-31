#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直接嵌入程序

将生成的理论数据和抽取的概念公式直接嵌入到同一个高维向量空间，无需中间合并步骤。
"""

import os
import json
import pickle
import argparse
import asyncio
from theory_generation.llm_interface import LLMInterface

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] 创建目录: {directory}")

async def load_theory_fields(theory_file):
    """加载理论数据并分析字段"""
    if not os.path.exists(theory_file):
        print(f"[ERROR] 理论文件不存在: {theory_file}")
        return []
        
    try:
        with open(theory_file, 'r', encoding='utf-8') as f:
            theories = json.load(f)
        
        # 分析字段
        if theories and len(theories) > 0:
            first_theory = theories[0]
            meta_fields = ['id', 'metadata']
            content_fields = [field for field in first_theory.keys() if field not in meta_fields]
            print(f"[INFO] 发现以下内容字段用于嵌入: {', '.join(content_fields)}")
        
        return theories
    except Exception as e:
        print(f"[ERROR] 加载理论文件失败: {str(e)}")
        return []

async def load_extracted_data(extracted_file):
    """加载从文献中抽取的概念和公式"""
    if not os.path.exists(extracted_file):
        print(f"[ERROR] 抽取数据文件不存在: {extracted_file}")
        return [], []
        
    try:
        with open(extracted_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        concepts = data.get("concepts", [])
        formulas = data.get("formulas", [])
        print(f"[INFO] 加载了 {len(concepts)} 个抽取概念，{len(formulas)} 个抽取公式")
        return concepts, formulas
    except Exception as e:
        print(f"[ERROR] 加载抽取数据失败: {str(e)}")
        return [], []

async def build_theory_description(theory):
    """构建理论的完整描述文本，除了元数据外包含所有字段"""
    description_parts = []
    
    # 理论名称始终包含
    description_parts.append(f"Theory Name: {theory.get('name', 'Unknown')}")
    
    # 元数据字段，不包含在描述中
    meta_fields = ['id', 'metadata', 'name']
    
    # 处理所有其他字段
    for field, value in theory.items():
        if field not in meta_fields and value:
            # 将字段名转换为可读的标题格式
            field_title = field.replace('_', ' ').title()
            
            # 如果是字符串，直接添加
            if isinstance(value, str):
                description_parts.append(f"{field_title}:\n{value}")
            # 如果是列表，尝试处理
            elif isinstance(value, list):
                # 尝试处理概念列表
                if field == "core_concepts" and value:
                    concepts_text = []
                    for item in value:
                        if isinstance(item, dict):
                            name = item.get('name', '')
                            desc = item.get('description', '')
                            if name and desc:
                                concepts_text.append(f"- {name}: {desc}")
                            elif name:
                                concepts_text.append(f"- {name}")
                        elif isinstance(item, str):
                            concepts_text.append(f"- {item}")
                    
                    if concepts_text:
                        description_parts.append(f"{field_title}:\n" + "\n".join(concepts_text))
                else:
                    # 其他类型的列表，简单地连接
                    list_text = "\n".join([f"- {item}" for item in value if item])
                    if list_text:
                        description_parts.append(f"{field_title}:\n{list_text}")
            # 其他类型，转为字符串
            elif value:
                description_parts.append(f"{field_title}:\n{str(value)}")
    
    # 将所有部分连接成一个字符串，用双换行分隔
    return "\n\n".join(description_parts)

class DirectEmbedder:
    """直接嵌入器，处理理论、概念和公式"""
    
    def __init__(self, llm):
        self.llm = llm
        self.embeddings = {
            "theories": {},
            "concepts": {},
            "formulas": {},
            "metadata": {
                "model_source": llm.model_source,
                "model_name": llm.model_name,
                "embedding_date": str(asyncio.get_event_loop().time())
            }
        }
    
    async def embed_theories(self, theories):
        """嵌入所有理论"""
        print(f"[步骤1] 嵌入 {len(theories)} 个理论")
        for i, theory in enumerate(theories, 1):
            try:
                theory_name = theory.get('name')
                if not theory_name:
                    continue
                    
                description = await build_theory_description(theory)
                embedding = await self.llm.get_embedding(description)
                self.embeddings["theories"][theory_name] = embedding
                print(f"[INFO] ({i}/{len(theories)}) 已嵌入理论: {theory_name}")
            except Exception as e:
                print(f"[ERROR] 嵌入理论 {theory.get('name', 'Unknown')} 失败: {str(e)}")
        
        return len(self.embeddings["theories"])
    
    async def embed_concepts(self, concepts):
        """嵌入所有概念"""
        print(f"\n[步骤2] 嵌入 {len(concepts)} 个概念")
        for i, concept in enumerate(concepts, 1):
            try:
                concept_name = concept.get('name')
                if not concept_name:
                    continue
                
                # 构建概念描述    
                description = f"Concept: {concept_name}\n\n"
                description += f"Description: {concept.get('description', '')}\n\n"
                
                # 添加领域信息
                if concept.get('domain'):
                    description += f"Domain: {concept.get('domain')}\n\n"
                
                # 添加相关理论
                if concept.get('theories') and isinstance(concept.get('theories'), list):
                    related_theories = ", ".join(concept.get('theories'))
                    description += f"Related Theories: {related_theories}"
                
                embedding = await self.llm.get_embedding(description)
                self.embeddings["concepts"][concept_name] = embedding
                
                if i % 10 == 0:
                    print(f"[INFO] 已嵌入 {i}/{len(concepts)} 个概念")
            except Exception as e:
                print(f"[ERROR] 嵌入概念 {concept.get('name', 'Unknown')} 失败: {str(e)}")
        
        return len(self.embeddings["concepts"])
    
    async def embed_formulas(self, formulas):
        """嵌入所有公式"""
        print(f"\n[步骤3] 嵌入 {len(formulas)} 个公式")
        for i, formula in enumerate(formulas, 1):
            try:
                formula_name = formula.get('name')
                if not formula_name:
                    continue
                    
                # 构建公式描述
                formula_text = f"Formula: {formula_name}\n\n"
                formula_text += f"Expression: {formula.get('expression', '')}\n\n"
                formula_text += f"Description: {formula.get('description', '')}\n\n"
                
                # 添加变量信息
                if formula.get('variables') and isinstance(formula.get('variables'), list):
                    variables_text = "\n".join([f"- {var.get('symbol')}: {var.get('description')}" 
                                              for var in formula.get('variables') if isinstance(var, dict)])
                    if variables_text:
                        formula_text += f"Variables:\n{variables_text}"
                
                embedding = await self.llm.get_embedding(formula_text)
                self.embeddings["formulas"][formula_name] = embedding
                
                if i % 10 == 0:
                    print(f"[INFO] 已嵌入 {i}/{len(formulas)} 个公式")
            except Exception as e:
                print(f"[ERROR] 嵌入公式 {formula.get('name', 'Unknown')} 失败: {str(e)}")
        
        return len(self.embeddings["formulas"])
    
    def save_embeddings(self, output_dir):
        """保存所有嵌入向量"""
        ensure_directory_exists(output_dir)
        
        # 保存统一的嵌入文件
        unified_path = os.path.join(output_dir, "unified_embeddings.pkl")
        with open(unified_path, 'wb') as f:
            pickle.dump(self.embeddings, f)
        print(f"[INFO] 已保存统一嵌入文件: {unified_path}")
        
        # 保存各个类型的嵌入文件(向后兼容)
        theory_path = os.path.join(output_dir, "theory_embeddings.pkl")
        with open(theory_path, 'wb') as f:
            pickle.dump(self.embeddings["theories"], f)
        
        concept_path = os.path.join(output_dir, "concept_embeddings.pkl")
        with open(concept_path, 'wb') as f:
            pickle.dump(self.embeddings["concepts"], f)
        
        formula_path = os.path.join(output_dir, "formula_embeddings.pkl")
        with open(formula_path, 'wb') as f:
            pickle.dump(self.embeddings["formulas"], f)
        
        # 保存统计信息
        stats = {
            "theories_count": len(self.embeddings["theories"]),
            "concepts_count": len(self.embeddings["concepts"]),
            "formulas_count": len(self.embeddings["formulas"]),
            "model_info": {
                "model_source": self.embeddings["metadata"]["model_source"],
                "model_name": self.embeddings["metadata"]["model_name"]
            }
        }
        
        stats_path = os.path.join(output_dir, "embedding_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        return stats

async def main():
    parser = argparse.ArgumentParser(description="直接嵌入程序")
    
    # LLM接口参数
    parser.add_argument("--model_source", type=str, default="openai",
                        choices=["openai", "deepseek"],
                        help="嵌入模型来源")
    parser.add_argument("--model_name", type=str, default="text-embedding-3-small",
                        help="嵌入模型名称")
    
    # 输入参数
    parser.add_argument("--theories_file", type=str, 
                        default="data/generated_theories/all_quantum_interpretations.json",
                        help="理论数据文件")
    parser.add_argument("--extracted_file", type=str, 
                        default="data/extracted_concepts/extracted_data.json",
                        help="抽取的概念和公式数据")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="data/embeddings",
                        help="嵌入向量的输出目录")
    parser.add_argument("--request_interval", type=float, default=1.0,
                        help="API请求之间的最小间隔(秒)")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    ensure_directory_exists(args.output_dir)
    
    # 初始化LLM接口
    llm = LLMInterface(
        model_source=args.model_source,
        model_name=args.model_name,
        request_interval=args.request_interval
    )
    
    # 显示当前使用的模型信息
    model_info = llm.get_current_model_info()
    print(f"[INFO] 当前使用的模型: {model_info['source']} - {model_info['name']}")
    
    # 加载理论数据
    print(f"\n[步骤1] 加载理论数据: {args.theories_file}")
    theories = await load_theory_fields(args.theories_file)
    
    # 加载抽取的概念和公式
    print(f"\n[步骤2] 加载抽取的概念和公式: {args.extracted_file}")
    concepts, formulas = await load_extracted_data(args.extracted_file)
    
    # 创建直接嵌入器
    embedder = DirectEmbedder(llm)
    
    # 嵌入所有数据
    print("\n[步骤3] 开始嵌入数据")
    
    # 嵌入理论
    theories_count = await embedder.embed_theories(theories)
    
    # 嵌入概念
    concepts_count = await embedder.embed_concepts(concepts)
    
    # 嵌入公式
    formulas_count = await embedder.embed_formulas(formulas)
    
    # 保存嵌入向量
    print("\n[步骤4] 保存嵌入向量")
    stats = embedder.save_embeddings(args.output_dir)
    
    print("\n[完成] 直接嵌入流程已完成!")
    print(f"嵌入统计:")
    print(f"- 理论数量: {stats['theories_count']}")
    print(f"- 概念数量: {stats['concepts_count']}")
    print(f"- 公式数量: {stats['formulas_count']}")
    print(f"- 使用模型: {stats['model_info']['model_source']}/{stats['model_info']['model_name']}")

if __name__ == "__main__":
    asyncio.run(main())
