#!/usr/bin/env python3
"""
Knowledge Graph Builder - 知识图谱构建器

构建概念、公式、理论之间的知识图谱，支持关系推理和查询。
整合概念提取、公式提取的结果，形成结构化的知识网络。

Author: UniversalTheoryGen Team
Date: 2025-01-24
"""

import json
import os
import sys
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import networkx as nx
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logging_config import get_logger

# 初始化日志
logger = get_logger('knowledge_graph_builder')

@dataclass
class Node:
    """图节点"""
    id: str
    type: str  # concept, formula, theory
    name: str
    attributes: Dict[str, Any]

@dataclass
class Edge:
    """图边"""
    source: str
    target: str
    relation: str
    weight: float = 1.0
    attributes: Dict[str, Any] = None

class KnowledgeGraph:
    """知识图谱"""
    
    def __init__(self):
        """初始化知识图谱"""
        self.graph = nx.MultiDiGraph()  # 有向多重图
        self.nodes = {}  # id -> Node
        self.edges = []  # Edge列表
        
        # 节点类型统计
        self.node_types = defaultdict(int)
        
        # 关系类型统计
        self.relation_types = defaultdict(int)
        
        # 索引
        self.name_to_id = {}  # 名称到ID的映射
        self.type_index = defaultdict(list)  # 类型索引
    
    def add_node(self, node: Node):
        """添加节点"""
        self.nodes[node.id] = node
        self.node_types[node.type] += 1
        self.type_index[node.type].append(node.id)
        
        # 更新名称索引
        self.name_to_id[node.name.lower()] = node.id
        
        # 添加到NetworkX图
        self.graph.add_node(node.id, 
                           type=node.type,
                           name=node.name,
                           **node.attributes)
    
    def add_edge(self, edge: Edge):
        """添加边"""
        self.edges.append(edge)
        self.relation_types[edge.relation] += 1
        
        # 添加到NetworkX图
        self.graph.add_edge(edge.source, edge.target,
                           relation=edge.relation,
                           weight=edge.weight,
                           **(edge.attributes or {}))
    
    def find_node_by_name(self, name: str) -> Optional[Node]:
        """通过名称查找节点"""
        node_id = self.name_to_id.get(name.lower())
        return self.nodes.get(node_id) if node_id else None
    
    def get_neighbors(self, node_id: str, relation: Optional[str] = None) -> List[str]:
        """获取邻居节点"""
        if relation:
            return [target for _, target, data in self.graph.edges(node_id, data=True)
                   if data.get('relation') == relation]
        else:
            return list(self.graph.neighbors(node_id))
    
    def find_path(self, source: str, target: str, max_length: int = 5) -> Optional[List[str]]:
        """查找两个节点之间的路径"""
        try:
            return nx.shortest_path(self.graph, source, target, weight='weight')[:max_length]
        except nx.NetworkXNoPath:
            return None
    
    def get_subgraph(self, node_ids: List[str], depth: int = 1) -> nx.DiGraph:
        """获取子图"""
        # 收集所有相关节点
        all_nodes = set(node_ids)
        for _ in range(depth):
            new_nodes = set()
            for node in all_nodes:
                new_nodes.update(self.graph.neighbors(node))
                new_nodes.update(self.graph.predecessors(node))
            all_nodes.update(new_nodes)
        
        return self.graph.subgraph(all_nodes)
    
    def compute_centrality(self) -> Dict[str, float]:
        """计算节点中心性"""
        return nx.pagerank(self.graph, weight='weight')
    
    def find_communities(self) -> List[Set[str]]:
        """发现社区结构"""
        # 转为无向图进行社区检测
        undirected = self.graph.to_undirected()
        communities = nx.community.louvain_communities(undirected)
        return [set(community) for community in communities]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取图统计信息"""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'node_types': dict(self.node_types),
            'relation_types': dict(self.relation_types),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
            'num_components': nx.number_weakly_connected_components(self.graph)
        }

class KnowledgeGraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self, output_dir: str = "data/knowledge_graph"):
        """初始化构建器"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.kg = KnowledgeGraph()
        
        # ID计数器
        self.id_counters = {
            'concept': 0,
            'formula': 0,
            'theory': 0
        }
    
    def load_concepts(self, concepts_file: str):
        """加载概念"""
        logger.info(f"Loading concepts from {concepts_file}")
        
        with open(concepts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理概念节点
        for concept_data in data:
            # 创建概念节点
            node = Node(
                id=f"C{self.id_counters['concept']:04d}",
                type='concept',
                name=concept_data['name'],
                attributes={
                    'description': concept_data.get('description', ''),
                    'category': concept_data.get('category', 'fundamental'),
                    'domain': concept_data.get('domain', ''),
                    'confidence': concept_data.get('confidence', 0.8)
                }
            )
            self.kg.add_node(node)
            self.id_counters['concept'] += 1
            
            # 处理前置概念关系
            for prereq in concept_data.get('prerequisites', []):
                prereq_node = self.kg.find_node_by_name(prereq)
                if prereq_node:
                    self.kg.add_edge(Edge(
                        source=node.id,
                        target=prereq_node.id,
                        relation='depends_on',
                        weight=0.8
                    ))
    
    def load_concept_relations(self, relations_file: str):
        """加载概念关系"""
        logger.info(f"Loading concept relations from {relations_file}")
        
        with open(relations_file, 'r', encoding='utf-8') as f:
            relations = json.load(f)
        
        for rel_data in relations:
            # 查找概念节点
            concept1 = self.kg.find_node_by_name(rel_data['concept1'])
            concept2 = self.kg.find_node_by_name(rel_data['concept2'])
            
            if concept1 and concept2:
                self.kg.add_edge(Edge(
                    source=concept1.id,
                    target=concept2.id,
                    relation=rel_data['relation_type'],
                    weight=rel_data.get('confidence', 0.7),
                    attributes={'evidence': rel_data.get('evidence', '')}
                ))
    
    def load_formulas(self, formulas_file: str):
        """加载公式"""
        logger.info(f"Loading formulas from {formulas_file}")
        
        with open(formulas_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        formulas = data.get('formulas', [])
        
        # 创建公式节点
        formula_nodes = {}
        for formula_data in formulas:
            node = Node(
                id=formula_data.get('id', f"F{self.id_counters['formula']:04d}"),
                type='formula',
                name=formula_data['name'],
                attributes={
                    'expression': formula_data['expression'],
                    'category': formula_data.get('category', 'derivation'),
                    'domain': formula_data.get('domain', ''),
                    'variables': formula_data.get('variables', {}),
                    'assumptions': formula_data.get('assumptions', [])
                }
            )
            self.kg.add_node(node)
            formula_nodes[formula_data.get('id', node.id)] = node
            self.id_counters['formula'] += 1
            
            # 链接到相关概念
            for concept_name in formula_data.get('related_concepts', []):
                concept_node = self.kg.find_node_by_name(concept_name)
                if concept_node:
                    self.kg.add_edge(Edge(
                        source=node.id,
                        target=concept_node.id,
                        relation='uses_concept',
                        weight=0.9
                    ))
    
    def load_theories(self, theories_dir: str):
        """加载理论"""
        logger.info(f"Loading theories from {theories_dir}")
        
        theory_files = []
        if os.path.exists(theories_dir):
            for file in os.listdir(theories_dir):
                if file.endswith('.json'):
                    theory_files.append(os.path.join(theories_dir, file))
        
        for theory_file in theory_files:
            with open(theory_file, 'r', encoding='utf-8') as f:
                theory_data = json.load(f)
            
            # 创建理论节点
            node = Node(
                id=f"T{self.id_counters['theory']:04d}",
                type='theory',
                name=theory_data['name'],
                attributes={
                    'description': theory_data.get('description', ''),
                    'core_assumptions': theory_data.get('core_assumptions', []),
                    'mathematical_formalism': theory_data.get('mathematical_formalism', ''),
                    'source_file': os.path.basename(theory_file)
                }
            )
            self.kg.add_node(node)
            self.id_counters['theory'] += 1
            
            # 提取理论中的概念并建立关系
            self._extract_theory_concepts(node, theory_data)
    
    def _extract_theory_concepts(self, theory_node: Node, theory_data: Dict):
        """从理论中提取概念并建立关系"""
        # 从各个字段中提取概念
        text_fields = [
            theory_data.get('description', ''),
            ' '.join(theory_data.get('core_assumptions', [])),
            theory_data.get('mathematical_formalism', ''),
            theory_data.get('content', '')
        ]
        
        full_text = ' '.join(text_fields).lower()
        
        # 查找已知概念
        for concept_name, concept_id in self.kg.name_to_id.items():
            if concept_name in full_text:
                self.kg.add_edge(Edge(
                    source=theory_node.id,
                    target=concept_id,
                    relation='contains_concept',
                    weight=0.8
                ))
    
    def infer_implicit_relations(self):
        """推理隐含关系"""
        logger.info("Inferring implicit relations")
        
        # 传递性关系推理
        self._infer_transitive_relations()
        
        # 基于共同邻居推理
        self._infer_similarity_relations()
        
        # 层次关系推理
        self._infer_hierarchical_relations()
    
    def _infer_transitive_relations(self):
        """推理传递性关系"""
        # depends_on 的传递性
        depends_edges = [(s, t) for s, t, d in self.kg.graph.edges(data=True) 
                        if d.get('relation') == 'depends_on']
        
        for a, b in depends_edges:
            for c in self.kg.get_neighbors(b, 'depends_on'):
                # 如果 A depends_on B 且 B depends_on C，则 A depends_on C
                if not self.kg.graph.has_edge(a, c):
                    self.kg.add_edge(Edge(
                        source=a,
                        target=c,
                        relation='depends_on',
                        weight=0.6,  # 降低权重表示推理得出
                        attributes={'inferred': True}
                    ))
    
    def _infer_similarity_relations(self):
        """基于共同邻居推理相似性"""
        # 获取所有概念节点
        concept_nodes = self.kg.type_index['concept']
        
        for i, node1 in enumerate(concept_nodes):
            for node2 in concept_nodes[i+1:]:
                # 计算Jaccard相似度
                neighbors1 = set(self.kg.graph.neighbors(node1))
                neighbors2 = set(self.kg.graph.neighbors(node2))
                
                if neighbors1 and neighbors2:
                    jaccard = len(neighbors1 & neighbors2) / len(neighbors1 | neighbors2)
                    
                    if jaccard > 0.3:  # 阈值
                        self.kg.add_edge(Edge(
                            source=node1,
                            target=node2,
                            relation='similar_to',
                            weight=jaccard,
                            attributes={'similarity_type': 'structural'}
                        ))
    
    def _infer_hierarchical_relations(self):
        """推理层次关系"""
        # 基于入度和出度推理概念的层次
        for node_id in self.kg.type_index['concept']:
            in_degree = self.kg.graph.in_degree(node_id)
            out_degree = self.kg.graph.out_degree(node_id)
            
            # 高入度低出度的可能是基础概念
            if in_degree > 5 and out_degree < 2:
                node = self.kg.nodes[node_id]
                node.attributes['level'] = 'fundamental'
            
            # 低入度高出度的可能是高级概念
            elif in_degree < 2 and out_degree > 5:
                node = self.kg.nodes[node_id]
                node.attributes['level'] = 'advanced'
    
    def compute_concept_importance(self) -> Dict[str, float]:
        """计算概念重要性"""
        # 使用PageRank算法
        centrality = self.kg.compute_centrality()
        
        # 结合其他因素
        importance = {}
        for node_id, node in self.kg.nodes.items():
            if node.type == 'concept':
                base_score = centrality.get(node_id, 0)
                
                # 考虑节点属性
                confidence = node.attributes.get('confidence', 0.5)
                category_weight = {
                    'fundamental': 1.2,
                    'theoretical_framework': 1.1,
                    'derived': 0.9,
                    'experimental': 0.8
                }.get(node.attributes.get('category', 'derived'), 1.0)
                
                # 考虑连接数
                degree = self.kg.graph.degree(node_id)
                degree_factor = 1 + np.log1p(degree) * 0.1
                
                importance[node_id] = base_score * confidence * category_weight * degree_factor
        
        # 归一化
        max_importance = max(importance.values()) if importance else 1
        return {k: v/max_importance for k, v in importance.items()}
    
    def export_for_visualization(self, output_file: str):
        """导出用于可视化的数据"""
        # 准备节点数据
        nodes_data = []
        importance_scores = self.compute_concept_importance()
        
        for node_id, node in self.kg.nodes.items():
            node_data = {
                'id': node_id,
                'label': node.name,
                'type': node.type,
                'size': importance_scores.get(node_id, 0.5) * 50 + 10,
                'color': {
                    'concept': '#4CAF50',
                    'formula': '#2196F3',
                    'theory': '#FF9800'
                }.get(node.type, '#757575'),
                'attributes': node.attributes
            }
            nodes_data.append(node_data)
        
        # 准备边数据
        edges_data = []
        for edge in self.kg.edges:
            edge_data = {
                'source': edge.source,
                'target': edge.target,
                'label': edge.relation,
                'weight': edge.weight,
                'color': {
                    'depends_on': '#FF5252',
                    'contains_concept': '#448AFF',
                    'uses_concept': '#4CAF50',
                    'similar_to': '#FFC107',
                    'derives_from': '#9C27B0'
                }.get(edge.relation, '#BDBDBD')
            }
            edges_data.append(edge_data)
        
        # 导出
        viz_data = {
            'nodes': nodes_data,
            'edges': edges_data,
            'statistics': self.kg.get_statistics()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(viz_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Visualization data exported to {output_file}")
    
    def save_graph(self, prefix: str = "knowledge_graph"):
        """保存知识图谱"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存图结构
        graph_file = os.path.join(self.output_dir, f"{prefix}_{timestamp}.json")
        graph_data = {
            'nodes': {nid: asdict(node) for nid, node in self.kg.nodes.items()},
            'edges': [asdict(edge) for edge in self.kg.edges],
            'statistics': self.kg.get_statistics(),
            'timestamp': timestamp
        }
        
        with open(graph_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        # 保存可视化数据
        viz_file = os.path.join(self.output_dir, f"{prefix}_viz_{timestamp}.json")
        self.export_for_visualization(viz_file)
        
        # 保存NetworkX格式（用于高级分析）
        nx_file = os.path.join(self.output_dir, f"{prefix}_{timestamp}.gml")
        nx.write_gml(self.kg.graph, nx_file)
        
        logger.info(f"Knowledge graph saved to {self.output_dir}")
        return graph_file


def main():
    """主函数 - 构建知识图谱"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Knowledge Graph Builder')
    parser.add_argument('--concepts_file', type=str,
                       help='Enhanced concepts JSON file')
    parser.add_argument('--relations_file', type=str,
                       help='Concept relations JSON file')
    parser.add_argument('--formulas_file', type=str,
                       help='Formulas JSON file')
    parser.add_argument('--theories_dir', type=str, default='data/theories_v2.1',
                       help='Directory containing theory files')
    parser.add_argument('--output_dir', type=str, default='data/knowledge_graph',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # 初始化构建器
    builder = KnowledgeGraphBuilder(output_dir=args.output_dir)
    
    # 加载数据
    if args.concepts_file and os.path.exists(args.concepts_file):
        builder.load_concepts(args.concepts_file)
    
    if args.relations_file and os.path.exists(args.relations_file):
        builder.load_concept_relations(args.relations_file)
    
    if args.formulas_file and os.path.exists(args.formulas_file):
        builder.load_formulas(args.formulas_file)
    
    if os.path.exists(args.theories_dir):
        builder.load_theories(args.theories_dir)
    
    # 推理隐含关系
    builder.infer_implicit_relations()
    
    # 计算概念重要性
    importance = builder.compute_concept_importance()
    print("\nTop 10 Most Important Concepts:")
    for node_id, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        node = builder.kg.nodes[node_id]
        print(f"  {node.name}: {score:.3f}")
    
    # 保存图谱
    graph_file = builder.save_graph()
    
    # 打印统计信息
    stats = builder.kg.get_statistics()
    print("\n=== Knowledge Graph Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()