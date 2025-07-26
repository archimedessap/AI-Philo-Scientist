#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
概念空间可视化器

提供高维概念空间的多种可视化方法
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import networkx as nx
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches

from utils.logging_config import get_logger


class ConceptSpaceVisualizer:
    """概念空间可视化器"""
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.logger = get_logger('concept_space_visualizer')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置配色方案
        self.color_palette = sns.color_palette("husl", 10)
        sns.set_style("whitegrid")
        
        self.logger.info(f"初始化概念空间可视化器，输出目录: {output_dir}")
    
    def visualize_concept_space_2d(
        self,
        embeddings: Dict[str, np.ndarray],
        labels: Optional[Dict[str, str]] = None,
        method: str = 'pca',
        title: str = "概念空间二维投影",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        二维概念空间可视化
        
        Args:
            embeddings: 概念嵌入向量字典
            labels: 概念标签（用于着色）
            method: 降维方法 ('pca', 'tsne')
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        self.logger.info(f"生成二维概念空间可视化 (方法: {method})")
        
        # 准备数据
        concept_names = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[name] for name in concept_names])
        
        # 降维
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(embedding_matrix)
            explained_var = reducer.explained_variance_ratio_
            subtitle = f"PCA (解释方差: {explained_var[0]:.2%} + {explained_var[1]:.2%})"
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(concept_names)-1))
            coords_2d = reducer.fit_transform(embedding_matrix)
            subtitle = "t-SNE 投影"
        else:
            raise ValueError(f"不支持的降维方法: {method}")
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 如果有标签，按标签着色
        if labels:
            unique_labels = list(set(labels.values()))
            colors = {label: self.color_palette[i % len(self.color_palette)] 
                     for i, label in enumerate(unique_labels)}
            
            for label in unique_labels:
                mask = [labels.get(name, 'unknown') == label for name in concept_names]
                points = coords_2d[mask]
                names = [name for name, m in zip(concept_names, mask) if m]
                
                ax.scatter(points[:, 0], points[:, 1], 
                          c=[colors[label]], label=label, 
                          alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
                
                # 添加文本标签
                for i, name in enumerate(names):
                    ax.annotate(name, (points[i, 0], points[i, 1]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
        else:
            # 无标签，使用默认颜色
            scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                               c=range(len(concept_names)), 
                               cmap='viridis', alpha=0.7, s=100,
                               edgecolors='black', linewidth=0.5)
            
            # 添加文本标签
            for i, name in enumerate(concept_names):
                ax.annotate(name, (coords_2d[i, 0], coords_2d[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
        
        ax.set_xlabel('维度 1', fontsize=12)
        ax.set_ylabel('维度 2', fontsize=12)
        ax.set_title(f"{title}\n{subtitle}", fontsize=14, fontweight='bold')
        
        if labels:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # 保存图形
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"图形已保存: {save_path}")
        
        return fig
    
    def visualize_concept_clusters(
        self,
        embeddings: Dict[str, np.ndarray],
        n_clusters: int = 5,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, Dict[str, int]]:
        """
        概念聚类可视化
        
        Args:
            embeddings: 概念嵌入向量
            n_clusters: 聚类数量
            save_path: 保存路径
            
        Returns:
            图形对象和聚类结果
        """
        self.logger.info(f"生成概念聚类可视化 (聚类数: {n_clusters})")
        
        # 准备数据
        concept_names = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[name] for name in concept_names])
        
        # 聚类
        kmeans = KMeans(n_clusters=min(n_clusters, len(concept_names)), 
                       random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embedding_matrix)
        
        # PCA降维用于可视化
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(embedding_matrix)
        centers_2d = pca.transform(kmeans.cluster_centers_)
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 绘制每个聚类
        for i in range(n_clusters):
            mask = cluster_labels == i
            points = coords_2d[mask]
            
            # 绘制点
            ax.scatter(points[:, 0], points[:, 1], 
                      c=[self.color_palette[i]], 
                      label=f'聚类 {i+1}',
                      alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
            
            # 绘制聚类中心
            ax.scatter(centers_2d[i, 0], centers_2d[i, 1], 
                      c='black', marker='*', s=500, 
                      edgecolors='white', linewidth=2)
            
            # 绘制聚类边界（椭圆）
            if len(points) > 1:
                cov = np.cov(points.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                width, height = 2 * np.sqrt(eigenvalues)
                
                ellipse = Ellipse(centers_2d[i], width, height, angle,
                                 facecolor=self.color_palette[i], alpha=0.2,
                                 edgecolor=self.color_palette[i], linewidth=2)
                ax.add_patch(ellipse)
        
        # 添加概念标签
        for i, name in enumerate(concept_names):
            ax.annotate(name, (coords_2d[i, 0], coords_2d[i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        ax.set_xlabel('主成分 1', fontsize=12)
        ax.set_ylabel('主成分 2', fontsize=12)
        ax.set_title(f"概念空间聚类分析\n({n_clusters} 个聚类)", 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # 保存图形
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"聚类图已保存: {save_path}")
        
        # 创建聚类结果字典
        cluster_assignments = {name: int(label) for name, label in 
                             zip(concept_names, cluster_labels)}
        
        return fig, cluster_assignments
    
    def visualize_concept_network(
        self,
        embeddings: Dict[str, np.ndarray],
        threshold: float = 0.7,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        概念网络可视化（基于相似度）
        
        Args:
            embeddings: 概念嵌入向量
            threshold: 相似度阈值（用于连接边）
            save_path: 保存路径
            
        Returns:
            图形对象
        """
        self.logger.info(f"生成概念网络可视化 (阈值: {threshold})")
        
        # 计算相似度矩阵
        concept_names = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[name] for name in concept_names])
        
        # 归一化向量
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        normalized = embedding_matrix / (norms + 1e-8)
        
        # 计算余弦相似度
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # 创建网络图
        G = nx.Graph()
        
        # 添加节点
        for name in concept_names:
            G.add_node(name)
        
        # 添加边（基于相似度阈值）
        for i in range(len(concept_names)):
            for j in range(i+1, len(concept_names)):
                sim = similarity_matrix[i, j]
                if sim >= threshold:
                    G.add_edge(concept_names[i], concept_names[j], 
                             weight=sim, similarity=sim)
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # 使用spring布局
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # 计算节点大小（基于度）
        node_sizes = [300 + 100 * G.degree(node) for node in G.nodes()]
        
        # 绘制边
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=weights,
                             edge_color='gray', ax=ax)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                             node_color='lightblue', alpha=0.9,
                             edgecolors='black', linewidths=1, ax=ax)
        
        # 添加标签
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        
        # 添加边权重标签
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" 
                      for u, v in edges if G[u][v]['weight'] >= threshold + 0.1}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
        
        ax.set_title(f"概念相似度网络\n(相似度阈值 ≥ {threshold})", 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # 添加统计信息
        info_text = f"节点数: {G.number_of_nodes()}\n边数: {G.number_of_edges()}\n连通分量: {nx.number_connected_components(G)}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存图形
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"网络图已保存: {save_path}")
        
        return fig
    
    def visualize_concept_density(
        self,
        embeddings: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        概念空间密度可视化
        
        Args:
            embeddings: 概念嵌入向量
            save_path: 保存路径
            
        Returns:
            图形对象
        """
        self.logger.info("生成概念空间密度可视化")
        
        # 准备数据
        concept_names = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[name] for name in concept_names])
        
        # PCA降维
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(embedding_matrix)
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：密度热力图
        from scipy.stats import gaussian_kde
        
        x = coords_2d[:, 0]
        y = coords_2d[:, 1]
        
        # 计算密度
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        
        # 绘制密度图
        scatter = ax1.scatter(x, y, c=z, s=100, cmap='YlOrRd', 
                            edgecolors='black', linewidth=0.5, alpha=0.8)
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('密度', fontsize=10)
        
        ax1.set_xlabel('主成分 1', fontsize=12)
        ax1.set_ylabel('主成分 2', fontsize=12)
        ax1.set_title('概念空间密度分布', fontsize=14, fontweight='bold')
        
        # 右图：等高线图
        xi = np.linspace(x.min()-1, x.max()+1, 100)
        yi = np.linspace(y.min()-1, y.max()+1, 100)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = gaussian_kde(xy)(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
        
        contour = ax2.contourf(Xi, Yi, Zi, levels=15, cmap='YlOrRd', alpha=0.6)
        ax2.contour(Xi, Yi, Zi, levels=10, colors='black', alpha=0.4, linewidths=0.5)
        
        # 添加概念点
        ax2.scatter(x, y, c='blue', s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # 标注高密度区域的概念
        high_density_idx = np.argsort(z)[-5:]  # 前5个高密度点
        for idx in high_density_idx:
            ax2.annotate(concept_names[idx], (x[idx], y[idx]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax2.set_xlabel('主成分 1', fontsize=12)
        ax2.set_ylabel('主成分 2', fontsize=12)
        ax2.set_title('概念空间密度等高线', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图形
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"密度图已保存: {save_path}")
        
        return fig
    
    def visualize_concept_evolution(
        self,
        embeddings_timeline: List[Tuple[str, Dict[str, np.ndarray]]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        概念空间演化可视化
        
        Args:
            embeddings_timeline: [(时间标签, 嵌入字典), ...]
            save_path: 保存路径
            
        Returns:
            图形对象
        """
        self.logger.info("生成概念空间演化可视化")
        
        n_timepoints = len(embeddings_timeline)
        fig, axes = plt.subplots(1, n_timepoints, figsize=(6*n_timepoints, 6))
        
        if n_timepoints == 1:
            axes = [axes]
        
        # 收集所有概念用于一致的降维
        all_concepts = set()
        for _, embeddings in embeddings_timeline:
            all_concepts.update(embeddings.keys())
        
        # 为每个时间点创建可视化
        for i, (time_label, embeddings) in enumerate(embeddings_timeline):
            ax = axes[i]
            
            # 降维
            concept_names = list(embeddings.keys())
            embedding_matrix = np.array([embeddings[name] for name in concept_names])
            
            pca = PCA(n_components=2, random_state=42)
            coords_2d = pca.fit_transform(embedding_matrix)
            
            # 绘制
            ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                      c=range(len(concept_names)), cmap='viridis',
                      s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # 添加标签
            for j, name in enumerate(concept_names):
                ax.annotate(name, (coords_2d[j, 0], coords_2d[j, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
            
            ax.set_xlabel('维度 1', fontsize=10)
            ax.set_ylabel('维度 2', fontsize=10)
            ax.set_title(f'{time_label}\n({len(concept_names)} 概念)', 
                        fontsize=12, fontweight='bold')
        
        fig.suptitle('概念空间演化', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图形
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"演化图已保存: {save_path}")
        
        return fig
    
    def create_comprehensive_report(
        self,
        embeddings: Dict[str, np.ndarray],
        labels: Optional[Dict[str, str]] = None,
        output_prefix: str = "concept_space"
    ):
        """
        创建综合的概念空间分析报告
        
        Args:
            embeddings: 概念嵌入向量
            labels: 概念标签
            output_prefix: 输出文件前缀
        """
        self.logger.info("生成综合概念空间分析报告")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"{output_prefix}_{timestamp}"
        report_dir.mkdir(exist_ok=True)
        
        # 1. 二维投影（PCA和t-SNE）
        self.visualize_concept_space_2d(
            embeddings, labels, method='pca',
            title="概念空间 PCA 投影",
            save_path=str(report_dir / "pca_projection.png")
        )
        
        if len(embeddings) > 5:  # t-SNE需要足够的样本
            self.visualize_concept_space_2d(
                embeddings, labels, method='tsne',
                title="概念空间 t-SNE 投影",
                save_path=str(report_dir / "tsne_projection.png")
            )
        
        # 2. 聚类分析
        if len(embeddings) >= 3:
            n_clusters = min(5, len(embeddings) // 2)
            fig, clusters = self.visualize_concept_clusters(
                embeddings, n_clusters=n_clusters,
                save_path=str(report_dir / "cluster_analysis.png")
            )
            
            # 保存聚类结果
            with open(report_dir / "cluster_assignments.json", 'w', encoding='utf-8') as f:
                json.dump(clusters, f, ensure_ascii=False, indent=2)
        
        # 3. 概念网络
        self.visualize_concept_network(
            embeddings, threshold=0.7,
            save_path=str(report_dir / "concept_network.png")
        )
        
        # 4. 密度分析
        if len(embeddings) >= 5:
            self.visualize_concept_density(
                embeddings,
                save_path=str(report_dir / "density_analysis.png")
            )
        
        # 5. 生成文本报告
        self._generate_text_report(embeddings, labels, report_dir)
        
        self.logger.info(f"综合报告已生成: {report_dir}")
        print(f"📊 概念空间分析报告已生成: {report_dir}")
    
    def _generate_text_report(
        self,
        embeddings: Dict[str, np.ndarray],
        labels: Optional[Dict[str, str]],
        output_dir: Path
    ):
        """生成文本分析报告"""
        report_lines = []
        report_lines.append("# 概念空间分析报告")
        report_lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"\n## 基本统计")
        report_lines.append(f"- 概念总数: {len(embeddings)}")
        
        if labels:
            label_counts = {}
            for label in labels.values():
                label_counts[label] = label_counts.get(label, 0) + 1
            report_lines.append(f"- 标签分布:")
            for label, count in sorted(label_counts.items()):
                report_lines.append(f"  - {label}: {count}")
        
        # 计算统计信息
        embedding_matrix = np.array(list(embeddings.values()))
        
        report_lines.append(f"\n## 嵌入空间统计")
        report_lines.append(f"- 嵌入维度: {embedding_matrix.shape[1]}")
        report_lines.append(f"- 平均范数: {np.mean(np.linalg.norm(embedding_matrix, axis=1)):.3f}")
        report_lines.append(f"- 标准差: {np.std(embedding_matrix):.3f}")
        
        # 计算相似度统计
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        normalized = embedding_matrix / (norms + 1e-8)
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # 去除对角线
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        similarities = similarity_matrix[mask]
        
        report_lines.append(f"\n## 相似度分析")
        report_lines.append(f"- 平均相似度: {np.mean(similarities):.3f}")
        report_lines.append(f"- 最高相似度: {np.max(similarities):.3f}")
        report_lines.append(f"- 最低相似度: {np.min(similarities):.3f}")
        
        # 找出最相似的概念对
        concept_names = list(embeddings.keys())
        top_k = min(5, len(concept_names) * (len(concept_names) - 1) // 2)
        
        report_lines.append(f"\n## 最相似的概念对 (Top {top_k})")
        
        similarity_pairs = []
        for i in range(len(concept_names)):
            for j in range(i+1, len(concept_names)):
                similarity_pairs.append((
                    concept_names[i],
                    concept_names[j],
                    similarity_matrix[i, j]
                ))
        
        similarity_pairs.sort(key=lambda x: x[2], reverse=True)
        
        for i, (concept1, concept2, sim) in enumerate(similarity_pairs[:top_k]):
            report_lines.append(f"{i+1}. {concept1} ↔ {concept2}: {sim:.3f}")
        
        # 保存报告
        with open(output_dir / "analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))


# 便捷函数
def visualize_unified_theory_space(
    unified_generator,
    output_dir: str = "unified_space_visualization"
):
    """
    可视化统一理论生成器的概念空间
    
    Args:
        unified_generator: UnifiedTheoryGenerator实例
        output_dir: 输出目录
    """
    visualizer = ConceptSpaceVisualizer(output_dir)
    
    # 获取概念嵌入
    concept_embeddings = unified_generator.concept_embeddings
    
    # 获取理论嵌入
    theory_embeddings = unified_generator.theory_embeddings
    
    # 为概念和理论创建标签
    concept_labels = {name: 'concept' for name in concept_embeddings.keys()}
    theory_labels = {name: 'theory' for name in theory_embeddings.keys()}
    
    # 合并嵌入和标签
    all_embeddings = {**concept_embeddings, **theory_embeddings}
    all_labels = {**concept_labels, **theory_labels}
    
    # 生成综合报告
    visualizer.create_comprehensive_report(
        all_embeddings,
        labels=all_labels,
        output_prefix="unified_concept_space"
    )


if __name__ == "__main__":
    # 示例：创建一些模拟数据进行测试
    np.random.seed(42)
    
    # 生成模拟嵌入
    test_embeddings = {
        f"概念_{i}": np.random.randn(512) for i in range(20)
    }
    
    # 添加一些聚类结构
    for i in range(5):
        cluster_center = np.random.randn(512) * 2
        for j in range(3):
            test_embeddings[f"聚类{i}_概念{j}"] = cluster_center + np.random.randn(512) * 0.3
    
    # 创建标签
    test_labels = {}
    for name in test_embeddings:
        if "聚类" in name:
            test_labels[name] = name.split('_')[0]
        else:
            test_labels[name] = "通用概念"
    
    # 创建可视化器并生成报告
    visualizer = ConceptSpaceVisualizer("test_visualizations")
    visualizer.create_comprehensive_report(test_embeddings, test_labels, "test_space")