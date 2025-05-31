#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析概念嵌入空间

分析嵌入向量空间中的概念结构、关系和聚类。
生成可视化和相似度矩阵。
"""

import os
import argparse
import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns

def load_embeddings(embedding_dir):
    """加载嵌入向量"""
    concept_path = os.path.join(embedding_dir, "concept_embeddings.pkl")
    formula_path = os.path.join(embedding_dir, "formula_embeddings.pkl")
    theory_path = os.path.join(embedding_dir, "theory_embeddings.pkl")
    
    concept_embeddings = {}
    formula_embeddings = {}
    theory_embeddings = {}
    
    # 加载概念嵌入
    if os.path.exists(concept_path):
        with open(concept_path, 'rb') as f:
            concept_embeddings = pickle.load(f)
    
    # 加载公式嵌入
    if os.path.exists(formula_path):
        with open(formula_path, 'rb') as f:
            formula_embeddings = pickle.load(f)
    
    # 加载理论嵌入
    if os.path.exists(theory_path):
        with open(theory_path, 'rb') as f:
            theory_embeddings = pickle.load(f)
    
    return concept_embeddings, formula_embeddings, theory_embeddings

def compute_similarity_matrix(embeddings):
    """计算嵌入向量之间的相似度矩阵"""
    names = list(embeddings.keys())
    vectors = [embeddings[name] for name in names]
    vectors = np.array(vectors)
    
    # 计算余弦相似度
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity_matrix = np.dot(vectors_norm, vectors_norm.T)
    
    return similarity_matrix, names

def visualize_embeddings(embeddings, output_path, method="tsne", perplexity=5):
    """降维可视化嵌入向量"""
    if not embeddings:
        print("[警告] 没有嵌入向量可供可视化")
        return None
    
    names = list(embeddings.keys())
    vectors = [embeddings[name] for name in names]
    vectors = np.array(vectors)
    
    # 执行降维
    if method == "tsne":
        model = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, len(vectors)-1))
        embeddings_2d = model.fit_transform(vectors)
    else:
        raise ValueError(f"不支持的降维方法: {method}")
    
    # 绘制散点图
    plt.figure(figsize=(12, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    # 添加标签
    for i, name in enumerate(names):
        plt.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=10)
    
    plt.title("概念嵌入空间可视化")
    plt.tight_layout()
    plt.savefig(output_path)
    
    return embeddings_2d, names

def cluster_embeddings(embeddings, num_clusters=3):
    """聚类分析嵌入向量"""
    names = list(embeddings.keys())
    vectors = [embeddings[name] for name in names]
    vectors = np.array(vectors)
    
    # K-means聚类
    kmeans = KMeans(n_clusters=min(num_clusters, len(vectors)), random_state=42)
    clusters = kmeans.fit_predict(vectors)
    
    # DBSCAN密度聚类
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    dbscan_clusters = dbscan.fit_predict(vectors)
    
    results = {
        "kmeans": {name: int(cluster) for name, cluster in zip(names, clusters)},
        "dbscan": {name: int(cluster) for name, cluster in zip(names, dbscan_clusters)}
    }
    
    return results

def analyze_concept_relationships(embeddings, similarity_threshold=0.8):
    """分析概念间的关系"""
    similarity_matrix, names = compute_similarity_matrix(embeddings)
    
    relationships = []
    
    # 寻找相似概念对
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            sim = similarity_matrix[i, j]
            if sim >= similarity_threshold:
                relationships.append({
                    "concept1": names[i],
                    "concept2": names[j],
                    "similarity": float(sim),
                    "relationship_type": "similar"
                })
    
    # 对每个概念找到最相似的概念
    for i in range(len(names)):
        # 排除自身
        similarities = [(j, similarity_matrix[i, j]) for j in range(len(names)) if j != i]
        if similarities:
            most_similar_idx, sim = max(similarities, key=lambda x: x[1])
            relationships.append({
                "concept1": names[i],
                "concept2": names[most_similar_idx],
                "similarity": float(sim),
                "relationship_type": "most_similar"
            })
    
    return relationships

def save_results(output_dir, embeddings_2d, names, similarity_matrix, clusters, relationships):
    """保存分析结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存2D嵌入结果
    embeddings_2d_path = os.path.join(output_dir, "embeddings_2d.json")
    with open(embeddings_2d_path, 'w', encoding='utf-8') as f:
        json.dump({
            "names": names,
            "coordinates": embeddings_2d.tolist()
        }, f, ensure_ascii=False, indent=2)
    
    # 保存相似度矩阵
    similarity_path = os.path.join(output_dir, "similarity_matrix.json")
    with open(similarity_path, 'w', encoding='utf-8') as f:
        json.dump({
            "names": names,
            "matrix": similarity_matrix.tolist()
        }, f, ensure_ascii=False, indent=2)
    
    # 保存聚类结果
    clusters_path = os.path.join(output_dir, "clusters.json")
    with open(clusters_path, 'w', encoding='utf-8') as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    
    # 保存关系分析
    relationships_path = os.path.join(output_dir, "relationships.json")
    with open(relationships_path, 'w', encoding='utf-8') as f:
        json.dump(relationships, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] 分析结果已保存到: {output_dir}")

def visualize_similarity_matrix(similarity_matrix, names, output_path):
    """可视化相似度矩阵"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix, 
        annot=True, 
        xticklabels=names, 
        yticklabels=names,
        cmap="viridis",
        vmin=0, 
        vmax=1
    )
    plt.title("概念相似度矩阵")
    plt.tight_layout()
    plt.savefig(output_path)

def main():
    parser = argparse.ArgumentParser(description="分析概念嵌入空间")
    parser.add_argument("--embedding_dir", type=str, default="data/embeddings",
                      help="嵌入向量目录")
    parser.add_argument("--output_dir", type=str, default="data/analysis",
                      help="分析结果输出目录")
    parser.add_argument("--num_clusters", type=int, default=3,
                      help="聚类数量")
    parser.add_argument("--similarity_threshold", type=float, default=0.8,
                      help="相似度阈值")
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载嵌入向量
    concept_embeddings, formula_embeddings, theory_embeddings = load_embeddings(args.embedding_dir)
    
    print(f"[INFO] 加载了 {len(concept_embeddings)} 个概念嵌入, "
          f"{len(formula_embeddings)} 个公式嵌入, "
          f"{len(theory_embeddings)} 个理论嵌入")
    
    # 分析概念嵌入
    if concept_embeddings:
        print("\n[步骤1] 分析概念嵌入")
        
        # 可视化概念嵌入
        viz_path = os.path.join(args.output_dir, "concept_embeddings_2d.png")
        embeddings_2d, names = visualize_embeddings(concept_embeddings, viz_path)
        print(f"[INFO] 概念嵌入可视化已保存到: {viz_path}")
        
        # 计算相似度矩阵
        similarity_matrix, names = compute_similarity_matrix(concept_embeddings)
        
        # 可视化相似度矩阵
        sim_viz_path = os.path.join(args.output_dir, "concept_similarity_matrix.png")
        visualize_similarity_matrix(similarity_matrix, names, sim_viz_path)
        print(f"[INFO] 相似度矩阵可视化已保存到: {sim_viz_path}")
        
        # 聚类分析
        clusters = cluster_embeddings(concept_embeddings, args.num_clusters)
        print(f"[INFO] 聚类分析完成，发现 {len(set(clusters['kmeans'].values()))} 个K-means聚类")
        
        # 关系分析
        relationships = analyze_concept_relationships(concept_embeddings, args.similarity_threshold)
        print(f"[INFO] 关系分析完成，发现 {len(relationships)} 个关系")
        
        # 保存结果
        save_results(args.output_dir, embeddings_2d, names, similarity_matrix, clusters, relationships)
    
    # 分析公式嵌入
    if formula_embeddings:
        print("\n[步骤2] 分析公式嵌入")
        
        # 可视化公式嵌入
        viz_path = os.path.join(args.output_dir, "formula_embeddings_2d.png")
        embeddings_2d, names = visualize_embeddings(formula_embeddings, viz_path)
        print(f"[INFO] 公式嵌入可视化已保存到: {viz_path}")
    
    # 分析理论嵌入
    if theory_embeddings:
        print("\n[步骤3] 分析理论嵌入")
        
        # 可视化理论嵌入
        viz_path = os.path.join(args.output_dir, "theory_embeddings_2d.png")
        embeddings_2d, names = visualize_embeddings(theory_embeddings, viz_path)
        print(f"[INFO] 理论嵌入可视化已保存到: {viz_path}")
    
    print("\n[分析完成] 概念空间分析已完成")

if __name__ == "__main__":
    main()
