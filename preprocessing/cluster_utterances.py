#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
硬聚类工具
使用MiniBatchKMeans对PersonaChat utterances进行硬聚类
"""

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import os
import sys
import json
import glob
from datetime import datetime
from pathlib import Path

# 添加上级目录到路径，以便导入global_config
sys.path.append(str(Path(__file__).parent.parent))
from global_config import (
    EMBEDDING_DATA_DIR, EXTRACTED_DATA_DIR, CLUSTER_DATA_DIR,
    get_latest_embeddings_file, EXTRACTED_UTTERANCES_CSV,
    CLUSTERING_CONFIG
)

def load_data():
    """加载embeddings和元数据"""
    # 使用global_config中的函数获取最新文件
    embeddings_file = get_latest_embeddings_file()
    
    if not embeddings_file or not embeddings_file.exists():
        raise FileNotFoundError(f"在 {EMBEDDING_DATA_DIR} 中找不到embeddings文件。请先运行 encode_utterances.py")
    
    print(f"✅ 找到embeddings文件: {embeddings_file}")
    embeddings = np.load(embeddings_file)
    print(f"✅ 加载embeddings: {embeddings.shape}")
    
    # 加载CSV文件
    if not EXTRACTED_UTTERANCES_CSV.exists():
        raise FileNotFoundError(f"找不到CSV文件: {EXTRACTED_UTTERANCES_CSV}")
    
    df = pd.read_csv(EXTRACTED_UTTERANCES_CSV)
    print(f"✅ 加载CSV数据: {len(df)} 条记录")
    
    return embeddings, df

def perform_hard_clustering(embeddings, n_clusters, random_state=42, batch_size=1000, max_iter=100):
    """
    使用MiniBatchKMeans进行硬聚类
    
    Args:
        embeddings: 句向量数组
        n_clusters: 聚类数量
        random_state: 随机种子
        batch_size: 批处理大小
        max_iter: 最大迭代次数
    
    Returns:
        kmeans: 训练好的KMeans模型
        hard_labels: 硬分配标签 (n_samples,)
    """
    print(f"\n开始MiniBatchKMeans硬聚类...")
    print(f"聚类数量: {n_clusters}")
    print(f"批处理大小: {batch_size}")
    print(f"最大迭代次数: {max_iter}")
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=max_iter,
        init='k-means++',
        random_state=random_state,
        verbose=1
    )
    
    start_time = datetime.now()
    print("正在训练MiniBatchKMeans模型...")
    hard_labels = kmeans.fit_predict(embeddings)
    end_time = datetime.now()
    clustering_time = (end_time - start_time).total_seconds()
    print(f"硬聚类完成，耗时: {clustering_time:.2f} 秒")
    
    return kmeans, hard_labels

def evaluate_clustering_quality(embeddings, hard_labels, sample_size=10000):
    """
    评估聚类质量（随机采样评估）
    
    Args:
        embeddings: 句向量数组
        hard_labels: 聚类标签
        sample_size: 采样大小
    """
    print(f"\n评估聚类质量...")
    
    # 如果数据太大，使用采样评估
    if len(embeddings) > sample_size:
        print(f"使用 {sample_size:,} 个样本进行评估")
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[indices]
        sample_labels = hard_labels[indices]
    else:
        sample_embeddings = embeddings
        sample_labels = hard_labels
    
    # 计算评估指标
    try:
        silhouette_avg = silhouette_score(sample_embeddings, sample_labels)
        calinski_harabasz = calinski_harabasz_score(sample_embeddings, sample_labels)
        
        print(f"轮廓系数 (Silhouette Score): {silhouette_avg:.4f}")
        print(f"Calinski-Harabasz指数: {calinski_harabasz:.2f}")
        
        return silhouette_avg, calinski_harabasz
    except Exception as e:
        print(f"评估过程出错: {e}")
        return None, None

def create_clustering_dataset(metadata, hard_labels):
    """
    创建聚类数据集
    
    Args:
        metadata: 原始元数据
        hard_labels: 硬聚类标签
    
    Returns:
        clustering_df: 包含聚类信息的DataFrame
        summary_df: 每个utterance的摘要信息
    """
    print(f"\n创建聚类数据集...")
    
    clustering_records = []
    summary_records = []
    
    for i, (_, row) in enumerate(metadata.iterrows()):
        # 基础信息
        base_info = {
            'utterance_id': row['utterance_id'],
            'conv_id': row['conv_id'],
            'turn_id': row['turn_id'],
            'speaker': row['speaker'],
            'text': row['text'],
            'source': row['source'],
            'length': row['length']
        }
        
        # 摘要信息（每个utterance一行）
        summary_info = base_info.copy()
        summary_info.update({
            'cluster_id': int(hard_labels[i]),
            'confidence': 1.0  # 硬聚类的置信度为1
        })
        summary_records.append(summary_info)
        
        # 详细信息（与摘要相同，保持兼容性）
        detail_info = base_info.copy()
        detail_info.update({
            'cluster_id': int(hard_labels[i]),
            'confidence': 1.0,
            'rank_in_utterance': 1,
            'is_primary_cluster': True,
            'is_single_label': True
        })
        clustering_records.append(detail_info)
    
    clustering_df = pd.DataFrame(clustering_records)
    summary_df = pd.DataFrame(summary_records)
    
    print(f"聚类数据集创建完成:")
    print(f"  详细记录数: {len(clustering_df):,}")
    print(f"  摘要记录数: {len(summary_df):,}")
    
    return clustering_df, summary_df

def analyze_cluster_themes(clustering_df, top_n_words=10):
    """
    分析每个聚类的主题特征
    
    Args:
        clustering_df: 包含聚类信息的DataFrame
        top_n_words: 提取的主题词数量
    """
    print(f"\n分析聚类主题...")
    
    cluster_analysis = []
    unique_clusters = sorted(clustering_df['cluster_id'].unique())
    
    for cluster_id in unique_clusters:
        cluster_data = clustering_df[clustering_df['cluster_id'] == cluster_id]
        
        # 基本统计
        total_utterances = len(cluster_data)
        
        # 来源分布
        source_dist = cluster_data['source'].value_counts()
        speaker_dist = cluster_data['speaker'].value_counts()
        
        # 简单的主题词提取（基于词频）
        texts = cluster_data['text'].tolist()
        all_words = ' '.join(texts).lower().split()
        word_freq = pd.Series(all_words).value_counts().head(top_n_words)
        
        cluster_info = {
            'cluster_id': cluster_id,
            'total_utterances': total_utterances,
            'top_sources': source_dist.to_dict(),
            'top_speakers': speaker_dist.to_dict(),
            'top_words': word_freq.to_dict()
        }
        
        cluster_analysis.append(cluster_info)
    
    print(f"聚类主题分析完成，共分析 {len(unique_clusters)} 个聚类")
    return cluster_analysis

def save_clustering_results(kmeans, summary_df, silhouette_score, calinski_harabasz_score, n_clusters,
                          cluster_labels, output_dir=None):
    """
    保存核心聚类结果
    
    Args:
        kmeans: 训练好的KMeans模型
        summary_df: 聚类摘要数据
        silhouette_score: 轮廓系数
        calinski_harabasz_score: CH指数
        n_clusters: 聚类数量
        cluster_labels: 聚类标签数组
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = str(CLUSTER_DATA_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n保存聚类结果...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 清理旧文件
    print("清理旧的聚类文件...")
    patterns_to_clean = [
        f"{output_dir}/kmeans_model_*.pkl",
        f"{output_dir}/clustering_summary_*.csv",
        f"{output_dir}/cluster_labels_*.npy"  # 新增：清理旧的标签文件
    ]
    
    cleaned_count = 0
    for pattern in patterns_to_clean:
        old_files = glob.glob(pattern)
        for old_file in old_files:
            try:
                os.remove(old_file)
                print(f"  已删除: {os.path.basename(old_file)}")
                cleaned_count += 1
            except Exception as e:
                print(f"  删除失败: {os.path.basename(old_file)} - {e}")
    
    if cleaned_count > 0:
        print(f"共清理了 {cleaned_count} 个旧文件")
    else:
        print("没有发现需要清理的旧文件")
    
    # 保存KMeans模型
    import pickle
    kmeans_file = os.path.join(output_dir, f"kmeans_model_k{n_clusters}_{timestamp}.pkl")
    with open(kmeans_file, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"KMeans模型已保存: {kmeans_file}")
    
    # 保存聚类标签数组
    labels_file = os.path.join(output_dir, f"cluster_labels_k{n_clusters}_{timestamp}.npy")
    np.save(labels_file, cluster_labels)
    print(f"聚类标签已保存: {labels_file}")
    
    # 保存聚类摘要数据
    summary_file = os.path.join(output_dir, f"clustering_summary_k{n_clusters}_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"聚类摘要已保存: {summary_file}")
    
    print(f"\n核心文件保存完成!")
    print(f"  模型文件: {os.path.basename(kmeans_file)}")
    print(f"  标签文件: {os.path.basename(labels_file)}")
    print(f"  数据文件: {os.path.basename(summary_file)}")
    if silhouette_score and calinski_harabasz_score:
        print(f"  轮廓系数: {silhouette_score:.4f}")
        print(f"  CH指数: {calinski_harabasz_score:.2f}")

def main():
    """
    主函数
    """
    print("=== PersonaChat Utterances 硬聚类工具 ===")
    
    # 从global_config获取聚类参数
    n_clusters = CLUSTERING_CONFIG['n_clusters']
    batch_size = CLUSTERING_CONFIG['batch_size']
    max_iter = CLUSTERING_CONFIG['max_iter']
    random_state = CLUSTERING_CONFIG['random_state']
    sample_size = CLUSTERING_CONFIG['evaluation_sample_size']
    output_dir = str(CLUSTER_DATA_DIR)
    
    print(f"聚类参数:")
    print(f"  聚类数量: {n_clusters}")
    print(f"  批处理大小: {batch_size}")
    print(f"  最大迭代次数: {max_iter}")
    print(f"  评估采样大小: {sample_size}")
    print(f"  输出目录: {output_dir}")
    
    # 1. 加载数据
    embeddings, metadata = load_data()
    if embeddings is None:
        return
    
    print(f"  预计每聚类平均: {len(embeddings)//n_clusters} utterances")
    
    # 2. 执行硬聚类
    kmeans, hard_labels = perform_hard_clustering(
        embeddings, n_clusters, random_state, batch_size, max_iter
    )
    
    # 3. 评估聚类质量（采样评估）
    silhouette_avg, calinski_harabasz = evaluate_clustering_quality(
        embeddings, hard_labels, sample_size=sample_size
    )
    
    # 4. 创建聚类数据集（只需要摘要）
    _, summary_df = create_clustering_dataset(metadata, hard_labels)
    
    # 5. 保存核心结果
    save_clustering_results(
        kmeans, summary_df, silhouette_avg, calinski_harabasz, n_clusters, hard_labels, output_dir
    )
    
    print(f"\n硬聚类分析完成")
    print(f"总utterances数: {len(summary_df):,}")
    print(f"聚类数: {n_clusters}")
    print(f"平均每聚类: {len(summary_df)/n_clusters:.1f} utterances")

if __name__ == "__main__":
    main() 