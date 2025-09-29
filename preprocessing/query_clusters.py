#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#注意，聚类质量评估直接用的CPU计算，所以谨慎设置抽样数量

"""
聚类结果查询工具
用于快速查询和分析聚类结果
"""

import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# 添加上级目录到路径，以便导入global_config
sys.path.append(str(Path(__file__).parent.parent))
from global_config import (
    CLUSTER_DATA_DIR, get_latest_cluster_model, get_latest_cluster_labels, 
    get_latest_cluster_summary, get_latest_embeddings_file, EXTRACTED_UTTERANCES_CSV
)

class ClusterQuery:
    """聚类结果查询类"""
    
    def __init__(self):
        """初始化查询器"""
        self.model = None
        self.labels = None
        self.summary_df = None
        self.embeddings = None
        self.load_data()
    
    def load_data(self):
        """加载聚类数据"""
        print("正在加载聚类数据...")
        
        # 加载聚类模型
        model_file = get_latest_cluster_model()
        if model_file and model_file.exists():
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            print(f"聚类模型已加载: {model_file.name}")
        else:
            print("未找到聚类模型文件")
            return
        
        # 加载聚类标签
        labels_file = get_latest_cluster_labels()
        if labels_file and labels_file.exists():
            self.labels = np.load(labels_file)
            print(f"聚类标签已加载: {labels_file.name}")
        else:
            print("未找到聚类标签文件")
            return
        
        # 加载聚类摘要
        summary_file = get_latest_cluster_summary()
        if summary_file and summary_file.exists():
            self.summary_df = pd.read_csv(summary_file)
            print(f"聚类摘要已加载: {summary_file.name}")
        else:
            print("未找到聚类摘要文件")
            return
        
        # 加载embeddings用于质量评估
        embeddings_file = get_latest_embeddings_file()
        if embeddings_file and embeddings_file.exists():
            self.embeddings = np.load(embeddings_file)
            print(f"Embeddings已加载: {embeddings_file.name}")
        else:
            print("未找到embeddings文件")
            return
        
        print(f"数据加载完成！共 {len(self.summary_df):,} 个utterances，{self.model.n_clusters} 个聚类")
    
    def get_cluster_info(self, cluster_id):
        """获取指定聚类的详细信息"""
        if self.summary_df is None:
            print("数据未加载")
            return
        
        cluster_data = self.summary_df[self.summary_df['cluster_id'] == cluster_id]
        if len(cluster_data) == 0:
            print(f"聚类 {cluster_id} 不存在")
            return
        
        print(f"\n=== 聚类 {cluster_id} 信息 ===")
        print(f"utterances数量: {len(cluster_data):,}")
        
        # 来源分布
        source_dist = cluster_data['source'].value_counts()
        print(f"来源分布:")
        for source, count in source_dist.head(5).items():
            print(f"  {source}: {count:,}")
        
        # 说话者分布
        speaker_dist = cluster_data['speaker'].value_counts()
        print(f"说话者分布:")
        for speaker, count in speaker_dist.head(5).items():
            print(f"  {speaker}: {count:,}")
        
        # 显示一些示例utterances（如果文本列存在）
        if 'text' in cluster_data.columns:
            print(f"\n示例utterances:")
            for i, (_, row) in enumerate(cluster_data.sample(20).iterrows()):
                print(f"  {i+1}. [{row['speaker']}] {row['text'][:200]}...")
        else:
            print(f"\n注意: 聚类摘要中未包含文本内容，仅包含元数据")
    
    def evaluate_clustering_quality(self, sample_size=10000):
        """评估聚类质量（随机采样评估）"""
        if self.embeddings is None or self.labels is None:
            print("数据未加载")
            return
        
        print(f"\n=== 聚类质量评估 ===")
        
        # 如果数据太大，使用采样评估
        if len(self.embeddings) > sample_size:
            print(f"使用 {sample_size:,} 个样本进行评估")
            indices = np.random.choice(len(self.embeddings), sample_size, replace=False)
            sample_embeddings = self.embeddings[indices]
            sample_labels = self.labels[indices]
        else:
            sample_embeddings = self.embeddings
            sample_labels = self.labels
        
        # 计算评估指标
        try:
            silhouette_avg = silhouette_score(sample_embeddings, sample_labels)
            calinski_harabasz = calinski_harabasz_score(sample_embeddings, sample_labels)
            
            print(f"轮廓系数 (Silhouette Score): {silhouette_avg:.4f}")
            print(f"Calinski-Harabasz指数: {calinski_harabasz:.2f}")
            
            # 解释指标含义
            print(f"\n指标解释:")
            print(f"  轮廓系数: 范围[-1,1]，越接近1表示聚类质量越好")
            print(f"  CH指数: 值越大表示聚类质量越好")
            
            return silhouette_avg, calinski_harabasz
        except Exception as e:
            print(f"评估过程出错: {e}")
            return None, None
    
    def get_cluster_statistics(self):
        """获取聚类统计信息"""
        if self.summary_df is None:
            print("数据未加载")
            return
        
        print(f"\n=== 聚类统计信息 ===")
        print(f"总utterances数: {len(self.summary_df):,}")
        print(f"聚类数量: {self.model.n_clusters}")
        
        # 聚类大小分布
        cluster_sizes = self.summary_df['cluster_id'].value_counts()
        print(f"聚类大小统计:")
        print(f"  最大聚类: {cluster_sizes.max():,} utterances")
        print(f"  最小聚类: {cluster_sizes.min():,} utterances")
        print(f"  平均大小: {cluster_sizes.mean():.1f} utterances")
        print(f"  中位数: {cluster_sizes.median():.1f} utterances")
        
        # 来源分布
        source_dist = self.summary_df['source'].value_counts()
        print(f"\n来源分布:")
        for source, count in source_dist.items():
            print(f"  {source}: {count:,}")
        
        # 说话者分布
        speaker_dist = self.summary_df['speaker'].value_counts()
        print(f"\n说话者分布:")
        for speaker, count in speaker_dist.items():
            print(f"  {speaker}: {count:,}")
    
    def show_cluster_size_distribution(self):
        """显示聚类大小分布"""
        if self.summary_df is None:
            print("数据未加载")
            return
        
        print(f"\n=== 聚类大小分布 ===")
        
        cluster_sizes = self.summary_df['cluster_id'].value_counts()
        
        print(f"聚类大小统计:")
        print(f"  最大聚类: {cluster_sizes.max():,} utterances")
        print(f"  最小聚类: {cluster_sizes.min():,} utterances")
        print(f"  平均大小: {cluster_sizes.mean():.1f} utterances")
        print(f"  中位数: {cluster_sizes.median():.1f} utterances")
        
        # 显示大小分布区间
        print("\n大小分布区间:")
        size_ranges = [
            (1, 500, "1-500"),
            (501, 1000, "501-1000"),
            (1001, 1500, "1001-1500"),
            (1501, 2000, "1501-2000"),
            (2001, float('inf'), ">2000"),
        ]

        for min_size, max_size, label in size_ranges:
            if max_size == float('inf'):
                count = len(cluster_sizes[cluster_sizes >= min_size])
            else:
                count = len(cluster_sizes[(cluster_sizes >= min_size) & (cluster_sizes <= max_size)])
            print(f"  {label}: {count} 个聚类")

def main():
    """主函数"""
    print("=== 聚类结果查询工具 ===")
    
    query = ClusterQuery()
    
    if query.summary_df is None:
        print("数据加载失败，请检查文件是否存在")
        return
    
    # 显示基本统计信息
    query.get_cluster_statistics()
    
    # 交互式查询
    while True:
        print(f"\n" + "="*50)
        print("请选择操作:")
        print("1. 查看聚类统计信息")
        print("2. 查看聚类大小分布")
        print("3. 查看指定聚类详情") #推荐聚类实例：12（关于“喜欢阅读”）3（关于“喜欢、运动”）30（关于“警察、军队”）
        print("4. 评估聚类质量")
        print("5. 退出")
        
        choice = input("\n请输入选择 (1-5): ").strip()
        
        if choice == '1':
            query.get_cluster_statistics()
        
        elif choice == '2':
            query.show_cluster_size_distribution()
        
        elif choice == '3':
            cluster_id = input("请输入聚类ID: ").strip()
            if cluster_id.isdigit():
                query.get_cluster_info(int(cluster_id))
            else:
                print("请输入有效的聚类ID")
        
        elif choice == '4':
            sample_size = input("请输入评估采样大小 (默认10000): ").strip()
            sample_size = int(sample_size) if sample_size.isdigit() else 10000
            query.evaluate_clustering_quality(sample_size)
        
        elif choice == '5':
            print("退出查询工具")
            break
        
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main() 