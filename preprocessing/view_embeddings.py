#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#python preprocessing/view_embeddings.py


"""
PersonaChat Utterances 句向量查看工具
用于查看和分析已编码的句向量
"""

import numpy as np
import pandas as pd
import os
import sys
import json
from pathlib import Path
import random
from typing import List, Dict, Any, Tuple, Optional

# 添加上级目录到路径，以便导入global_config
sys.path.append(str(Path(__file__).parent.parent))
from global_config import (
    EMBEDDING_DATA_DIR, EXTRACTED_DATA_DIR,
    get_latest_embeddings_file, get_latest_embeddings_config,
    EXTRACTED_UTTERANCES_CSV
)

def load_embeddings_and_metadata() -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], Optional[Dict]]:
    """
    加载句向量和元数据
    
    Returns:
        Tuple[np.ndarray, pd.DataFrame, Dict]: 句向量数组, 元数据DataFrame, 配置信息
    """
    print("查找句向量文件...")
    
    # 使用global_config中的函数获取最新文件
    embeddings_file = get_latest_embeddings_file()
    config_file = get_latest_embeddings_config()
    
    if not embeddings_file or not embeddings_file.exists():
        print("错误: 找不到句向量文件")
        print(f"查找路径: {EMBEDDING_DATA_DIR}")
        return None, None, None
        
    print(f"找到句向量文件: {embeddings_file}")
    
    # 加载句向量
    try:
        embeddings = np.load(embeddings_file)
        print(f"成功加载句向量，形状: {embeddings.shape}")
    except Exception as e:
        print(f"加载句向量失败: {e}")
        return None, None, None
    
    # 加载元数据
    metadata_df = None
    if EXTRACTED_UTTERANCES_CSV.exists():
        try:
            metadata_df = pd.read_csv(EXTRACTED_UTTERANCES_CSV)
            print(f"成功加载元数据，共 {len(metadata_df)} 条记录")
            print(f"元数据文件位置: {EXTRACTED_UTTERANCES_CSV}")
            
            # 检查记录数是否匹配
            if len(metadata_df) != embeddings.shape[0]:
                print(f"警告: 元数据记录数 ({len(metadata_df)}) 与句向量数量 ({embeddings.shape[0]}) 不匹配")
        except Exception as e:
            print(f"加载元数据失败: {e}")
    else:
        print(f"找不到元数据文件: {EXTRACTED_UTTERANCES_CSV}")
    
    # 加载配置信息
    config = None
    if config_file and config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"成功加载配置信息")
        except Exception as e:
            print(f"加载配置信息失败: {e}")
    
    return embeddings, metadata_df, config

def show_basic_stats(embeddings: np.ndarray, metadata_df: pd.DataFrame, config: Dict):
    """
    显示基本统计信息
    """
    print("\n=== 句向量基本统计 ===")
    
    # 句向量统计
    print(f"句向量数量: {embeddings.shape[0]:,}")
    print(f"向量维度: {embeddings.shape[1]}")
    print(f"内存占用: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    
    # 向量值统计
    print(f"\n向量值统计:")
    print(f"  最小值: {embeddings.min():.4f}")
    print(f"  最大值: {embeddings.max():.4f}")
    print(f"  均值: {embeddings.mean():.4f}")
    print(f"  标准差: {embeddings.std():.4f}")
    
    # 向量长度统计
    vector_norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n向量长度统计:")
    print(f"  最小长度: {vector_norms.min():.4f}")
    print(f"  最大长度: {vector_norms.max():.4f}")
    print(f"  平均长度: {vector_norms.mean():.4f}")
    print(f"  标准差: {vector_norms.std():.4f}")
    
    # 元数据统计
    if metadata_df is not None:
        print(f"\n元数据统计:")
        
        # 按来源分布
        print("\n按来源分布:")
        source_counts = metadata_df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count:,} 个 ({count/len(metadata_df)*100:.1f}%)")
        
        # 按说话人分布
        print("\n按说话人分布:")
        speaker_counts = metadata_df['speaker'].value_counts()
        for speaker, count in speaker_counts.items():
            print(f"  {speaker}: {count:,} 个 ({count/len(metadata_df)*100:.1f}%)")
        
        # 文本长度分布
        print("\n文本长度分布:")
        length_stats = metadata_df['length'].describe()
        print(f"  最短: {length_stats['min']:.0f} 个词")
        print(f"  最长: {length_stats['max']:.0f} 个词")
        print(f"  平均: {length_stats['mean']:.1f} 个词")
        print(f"  中位数: {length_stats['50%']:.0f} 个词")
    
    # 配置信息
    if config:
        print(f"\n编码配置:")
        print(f"  模型: {config.get('model_name', '未知')}")
        print(f"  编码时间: {config.get('encoding_timestamp', '未知')}")

def show_random_examples(embeddings: np.ndarray, metadata_df: pd.DataFrame, n_samples: int = 5):
    """
    显示随机样本及其向量
    """
    if metadata_df is None:
        print("无法显示样本，缺少元数据")
        return
        
    print(f"\n=== 随机 {n_samples} 个样本 ===")
    
    # 随机选择样本
    sample_indices = random.sample(range(len(metadata_df)), min(n_samples, len(metadata_df)))
    
    for i, idx in enumerate(sample_indices):
        text = metadata_df.iloc[idx]['text']
        speaker = metadata_df.iloc[idx]['speaker']
        conv_id = metadata_df.iloc[idx]['conv_id']
        
        # 获取向量的前5个值
        vector_preview = embeddings[idx][:5]
        
        print(f"\n样本 {i+1}:")
        print(f"  对话ID: {conv_id}")
        print(f"  说话人: {speaker}")
        print(f"  文本: {text}")
        print(f"  向量前5维: {vector_preview}")

def main():
    """
    主函数
    """
    print("=== PersonaChat Utterances 句向量查看工具 ===")
    
    # 加载句向量和元数据
    embeddings, metadata_df, config = load_embeddings_and_metadata()
    if embeddings is None:
        return
    
    # 显示基本统计信息
    show_basic_stats(embeddings, metadata_df, config)
    
    # 显示随机样本
    show_random_examples(embeddings, metadata_df, n_samples=2)
    
    print("\n查看完成!")

if __name__ == "__main__":
    main() 