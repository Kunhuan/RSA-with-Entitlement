#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#python preprocessing/encode_utterances.py
#这是个预处理脚本，几乎是过程性的、一次性的使用。我们使用的模型是SBERT，作用是将自然语言处理为语义向量
#同样，生成结果时会删除旧的文件！

"""
PersonaChat Utterances 句向量编码工具
使用SBERT模型将所有utterances编码为句向量，保存为npy格式
"""

import os
import sys

# 设置清华源和离线模式（在其他导入之前）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from sentence_transformers import SentenceTransformer
from datetime import datetime
import json
from typing import List, Dict, Any
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# 添加项目根目录到Python路径
PREPROCESSING_DIR = Path(__file__).resolve().parent
RSAE_DIR = PREPROCESSING_DIR.parent
sys.path.insert(0, str(RSAE_DIR))

# 从全局配置导入设置
from global_config import (
    EXTRACTED_UTTERANCES_CSV, EMBEDDING_DATA_DIR, 
    SBERT_CONFIG, get_timestamp
)

def find_utterances_file():
    """查找utterances文件"""
    if EXTRACTED_UTTERANCES_CSV.exists():
        return str(EXTRACTED_UTTERANCES_CSV)
    
    raise FileNotFoundError(f"找不到extracted_utterances.csv文件: {EXTRACTED_UTTERANCES_CSV}")

def load_utterances(results_dir: str = "results") -> pd.DataFrame:
    """
    加载已提取的utterances数据
    
    Args:
        results_dir: 结果目录路径
        
    Returns:
        pandas.DataFrame: 包含utterances数据的DataFrame
    """
    try:
        utterances_file = find_utterances_file()
        print(f"找到utterances文件: {utterances_file}")
        
        df = pd.read_csv(utterances_file)
        print(f"成功加载 {len(df)} 条utterances")
        return df
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        raise

def encode_utterances_with_sbert(df: pd.DataFrame, batch_size: int = 32):
    """使用SBERT模型对utterances进行批量编码"""
    model_name = SBERT_CONFIG['model_name']
    
    print(f"\n=== 开始使用SBERT编码utterances ===")
    print(f"模型: {model_name}")
    print(f"批处理大小: {batch_size}")
    
    # 检测GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载SBERT模型
    print("正在加载SBERT模型...")
    try:
        cache_dir = SBERT_CONFIG['cache_dir']
        model = SentenceTransformer(model_name, cache_folder=cache_dir, device=device)
        print(f"成功加载模型: {model_name}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None
    
    # 准备文本数据
    utterance_texts = df['text'].tolist()
    
    print(f"准备编码 {len(utterance_texts)} 个utterances...")
    
    # 批量编码
    print("开始编码...")
    start_time = datetime.now()
    
    try:
        total = len(utterance_texts)
        embeddings_list = []
        
        for i in range(0, total, batch_size * 100):
            end_idx = min(i + batch_size * 100, total)
            current_batch = utterance_texts[i:end_idx]
            batch_num = i//batch_size//100 + 1
            total_batches = (total-1)//(batch_size*100) + 1
            
            # 每10个批次显示一次进度，或者是第一个/最后一个批次
            if batch_num % 10 == 1 or batch_num == total_batches or batch_num == 1:
                print(f"处理批次 {batch_num}/{total_batches}: {i} 到 {end_idx} (共 {total})")
            
            current_embeddings = model.encode(current_batch, 
                            batch_size=batch_size,
                            show_progress_bar=False,  # 关闭内部进度条
                            convert_to_numpy=True)
            
            embeddings_list.append(current_embeddings)
        
        
        # 合并所有批次的结果
        embeddings = np.vstack(embeddings_list)
        
        end_time = datetime.now()
        encoding_time = (end_time - start_time).total_seconds()
        
        print(f"编码完成！耗时: {encoding_time:.2f} 秒")
        print(f"句向量维度: {embeddings.shape}")
        
        # 保存结果
        save_embeddings(embeddings, df, model_name)
        
        return embeddings
        
    except Exception as e:
        print(f"编码过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_embeddings(embeddings, texts, model_name):
    """保存编码结果和元数据"""
    timestamp = get_timestamp()
    
    # 确保输出目录存在
    EMBEDDING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 保存向量文件
    base_filename = f"utterances_embeddings_{model_name}_{timestamp}"
    npy_path = EMBEDDING_DATA_DIR / f"{base_filename}.npy"
    np.save(npy_path, embeddings)
    print(f"句向量已保存到: {npy_path}")
    
    # 保存配置文件
    config = {
        'model_name': model_name,
        'embedding_dim': embeddings.shape[1],
        'num_texts': len(texts),
        'timestamp': timestamp,
        'data_source': 'extracted_utterances.csv',
        'file_format': 'numpy_array'
    }
    
    config_path = EMBEDDING_DATA_DIR / f"{base_filename}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"配置信息已保存到: {config_path}")
    
    return npy_path, config_path

def load_embeddings(file_path: str):
    """
    加载已保存的嵌入向量
    """
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.npz'):
        data = np.load(file_path)
        return data['embeddings'], {key: data[key] for key in data.files if key != 'embeddings'}
    else:
        raise ValueError("不支持的文件格式，请使用 .npy 或 .npz 格式")

def main():
    """主函数"""
    print("=== PersonaChat Utterances SBERT编码工具 ===")
    
    # 加载utterances数据
    df = load_utterances()
    if df is None:
        return
    
    # 显示基本数据信息
    print(f"\n数据信息:")
    print(f"总utterances数: {len(df)}")
    print(f"来源分布: {dict(df['source'].value_counts())}")
    print(f"说话人分布: {dict(df['speaker'].value_counts())}")
    
    # 设置批处理大小
    batch_size = SBERT_CONFIG['batch_size']
    
    
    print(f"使用批处理大小: {batch_size}")
    
    # 开始编码
    embeddings = encode_utterances_with_sbert(df, batch_size)
    
    if embeddings is not None:
        print(f"\n编码完成")
    else:
        print(f"\n编码失败")

if __name__ == "__main__":
    main() 