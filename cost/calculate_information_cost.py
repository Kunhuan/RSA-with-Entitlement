#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算Utterance的信息量成本

该脚本加载所有提取的utterances，基于Unigram模型计算每个utterance的信息量，
并将其作为成本保存到一个文件中。
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import Counter
import argparse
from pathlib import Path
from datetime import datetime

# 将项目根目录添加到Python路径中，以便导入其他模块
COST_DIR = Path(__file__).resolve().parent
RSAE_DIR = COST_DIR.parent
sys.path.insert(0, str(RSAE_DIR))

# 从全局配置导入设置
from global_config import (
    EXTRACTED_UTTERANCES_CSV, COST_RESULTS_DIR, 
    COST_PARAMS, get_timestamp
)

def find_utterances_file():
    """查找utterances文件"""
    if EXTRACTED_UTTERANCES_CSV.exists():
        return str(EXTRACTED_UTTERANCES_CSV)
    
    raise FileNotFoundError(f"找不到extracted_utterances.csv文件: {EXTRACTED_UTTERANCES_CSV}")

def load_all_utterances():
    """
    加载所有utterances数据，用于构建语言模型
    """
    utterances_file = find_utterances_file()
    
    print(f"正在加载数据: {utterances_file}")
    df = pd.read_csv(utterances_file)
    print(f"✅ 成功加载 {len(df)} 条utterances")
    
    return df

def preprocess_text(text: str) -> str:
    """
    对文本进行简单的预处理。
    """
    if not isinstance(text, str):
        return ""
    return text.lower()

def build_word_probability_model(texts: pd.Series, smoothing_alpha: float = 1.0):
    """
    构建Unigram词概率模型。

    Args:
        texts (pd.Series): 包含所有文本的Pandas Series。
        smoothing_alpha (float): 拉普拉斯平滑的alpha值。

    Returns:
        dict: 词到其对数概率的映射。
    """
    print("构建词概率模型...")
    all_words = " ".join(texts).split()
    word_counts = Counter(all_words)
    
    vocab_size = len(word_counts)
    total_words = len(all_words)
    
    print(f"词汇表大小 (V): {vocab_size}")
    print(f"总词数 (N): {total_words}")
    
    # 计算带有拉普拉斯平滑的对数概率
    # P(w) = (count(w) + alpha) / (N + alpha * V)
    # log(P(w)) = log(count(w) + alpha) - log(N + alpha * V)
    denominator_log = np.log(total_words + smoothing_alpha * vocab_size)
    
    word_log_probs = {
        word: np.log(count + smoothing_alpha) - denominator_log
        for word, count in word_counts.items()
    }
    
    # 为未登录词（OOV）计算一个固定的对数概率
    oov_log_prob = np.log(smoothing_alpha) - denominator_log
    word_log_probs['<oov>'] = oov_log_prob
    
    print("词概率模型构建完成。")
    return word_log_probs

def calculate_utterance_cost(text: str, word_log_probs: dict) -> float:
    """
    计算单个utterance的信息量成本。

    成本定义为句子的负对数概率： Cost(u) = -log(P(u))
    -log(P(u)) = -log(Π P(w_i)) = -Σ log(P(w_i))
    """
    log_prob = 0
    oov_log_prob = word_log_probs.get('<oov>', -20) # 如果<oov>不存在，给一个大的负数
    
    for word in text.split():
        log_prob += word_log_probs.get(word, oov_log_prob)
        
    # 成本是负对数概率
    return -log_prob

def main(args):
    """
    主函数
    """
    print("=== Utterance信息量成本计算工具 ===")
    
    # 1. 加载数据
    utterances_df = load_all_utterances()
    
    # 2. 预处理文本
    print("预处理文本...")
    utterances_df['processed_text'] = utterances_df['text'].apply(preprocess_text)
    
    # 3. 构建词概率模型
    word_log_probs = build_word_probability_model(
        utterances_df['processed_text'],
        smoothing_alpha=args.alpha
    )
    
    # 4. 计算每个utterance的成本
    print("计算每个utterance的成本...")
    utterances_df['information_cost'] = utterances_df['processed_text'].apply(
        lambda text: calculate_utterance_cost(text, word_log_probs)
    )
    
    # 5. 保存结果
    output_df = utterances_df[['utterance_id', 'information_cost']]
    
    # 使用全局配置的输出目录，但支持命令行覆盖
    if args.output_dir == str(COST_RESULTS_DIR):
        output_dir = COST_RESULTS_DIR
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = get_timestamp()
    output_path = output_dir / f"utterance_costs_{timestamp}.csv"
    
    output_df.to_csv(output_path, index=False)
    
    print(f"\n成本数据已成功保存到: {output_path}")
    
    # 显示一些统计信息
    print("\n成本统计摘要:")
    print(output_df['information_cost'].describe())
    
    print("\n成本最高的5个utterances:")
    highest_cost_samples = utterances_df.nlargest(5, 'information_cost')
    for _, row in highest_cost_samples.iterrows():
        print(f"  Cost: {row['information_cost']:.2f}, Text: '{row['text']}'")
        
    print("\n成本最低的5个utterances:")
    lowest_cost_samples = utterances_df.nsmallest(5, 'information_cost')
    for _, row in lowest_cost_samples.iterrows():
        print(f"  Cost: {row['information_cost']:.2f}, Text: '{row['text']}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='计算Utterance的信息量成本')
    
    parser.add_argument('--alpha', type=float, default=COST_PARAMS['smoothing_alpha'],
                        help=f'拉普拉斯平滑的alpha值，默认：{COST_PARAMS["smoothing_alpha"]}')
                        
    parser.add_argument('--output-dir', type=str, default=str(COST_RESULTS_DIR),
                        help=f'保存成本文件的目录，默认：{COST_RESULTS_DIR}')

    args = parser.parse_args()
    main(args) 