#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#python preprocessing/extract_utterances.py

#这是个预处理脚本，几乎是一次性的、过程性的使用
#同样，生成结果时会删除旧的文件！

"""
PersonaChat 对话数据抽取脚本
用于从PersonaChat数据集中抽取每一句发言，以供词向量分析和意义分类使用
"""

import pandas as pd
import json
import os
import sys
from typing import List, Dict, Any
import re
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
PREPROCESSING_DIR = Path(__file__).resolve().parent
RSAE_DIR = PREPROCESSING_DIR.parent
sys.path.insert(0, str(RSAE_DIR))

# 从全局配置导入设置
from global_config import (
    PERSONACHAT_DATA_FILE, EXTRACTED_DATA_DIR, 
    FILE_SEARCH_PATHS, EXTRACTION_CONFIG
)

def load_local_dataset(parquet_path: str) -> pd.DataFrame: # 加载本地的parquet文件
    
    """
    加载本地的parquet文件
    """
    try:
        df = pd.read_parquet(parquet_path)
        print(f"成功加载数据集，共 {len(df)} 条对话")
        return df
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return None

def clean_text(text: str) -> str:
    """
    清理文本，去除多余的空格和特殊字符
    """
    if not isinstance(text, str):
        return str(text)
    
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 去除一些常见的聊天特殊标记
    text = re.sub(r'<.*?>', '', text)  # 去除尖括号标记
    
    return text

def filter_dialogues_with_reference(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤出有参考回复的对话，并移除不合规的对话
    """
    # 检查reference字段是否存在且非空
    has_reference = df['reference'].apply(lambda x: isinstance(x, str) and x.strip() != '')
    
    # 移除特定的不合规对话（不以A结尾的对话）
    invalid_conv_ids = ['conv63608', 'conv50220', 'conv63602', 'conv63603', 'conv63605', 'conv63609']
    is_valid_conv = ~df['conv_id'].isin(invalid_conv_ids)
    
    # 综合过滤条件
    filtered_df = df[has_reference & is_valid_conv]
    
    print(f"过滤前对话数: {len(df)}")
    print(f"移除无参考回复的对话: {len(df) - df[has_reference].shape[0]} 个")
    print(f"移除不合规对话 {invalid_conv_ids}: {len(df) - df[is_valid_conv].shape[0]} 个")
    print(f"过滤后对话数: {len(filtered_df)}")
    print(f"总共移除: {len(df) - len(filtered_df)} 个对话")
    
    return filtered_df

def extract_utterances_from_personachat(df: pd.DataFrame, include_persona: bool = True) -> List[Dict[str, Any]]:
    """
    从PersonaChat数据集中抽取所有发言
    
    Args:
        df: PersonaChat数据集
        include_persona: 是否包含persona描述
    
    官方说明的数据集结构：
    - conv_id: 对话唯一标识符
    - persona_b: 仅有角色B的人格描述列表
    - dialogue: 对话轮次列表，A开始A结束
    - reference: 基于角色B人格的参考回复
    """
    all_utterances = []
    utterance_id = 0
    
    print("开始抽取发言...")
    
    for idx, row in df.iterrows():
        conv_id = row['conv_id']  # 对话唯一标识符
        
        # 1. 抽取dialogue中的每句对话
        if 'dialogue' in row and hasattr(row['dialogue'], '__len__'):
            dialogue_array = row['dialogue']
            
            for turn_idx, utterance_text in enumerate(dialogue_array):
                if isinstance(utterance_text, str) and utterance_text.strip():
                    # 解析说话人和内容
                    speaker = None
                    clean_utterance_text = utterance_text.strip()
                    
                    if clean_utterance_text.startswith('Persona A:'):
                        speaker = 'Persona_A'
                        clean_utterance_text = clean_utterance_text[len('Persona A:'):].strip()
                    elif clean_utterance_text.startswith('Persona B:'):
                        speaker = 'Persona_B'
                        clean_utterance_text = clean_utterance_text[len('Persona B:'):].strip()
                    else:
                        # 根据官方说明：对话A开始A结束，所以偶数轮次是A，奇数轮次是B
                        speaker = 'Persona_A' if turn_idx % 2 == 0 else 'Persona_B'
                    
                    # 清理文本
                    cleaned_utterance = clean_text(clean_utterance_text)
                    
                    if cleaned_utterance:  # 只保留非空发言
                        all_utterances.append({
                            'utterance_id': utterance_id,
                            'conv_id': conv_id,
                            'turn_id': turn_idx,
                            'speaker': speaker,
                            'text': cleaned_utterance,
                            'source': 'dialogue',
                            'length': len(cleaned_utterance.split()),
                            'original_text': utterance_text.strip()
                        })
                        utterance_id += 1
        
        # 2. 抽取reference回复（B的参考回答）
        if 'reference' in row and isinstance(row['reference'], str) and row['reference'].strip():
            reference_text = row['reference'].strip()
            cleaned_reference = clean_text(reference_text)
            
            if cleaned_reference:
                all_utterances.append({
                    'utterance_id': utterance_id,
                    'conv_id': conv_id,
                    'turn_id': -1,  # reference没有轮次概念
                    'speaker': 'Persona_B_reference',
                    'text': cleaned_reference,
                    'source': 'reference',
                    'length': len(cleaned_reference.split()),
                    'original_text': reference_text
                })
                utterance_id += 1
        
        # 3. 抽取persona_b（仅角色B的人格描述）
        if include_persona and 'persona_b' in row and hasattr(row['persona_b'], '__len__'):
            persona_array = row['persona_b']
            
            for pers_idx, persona_text in enumerate(persona_array):
                if isinstance(persona_text, str) and persona_text.strip():
                    cleaned_persona = clean_text(persona_text.strip())
                    
                    if cleaned_persona:
                        all_utterances.append({
                            'utterance_id': utterance_id,
                            'conv_id': conv_id,
                            'turn_id': -2,  # 人格描述用-2标识
                            'speaker': 'Persona_B_persona',
                            'text': cleaned_persona,
                            'source': 'persona_description',
                            'length': len(cleaned_persona.split()),
                            'original_text': persona_text.strip()
                        })
                        utterance_id += 1
        
        if (idx + 1) % 1000 == 0:
            print(f"已处理 {idx + 1} 条对话，抽取了 {len(all_utterances)} 句发言")
    
    print(f"抽取完成！总共抽取了 {len(all_utterances)} 句发言")
    return all_utterances

def save_utterances(utterances: List[Dict]):
    """保存抽取的发言到不同格式的文件中"""
    # 确保目录存在
    EXTRACTED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 转换为DataFrame
    df_utterances = pd.DataFrame(utterances)
    
    # 保存CSV文件
    csv_path = EXTRACTED_DATA_DIR / "extracted_utterances.csv"
    df_utterances.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✅ 已保存 {len(df_utterances)} 个utterances到: {csv_path}")
    
    # 保存JSON文件
    json_path = EXTRACTED_DATA_DIR / "extracted_utterances.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(utterances, f, ensure_ascii=False, indent=2)
    print(f"✅ 已保存 {len(utterances)} 个utterances到: {json_path}")
    
    # 打印统计信息
    print("\n=== 抽取统计 ===")
    print(f"总发言数: {len(utterances)}")
    
    if len(utterances) > 0:
        print("按来源分布:")
        source_counts = df_utterances['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count} 句")
        
        print("按说话人分布:")
        speaker_counts = df_utterances['speaker'].value_counts()
        for speaker, count in speaker_counts.items():
            print(f"  {speaker}: {count} 句")
        
        print(f"平均句子长度: {df_utterances['length'].mean():.2f} 词")
        print(f"句子长度范围: {df_utterances['length'].min()} - {df_utterances['length'].max()} 词")
        
        # 按对话ID统计
        conv_counts = df_utterances['conv_id'].value_counts()
        print(f"涉及对话数: {len(conv_counts)} 个")
        print(f"平均每个对话的发言数: {conv_counts.mean():.2f} 句")
    else:
        print("没有抽取到任何发言，请检查数据格式！")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="PersonaChat 发言抽取工具")
    parser.add_argument("--no-filter-reference", action="store_true", help="不过滤无参考回复的对话（默认会过滤）")
    parser.add_argument("--include-persona", action="store_true", help="包含persona描述（默认不包含）")
    args = parser.parse_args()
    
    # 默认启用过滤功能
    filter_reference = not args.no_filter_reference
    include_persona = args.include_persona
    
    print("=== PersonaChat 发言抽取工具 ===")
    print(f"过滤无参考回复对话: {'是' if filter_reference else '否'}")
    print(f"过滤不合规对话: {'是' if filter_reference else '否'}")
    print(f"包含persona描述: {'是' if include_persona else '否'}")
    
    # 检查数据文件
    if not PERSONACHAT_DATA_FILE.exists():
        print(f"错误: 找不到数据文件 {PERSONACHAT_DATA_FILE}")
        return
    
    print(f"找到数据文件: {PERSONACHAT_DATA_FILE}")
    
    # 加载数据集
    df = load_local_dataset(str(PERSONACHAT_DATA_FILE))
    if df is None:
        return
    
    # 显示数据集基本信息
    print(f"数据集列名: {list(df.columns)}")
    print(f"数据集形状: {df.shape}")
    
    # 过滤无参考回复的对话和不合规对话（默认启用）
    if filter_reference:
        df = filter_dialogues_with_reference(df)
    
    # 查看第一条数据的结构
    print("\n第一条数据样例:")
    first_row = df.iloc[0]
    for col in df.columns:
        value = first_row[col]
        if isinstance(value, list):
            print(f"  {col}: list (长度: {len(value)}) - {value[:2]}...")
        elif hasattr(value, '__len__') and not isinstance(value, str):
            print(f"  {col}: {type(value).__name__} (长度: {len(value)}) - {value[:2] if len(value) > 0 else 'empty'}")
        else:
            print(f"  {col}: {type(value).__name__} - {str(value)[:100]}...")
    
    # 详细查看dialogue字段内容
    print("\n=== 详细查看dialogue字段 ===")
    dialogue_sample = first_row['dialogue']
    print(f"dialogue类型: {type(dialogue_sample)}")
    if hasattr(dialogue_sample, '__len__'):
        print(f"dialogue长度: {len(dialogue_sample)}")
        print("前5个对话内容:")
        for i, turn in enumerate(dialogue_sample[:5]):
            print(f"  {i}: {turn}")
    
    # 详细查看persona_b字段内容
    print("\n=== 详细查看persona_b字段 ===")
    persona_sample = first_row['persona_b']
    print(f"persona_b类型: {type(persona_sample)}")
    if hasattr(persona_sample, '__len__'):
        print(f"persona_b长度: {len(persona_sample)}")
        print("人格描述内容:")
        for i, persona in enumerate(persona_sample):
            print(f"  {i}: {persona}")
    
    # 详细查看reference字段内容
    print("\n=== 详细查看reference字段 ===")
    reference_sample = first_row['reference']
    print(f"reference类型: {type(reference_sample)}")
    print(f"reference内容: {reference_sample}")
            
    print("\n=== 开始抽取utterances ===")
    
    # 抽取发言
    utterances = extract_utterances_from_personachat(df, include_persona=include_persona)
    
    # 保存结果
    save_utterances(utterances)
    
    print("\n发言抽取完成！")

if __name__ == "__main__":
    main() 