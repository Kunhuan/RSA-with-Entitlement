#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#python preprocessing/query_utterances.py

#这个使用后在终端根据指示输入对应内容查询即可


"""
PersonaChat 发言查询工具
用于查询特定对话中的utterances信息
"""

import pandas as pd
import json
import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Any

# 添加上级目录到路径，以便导入global_config
sys.path.append(str(Path(__file__).parent.parent))
from global_config import EXTRACTED_UTTERANCES_CSV

def load_utterances() -> pd.DataFrame:
    """加载已提取的utterances数据"""
    try:
        if not EXTRACTED_UTTERANCES_CSV.exists():
            print(f"错误: 找不到文件 {EXTRACTED_UTTERANCES_CSV}")
            print("请确保已运行extract_utterances.py生成数据")
            return None
            
        print(f"找到utterances文件: {EXTRACTED_UTTERANCES_CSV}")
        df = pd.read_csv(EXTRACTED_UTTERANCES_CSV)
        print(f"成功加载 {len(df)} 条utterances")
        return df
        
    except Exception as e:
        print(f"加载utterances数据失败: {e}")
        return None

def show_conversation_summary(df: pd.DataFrame):
    """
    显示概览信息
    """
    print("\n=== 对话概览 ===")
    
    # 按对话统计
    conv_stats = df.groupby('conv_id').agg({
        'utterance_id': 'count',
        'speaker': lambda x: list(x.unique()),
        'source': lambda x: list(x.unique())
    }).rename(columns={'utterance_id': 'total_utterances'})
    
    print(f"总对话数: {len(conv_stats)}")
    print(f"总utterances数: {len(df)}")
    print(f"平均每个对话的utterances数: {conv_stats['total_utterances'].mean():.2f}")
    
    # 按说话人统计
    print("\n=== 按说话人统计 ===")
    speaker_counts = df['speaker'].value_counts()
    persona_a_count = speaker_counts.get('Persona_A', 0)
    persona_b_count = speaker_counts.get('Persona_B', 0)
    persona_b_ref_count = speaker_counts.get('Persona_B_reference', 0)

    
    print(f"Persona A的utterances数量: {persona_a_count}")
    print(f"Persona B的utterances数量: {persona_b_count}")
    print(f"Persona B的参考答案数量: {persona_b_ref_count}")

    
    # 按来源统计
    print("\n=== 按来源统计 ===")
    source_counts = df['source'].value_counts()
    for source, count in source_counts.items():
        print(f"{source}: {count} 条")
    
    
    # 显示utterances数量分布
    utterance_distribution = conv_stats['total_utterances'].value_counts().sort_index()
    print(f"\nutterances数量分布:")
    
    # 按频次排序显示最多的5个
    top_distributions = utterance_distribution.sort_values(ascending=False).head(10)
    print("最多的10个utterances数量分布:")
    for count, freq in top_distributions.items():
        print(f"  {count}个utterances的对话: {freq} 个")
    
    
    print(f"\n总共有 {len(utterance_distribution)} 种不同的utterances数量分布")


def query_conversation(df: pd.DataFrame, conv_id: str):
    """
    查询特定对话的详细信息
    """
    # 筛选特定对话
    conv_data = df[df['conv_id'] == conv_id].copy()
    
    if len(conv_data) == 0:
        print(f"错误: 找不到对话 ID '{conv_id}'")
        return
    
    print(f"\n=== 对话 {conv_id} 详细信息 ===")
    print(f"总utterances数: {len(conv_data)}")
    
    # 按来源分类统计
    print("\n按来源分布:")
    source_counts = conv_data['source'].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count} 条")
    
    # 按说话人分类统计
    print("\n按说话人分布:")
    speaker_counts = conv_data['speaker'].value_counts()
    for speaker, count in speaker_counts.items():
        print(f"  {speaker}: {count} 条")
    
    # 显示对话内容（按turn_id排序）
    print("\n=== 对话内容 ===")
    
    # 分别显示dialogue、reference和persona
    dialogue_data = conv_data[conv_data['source'] == 'dialogue'].sort_values('turn_id')
    reference_data = conv_data[conv_data['source'] == 'reference']
    persona_data = conv_data[conv_data['source'] == 'persona_description']
    
    # 显示对话轮次
    if len(dialogue_data) > 0:
        print("\n对话轮次:")
        for _, row in dialogue_data.iterrows():
            print(f"  轮次 {row['turn_id']} ({row['speaker']}): {row['text']}")
    
    # 显示参考回复
    if len(reference_data) > 0:
        print("\n参考回复 (Persona_B_reference):")
        for _, row in reference_data.iterrows():
            print(f"  {row['text']}")
    
    # 显示人格描述
    if len(persona_data) > 0:
        print("\n人格描述 (Persona_B):")
        for _, row in persona_data.iterrows():
            print(f"  {row['text']}")

def query_by_number(df: pd.DataFrame, conv_number: int):
    """
    根据对话编号查询（第N个对话）
    """
    unique_convs = df['conv_id'].unique()
    
    if conv_number < 1 or conv_number > len(unique_convs):
        print(f"错误: 对话编号应在 1 到 {len(unique_convs)} 之间")
        return
    
    conv_id = unique_convs[conv_number - 1]
    print(f"第 {conv_number} 个对话的ID是: {conv_id}")
    query_conversation(df, conv_id)

def search_conversations(df: pd.DataFrame, keyword: str):
    """
    搜索包含特定关键词的对话
    """
    matching_convs = df[df['text'].str.contains(keyword, case=False, na=False)]['conv_id'].unique()
    
    print(f"\n包含关键词 '{keyword}' 的对话:")
    if len(matching_convs) == 0:
        print("  没有找到匹配的对话")
        return
    
    for i, conv_id in enumerate(matching_convs[:10]):  # 只显示前10个
        conv_data = df[df['conv_id'] == conv_id]
        matching_utterances = conv_data[conv_data['text'].str.contains(keyword, case=False, na=False)]
        print(f"  {i+1}. {conv_id}: {len(matching_utterances)} 个匹配的utterances")

def main():
    """
    主函数 - 交互式查询
    """
    print("=== PersonaChat Utterances 查询工具 ===")
    
    # 加载数据
    df = load_utterances()
    if df is None:
        return
    
    # 显示使用说明
    print("\n可用命令:")
    print("  summary                          - 显示处理后数据集概览")
    print("  conv <conv_id>，如conv conv8747  - 查看特定对话详情")
    print("  num <number>                     - 查看第N个对话")
    print("  search <keyword>                 - 搜索包含关键词的对话")
    print("  list                             - 随机列出10个对话ID")
    print("  exit                             - 退出程序")
    
    while True:
        try:
            command = input("\n请输入命令: ").strip()
            
            if command in ['exit']:
                print("退出")
                break
            elif command == 'summary':
                show_conversation_summary(df)
            elif command == 'list':
                unique_convs = df['conv_id'].unique()
                if len(unique_convs) > 0:
                    random_convs = random.sample(list(unique_convs), min(10, len(unique_convs)))
                    print(f"\n随机显示10个对话ID:")
                    for i, conv_id in enumerate(random_convs):
                        print(f"  {i+1}. {conv_id}")
                else:
                    print("  没有可显示的对话ID")
            elif command.startswith('conv '):
                conv_id = command[5:].strip()
                query_conversation(df, conv_id)
            elif command.startswith('num '):
                try:
                    conv_number = int(command[4:].strip())
                    query_by_number(df, conv_number)
                except ValueError:
                    print("错误: 请输入有效的数字")
            elif command.startswith('search '):
                keyword = command[7:].strip()
                if keyword:
                    search_conversations(df, keyword)
                else:
                    print("错误: 请提供搜索关键词")
            elif command == '':
                continue
            else:
                print("未知命令，请重新输入；若需要重新运行本文档，则需要输出exit退出先")
                
        except KeyboardInterrupt:
            print("\n\n程序中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    # 支持命令行参数
    if len(sys.argv) > 1:
        df = load_utterances()
        if df is not None:
            if sys.argv[1] == 'summary':
                show_conversation_summary(df)
            elif sys.argv[1] == 'conv' and len(sys.argv) > 2:
                query_conversation(df, sys.argv[2])
            elif sys.argv[1] == 'num' and len(sys.argv) > 2:
                try:
                    query_by_number(df, int(sys.argv[2]))
                except ValueError:
                    print("错误: 请提供有效的数字")
            else:
                print("用法: python query_utterances.py [summary|conv <conv_id>|num <number>]")
    else:
        main() 