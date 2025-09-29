#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#使用示例：python models/rsa/generate_rsa.py --max-conversations 100 --batch-size 64
#参数可调节，最大对话数为0时即处理全部对话

"""
RSA
"""

import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
from tqdm import tqdm
import torch
import torch.nn.functional as F

from config import (
    get_current_file_paths,
    RSA_PARAMS,
    RSA_RESULTS_DIR,
    RUNTIME_PARAMS
)

# 确保评估结果目录存在
RSA_RESULTS_DIR.mkdir(exist_ok=True)

def clear_old_results(): 
    """清理旧结果文件"""
    old_files = list(RSA_RESULTS_DIR.glob("rsa_evaluation_*.csv"))
    for file in old_files:
        file.unlink()

# GPU设备检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class GPUCorrectedRSA:
    """GPU加速0731"""
    
    def __init__(self, meaning_space, utterance_space, cluster_labels, costs_df, utterance_ids, alpha=0.8, fusion_weight=0.5):
        self.device = device
        self.alpha = alpha
        self.fusion_weight = fusion_weight
        
        # 转换为张量
        self.meaning_space = torch.from_numpy(meaning_space).float().to(self.device)
        self.utterance_space = torch.from_numpy(utterance_space).float().to(self.device)
        
        # 保存聚类标签（用于索引）
        self.cluster_labels = cluster_labels
        
        # 计算全局先验信念（用于字面听者）
        cluster_counts = np.bincount(cluster_labels, minlength=len(meaning_space))
        global_prior = cluster_counts / np.sum(cluster_counts)
        self.global_prior = torch.from_numpy(global_prior).float().to(self.device)
        
        # 处理cost参数
        self.setup_costs(costs_df, utterance_ids)
        
        # 预计算字面听者概率
        self.compute_literal_listener()
    
    def setup_costs(self, costs_df, utterance_ids):
        """设置成本函数"""
        if costs_df is not None and utterance_ids is not None:
            costs_map = costs_df.set_index('utterance_id')['information_cost']
            costs = costs_map.reindex(utterance_ids).fillna(costs_map.mean()).values
            self.costs = torch.from_numpy(costs).float().to(self.device)
        else:
            self.costs = torch.zeros(len(self.utterance_space)).float().to(self.device)
    
    def compute_literal_listener(self):
        """计算字面听者概率矩阵（delta=1）"""
        n_utterances = len(self.utterance_space)
        n_meanings = len(self.meaning_space)
        
        self.literal_listener_probs = torch.zeros(n_utterances, n_meanings).to(self.device)
        cluster_labels_tensor = torch.from_numpy(self.cluster_labels).long().to(self.device)
        
        for meaning_idx in range(n_meanings):
            meaning_mask = (cluster_labels_tensor == meaning_idx)
            self.literal_listener_probs[meaning_mask, meaning_idx] = 1.0  # delta=1
    
    def pragmatic_speaker(self, meaning_idx):
        """计算语用说者概率"""
        # 使用预计算的字面听者概率（delta=1）
        literal_probs = self.literal_listener_probs[:, meaning_idx]
        info_utility = torch.log(torch.clamp(literal_probs, min=1e-10))
        utility = info_utility - self.costs
        return F.softmax(self.alpha * utility, dim=0)
    
    def pragmatic_listener(self, utterance_idx, prior):
        """计算语用听者概率"""
        n_meanings = len(self.meaning_space)
        
        # 一次性计算所有meaning的语用说者概率
        all_speaker_probs = torch.zeros(n_meanings, len(self.utterance_space), device=self.device)
        
        for m_idx in range(n_meanings):
            all_speaker_probs[m_idx] = self.pragmatic_speaker(m_idx)
        
        # 贝叶斯更新
        joint_probs = all_speaker_probs[:, utterance_idx] * prior
        
        total_prob = torch.sum(joint_probs)
        if total_prob > 0:
            return joint_probs / total_prob
        return prior.clone()
    
    def compute_dynamic_prior(self, previous_belief=None):
        """计算动态先验信念（融合全局先验和上一轮信念）"""
        if previous_belief is None:
            return self.global_prior
        
        previous_belief_tensor = torch.from_numpy(previous_belief).float().to(self.device) if isinstance(previous_belief, np.ndarray) else previous_belief
        
        dynamic_prior = (self.fusion_weight * self.global_prior + 
                        (1 - self.fusion_weight) * previous_belief_tensor)
        
        # 重新归一化
        total_prob = torch.sum(dynamic_prior)
        if total_prob > 0:
            return dynamic_prior / total_prob
        return dynamic_prior
    
    def process_conversation_sequence(self, utterance_indices):
        """处理对话序列（动态先验版本）"""
        belief_sequence = [self.global_prior.clone()]
        
        for i, u_idx in enumerate(utterance_indices):
            # 使用动态先验，第一轮时 previous_belief 为 None，将自动使用全局先验
            current_prior = self.compute_dynamic_prior(belief_sequence[-1] if i > 0 else None)
            
            # 使用当前先验计算B对A当前utterance的语用听者信念
            pragmatic_result = self.pragmatic_listener(u_idx, current_prior)
            belief_sequence.append(pragmatic_result.clone())
        
        return belief_sequence


def find_latest_cost_file(cost_dir: Path) -> Path:
    """查找最新cost文件"""
    files = list(cost_dir.glob("utterance_costs_*.csv"))
    return max(files, key=lambda f: f.stat().st_mtime)

def load_data():
    """加载数据"""
    file_paths = get_current_file_paths()
    
    costs_df = pd.read_csv(file_paths["cost_file"])
    embeddings = np.load(file_paths["embeddings_file"])
    metadata = pd.read_csv(file_paths["metadata_file"])
    
    with open(file_paths["cluster_model_file"], 'rb') as f:
        cluster_model = pickle.load(f)
    cluster_centers = cluster_model.cluster_centers_
    
    if file_paths["cluster_labels_file"] is not None and file_paths["cluster_labels_file"].exists():
        cluster_labels = np.load(file_paths["cluster_labels_file"])
    else:
        cluster_labels = cluster_model.predict(embeddings)

    return embeddings, metadata, cluster_centers, cluster_labels, costs_df

def process_conversations_batch(conversations_batch, gpu_rsa, utterance_id_to_idx, cluster_labels):
    """批量处理对话"""
    results = []
    
    for conv_id, dialogue_df in conversations_batch:
        dialogue_df = dialogue_df.sort_values('turn_id')
        a_utterances = dialogue_df[dialogue_df['speaker'] == 'Persona_A'].sort_values('turn_id')
        reference_row = dialogue_df[dialogue_df['speaker'] == 'Persona_B_reference']
        
        if len(reference_row) == 0 or len(a_utterances) == 0:
            continue
            
        reference_row = reference_row.iloc[0]
        
        a_indices = []
        for _, a_row in a_utterances.iterrows():
            u_idx = utterance_id_to_idx.get(a_row['utterance_id'])
            if u_idx is not None:
                a_indices.append(u_idx)
        
        if not a_indices:
            continue
        
        belief_sequence = gpu_rsa.process_conversation_sequence(a_indices)
        
        ref_id = reference_row['utterance_id']
        ref_idx = utterance_id_to_idx.get(ref_id)
        if ref_idx is None:
            continue
        
        final_belief = belief_sequence[-1].cpu().numpy()
        initial_belief = belief_sequence[0].cpu().numpy()
        
        predicted_meaning_idx = np.argmax(final_belief)
        actual_meaning_idx = cluster_labels[ref_idx]
        
        sorted_indices = np.argsort(final_belief)[::-1]
        actual_meaning_rank = np.where(sorted_indices == actual_meaning_idx)[0][0] + 1
        
        final_entropy = -np.sum(final_belief * np.log(final_belief + 1e-9))
        initial_entropy = -np.sum(initial_belief * np.log(initial_belief + 1e-9))
        
        results.append({
            'conv_id': conv_id,
            'predicted_meaning_idx': predicted_meaning_idx,
            'actual_meaning_idx': actual_meaning_idx,
            'top_1_accuracy': 1 if actual_meaning_rank <= 1 else 0,
            'top_3_accuracy': 1 if actual_meaning_rank <= 3 else 0,
            'top_5_accuracy': 1 if actual_meaning_rank <= 5 else 0,
            'initial_entropy': initial_entropy,
            'final_entropy': final_entropy,
            'num_A_utterances': len(a_utterances)
        })
    
    return results

def test_gpu_corrected_rsa(args):
    """测试RSA模型"""
    clear_old_results()
    
    data = load_data()
    embeddings, metadata, cluster_centers, cluster_labels, costs_df = data
    
    utterance_id_to_idx = {uid: i for i, uid in enumerate(metadata['utterance_id'])}

    gpu_rsa = GPUCorrectedRSA(
        meaning_space=cluster_centers,
        utterance_space=embeddings,
        cluster_labels=cluster_labels,
        costs_df=costs_df,
        utterance_ids=metadata['utterance_id'].tolist(),
        alpha=RSA_PARAMS['alpha'],
        fusion_weight=RSA_PARAMS['fusion_weight'] # 使用配置中的融合权重
    )
    
    print(f"模型初始化完成:")
    print(f" 意义空间维度: {gpu_rsa.meaning_space.shape}")
    print(f" 言说空间维度: {gpu_rsa.utterance_space.shape}")
    global_prior_cpu = gpu_rsa.global_prior.cpu().numpy()
    print(f" 全局先验信念熵: {-np.sum(global_prior_cpu * np.log(global_prior_cpu + 1e-9)):.3f}")

    dialogues = metadata.groupby('conv_id')
    max_conversations = args.max_conversations if args.max_conversations > 0 else len(dialogues)
    
    import random
    all_conv_ids = list(dialogues.groups.keys())
    if max_conversations < len(all_conv_ids):
        conv_ids = random.sample(all_conv_ids, max_conversations)
        print(f"随机选择 {max_conversations} 个对话进行处理")
    else:
        conv_ids = all_conv_ids
        print(f"处理全部 {len(conv_ids)} 个对话")
    
    print(f"开始处理 {len(conv_ids)} 个对话...")
    
    evaluation_results = []
    processed_count = 0
    batch_size = args.batch_size
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") #生成时间戳
    checkpoint_file = RSA_RESULTS_DIR / f"checkpoint_rsa_{timestamp}.csv" #检查点文件
    
    for i in tqdm(range(0, len(conv_ids), batch_size), desc="处理中"): #tqdm是进度条
        batch_conv_ids = conv_ids[i:i+batch_size]
        conversations_batch = [(conv_id, dialogues.get_group(conv_id)) for conv_id in batch_conv_ids]
        
        batch_results = process_conversations_batch(
            conversations_batch, gpu_rsa, utterance_id_to_idx, cluster_labels
        )
        evaluation_results.extend(batch_results)
        processed_count += len(batch_results)
        
        if processed_count % RUNTIME_PARAMS['checkpoint_frequency'] == 0:
            temp_df = pd.DataFrame(evaluation_results)
            temp_df.to_csv(checkpoint_file, index=False)
            print(f"\n检查点保存: 已处理{processed_count}个对话")
    
    results_df = pd.DataFrame(evaluation_results)
    
    print(f"\n=== RSA模型评估结果 ===")
    print(f"处理对话: {len(evaluation_results)}/{len(conv_ids)} (成功率: {len(evaluation_results)/len(conv_ids)*100:.1f}%)")
    print(f"Top-1准确率: {(results_df['top_1_accuracy'] == 1).mean()*100:.1f}%")
    print(f"Top-3准确率: {(results_df['top_3_accuracy'] == 1).mean()*100:.1f}%")
    print(f"Top-5准确率: {(results_df['top_5_accuracy'] == 1).mean()*100:.1f}%")
    print(f"平均初始熵: {results_df['initial_entropy'].mean():.3f}")
    print(f"平均最终熵: {results_df['final_entropy'].mean():.3f}")
    
    output_filename = RSA_RESULTS_DIR / f"rsa_evaluation_{timestamp}.csv"
    results_df.to_csv(output_filename, index=False)
    
    print(f"\n结果保存到: {output_filename}")
    
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="RSA模型评估")
    parser.add_argument('--max-conversations', type=int, default=RUNTIME_PARAMS['default_max_conversations'], 
                       help=f'最大处理对话数量去global_config中调')
    parser.add_argument('--batch-size', type=int, default=RUNTIME_PARAMS['default_batch_size'],
                       help=f'批处理大小也一样去global_config中调')
    args = parser.parse_args()
    
    print(f"=== RSA模型配置 ===")
    print(f"理性参数alpha: {RSA_PARAMS['alpha']}")
    print(f"融合权重: {RSA_PARAMS['fusion_weight']}")
    print(f"处理对话数: {'全部' if args.max_conversations == 0 else args.max_conversations}")
    print(f"批处理大小: {args.batch_size}")
    print()
    
    test_gpu_corrected_rsa(args)

if __name__ == "__main__":
    main() 