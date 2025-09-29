#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#python models/rsae/generate_rsae.py --max-conversations 100 --batch-size 64
#参数可调节，最大对话数为0时即处理全部对话

"""注意！！！会默认删除旧的生成文件，注意备份"""

"""
RSAE
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
    RSAE_PARAMS,
    RSAE_RESULTS_DIR,
    RUNTIME_PARAMS
)

# 确保评估结果目录存在
RSAE_RESULTS_DIR.mkdir(exist_ok=True)

def clear_old_results():
    """清理旧结果文件"""
    old_files = list(RSAE_RESULTS_DIR.glob("rsae_evaluation_*.csv"))
    for file in old_files:
        file.unlink()

# GPU设备检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class GPUAcceleratedRSAE:
    """RSAE"""
    
    def __init__(self, meaning_space, utterance_space, cluster_labels, costs_df, utterance_ids, 
                 alpha=0.8, initial_E=0.8, lambda_E=0.8, beta=0.3, fusion_weight=0.5,
                 E_min=0.1, E_max=1.0, performance_weights=None, S_score=0.8):
        self.device = device
        self.alpha = alpha
        self.fusion_weight = fusion_weight
        
        # RSAE参数
        self.current_E = initial_E
        self.lambda_E = lambda_E
        self.beta = beta
        self.performance_weights = performance_weights if performance_weights else {
            "w_1": 0.35, "w_2": 0.35, "w_3": 0.15, "w_4": 0.15
        }
        self.S_score = S_score
        
        # 控制参数
        self.E_min = E_min
        self.E_max = E_max
        
        # 数据设置
        self.meaning_space = meaning_space
        self.utterance_space = utterance_space
        self.cluster_labels = cluster_labels
        self.cluster_labels_tensor = torch.from_numpy(cluster_labels).long().to(self.device)
        
        # 计算全局先验信念
        cluster_counts = np.bincount(cluster_labels, minlength=len(meaning_space))
        global_prior = cluster_counts / np.sum(cluster_counts)
        self.global_prior = torch.from_numpy(global_prior).float().to(self.device)
        
        # 处理成本
        self.setup_costs(costs_df, utterance_ids)
        
        # 预计算字面听者概率（简化版）
        self.compute_literal_listener()
        
        # 记录历史
        self.E_history = [initial_E]
        self.score_history = []
    
    def setup_costs(self, costs_df, utterance_ids):
        """设置成本函数"""
        if costs_df is not None and utterance_ids is not None:
            costs_map = costs_df.set_index('utterance_id')['information_cost']
            costs = costs_map.reindex(utterance_ids).fillna(costs_map.mean()).values
            self.costs = torch.from_numpy(costs).float().to(self.device)
        else:
            self.costs = torch.zeros(len(self.utterance_space)).float().to(self.device)
    
    def compute_literal_listener(self):
        """计算字面听者概率矩阵 P_L0(m|u)"""
        n_utterances = len(self.utterance_space)
        n_meanings = len(self.meaning_space)
        
        self.literal_listener_probs = torch.zeros(n_utterances, n_meanings).to(self.device)
        
        for meaning_idx in range(n_meanings):
            meaning_mask = (self.cluster_labels_tensor == meaning_idx)
            self.literal_listener_probs[meaning_mask, meaning_idx] = 1.0
    
    def get_literal_listener_prob(self, utterance_idx, meaning_idx):
        """获取单个字面听者概率 P_L0(m|u)"""
        return self.literal_listener_probs[utterance_idx, meaning_idx]

    def pragmatic_speaker(self, meaning_idx, E=None):
        """计算语用说者概率 P_S1(u|m)"""
        if E is None:
            E = self.current_E
            
        # 计算效用函数 U(u,m) = log(P_L0(m|u)) - cost(u)
        literal_probs = self.literal_listener_probs[:, meaning_idx]
        info_utility = torch.log(torch.clamp(literal_probs, min=1e-10))
        utility = info_utility - self.costs
        
        # 标准语用说者概率 P_standard(u|m) = exp(α × U(u,m)) / Σ exp(α × U(u',m))
        standard_speaker_probs = F.softmax(self.alpha * utility, dim=0)
        
        # 第二项：(1-E) × (α × β × U(u,m))
        # 注意：这里需要将效用转换为概率分布
        utility_component = self.alpha * self.beta * utility
        utility_probs = F.softmax(utility_component, dim=0)
        
        # 最终概率：P_S1(u|m) = E × P_standard(u|m) + (1-E) × utility_probs
        pragmatic_speaker_probs = E * standard_speaker_probs + (1 - E) * utility_probs
        
        # 归一化确保概率和为1
        total_prob = torch.sum(pragmatic_speaker_probs)
        if total_prob > 0:
            return pragmatic_speaker_probs / total_prob
        return pragmatic_speaker_probs
    
    def pragmatic_listener(self, utterance_idx, prior=None):
        """计算语用听者概率 P_L1(m|u)"""
        if prior is None:
            prior = self.global_prior
        
        prior_tensor = torch.from_numpy(prior).float().to(self.device) if isinstance(prior, np.ndarray) else prior
        
        n_meanings = len(self.meaning_space)
        all_speaker_probs = torch.zeros(n_meanings, len(self.utterance_space), device=self.device)
        
        for m_idx in range(n_meanings):
            all_speaker_probs[m_idx] = self.pragmatic_speaker(m_idx, self.current_E)
        
        # 贝叶斯更新：P_L1(m|u) = P_S1(u|m) * P(m) / sum
        joint_probs = all_speaker_probs[:, utterance_idx] * prior_tensor
        
        total_prob = torch.sum(joint_probs)
        if total_prob > 0:
            result = joint_probs / total_prob
        else:
            result = prior_tensor.clone()
        
        return result.cpu().numpy(), all_speaker_probs[:, utterance_idx]

    def evaluate_speaker_performance(self, utterance_idx, listener_belief, prior, speaker_probs_for_utterance, actual_meaning_idx):
        """评估说者表现"""
        # 1. 理解效果 (Information Gain)
        listener_entropy = -torch.sum(listener_belief * torch.log(listener_belief + 1e-9))
        prior_entropy = -torch.sum(prior * torch.log(prior + 1e-9))
        understanding_effect = torch.clamp(prior_entropy - listener_entropy, min=0).item()

        # 2. 表达效果 (Clarity Score) - P_S1(u|m) / P_L0(m|u)
        pragmatic_speaker_prob_for_meaning = speaker_probs_for_utterance[actual_meaning_idx].item()
        literal_listener_prob_for_meaning = self.get_literal_listener_prob(utterance_idx, actual_meaning_idx).item()
        
        expression_effect = 0.0
        if literal_listener_prob_for_meaning > 1e-9:
            expression_effect = pragmatic_speaker_prob_for_meaning / literal_listener_prob_for_meaning
        
        # 3. 成本效率
        utterance_cost = self.costs[utterance_idx].item()
        cost_efficiency = 0.0
        if utterance_cost > 1e-9:
            cost_efficiency = understanding_effect / utterance_cost
            
        # 4. 身份-主题契合度
        identity_fit_score = self.S_score

        # 综合评分
        w = self.performance_weights
        performance_score = (
            w.get("w_1", 0.35) * torch.clamp(torch.tensor(understanding_effect / 2.0), 0, 1).item() +
            w.get("w_2", 0.35) * torch.clamp(torch.tensor(expression_effect / 2.0), 0, 1).item() +
            w.get("w_3", 0.15) * torch.clamp(torch.tensor(cost_efficiency / 5.0), 0, 1).item() +
            w.get("w_4", 0.15) * identity_fit_score
        )
        
        return np.clip(performance_score, 0.0, 1.0)
    
    def update_entitlement_with_performance(self, previous_utterance_idx, 
                                          previous_B_prediction,
                                          speaker_probs_for_utterance,
                                          actual_meaning_idx):
        """更新资格参数"""
        performance_score = self.evaluate_speaker_performance(
            previous_utterance_idx, 
            torch.from_numpy(previous_B_prediction).float().to(self.device),
            self.global_prior,
            speaker_probs_for_utterance,
            actual_meaning_idx
        )
        
        new_E = (1 - self.lambda_E) * self.current_E + self.lambda_E * performance_score
        new_E = np.clip(new_E, self.E_min, self.E_max)
        
        self.current_E = new_E
        self.E_history.append(new_E)
        self.score_history.append(performance_score)
        
        return new_E
    
    def compute_dynamic_prior(self, previous_belief=None):
        """计算动态先验信念"""
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
    
    def process_conversation_sequence(self, utterance_indices, update_entitlement=True):
        """处理对话序列"""
        belief_sequence = [self.global_prior.clone()]
        E_sequence = [self.current_E]
        
        previous_utterance_idx = None
        previous_B_prediction = None
        previous_speaker_probs = None
        
        for i, u_idx in enumerate(utterance_indices):
            if update_entitlement and i > 0 and previous_utterance_idx is not None:
                self.update_entitlement_with_performance(
                    previous_utterance_idx=previous_utterance_idx,
                    previous_B_prediction=previous_B_prediction,
                    speaker_probs_for_utterance=previous_speaker_probs,
                    actual_meaning_idx=np.argmax(previous_B_prediction).item()
                )
                E_sequence.append(self.current_E)
            
            current_prior = self.compute_dynamic_prior(belief_sequence[-1] if i > 0 else None)
            
            pragmatic_result, speaker_probs = self.pragmatic_listener(u_idx, current_prior)
            belief_sequence.append(pragmatic_result.copy())
            
            previous_utterance_idx = u_idx
            previous_B_prediction = pragmatic_result.copy()
            previous_speaker_probs = speaker_probs.clone()
        
        return {
            'belief_sequence': belief_sequence,
            'E_sequence': E_sequence,
            'final_belief': belief_sequence[-1],
            'final_E': self.current_E
        }


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

def process_conversations_batch(conversations_batch, gpu_rsae, utterance_id_to_idx, cluster_labels, cluster_centers):
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
        
        initial_E = gpu_rsae.E_history[0]
        gpu_rsae.current_E = initial_E
        gpu_rsae.E_history = [initial_E]
        gpu_rsae.score_history = []
        
        rsae_result = gpu_rsae.process_conversation_sequence(a_indices)
        
        ref_id = reference_row['utterance_id']
        ref_idx = utterance_id_to_idx.get(ref_id)
        if ref_idx is None:
            continue
        
        final_belief = rsae_result['final_belief']
        initial_belief = rsae_result['belief_sequence'][0]
        
        predicted_meaning_idx = np.argmax(final_belief)
        actual_meaning_idx = cluster_labels[ref_idx]
        
        sorted_indices = np.argsort(final_belief)[::-1]
        actual_meaning_rank = np.where(sorted_indices == actual_meaning_idx)[0][0] + 1
        
        if isinstance(final_belief, torch.Tensor):
            final_belief_cpu = final_belief.cpu().numpy()
        else:
            final_belief_cpu = final_belief
            
        if isinstance(initial_belief, torch.Tensor):
            initial_belief_cpu = initial_belief.cpu().numpy()
        else:
            initial_belief_cpu = initial_belief
            
        final_entropy = -np.sum(final_belief_cpu * np.log(final_belief_cpu + 1e-9))
        initial_entropy = -np.sum(initial_belief_cpu * np.log(initial_belief_cpu + 1e-9))
        
        final_E = rsae_result['final_E']
        E_change = final_E - initial_E
        avg_score = np.mean(gpu_rsae.score_history) if gpu_rsae.score_history else 0.0
        
        results.append({  # 记录结果
            'conv_id': conv_id,
            'predicted_meaning_idx': predicted_meaning_idx,
            'actual_meaning_idx': actual_meaning_idx,
            'top_1_accuracy': 1 if actual_meaning_rank <= 1 else 0,
            'top_3_accuracy': 1 if actual_meaning_rank <= 3 else 0,
            'top_5_accuracy': 1 if actual_meaning_rank <= 5 else 0,
            'initial_entropy': initial_entropy,
            'final_entropy': final_entropy,
            'initial_E': initial_E,
            'final_E': final_E,
            'E_change': E_change,
            'avg_comprehensive_score': avg_score,
            'num_A_utterances': len(a_utterances)
        })
    
    return results

def test_gpu_rsae(args): 
    """测试RSAE模型"""
    clear_old_results()
    
    data = load_data()
    embeddings, metadata, cluster_centers, cluster_labels, costs_df = data
    
    utterance_id_to_idx = {uid: i for i, uid in enumerate(metadata['utterance_id'])}

    gpu_rsae = GPUAcceleratedRSAE(
        meaning_space=cluster_centers,
        utterance_space=embeddings,
        cluster_labels=cluster_labels,
        costs_df=costs_df,
        utterance_ids=metadata['utterance_id'].tolist(),
        alpha=RSAE_PARAMS['alpha'],
        initial_E=RSAE_PARAMS['initial_E'],
        lambda_E=RSAE_PARAMS['lambda_E'],
        beta=RSAE_PARAMS['beta'],
        fusion_weight=RSAE_PARAMS['fusion_weight'],
        E_min=RSAE_PARAMS['E_min'],
        E_max=RSAE_PARAMS['E_max'],
        performance_weights=RSAE_PARAMS.get('performance_weights'),
        S_score=RSAE_PARAMS.get('S_score')
    )
    
    print(f"模型初始化完成:")
    print(f"  意义空间维度: {gpu_rsae.meaning_space.shape}")
    print(f"  言说空间维度: {gpu_rsae.utterance_space.shape}")
    global_prior_cpu = gpu_rsae.global_prior.cpu().numpy()
    print(f"  全局先验信念熵: {-np.sum(global_prior_cpu * np.log(global_prior_cpu + 1e-9)):.3f}")
    print(f"  初始资格参数E: {gpu_rsae.current_E:.3f}")
    print(f"  资格更新学习率λ: {gpu_rsae.lambda_E:.3f}")
    print(f"  语用说者beta参数: {gpu_rsae.beta:.1f}")
    print(f"  动态先验融合权重: {gpu_rsae.fusion_weight:.1f}")

    dialogues = metadata.groupby('conv_id')
    max_conversations = args.max_conversations if args.max_conversations > 0 else len(dialogues)
    
    import random
    all_conv_ids = list(dialogues.groups.keys())
    if max_conversations < len(all_conv_ids):
        conv_ids = random.sample(all_conv_ids, max_conversations)
        print(f"随机选择了 {max_conversations} 个对话进行处理")
    else:
        conv_ids = all_conv_ids
        print(f"处理全部 {len(conv_ids)} 个对话")
    
    print(f"开始处理 {len(conv_ids)} 个对话...")
    
    evaluation_results = []
    processed_count = 0
    batch_size = args.batch_size
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = RSAE_RESULTS_DIR / f"checkpoint_rsae_{timestamp}.csv"
    
    for i in tqdm(range(0, len(conv_ids), batch_size), desc="处理中"):
        batch_conv_ids = conv_ids[i:i+batch_size]
        conversations_batch = [(conv_id, dialogues.get_group(conv_id)) for conv_id in batch_conv_ids]
        
        batch_results = process_conversations_batch(
            conversations_batch, gpu_rsae, utterance_id_to_idx, cluster_labels, cluster_centers
        )
        evaluation_results.extend(batch_results)
        processed_count += len(batch_results)
        
        if processed_count % RUNTIME_PARAMS['checkpoint_frequency'] == 0:
            temp_df = pd.DataFrame(evaluation_results)
            temp_df.to_csv(checkpoint_file, index=False)
            print(f"\n检查点保存: 已处理{processed_count}个对话")

    if len(evaluation_results) == 0:
        print("没有成功处理的对话，无法生成统计结果。")
        return
    
    results_df = pd.DataFrame(evaluation_results)
    
    print(f"\n=== RSAE模型评估结果 ===")
    print(f"处理对话: {len(evaluation_results)}/{len(conv_ids)} (成功率: {len(evaluation_results)/len(conv_ids)*100:.1f}%)")
    print(f"Top-1准确率: {(results_df['top_1_accuracy'] == 1).mean()*100:.1f}%")
    print(f"Top-3准确率: {(results_df['top_3_accuracy'] == 1).mean()*100:.1f}%")
    print(f"Top-5准确率: {(results_df['top_5_accuracy'] == 1).mean()*100:.1f}%")
    print(f"平均初始熵: {results_df['initial_entropy'].mean():.3f}")
    print(f"平均最终熵: {results_df['final_entropy'].mean():.3f}")
    print(f"平均初始资格参数: {results_df['initial_E'].mean():.3f}")
    print(f"平均最终资格参数: {results_df['final_E'].mean():.3f}")
    print(f"平均资格参数变化: {results_df['E_change'].mean():.3f}")
    print(f"平均综合得分: {results_df['avg_comprehensive_score'].mean():.3f}")
    
    output_filename = RSAE_RESULTS_DIR / f"rsae_evaluation_{timestamp}.csv"
    results_df.to_csv(output_filename, index=False)
    
    print(f"\n结果已保存到: {output_filename}")
    
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    return results_df

def main(): # 主函数
    parser = argparse.ArgumentParser(description="RSAE模型评估")
    parser.add_argument('--max-conversations', type=int, default=RUNTIME_PARAMS['default_max_conversations'], 
                       help=f'最大处理对话数量去global_config中调')
    parser.add_argument('--batch-size', type=int, default=RUNTIME_PARAMS['default_batch_size'],
                       help=f'批处理大小去global_config中调')
    
    args = parser.parse_args() # 解析命令行参数
    
    print(f"=== RSAE模型配置 ===")
    print(f"理性参数alpha: {RSAE_PARAMS['alpha']}")
    print(f"初始资格参数E: {RSAE_PARAMS['initial_E']}")
    print(f"学习率lambda: {RSAE_PARAMS['lambda_E']}")
    print(f"语用说者beta参数: {RSAE_PARAMS['beta']}")
    print(f"动态先验融合权重: {RSAE_PARAMS['fusion_weight']}")
    print(f"处理对话数: {'全部' if args.max_conversations == 0 else args.max_conversations}")
    print(f"批处理大小: {args.batch_size}")
    print()
    
    test_gpu_rsae(args)

if __name__ == "__main__":
    main() 