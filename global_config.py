#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSAE项目全局配置文件
统一管理所有文件路径、参数设置和模型配置
"""

import os
from pathlib import Path
from datetime import datetime

# ==================== 项目根目录配置 ====================
# 自动检测项目根目录，支持不同的运行环境
def get_project_root():
    """自动检测项目根目录"""
    current_file = Path(__file__).resolve()
    
    # 如果当前文件就在RSAE目录内，直接返回当前目录
    if current_file.name == "global_config.py" and current_file.parent.name == "RSAE":
        return current_file.parent
    
    # 从当前文件开始向上查找RSAE目录
    for parent in current_file.parents:
        if parent.name == "RSAE":
            return parent
    
    # 如果找不到，使用当前文件的父目录（假设在RSAE目录内）
    return current_file.parent

ROOT_DIR = get_project_root()
RSAE_DIR = ROOT_DIR

# ==================== 主要目录配置 ====================
# 数据目录
DATA_DIR = RSAE_DIR / "data"
RESULTS_DIR = RSAE_DIR / "results"
MODELS_DIR = RSAE_DIR / "models"
PREPROCESSING_DIR = RSAE_DIR / "preprocessing"
COST_DIR = RSAE_DIR / "cost"

# 结果子目录
EXTRACTED_DATA_DIR = RESULTS_DIR / "extracted_u"
EMBEDDING_DATA_DIR = RESULTS_DIR / "embedding_u"
CLUSTER_DATA_DIR = RESULTS_DIR / "cluster"
RSA_RESULTS_DIR = RESULTS_DIR / "rsa"
RSAE_RESULTS_DIR = RESULTS_DIR / "rsae"
RSA_OLD_RESULTS_DIR = RESULTS_DIR / "dynamicornot"  # RSA旧版本结果目录

# 模型子目录
RSA_MODEL_DIR = MODELS_DIR / "rsa"
RSAE_MODEL_DIR = MODELS_DIR / "rsae"

# 成本计算目录
COST_RESULTS_DIR = COST_DIR / "results"

# ==================== 数据文件配置 ====================
# 原始数据文件
PERSONACHAT_DATA_FILE = DATA_DIR / "train-00000-of-00001.parquet"

# 提取的utterances文件
EXTRACTED_UTTERANCES_CSV = EXTRACTED_DATA_DIR / "extracted_utterances.csv"
EXTRACTED_UTTERANCES_JSON = EXTRACTED_DATA_DIR / "extracted_utterances.json"

# 嵌入向量文件（动态查找最新文件）
def get_latest_embeddings_file():
    """获取最新的嵌入向量文件"""
    pattern = "utterances_embeddings_all-mpnet-base-v2_*.npy"
    files = list(EMBEDDING_DATA_DIR.glob(pattern))
    if files:
        return max(files, key=lambda f: f.stat().st_mtime)
    return None

def get_latest_embeddings_config():
    """获取最新的嵌入向量配置文件"""
    pattern = "utterances_embeddings_all-mpnet-base-v2_*_config.json"
    files = list(EMBEDDING_DATA_DIR.glob(pattern))
    if files:
        return max(files, key=lambda f: f.stat().st_mtime)
    return None

# 聚类文件（动态查找最新文件）
def get_latest_cluster_model():
    """获取最新的聚类模型文件"""
    pattern = "kmeans_model_k*_*.pkl"
    files = list(CLUSTER_DATA_DIR.glob(pattern))
    if files:
        return max(files, key=lambda f: f.stat().st_mtime)
    return None

def get_latest_cluster_labels():
    """获取最新的聚类标签文件"""
    pattern = "cluster_labels_k*_*.npy"
    files = list(CLUSTER_DATA_DIR.glob(pattern))
    if files:
        return max(files, key=lambda f: f.stat().st_mtime)
    return None

def get_latest_cluster_summary():
    """获取最新的聚类摘要文件"""
    pattern = "clustering_summary_k*_*.csv"
    files = list(CLUSTER_DATA_DIR.glob(pattern))
    if files:
        return max(files, key=lambda f: f.stat().st_mtime)
    return None

# 成本计算文件（动态查找最新文件）
def get_latest_cost_file():
    """获取最新的成本计算文件"""
    pattern = "utterance_costs_*.csv"
    files = list(COST_RESULTS_DIR.glob(pattern))
    if files:
        return max(files, key=lambda f: f.stat().st_mtime)
    return None

# ==================== 模型参数配置 ====================
# SBERT参数
SBERT_CONFIG = {
    "model_name": "all-mpnet-base-v2",
    "cache_dir": str(Path.home() / ".cache" / "sentence-transformers"),
    "batch_size": 32,
    "device": "auto",  # auto, cpu, cuda
}

# 聚类参数
CLUSTERING_CONFIG = {
    "n_clusters": 150,
    "batch_size": 1000,
    "max_iter": 100,
    "random_state": 42,
    "evaluation_sample_size": 10000,
    "output_dir": "results/cluster",
}

# RSA模型参数
RSA_PARAMS = {
    "alpha": 0.8,  # 理性参数
    "fusion_weight": 0.5,  # 动态先验融合权重 (全局先验 vs 上一轮信念)
}

# RSAE模型参数
RSAE_PARAMS = {
    # 继承RSA参数
    **RSA_PARAMS,
    
    # 资格参数相关
    "initial_E": 0.8,  # 初始资格参数值 
    "lambda_E": 0.8,   # 资格更新学习率 
    
    # 语用说者计算参数
    "beta": 0.3,  # 语用说者公式中的调节系数
    
    # 资格更新权重 (performance_score)
    "performance_weights": {
        "w_1": 0.35,  # 理解效果 (信息增益)
        "w_2": 0.35,  # 表达效果 (清晰度)
        "w_3": 0.15,  # 成本效率
        "w_4": 0.15,  # 身份-主题契合度
    },
    "S_score": 0.8,  # 身份-主题契合度评分
    
    # 控制参数
    "E_min": 0.1,   # 资格参数最小值
    "E_max": 1.0,   # 资格参数最大值
}

# 成本计算参数
COST_PARAMS = {
    "smoothing_alpha": 1.0,  # 拉普拉斯平滑参数
}

# ==================== 处理流程配置 ====================
# 数据提取配置
EXTRACTION_CONFIG = {
    "filter_reference": True,   # 是否过滤无参考回答的对话
    "include_persona": False,   # 是否包含persona描述
    "output_formats": ["csv", "json"],  # 输出格式
}

# 文件查找配置
FILE_SEARCH_PATHS = {
    "personachat_data": [
        "data/train-00000-of-00001.parquet",
        str(RSAE_DIR / "data" / "train-00000-of-00001.parquet")
    ],
    "extracted_utterances": [
        "results/extracted_u/extracted_utterances.csv",
        str(RSAE_DIR / "results" / "extracted_u" / "extracted_utterances.csv")
    ]
}

# 运行时参数
RUNTIME_PARAMS = {
    "default_batch_size": 128,  # 默认批处理大小
    "default_max_conversations": 0,  # 默认最大对话数（0=全部）
    "checkpoint_frequency": 8000,  # 检查点保存频率
    "verbose": False,  # 默认详细输出设置
}

# ==================== 工具函数 ====================
def ensure_directories():
    """确保所有必要的目录存在"""
    directories = [
        DATA_DIR, RESULTS_DIR, MODELS_DIR, PREPROCESSING_DIR, COST_DIR,
        EXTRACTED_DATA_DIR, EMBEDDING_DATA_DIR, CLUSTER_DATA_DIR,
        RSA_RESULTS_DIR, RSAE_RESULTS_DIR, RSA_MODEL_DIR, RSAE_MODEL_DIR,
        COST_RESULTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_file_paths():
    """获取当前所有重要文件的路径"""
    return {
        "personachat_data": PERSONACHAT_DATA_FILE,
        "extracted_utterances_csv": EXTRACTED_UTTERANCES_CSV,
        "extracted_utterances_json": EXTRACTED_UTTERANCES_JSON,
        "latest_embeddings": get_latest_embeddings_file(),
        "latest_embeddings_config": get_latest_embeddings_config(),
        "latest_cluster_model": get_latest_cluster_model(),
        "latest_cluster_labels": get_latest_cluster_labels(),
        "latest_cluster_summary": get_latest_cluster_summary(),
        "latest_cost_file": get_latest_cost_file(),
    }

def check_required_files():
    """检查必需文件是否存在"""
    files = get_file_paths()
    missing_files = []
    
    required_files = [
        "personachat_data",
        "extracted_utterances_csv",
        "latest_embeddings",
        "latest_cluster_model",
        "latest_cost_file"
    ]
    
    for file_key in required_files:
        file_path = files[file_key]
        if file_path is None or not file_path.exists():
            missing_files.append(file_key)
    
    return missing_files

def print_file_status():
    """打印所有文件的状态"""
    files = get_file_paths()
    print("=== 文件状态检查 ===")
    for key, path in files.items():
        if path is None:
            status = "未找到"
        elif path.exists():
            status = f"存在 ({path.stat().st_size / 1024**2:.1f} MB)"
        else:
            status = "不存在"
        print(f"{key}: {status}")
        if path:
            print(f"    路径: {path}")

def get_timestamp():
    """获取当前时间戳（用于文件命名）"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ==================== 模块初始化 ====================
# 在导入时自动创建目录
ensure_directories()

# 导出常用功能
__all__ = [
    # 目录配置
    'RSAE_DIR', 'DATA_DIR', 'RESULTS_DIR', 'MODELS_DIR', 'PREPROCESSING_DIR',
    'EXTRACTED_DATA_DIR', 'EMBEDDING_DATA_DIR', 'CLUSTER_DATA_DIR',
    'RSA_RESULTS_DIR', 'RSAE_RESULTS_DIR', 'RSA_OLD_RESULTS_DIR', 'COST_RESULTS_DIR',
    
    # 文件路径
    'PERSONACHAT_DATA_FILE', 'EXTRACTED_UTTERANCES_CSV', 'EXTRACTED_UTTERANCES_JSON',
    
    # 动态文件获取
    'get_latest_embeddings_file', 'get_latest_cluster_model', 'get_latest_cluster_labels',
    'get_latest_cost_file', 'get_latest_embeddings_config', 'get_latest_cluster_summary',
    
    # 参数配置
    'SBERT_CONFIG', 'CLUSTERING_CONFIG', 'RSA_PARAMS', 'RSAE_PARAMS', 'COST_PARAMS',
    'EXTRACTION_CONFIG', 'FILE_SEARCH_PATHS', 'RUNTIME_PARAMS',
    
    # 工具函数
    'ensure_directories', 'get_file_paths', 'check_required_files', 
    'print_file_status', 'get_timestamp'
] 