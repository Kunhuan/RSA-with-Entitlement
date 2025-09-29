#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSA模型配置文件
从全局配置导入所有设置
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
RSAE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RSAE_DIR))

# 从全局配置导入所有设置
from global_config import *

# RSA模型特定设置
RSA_MODEL_DIR = MODELS_DIR / "rsa"
RSA_RESULTS_DIR = RESULTS_DIR / "rsa"
RSA_OLD_RESULTS_DIR = RSA_OLD_RESULTS_DIR  # 从全局配置导入

# 确保RSA目录存在
RSA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
RSA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RSA_OLD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 获取当前文件路径（用于RSA模型）
def get_current_file_paths():
    """获取RSA模型需要的当前文件路径"""
    return {
        "embeddings_file": get_latest_embeddings_file(),
        "metadata_file": EXTRACTED_UTTERANCES_CSV,
        "cluster_model_file": get_latest_cluster_model(),
        "cluster_labels_file": get_latest_cluster_labels(),
        "cost_file": get_latest_cost_file()
    }

# 文件检查函数
def check_files_exist():
    """检查RSA模型必要的文件是否存在"""
    files = get_current_file_paths()
    missing_files = []
    
    for key, file_path in files.items():
        if file_path is None or not file_path.exists():
            missing_files.append(key)
    
    if missing_files:
        print("错误: 以下文件不存在:")
        for file_key in missing_files:
            file_path = files[file_key]
            print(f"  - {file_key}: {file_path}")
        return False
    
    return True 