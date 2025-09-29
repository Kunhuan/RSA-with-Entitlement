#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSA模型版本对比分析工具（有无动态先验）

该脚本用于加载、分析和对比RSA模型有无动态先验版本的实验结果，
生成详细的对比报告和表格，论证动态先验机制的合理性。

功能：
1. 加载RSA旧版本（无动态先验）和新版本（有动态先验）的评估结果
2. 计算统计指标对比
3. 输出详细的分析报告
4. 导出对比表格
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
MODELS_DIR = Path(__file__).resolve().parent
RSAE_DIR = MODELS_DIR.parent
ROOT_DIR = RSAE_DIR.parent
sys.path.insert(0, str(RSAE_DIR))
sys.path.insert(0, str(ROOT_DIR))

# 从全局配置导入设置
try:
    from global_config import (
        RSA_RESULTS_DIR, RESULTS_DIR, RSA_OLD_RESULTS_DIR,
        get_timestamp
    )
except ImportError:
    # 如果直接导入失败，尝试从RSAE目录导入
    sys.path.insert(0, str(RSAE_DIR))
    from global_config import (
        RSA_RESULTS_DIR, RESULTS_DIR, RSA_OLD_RESULTS_DIR,
        get_timestamp
    )

class RSAVersionComparator:
    """RSA模型版本对比分析器（有无动态先验）"""
    
    def __init__(self):
        self.rsa_old_data = None  # 无动态先验版本
        self.rsa_new_data = None  # 有动态先验版本
        self.comparison_results = {}
        
    def load_latest_results(self):
        """加载最新的RSA版本结果文件"""
        print("=== 加载RSA版本结果 ===")
        
        # 查找最新的RSA旧版本结果文件（无动态先验）
        rsa_old_files = list(RSA_OLD_RESULTS_DIR.glob("rsa_old_evaluation_*.csv"))
        if not rsa_old_files:
            raise FileNotFoundError(f"在 {RSA_OLD_RESULTS_DIR} 中找不到RSA旧版本结果文件")
        
        latest_rsa_old_file = max(rsa_old_files, key=lambda f: f.stat().st_mtime)
        self.rsa_old_data = pd.read_csv(latest_rsa_old_file)
        print(f" 加载RSA旧版本结果: {latest_rsa_old_file.name} ({len(self.rsa_old_data)} 条记录)")
        
        # 查找最新的RSA新版本结果文件（有动态先验）
        rsa_new_files = list(RSA_RESULTS_DIR.glob("rsa_evaluation_*.csv"))
        if not rsa_new_files:
            raise FileNotFoundError(f"在 {RSA_RESULTS_DIR} 中找不到RSA新版本结果文件")
        
        latest_rsa_new_file = max(rsa_new_files, key=lambda f: f.stat().st_mtime)
        self.rsa_new_data = pd.read_csv(latest_rsa_new_file)
        print(f" 加载RSA新版本结果: {latest_rsa_new_file.name} ({len(self.rsa_new_data)} 条记录)")
        
        return latest_rsa_old_file, latest_rsa_new_file
    
    def load_specific_results(self, rsa_old_file=None, rsa_new_file=None):
        """加载指定的结果文件"""
        print("=== 加载指定RSA版本结果 ===")
        
        if rsa_old_file:
            rsa_old_path = Path(rsa_old_file)
            if not rsa_old_path.is_absolute():
                rsa_old_path = RSA_OLD_RESULTS_DIR / rsa_old_path
            self.rsa_old_data = pd.read_csv(rsa_old_path)
            print(f" 加载RSA旧版本结果: {rsa_old_path.name} ({len(self.rsa_old_data)} 条记录)")
        
        if rsa_new_file:
            rsa_new_path = Path(rsa_new_file)
            if not rsa_new_path.is_absolute():
                rsa_new_path = RSA_RESULTS_DIR / rsa_new_path
            self.rsa_new_data = pd.read_csv(rsa_new_path)
            print(f" 加载RSA新版本结果: {rsa_new_path.name} ({len(self.rsa_new_data)} 条记录)")
    
    def compute_comparison_statistics(self):
        """计算对比统计指标"""
        print("\n=== 计算对比统计 ===")
        
        if self.rsa_old_data is None or self.rsa_new_data is None:
            raise ValueError("请先加载RSA旧版本和新版本结果数据")
        
        # 基础统计
        stats = {}
        
        # RSA旧版本统计（无动态先验）
        rsa_old_stats = {
            'sample_size': len(self.rsa_old_data),
            'top_1_accuracy': self.rsa_old_data['top_1_accuracy'].mean() * 100,
            'top_3_accuracy': self.rsa_old_data['top_3_accuracy'].mean() * 100,
            'top_5_accuracy': self.rsa_old_data['top_5_accuracy'].mean() * 100,
            'initial_entropy_mean': self.rsa_old_data['initial_entropy'].mean(),
            'final_entropy_mean': self.rsa_old_data['final_entropy'].mean(),
            'entropy_reduction': self.rsa_old_data['initial_entropy'].mean() - self.rsa_old_data['final_entropy'].mean(),
            'avg_num_A_utterances': self.rsa_old_data['num_A_utterances'].mean(),
        }
        
        # RSA新版本统计（有动态先验）
        rsa_new_stats = {
            'sample_size': len(self.rsa_new_data),
            'top_1_accuracy': self.rsa_new_data['top_1_accuracy'].mean() * 100,
            'top_3_accuracy': self.rsa_new_data['top_3_accuracy'].mean() * 100,
            'top_5_accuracy': self.rsa_new_data['top_5_accuracy'].mean() * 100,
            'initial_entropy_mean': self.rsa_new_data['initial_entropy'].mean(),
            'final_entropy_mean': self.rsa_new_data['final_entropy'].mean(),
            'entropy_reduction': self.rsa_new_data['initial_entropy'].mean() - self.rsa_new_data['final_entropy'].mean(),
            'avg_num_A_utterances': self.rsa_new_data['num_A_utterances'].mean(),
        }
        
        # 计算改进幅度
        improvements = {
            'top_1_accuracy': rsa_new_stats['top_1_accuracy'] - rsa_old_stats['top_1_accuracy'],
            'top_3_accuracy': rsa_new_stats['top_3_accuracy'] - rsa_old_stats['top_3_accuracy'],
            'top_5_accuracy': rsa_new_stats['top_5_accuracy'] - rsa_old_stats['top_5_accuracy'],
            'entropy_reduction': rsa_new_stats['entropy_reduction'] - rsa_old_stats['entropy_reduction'],
        }
        
        # 计算相对改进（百分比）
        relative_improvements = {
            'top_1_accuracy': (improvements['top_1_accuracy'] / rsa_old_stats['top_1_accuracy'] * 100) if rsa_old_stats['top_1_accuracy'] > 0 else float('inf'),
            'top_3_accuracy': (improvements['top_3_accuracy'] / rsa_old_stats['top_3_accuracy'] * 100) if rsa_old_stats['top_3_accuracy'] > 0 else float('inf'),
            'top_5_accuracy': (improvements['top_5_accuracy'] / rsa_old_stats['top_5_accuracy'] * 100) if rsa_old_stats['top_5_accuracy'] > 0 else float('inf'),
        }
        
        self.comparison_results = {
            'rsa_old': rsa_old_stats,
            'rsa_new': rsa_new_stats,
            'improvements': improvements,
            'relative_improvements': relative_improvements
        }
        
        print(" 统计计算完成")
        return self.comparison_results
    
    def print_comparison_report(self):
        """打印详细的对比报告"""
        if not self.comparison_results:
            self.compute_comparison_statistics()
        
        print("\n" + "="*60)
        print(" RSA模型版本对比报告（有无动态先验）")
        print("="*60)
        
        rsa_old = self.comparison_results['rsa_old']
        rsa_new = self.comparison_results['rsa_new']
        imp = self.comparison_results['improvements']
        rel_imp = self.comparison_results['relative_improvements']
        
        # 基础信息
        print(f"\n 样本数量:")
        print(f"  RSA（无动态先验）: {rsa_old['sample_size']:,} 个对话")
        print(f"  RSA（有动态先验）: {rsa_new['sample_size']:,} 个对话")
        
        # 准确率对比
        print(f"\n 准确率对比:")
        print(f"  指标            RSA(旧)     RSA(新)     改进      相对改进")
        print(f"  Top-1准确率    {rsa_old['top_1_accuracy']:6.2f}%   {rsa_new['top_1_accuracy']:6.2f}%   {imp['top_1_accuracy']:+6.2f}%   {rel_imp['top_1_accuracy']:+6.1f}%")
        print(f"  Top-3准确率    {rsa_old['top_3_accuracy']:6.2f}%   {rsa_new['top_3_accuracy']:6.2f}%   {imp['top_3_accuracy']:+6.2f}%   {rel_imp['top_3_accuracy']:+6.1f}%")
        print(f"  Top-5准确率    {rsa_old['top_5_accuracy']:6.2f}%   {rsa_new['top_5_accuracy']:6.2f}%   {imp['top_5_accuracy']:+6.2f}%   {rel_imp['top_5_accuracy']:+6.1f}%")
        
        # 信念熵对比
        print(f"\n 信念熵对比:")
        print(f"  初始熵         {rsa_old['initial_entropy_mean']:6.3f}     {rsa_new['initial_entropy_mean']:6.3f}     {rsa_new['initial_entropy_mean']-rsa_old['initial_entropy_mean']:+6.3f}")
        print(f"  最终熵         {rsa_old['final_entropy_mean']:6.3f}     {rsa_new['final_entropy_mean']:6.3f}     {rsa_new['final_entropy_mean']-rsa_old['final_entropy_mean']:+6.3f}")
        print(f"  熵减少量       {rsa_old['entropy_reduction']:6.3f}     {rsa_new['entropy_reduction']:6.3f}     {imp['entropy_reduction']:+6.3f}")
        
        # 对话复杂度
        print(f"\n 对话复杂度:")
        print(f"  平均A发言数    {rsa_old['avg_num_A_utterances']:6.2f}     {rsa_new['avg_num_A_utterances']:6.2f}")
        
        # 总结
        print(f"\n 关键发现:")
        top1_better = "✅" if imp['top_1_accuracy'] > 0 else "❌"
        top3_better = "✅" if imp['top_3_accuracy'] > 0 else "❌"
        entropy_better = "✅" if imp['entropy_reduction'] > 0 else "❌"
        
        print(f"  {top1_better} 动态先验在Top-1准确率上{'优于' if imp['top_1_accuracy'] > 0 else '不如'}固定先验")
        print(f"  {top3_better} 动态先验在Top-3准确率上{'优于' if imp['top_3_accuracy'] > 0 else '不如'}固定先验")
        print(f"  {entropy_better} 动态先验的信念收敛性{'更好' if imp['entropy_reduction'] > 0 else '较差'}")
        
        # 动态先验机制合理性论证
        print(f"\n 动态先验机制合理性论证:")
        if imp['top_1_accuracy'] > 0 or imp['top_3_accuracy'] > 0 or imp['entropy_reduction'] > 0:
            print(f"   主要改进:")
            if imp['top_1_accuracy'] > 0:
                print(f"    - Top-1准确率提升 {imp['top_1_accuracy']:.2f}% ({rel_imp['top_1_accuracy']:.1f}%)")
            if imp['top_3_accuracy'] > 0:
                print(f"    - Top-3准确率提升 {imp['top_3_accuracy']:.2f}% ({rel_imp['top_3_accuracy']:.1f}%)")
            if imp['entropy_reduction'] > 0:
                print(f"    - 信念收敛性提升 {imp['entropy_reduction']:.3f}")
        else:
            print(f"   动态先验未带来显著改进，需进一步分析原因")
            
    
    def export_comparison_table(self, save_csv=True):
        """导出对比结果表格"""
        if not self.comparison_results:
            self.compute_comparison_statistics()
        
        print("\n=== 导出对比表格 ===")
        
        # 清理旧的表格文件（防止遗漏）
        if save_csv:
            old_tables = list(RESULTS_DIR.glob("rsa_version_comparison_table_*.csv"))
            if old_tables:
                print(f"清理 {len(old_tables)} 个残留的旧表格文件...")
                for file in old_tables:
                    file.unlink()
        
        # 创建对比表格
        comparison_table = []
        
        metrics = [
            ('样本数量', 'sample_size', ''),
            ('Top-1准确率', 'top_1_accuracy', '%'),
            ('Top-3准确率', 'top_3_accuracy', '%'),
            ('Top-5准确率', 'top_5_accuracy', '%'),
            ('初始信念熵', 'initial_entropy_mean', ''),
            ('最终信念熵', 'final_entropy_mean', ''),
            ('熵减少量', 'entropy_reduction', ''),
            ('平均A发言数', 'avg_num_A_utterances', ''),
        ]
        
        for metric_name, metric_key, unit in metrics:
            rsa_old_val = self.comparison_results['rsa_old'][metric_key]
            rsa_new_val = self.comparison_results['rsa_new'][metric_key]
            improvement = rsa_new_val - rsa_old_val
            
            if unit == '%':
                rsa_old_str = f"{rsa_old_val:.2f}%"
                rsa_new_str = f"{rsa_new_val:.2f}%"
                imp_str = f"{improvement:+.2f}%"
            elif metric_key == 'sample_size':
                rsa_old_str = f"{rsa_old_val:,}"
                rsa_new_str = f"{rsa_new_val:,}"
                imp_str = f"{improvement:+,}"
            else:
                rsa_old_str = f"{rsa_old_val:.3f}"
                rsa_new_str = f"{rsa_new_val:.3f}"
                imp_str = f"{improvement:+.3f}"
            
            comparison_table.append({
                '指标': metric_name,
                'RSA(无动态先验)': rsa_old_str,
                'RSA(有动态先验)': rsa_new_str,
                '改进': imp_str
            })
        
        df_comparison = pd.DataFrame(comparison_table)
        
        # 打印表格
        print("\n RSA版本对比汇总表:")
        print(df_comparison.to_string(index=False))
        
        if save_csv:
            timestamp = get_timestamp()
            table_path = RESULTS_DIR / f"rsa_version_comparison_table_{timestamp}.csv"
            df_comparison.to_csv(table_path, index=False, encoding='utf-8-sig')
            print(f"\n 对比表格已保存: {table_path}")
        
        return df_comparison


def clear_old_comparison_reports():
    """清理旧的比较报告文件"""
    print("=== 清理旧的RSA版本比较报告 ===")
    
    # 清理对比表格文件
    old_tables = list(RESULTS_DIR.glob("rsa_version_comparison_table_*.csv"))
    if old_tables:
        print(f"删除 {len(old_tables)} 个旧的对比表格文件...")
        for file in old_tables:
            file.unlink()
    
    print(" 旧报告清理完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RSA模型版本对比分析（有无动态先验）")
    parser.add_argument('--rsa-old-file', type=str, help='指定RSA旧版本结果文件（可选，默认使用最新）')
    parser.add_argument('--rsa-new-file', type=str, help='指定RSA新版本结果文件（可选，默认使用最新）')
    parser.add_argument('--no-table', action='store_true', help='不导出对比表格')
    
    args = parser.parse_args()
    
    try:
        # 清理旧报告
        clear_old_comparison_reports()

        # 创建对比分析器
        comparator = RSAVersionComparator()
        
        # 加载数据
        if args.rsa_old_file or args.rsa_new_file:
            comparator.load_specific_results(args.rsa_old_file, args.rsa_new_file)
        else:
            comparator.load_latest_results()
        
        # 计算统计并生成报告
        comparator.compute_comparison_statistics()
        comparator.print_comparison_report()
        
        # 导出表格（如果需要）
        if not args.no_table:
            comparator.export_comparison_table()
        
        print(f"\n RSA版本对比分析完成！")
        print(f" 该分析结果可用于论证动态先验机制的合理性")
        
    except Exception as e: # expectation的作用是捕获所有异常
        print(f" 分析过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 