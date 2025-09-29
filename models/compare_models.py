#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#输入python models/compare_models.py  --help以查看代码可选项

#python models/compare_models.py
#注意默认对比最新的两份结果

#python compare_models.py --rsa-file rsa_evaluation_20240815_143045.csv --rsae-file rsae_evaluation_20240815_150230.csv
#这个是对比指定文件，可以只指定一个模型的文件，也可以两个都指定

#同样，生成图表时会删除旧图表！

"""
RSA vs RSAE 模型结果对比
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
MODELS_DIR = Path(__file__).resolve().parent
RSAE_DIR = MODELS_DIR.parent
sys.path.insert(0, str(RSAE_DIR))

# 从全局配置导入设置
from global_config import (
    RSA_RESULTS_DIR, RSAE_RESULTS_DIR, RESULTS_DIR,
    get_timestamp
)

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class ModelComparator:
    """RSA和RSAE模型结果对比分析器"""
    
    def __init__(self):
        self.rsa_data = None
        self.rsae_data = None
        self.comparison_results = {}
        
    def load_latest_results(self):
        """加载最新的RSA和RSAE结果文件"""
        print("=== 加载模型结果 ===")
        
        # 查找最新的RSA结果文件
        rsa_files = list(RSA_RESULTS_DIR.glob("rsa_evaluation_*.csv"))
        if not rsa_files:
            raise FileNotFoundError(f"在 {RSA_RESULTS_DIR} 中找不到RSA结果文件")
        
        latest_rsa_file = max(rsa_files, key=lambda f: f.stat().st_mtime)
        self.rsa_data = pd.read_csv(latest_rsa_file)
        print(f"加载RSA结果: {latest_rsa_file.name} ({len(self.rsa_data)} 条记录)")
        
        # 查找最新的RSAE结果文件
        rsae_files = list(RSAE_RESULTS_DIR.glob("rsae_evaluation_*.csv"))
        if not rsae_files:
            raise FileNotFoundError(f"在 {RSAE_RESULTS_DIR} 中找不到RSAE结果文件")
        
        latest_rsae_file = max(rsae_files, key=lambda f: f.stat().st_mtime)
        self.rsae_data = pd.read_csv(latest_rsae_file)
        print(f"加载RSAE结果: {latest_rsae_file.name} ({len(self.rsae_data)} 条记录)")
        
        return latest_rsa_file, latest_rsae_file
    
    def load_specific_results(self, rsa_file=None, rsae_file=None):
        """加载指定的结果文件"""
        print("=== 加载指定模型结果 ===")
        
        if rsa_file:
            rsa_path = Path(rsa_file)
            if not rsa_path.is_absolute():
                rsa_path = RSA_RESULTS_DIR / rsa_path
            self.rsa_data = pd.read_csv(rsa_path)
            print(f"加载RSA结果: {rsa_path.name} ({len(self.rsa_data)} 条记录)")
        
        if rsae_file:
            rsae_path = Path(rsae_file)
            if not rsae_path.is_absolute():
                rsae_path = RSAE_RESULTS_DIR / rsae_path
            self.rsae_data = pd.read_csv(rsae_path)
            print(f"加载RSAE结果: {rsae_path.name} ({len(self.rsae_data)} 条记录)")
    
    def compute_comparison_statistics(self):
        """计算对比统计指标"""
        print("\n=== 计算对比统计 ===")
        
        if self.rsa_data is None or self.rsae_data is None:
            raise ValueError("请先加载RSA和RSAE结果数据")
        
        # 基础统计
        stats = {}
        
        # RSA统计
        rsa_stats = {
            'sample_size': len(self.rsa_data),
            'top_1_accuracy': self.rsa_data['top_1_accuracy'].mean() * 100,
            'top_3_accuracy': self.rsa_data['top_3_accuracy'].mean() * 100,
            'top_5_accuracy': self.rsa_data['top_5_accuracy'].mean() * 100,
            'initial_entropy_mean': self.rsa_data['initial_entropy'].mean(),
            'final_entropy_mean': self.rsa_data['final_entropy'].mean(),
            'entropy_reduction': self.rsa_data['initial_entropy'].mean() - self.rsa_data['final_entropy'].mean(),
            'avg_num_A_utterances': self.rsa_data['num_A_utterances'].mean(),
        }
        
        # RSAE统计
        rsae_stats = {
            'sample_size': len(self.rsae_data),
            'top_1_accuracy': self.rsae_data['top_1_accuracy'].mean() * 100,
            'top_3_accuracy': self.rsae_data['top_3_accuracy'].mean() * 100,
            'top_5_accuracy': self.rsae_data['top_5_accuracy'].mean() * 100,
            'initial_entropy_mean': self.rsae_data['initial_entropy'].mean(),
            'final_entropy_mean': self.rsae_data['final_entropy'].mean(),
            'entropy_reduction': self.rsae_data['initial_entropy'].mean() - self.rsae_data['final_entropy'].mean(),
            'avg_num_A_utterances': self.rsae_data['num_A_utterances'].mean(),
            'initial_E_mean': self.rsae_data['initial_E'].mean(),
            'final_E_mean': self.rsae_data['final_E'].mean(),
            'E_change_mean': self.rsae_data['E_change'].mean(),
            'avg_comprehensive_score': self.rsae_data['avg_comprehensive_score'].mean(),
        }
        
        # 计算改进幅度
        improvements = {
            'top_1_accuracy': rsae_stats['top_1_accuracy'] - rsa_stats['top_1_accuracy'],
            'top_3_accuracy': rsae_stats['top_3_accuracy'] - rsa_stats['top_3_accuracy'],
            'top_5_accuracy': rsae_stats['top_5_accuracy'] - rsa_stats['top_5_accuracy'],
            'entropy_reduction': rsae_stats['entropy_reduction'] - rsa_stats['entropy_reduction'],
        }
        
        # 计算相对改进（百分比）
        relative_improvements = {
            'top_1_accuracy': (improvements['top_1_accuracy'] / rsa_stats['top_1_accuracy'] * 100) if rsa_stats['top_1_accuracy'] > 0 else float('inf'),
            'top_3_accuracy': (improvements['top_3_accuracy'] / rsa_stats['top_3_accuracy'] * 100) if rsa_stats['top_3_accuracy'] > 0 else float('inf'),
            'top_5_accuracy': (improvements['top_5_accuracy'] / rsa_stats['top_5_accuracy'] * 100) if rsa_stats['top_5_accuracy'] > 0 else float('inf'),
        }
        
        self.comparison_results = {
            'rsa': rsa_stats,
            'rsae': rsae_stats,
            'improvements': improvements,
            'relative_improvements': relative_improvements
        }
        
        print("统计计算完成")
        return self.comparison_results
    
    def print_comparison_report(self):
        """打印详细的对比报告"""
        if not self.comparison_results:
            self.compute_comparison_statistics()
        
        print("\n" + "="*60)
        print("RSA vs RSAE 模型对比报告")
        print("="*60)
        
        rsa = self.comparison_results['rsa']
        rsae = self.comparison_results['rsae']
        imp = self.comparison_results['improvements']
        rel_imp = self.comparison_results['relative_improvements']
        
        # 基础信息
        print(f"\n样本数量:")
        print(f"  RSA:  {rsa['sample_size']:,} 个对话")
        print(f"  RSAE: {rsae['sample_size']:,} 个对话")
        
        # 准确率对比
        print(f"\n准确率对比:")
        print(f"  指标            RSA        RSAE       改进      相对改进")
        print(f"  Top-1准确率    {rsa['top_1_accuracy']:6.2f}%   {rsae['top_1_accuracy']:6.2f}%   {imp['top_1_accuracy']:+6.2f}%   {rel_imp['top_1_accuracy']:+6.1f}%")
        print(f"  Top-3准确率    {rsa['top_3_accuracy']:6.2f}%   {rsae['top_3_accuracy']:6.2f}%   {imp['top_3_accuracy']:+6.2f}%   {rel_imp['top_3_accuracy']:+6.1f}%")
        print(f"  Top-5准确率    {rsa['top_5_accuracy']:6.2f}%   {rsae['top_5_accuracy']:6.2f}%   {imp['top_5_accuracy']:+6.2f}%   {rel_imp['top_5_accuracy']:+6.1f}%")
        
        # 信念熵对比
        print(f"\n信念熵对比:")
        print(f"  初始熵         {rsa['initial_entropy_mean']:6.3f}     {rsae['initial_entropy_mean']:6.3f}     {rsae['initial_entropy_mean']-rsa['initial_entropy_mean']:+6.3f}")
        print(f"  最终熵         {rsa['final_entropy_mean']:6.3f}     {rsae['final_entropy_mean']:6.3f}     {rsae['final_entropy_mean']-rsa['final_entropy_mean']:+6.3f}")
        print(f"  熵减少量       {rsa['entropy_reduction']:6.3f}     {rsae['entropy_reduction']:6.3f}     {imp['entropy_reduction']:+6.3f}")
        
        # RSAE特有指标
        if 'initial_E_mean' in rsae:
            print(f"\n RSAE资格参数:")
            print(f"  初始资格参数E  {rsae['initial_E_mean']:6.3f}")
            print(f"  最终资格参数E  {rsae['final_E_mean']:6.3f}")
            print(f"  资格参数变化   {rsae['E_change_mean']:+6.3f}")
            print(f"  平均综合得分   {rsae['avg_comprehensive_score']:6.3f}")
        
        # 对话复杂度
        print(f"\n 对话复杂度:")
        print(f"  平均A发言数    {rsa['avg_num_A_utterances']:6.2f}     {rsae['avg_num_A_utterances']:6.2f}")
        
        # 总结
        print(f"\n 主要对比:")
        top1_better = "更高" if imp['top_1_accuracy'] > 0 else "更低"
        top3_better = "更高" if imp['top_3_accuracy'] > 0 else "更低"
        entropy_better = "更高" if imp['entropy_reduction'] > 0 else "更低"
        
        print(f"  {top1_better} RSAE在Top-1准确率上{'优于' if imp['top_1_accuracy'] > 0 else '不如'}RSA")
        print(f"  {top3_better} RSAE在Top-3准确率上{'优于' if imp['top_3_accuracy'] > 0 else '不如'}RSA")
        print(f"  {entropy_better} RSAE的信念收敛性{'更好' if imp['entropy_reduction'] > 0 else '较差'}")
        
    def create_comparison_visualizations(self, save_plots=True):
        """创建对比可视化图表"""
        if not self.comparison_results:
            self.compute_comparison_statistics()
        
        print("\n=== 生成可视化图表 ===")
        
        # 创建图表目录
        plots_dir = RESULTS_DIR / "comparison_plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 确保目录是干净的（防止遗漏）
        existing_plots = list(plots_dir.glob("*.png"))
        if existing_plots:
            print(f"清理 {len(existing_plots)} 个残留的旧图表文件...")
            for file in existing_plots:
                file.unlink()
        
        timestamp = get_timestamp()
        
        # 1. 准确率对比柱状图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Top-K准确率对比
        metrics = ['Top-1', 'Top-3', 'Top-5']
        rsa_acc = [self.comparison_results['rsa']['top_1_accuracy'],
                   self.comparison_results['rsa']['top_3_accuracy'],
                   self.comparison_results['rsa']['top_5_accuracy']]
        rsae_acc = [self.comparison_results['rsae']['top_1_accuracy'],
                    self.comparison_results['rsae']['top_3_accuracy'],
                    self.comparison_results['rsae']['top_5_accuracy']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, rsa_acc, width, label='RSA', alpha=0.8, color='skyblue')
        ax1.bar(x + width/2, rsae_acc, width, label='RSAE', alpha=0.8, color='lightcoral')
        ax1.set_xlabel('准确率指标')
        ax1.set_ylabel('准确率 (%)')
        ax1.set_title('RSA vs RSAE 准确率对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签
        for i, (r, rs) in enumerate(zip(rsa_acc, rsae_acc)):
            ax1.text(i - width/2, r + 0.5, f'{r:.1f}%', ha='center', va='bottom')
            ax1.text(i + width/2, rs + 0.5, f'{rs:.1f}%', ha='center', va='bottom')
        
        # 2. 信念熵对比
        entropy_metrics = ['初始熵', '最终熵', '熵减少量']
        rsa_entropy = [self.comparison_results['rsa']['initial_entropy_mean'],
                       self.comparison_results['rsa']['final_entropy_mean'],
                       self.comparison_results['rsa']['entropy_reduction']]
        rsae_entropy = [self.comparison_results['rsae']['initial_entropy_mean'],
                        self.comparison_results['rsae']['final_entropy_mean'],
                        self.comparison_results['rsae']['entropy_reduction']]
        
        x_ent = np.arange(len(entropy_metrics))
        ax2.bar(x_ent - width/2, rsa_entropy, width, label='RSA', alpha=0.8, color='lightgreen')
        ax2.bar(x_ent + width/2, rsae_entropy, width, label='RSAE', alpha=0.8, color='orange')
        ax2.set_xlabel('熵指标')
        ax2.set_ylabel('熵值')
        ax2.set_title('RSA vs RSAE 信念熵对比')
        ax2.set_xticks(x_ent)
        ax2.set_xticklabels(entropy_metrics)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 改进幅度雷达图
        categories = ['Top-1\n准确率', 'Top-3\n准确率', 'Top-5\n准确率', '熵减少量']
        improvements = [self.comparison_results['improvements']['top_1_accuracy'],
                       self.comparison_results['improvements']['top_3_accuracy'],
                       self.comparison_results['improvements']['top_5_accuracy'],
                       self.comparison_results['improvements']['entropy_reduction'] * 10]  # 放大熵改进以便可视化
        
        # 标准化改进幅度（为了雷达图显示）
        max_imp = max(abs(i) for i in improvements) if improvements else 1
        norm_improvements = [i / max_imp for i in improvements]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        norm_improvements += norm_improvements[:1]  # 闭合图形
        angles += angles[:1]
        
        ax3 = plt.subplot(223, projection='polar')
        ax3.plot(angles, norm_improvements, 'o-', linewidth=2, label='RSAE改进', color='red')
        ax3.fill(angles, norm_improvements, alpha=0.25, color='red')
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories)
        ax3.set_ylim(-1, 1)
        ax3.set_title('RSAE相对RSA的改进幅度', pad=20)
        ax3.grid(True)
        
        # 4. 分布对比（箱线图）
        if len(self.rsa_data) > 1 and len(self.rsae_data) > 1:
            # 合并数据用于箱线图
            combined_top1 = pd.DataFrame({
                'Model': ['RSA'] * len(self.rsa_data) + ['RSAE'] * len(self.rsae_data),
                'Top-1 Accuracy': list(self.rsa_data['top_1_accuracy'] * 100) + list(self.rsae_data['top_1_accuracy'] * 100)
            })
            
            sns.boxplot(data=combined_top1, x='Model', y='Top-1 Accuracy', ax=ax4)
            ax4.set_title('Top-1准确率分布对比')
            ax4.set_ylabel('Top-1准确率 (%)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = plots_dir / f"rsa_rsae_comparison_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"可视化图表已保存: {plot_path}")
        
        plt.show()
        
        # 5. 单独的RSAE资格参数分析图
        if 'initial_E_mean' in self.comparison_results['rsae']:
            self._plot_rsae_entitlement_analysis(plots_dir, timestamp, save_plots)
        
        return plots_dir
    
    def _plot_rsae_entitlement_analysis(self, plots_dir, timestamp, save_plots):
        """绘制RSAE资格参数分析图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 资格参数变化分布
        ax1.hist(self.rsae_data['E_change'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax1.axvline(self.rsae_data['E_change'].mean(), color='red', linestyle='--', 
                   label=f'均值: {self.rsae_data["E_change"].mean():.3f}')
        ax1.set_xlabel('资格参数变化 (ΔE)')
        ax1.set_ylabel('频次')
        ax1.set_title('RSAE资格参数变化分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 综合得分vs准确率关系
        ax2.scatter(self.rsae_data['avg_comprehensive_score'], 
                   self.rsae_data['top_3_accuracy'] * 100, alpha=0.6, color='green')
        ax2.set_xlabel('平均综合得分')
        ax2.set_ylabel('Top-3准确率 (%)')
        ax2.set_title('综合得分与准确率关系')
        ax2.grid(True, alpha=0.3)
        
        # 计算相关系数
        correlation = np.corrcoef(self.rsae_data['avg_comprehensive_score'], 
                                 self.rsae_data['top_3_accuracy'])[0, 1]
        ax2.text(0.05, 0.95, f'相关系数: {correlation:.3f}', 
                transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = plots_dir / f"rsae_entitlement_analysis_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"RSAE资格参数分析图已保存: {plot_path}")
        
        plt.show()
    
    def export_comparison_table(self, save_csv=True):
        """导出对比结果表格"""
        if not self.comparison_results:
            self.compute_comparison_statistics()
        
        print("\n=== 导出对比表格 ===")
        
        # 清理旧的表格文件（防止遗漏）
        if save_csv:
            old_tables = list(RESULTS_DIR.glob("model_comparison_table_*.csv"))
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
            rsa_val = self.comparison_results['rsa'][metric_key]
            rsae_val = self.comparison_results['rsae'][metric_key]
            improvement = rsae_val - rsa_val
            
            if unit == '%':
                rsa_str = f"{rsa_val:.2f}%"
                rsae_str = f"{rsae_val:.2f}%"
                imp_str = f"{improvement:+.2f}%"
            elif metric_key == 'sample_size':
                rsa_str = f"{rsa_val:,}"
                rsae_str = f"{rsae_val:,}"
                imp_str = f"{improvement:+,}"
            else:
                rsa_str = f"{rsa_val:.3f}"
                rsae_str = f"{rsae_val:.3f}"
                imp_str = f"{improvement:+.3f}"
            
            comparison_table.append({
                '指标': metric_name,
                'RSA': rsa_str,
                'RSAE': rsae_str,
                '改进': imp_str
            })
        
        # 添加RSAE特有指标
        if 'initial_E_mean' in self.comparison_results['rsae']:
            rsae_metrics = [
                ('初始资格参数E', 'initial_E_mean'),
                ('最终资格参数E', 'final_E_mean'),
                ('资格参数变化', 'E_change_mean'),
                ('平均综合得分', 'avg_comprehensive_score'),
            ]
            
            for metric_name, metric_key in rsae_metrics:
                comparison_table.append({
                    '指标': metric_name,
                    'RSA': 'N/A',
                    'RSAE': f"{self.comparison_results['rsae'][metric_key]:.3f}",
                    '改进': 'N/A'
                })
        
        df_comparison = pd.DataFrame(comparison_table)
        
        # 打印表格
        print("\n模型对比汇总表:")
        print(df_comparison.to_string(index=False))
        
        if save_csv:
            timestamp = get_timestamp()
            table_path = RESULTS_DIR / f"model_comparison_table_{timestamp}.csv"
            df_comparison.to_csv(table_path, index=False, encoding='utf-8-sig')
            print(f"\n对比表格已保存: {table_path}")
        
        return df_comparison


def clear_old_comparison_reports(): #这个是全部清理哦！
    """清理旧的比较报告文件"""
    print("=== 清理旧的比较报告 ===")
    
    # 清理对比表格文件
    old_tables = list(RESULTS_DIR.glob("model_comparison_table_*.csv"))
    if old_tables:
        print(f"删除 {len(old_tables)} 个旧的对比表格文件...")
        for file in old_tables:
            file.unlink()
    
    # 清理可视化图表目录
    plots_dir = RESULTS_DIR / "comparison_plots"
    if plots_dir.exists():
        old_plots = list(plots_dir.glob("*.png"))
        if old_plots:
            print(f"删除 {len(old_plots)} 个旧的可视化图表文件...")
            for file in old_plots:
                file.unlink()
    
    print("旧报告清理完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RSA vs RSAE 模型结果对比分析")
    parser.add_argument('--rsa-file', type=str, help='指定RSA结果文件（可选，默认使用最新）')
    parser.add_argument('--rsae-file', type=str, help='指定RSAE结果文件（可选，默认使用最新）')
    parser.add_argument('--no-plots', action='store_true', help='不生成可视化图表')
    parser.add_argument('--no-table', action='store_true', help='不导出对比表格')
    
    args = parser.parse_args()
    
    try:
        # 清理旧报告
        clear_old_comparison_reports()

        # 创建对比分析器
        comparator = ModelComparator()
        
        # 加载数据
        if args.rsa_file or args.rsae_file:
            comparator.load_specific_results(args.rsa_file, args.rsae_file)
        else:
            comparator.load_latest_results()
        
        # 计算统计并生成报告
        comparator.compute_comparison_statistics()
        comparator.print_comparison_report()
        
        # 生成可视化（如果需要）
        if not args.no_plots:
            comparator.create_comparison_visualizations()
        
        # 导出表格（如果需要）
        if not args.no_table:
            comparator.export_comparison_table()
        
        print(f"\n对比分析完成")
        
    except Exception as e:
        print(f"分析过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 