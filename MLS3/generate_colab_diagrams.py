"""
创建 Colab 运行流程可视化图表

运行此脚本生成工作流程图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_workflow_diagram():
    """创建工作流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 标题
    ax.text(5, 11.5, 'MLS3 Colab 运行流程', 
            fontsize=20, weight='bold', ha='center')
    
    # 三种方法的列
    methods = ['方法 A: Notebook', '方法 B: 直接运行', '方法 C: Google Drive']
    colors = ['#3498db', '#2ecc71', '#f39c12']
    
    # 绘制三列
    for i, (method, color) in enumerate(zip(methods, colors)):
        x = 1 + i * 3
        
        # 方法标题
        box = FancyBboxPatch((x-0.4, 10), 2.8, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black',
                             linewidth=2, alpha=0.7)
        ax.add_patch(box)
        ax.text(x+1, 10.4, method, fontsize=11, weight='bold',
                ha='center', va='center', color='white')
        
        # 步骤
        if i == 0:  # Notebook 方法
            steps = [
                ('上传\ncolab_setup.ipynb', 9),
                ('选择上传方式\nGitHub/ZIP/手动', 7.5),
                ('运行所有单元格', 6),
                ('查看结果', 4.5),
                ('下载 ZIP', 3)
            ]
        elif i == 1:  # 直接运行方法
            steps = [
                ('新建 Colab\nNotebook', 9),
                ('粘贴一键\n运行代码', 7.5),
                ('等待完成', 6),
                ('查看输出', 4.5),
                ('打包下载', 3)
            ]
        else:  # Google Drive 方法
            steps = [
                ('挂载\nGoogle Drive', 9),
                ('上传项目\nZIP', 7.5),
                ('解压运行', 6),
                ('结果自动保存', 4.5),
                ('随时继续', 3)
            ]
        
        for step, y in steps:
            # 步骤框
            box = FancyBboxPatch((x-0.35, y-0.5), 2.7, 1,
                                boxstyle="round,pad=0.05",
                                facecolor='white', edgecolor=color,
                                linewidth=2)
            ax.add_patch(box)
            ax.text(x+1, y, step, fontsize=9, ha='center', va='center')
            
            # 箭头
            if y > 3:
                arrow = FancyArrowPatch((x+1, y-0.5), (x+1, y-1.4),
                                       arrowstyle='->', mutation_scale=20,
                                       linewidth=2, color=color, alpha=0.6)
                ax.add_patch(arrow)
    
    # 底部总结
    summary_box = FancyBboxPatch((0.5, 0.2), 9, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#ecf0f1', edgecolor='black',
                                 linewidth=2)
    ax.add_patch(summary_box)
    
    ax.text(5, 1.3, '✓ 完成！生成的文件：', fontsize=12, weight='bold', ha='center')
    ax.text(5, 0.8, 'optimized_models/ (优化模型) | results/ (性能报告) | logs/ (日志)',
            fontsize=9, ha='center')
    ax.text(5, 0.4, '下载 → 分析 → 编写报告',
            fontsize=9, ha='center', style='italic')
    
    plt.tight_layout()
    plt.savefig('colab_workflow.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ 工作流程图已保存: colab_workflow.png")
    plt.show()


def create_comparison_chart():
    """创建方法对比图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    methods = ['Notebook\n(colab_setup.ipynb)', 
               'GitHub\n直接克隆',
               'ZIP\n上传',
               'Google Drive\n持久化']
    
    # 评分标准 (1-5)
    metrics = {
        '易用性': [5, 4, 3, 3],
        '速度': [3, 5, 4, 3],
        '完整性': [5, 3, 4, 4],
        '灵活性': [4, 3, 3, 5],
        '适合初学者': [5, 2, 3, 2]
    }
    
    x = np.arange(len(methods))
    width = 0.15
    multiplier = 0
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    for i, (metric, scores) in enumerate(metrics.items()):
        offset = width * multiplier
        bars = ax.bar(x + offset, scores, width, label=metric, color=colors[i], alpha=0.8)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}',
                   ha='center', va='bottom', fontsize=8)
        
        multiplier += 1
    
    ax.set_xlabel('方法', fontsize=12, weight='bold')
    ax.set_ylabel('评分 (1-5)', fontsize=12, weight='bold')
    ax.set_title('Colab 运行方法对比', fontsize=16, weight='bold', pad=20)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(methods)
    ax.legend(loc='upper left', ncol=2)
    ax.set_ylim(0, 6)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('colab_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ 方法对比图已保存: colab_comparison.png")
    plt.show()


def create_time_estimate_chart():
    """创建时间估算图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # GPU vs CPU 时间对比
    tasks = ['环境设置', '基线训练', '优化流程', '结果分析']
    gpu_times = [5, 40, 25, 5]  # 分钟
    cpu_times = [5, 90, 50, 5]  # 分钟
    
    x = np.arange(len(tasks))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, gpu_times, width, label='使用 GPU', 
                    color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, cpu_times, width, label='使用 CPU',
                    color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('任务', fontsize=11, weight='bold')
    ax1.set_ylabel('时间 (分钟)', fontsize=11, weight='bold')
    ax1.set_title('GPU vs CPU 运行时间对比', fontsize=13, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}m',
                    ha='center', va='bottom', fontsize=9)
    
    # 总时间对比
    ax1.text(1.5, max(cpu_times) * 1.15, 
            f'总计: GPU ≈ {sum(gpu_times)}分钟 | CPU ≈ {sum(cpu_times)}分钟',
            ha='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 文件大小对比
    files = ['完整包\n(含模型)', '轻量级包\n(仅代码)', '结果文件\n(下载)']
    sizes = [150, 2, 50]  # MB
    colors_files = ['#3498db', '#2ecc71', '#f39c12']
    
    bars = ax2.bar(files, sizes, color=colors_files, alpha=0.8)
    ax2.set_ylabel('大小 (MB)', fontsize=11, weight='bold')
    ax2.set_title('文件大小估算', fontsize=13, weight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f} MB',
                ha='center', va='bottom', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig('colab_time_estimates.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ 时间估算图已保存: colab_time_estimates.png")
    plt.show()


def main():
    """生成所有图表"""
    print("生成 Colab 运行可视化图表...\n")
    
    print("[1/3] 创建工作流程图...")
    create_workflow_diagram()
    
    print("\n[2/3] 创建方法对比图...")
    create_comparison_chart()
    
    print("\n[3/3] 创建时间估算图...")
    create_time_estimate_chart()
    
    print("\n" + "="*70)
    print("✓ 所有图表生成完成！")
    print("="*70)
    print("\n生成的文件:")
    print("  - colab_workflow.png       : 工作流程图")
    print("  - colab_comparison.png     : 方法对比图")
    print("  - colab_time_estimates.png : 时间和文件大小估算")
    print("\n这些图表可以添加到文档中，帮助理解 Colab 运行流程。")


if __name__ == "__main__":
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    main()
