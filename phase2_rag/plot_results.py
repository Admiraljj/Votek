"""
实验结果可视化：生成论文级别的对比图表
用法: python -m phase2_rag.plot_results
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

OUTPUT_DIR = "phase2_rag/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    'topk': '#4E79A7',
    'reranker': '#F28E2B',
    'hybrid': '#E15759',
    'votek': '#59A14F',
}


def plot_phase1_accuracy():
    """图1: Phase 1 Vote-K vs Random 准确率对比"""
    fig, ax = plt.subplots(figsize=(7, 5))

    models = ['GPT-2-XL\n(1.5B)', 'Qwen3-4B\n(4B)']
    random_acc = [8.98, 50.39]
    votek_acc = [7.03, 60.16]

    x = np.arange(len(models))
    width = 0.25

    bars1 = ax.bar(x - width, random_acc, width, label='Random', color='#4E79A7', alpha=0.85)
    bars2 = ax.bar(x, votek_acc, width, label='Vote-K (ours)', color='#59A14F', alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.annotate('+9.77%', xy=(1, 60.16), xytext=(1.35, 65),
                fontsize=12, fontweight='bold', color='#E15759',
                arrowprops=dict(arrowstyle='->', color='#E15759', lw=1.5))
    ax.annotate('-1.95%', xy=(0, 7.03), xytext=(0.35, 12),
                fontsize=10, color='#999999',
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1))

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Phase 1: Vote-K vs Random on DBpedia-14', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0, 75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig1_phase1_accuracy.png')
    plt.savefig(path)
    plt.close()
    print(f"[Fig 1] Phase 1 accuracy -> {path}")


def plot_budget_sweep():
    """图2: Coverage 和 Redundancy 随预算变化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    budgets = [500, 800, 1500]
    topk_cov = [0.948, 0.977, 1.000]
    reranker_cov = [0.993, 1.000, 1.000]
    hybrid_cov = [0.860, 0.985, 0.996]
    votek_cov = [0.835, 0.927, 0.997]

    ax1.plot(budgets, topk_cov, 'o-', color=COLORS['topk'], label='Top-k', linewidth=2, markersize=7)
    ax1.plot(budgets, reranker_cov, 's-', color=COLORS['reranker'], label='Reranker', linewidth=2, markersize=7)
    ax1.plot(budgets, hybrid_cov, 'D-', color=COLORS['hybrid'], label='Hybrid', linewidth=2, markersize=7)
    ax1.plot(budgets, votek_cov, '^-', color=COLORS['votek'], label='Vote-K', linewidth=2, markersize=7)

    ax1.set_xlabel('Token Budget', fontsize=11)
    ax1.set_ylabel('Evidence Coverage', fontsize=11)
    ax1.set_title('Coverage vs Budget', fontsize=12, fontweight='bold')
    ax1.set_ylim(0.80, 1.02)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    topk_red = [0.446, 0.409, 0.346]
    reranker_red = [0.426, 0.395, 0.344]
    hybrid_red = [0.392, 0.385, 0.365]
    votek_red = [0.375, 0.367, 0.342]

    ax2.plot(budgets, topk_red, 'o-', color=COLORS['topk'], label='Top-k', linewidth=2, markersize=7)
    ax2.plot(budgets, reranker_red, 's-', color=COLORS['reranker'], label='Reranker', linewidth=2, markersize=7)
    ax2.plot(budgets, hybrid_red, 'D-', color=COLORS['hybrid'], label='Hybrid', linewidth=2, markersize=7)
    ax2.plot(budgets, votek_red, '^-', color=COLORS['votek'], label='Vote-K', linewidth=2, markersize=7)

    ax2.set_xlabel('Token Budget', fontsize=11)
    ax2.set_ylabel('Context Redundancy (lower=better)', fontsize=11)
    ax2.set_title('Redundancy vs Budget', fontsize=12, fontweight='bold')
    ax2.set_ylim(0.30, 0.48)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig2_budget_sweep.png')
    plt.savefig(path)
    plt.close()
    print(f"[Fig 2] Budget sweep -> {path}")


def plot_f1_em_500():
    """图3: 500样本 F1 和 EM 水平柱状图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    pipelines = ['Top-k (A)', 'Reranker (B)', 'Hybrid (D)', 'Vote-K (C)']
    colors = [COLORS['topk'], COLORS['reranker'], COLORS['hybrid'], COLORS['votek']]

    f1 = [0.391, 0.409, 0.407, 0.393]
    bars1 = ax1.barh(pipelines, f1, color=colors, alpha=0.85, height=0.5)
    for bar, v in zip(bars1, f1):
        ax1.text(v + 0.003, bar.get_y() + bar.get_height()/2,
                 f'{v:.3f}', va='center', fontsize=10)
    ax1.set_xlim(0.36, 0.44)
    ax1.set_title('F1-Score (500 samples, B=1500)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('F1-Score', fontsize=11)
    ax1.axvline(x=f1[0], color=COLORS['topk'], linestyle='--', alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='x', alpha=0.3)

    em = [0.270, 0.286, 0.278, 0.262]
    bars2 = ax2.barh(pipelines, em, color=colors, alpha=0.85, height=0.5)
    for bar, v in zip(bars2, em):
        ax2.text(v + 0.003, bar.get_y() + bar.get_height()/2,
                 f'{v:.3f}', va='center', fontsize=10)
    ax2.set_xlim(0.23, 0.32)
    ax2.set_title('Exact Match (500 samples, B=1500)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Exact Match', fontsize=11)
    ax2.axvline(x=em[0], color=COLORS['topk'], linestyle='--', alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig3_f1_em_500.png')
    plt.savefig(path)
    plt.close()
    print(f"[Fig 3] F1/EM 500-sample -> {path}")


def plot_efficiency_500():
    """图4: 效率指标对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    pipelines = ['Top-k', 'Reranker', 'Hybrid', 'Vote-K']
    colors = [COLORS['topk'], COLORS['reranker'], COLORS['hybrid'], COLORS['votek']]

    f1_per_1k = [0.315, 0.329, 0.424, 0.317]
    bars1 = ax1.bar(pipelines, f1_per_1k, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars1, f1_per_1k):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                 f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax1.set_ylabel('F1 per 1k Tokens', fontsize=11)
    ax1.set_title('F1 Efficiency (500 samples)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 0.55)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3)
    ax1.annotate(f'+{(0.424-0.329)/0.329*100:.1f}% vs Reranker',
                 xy=(2, 0.424), xytext=(2.5, 0.48),
                 fontsize=10, fontweight='bold', color='#E15759',
                 arrowprops=dict(arrowstyle='->', color='#E15759', lw=1.5))

    eff_per_1k = [0.857, 0.859, 1.075, 0.858]
    bars2 = ax2.bar(pipelines, eff_per_1k, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars2, eff_per_1k):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                 f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Evidence Eff. per 1k Tokens', fontsize=11)
    ax2.set_title('Evidence Efficiency (500 samples)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3)
    ax2.annotate(f'+{(1.075-0.859)/0.859*100:.1f}% vs Reranker',
                 xy=(2, 1.075), xytext=(2.5, 1.20),
                 fontsize=10, fontweight='bold', color='#E15759',
                 arrowprops=dict(arrowstyle='->', color='#E15759', lw=1.5))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig4_efficiency_500.png')
    plt.savefig(path)
    plt.close()
    print(f"[Fig 4] Efficiency -> {path}")


def plot_token_coverage_500():
    """图5: Token 用量 + Coverage"""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    pipelines = ['Top-k (A)', 'Reranker (B)', 'Hybrid (D)', 'Vote-K (C)']
    colors = [COLORS['topk'], COLORS['reranker'], COLORS['hybrid'], COLORS['votek']]
    x = np.arange(len(pipelines))

    tokens = [1242, 1242, 959, 1238]
    coverage = [0.998, 1.000, 0.996, 0.997]

    bars = ax1.bar(x, tokens, color=colors, alpha=0.7, width=0.5, label='Avg Tokens Used')
    for bar, t in zip(bars, tokens):
        ax1.text(bar.get_x() + bar.get_width()/2, t + 20, f'{t}', ha='center', fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(x, [c*100 for c in coverage], 'ko-', markersize=8, linewidth=2, label='Coverage (%)')
    for i, c in enumerate(coverage):
        ax2.text(i, c*100 + 0.1, f'{c*100:.1f}%', ha='center', fontsize=9)

    ax1.set_ylabel('Avg Tokens Used', fontsize=11)
    ax2.set_ylabel('Evidence Coverage (%)', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(pipelines, fontsize=10)
    ax1.set_ylim(0, 1500)
    ax2.set_ylim(98.5, 101)
    ax1.set_title('Token Usage vs Coverage (500 samples, B=1500)', fontsize=12, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig5_token_coverage.png')
    plt.savefig(path)
    plt.close()
    print(f"[Fig 5] Token+Coverage -> {path}")


if __name__ == '__main__':
    print("=" * 60)
    print("  Generating experiment visualization charts")
    print("=" * 60)
    plot_phase1_accuracy()
    plot_budget_sweep()
    plot_f1_em_500()
    plot_efficiency_500()
    plot_token_coverage_500()
    print(f"\nAll figures saved to {OUTPUT_DIR}/")
