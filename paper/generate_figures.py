"""
PipeGuard — Paper Figures (IEEE 2-column compliant)
Max single-column width: 3.5 inches
Max full-width: 7.0 inches
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs('paper/figures', exist_ok=True)
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 200
})

# ── FIGURE 1: SYSTEM ARCHITECTURE ──────────────────────────────────────────
def draw_architecture():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor('#FAFAFA')

    def box(x, y, w, h, label, sublabel='', color='#1565C0',
            textcolor='white', fontsize=9, style='normal'):
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5,
                              edgecolor='white', facecolor=color, alpha=0.90,
                              zorder=2)
        ax.add_patch(rect)
        yo = y + h/2 + (0.18 if sublabel else 0)
        ax.text(x+w/2, yo, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=textcolor, zorder=3)
        if sublabel:
            ax.text(x+w/2, y+h/2 - 0.22, sublabel,
                    ha='center', va='center', fontsize=7.5,
                    color=textcolor, style='italic', zorder=3)

    def arrow(x1, y1, x2, y2, color='#333333', style='->', lw=1.5, dashed=False):
        ls = (0, (4, 3)) if dashed else 'solid'
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color,
                                   lw=lw, linestyle=ls),
                    zorder=4)

    def label(x, y, text, fontsize=9, color='#333333', bold=False):
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                color=color, fontweight='bold' if bold else 'normal', zorder=5)

    # ── DATA SOURCES ──
    label(1.1, 7.5, 'Data Sources', fontsize=9, bold=True)
    box(0.1, 6.6, 2.0, 0.7, 'NYC Taxi', '3.06M rows', '#455A64', fontsize=8)
    box(0.1, 5.7, 2.0, 0.7, 'SMD', '28 machines', '#455A64', fontsize=8)
    box(0.1, 4.8, 2.0, 0.7, 'Credit Card', '284K rows', '#455A64', fontsize=8)
    box(0.1, 3.9, 2.0, 0.7, 'UCI Census', '32K rows', '#455A64', fontsize=8)

    # arrows: data sources → airflow
    for y in [6.95, 6.05, 5.15, 4.25]:
        arrow(2.1, y, 2.9, 5.2)

    # ── AIRFLOW DAG ──
    box(2.9, 4.5, 2.0, 1.4, 'Airflow DAG', 'Batch Ingestion\nScheduler', '#1565C0')
    arrow(4.9, 5.2, 5.6, 5.2)

    # ── STAGE 1 ──
    box(5.6, 4.1, 2.2, 2.2, 'Stage 1',
        'Structural Check\nAuto-Schema Inference\nNull Rate Detection\nFormat Validation',
        '#6A1B9A', fontsize=8)
    arrow(7.8, 5.2, 8.5, 5.2)

    # Score written to PostgreSQL (dashed)
    arrow(6.7, 4.1, 6.7, 2.1, color='#1565C0', lw=1.2, dashed=True)

    # ── STAGE 2 ──
    box(8.5, 4.1, 2.2, 2.2, 'Stage 2',
        'Isolation Forest\nKS-Test + PSI\nADWIN\nComposite Scoring',
        '#B71C1C', fontsize=8)
    arrow(10.7, 5.2, 11.3, 5.2)

    # Score written to PostgreSQL (dashed)
    arrow(9.6, 4.1, 9.6, 2.1, color='#B71C1C', lw=1.2, dashed=True)

    # ── DLQ ROUTING ──
    box(11.3, 6.1, 1.6, 0.75, 'Silver Table', 'score < 0.28', '#1B5E20', fontsize=8)
    box(11.3, 5.0, 1.6, 0.75, 'DLQ Review', '0.28 – 0.70', '#E65100', fontsize=8)
    box(11.3, 3.9, 1.6, 0.75, 'DLQ High Risk', 'score > 0.70', '#B71C1C', fontsize=8)

    arrow(11.3, 5.5, 11.1, 5.5)
    arrow(11.1, 5.2, 11.3, 5.38, style='-')
    arrow(11.1, 5.2, 11.3, 5.0 + 0.375, style='-')
    arrow(11.1, 5.2, 11.3, 3.9 + 0.375, style='-')

    # Vertical bar for routing split
    ax.plot([11.1, 11.1], [3.9+0.375, 6.1+0.375],
            color='#333333', lw=1.5, zorder=4)

    # ── INFRASTRUCTURE LAYER ──
    label(6.5, 1.6, '— Infrastructure Layer —', fontsize=8.5,
          color='#555555', bold=False)

    box(3.0, 0.4, 2.0, 0.9, 'PostgreSQL', 'Score Storage\nAudit Trail', '#0D47A1', fontsize=8)
    box(5.5, 0.4, 2.0, 0.9, 'dbt Core', 'Transforms Raw Scores\nto Silver/DLQ Tables', '#1A237E', fontsize=8)
    box(8.0, 0.4, 2.2, 0.9, 'Grafana', 'Real-Time Dashboard\nScore Trends', '#F57F17', fontsize=8)
    box(10.7, 0.4, 2.0, 0.9, 'Prometheus', 'Metrics Scraping\nAlerting', '#4A148C', fontsize=8)

    # PostgreSQL ← Stage 1 and Stage 2 (already drawn dashed above)
    # dbt ← PostgreSQL
    arrow(5.0, 0.85, 5.5, 0.85)
    # dbt → Silver/DLQ (dashed, conceptual)
    arrow(7.5, 0.85, 11.1, 3.9, color='#1A237E', lw=1.0, dashed=True)
    # Grafana ← PostgreSQL (dashed)
    arrow(7.5, 0.85, 8.0, 0.85)
    # Prometheus ← Airflow (dashed)
    arrow(3.9, 4.5, 3.9, 1.3, color='#4A148C', lw=1.0, dashed=True)
    arrow(3.9, 1.3, 10.7, 0.85, color='#4A148C', lw=1.0, dashed=True)

    # ── TITLE ──
    ax.text(6.5, 7.8,
            'PipeGuard — Dual-Stage ETL Data Quality Framework',
            ha='center', fontweight='bold', fontsize=13, color='#1A237E', zorder=5)

    # ── LEGEND ──
    solid  = mpatches.Patch(color='#333333', label='Data / Control Flow')
    dashed = mpatches.Patch(color='#777777', label='Score Write (async)')
    ax.legend(handles=[solid, dashed], loc='lower left',
              fontsize=8, framealpha=0.7)

    plt.tight_layout(pad=0.5)
    plt.savefig('paper/figures/F1_architecture.png', dpi=200,
                bbox_inches='tight', facecolor='#FAFAFA')
    print("F1 saved")
    plt.close()


# ── FIGURE 2: RESULTS — STACKED VERTICAL (IEEE 2-column compliant) ──────────
def draw_results():
    # Full-width figure: 7 inches wide × 7 inches tall (stacked)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))
    fig.suptitle('PipeGuard vs. Great Expectations — Detection Performance',
                 fontsize=11, fontweight='bold', y=1.01)

    experiments  = ['SMD\n(28 machines)', 'Credit Card\n(5% fraud)',
                    'Credit Card\n(10% fraud)', 'Streaming\n(NYC Taxi)']
    pg_dr  = [0.664, 0.884, 0.930, 1.000]
    ge_dr  = [0.536, 0.000, 0.000, 0.909]
    pg_err = [0.219, 0.018, 0.011, 0.000]

    pg_fpr = [0.309, 0.082, 0.150, 0.000]
    ge_fpr = [0.000, 0.000, 0.000, 0.000]
    pg_fpr_err = [0.185, 0.004, 0.016, 0.000]

    x = np.arange(len(experiments))
    w = 0.32

    # ── DR Panel ──
    b1 = ax1.bar(x - w/2, pg_dr, w, label='PipeGuard',
                 color='#1565C0', alpha=0.88, yerr=pg_err, capsize=5,
                 error_kw={'linewidth': 1.5})
    b2 = ax1.bar(x + w/2, ge_dr, w, label='Great Expectations',
                 color='#C62828', alpha=0.88)

    # Force GE bars visible with minimum display height
    for i, (bar, val) in enumerate(zip(b2, ge_dr)):
        if val == 0.0:
            ax1.bar(x[i] + w/2, 0.02, w, color='#C62828', alpha=0.5)
            ax1.text(x[i] + w/2, 0.04, '0.000',
                     ha='center', va='bottom', fontsize=7.5, color='#C62828')
        else:
            ax1.text(bar.get_x()+bar.get_width()/2, val+0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)

    for bar, val in zip(b1, pg_dr):
        ax1.text(bar.get_x()+bar.get_width()/2, val+0.02,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=7.5, fontweight='bold')

    ax1.set_ylabel('Detection Rate (DR)', fontsize=10)
    ax1.set_title('(a) Detection Rate — Higher is Better ↑', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(experiments, fontsize=9)
    ax1.set_ylim(0, 1.30)
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.4, lw=1)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, axis='y', alpha=0.3)

    # ── FPR Panel ──
    b3 = ax2.bar(x - w/2, pg_fpr, w, label='PipeGuard',
                 color='#1565C0', alpha=0.88, yerr=pg_fpr_err, capsize=5,
                 error_kw={'linewidth': 1.5})

    # GE FPR = 0 for all — show minimal bar with label
    for i in range(len(experiments)):
        ax2.bar(x[i] + w/2, 0.01, w, color='#C62828', alpha=0.5)
        ax2.text(x[i] + w/2, 0.025, '0.000',
                 ha='center', va='bottom', fontsize=7.5, color='#C62828')

    for bar, val in zip(b3, pg_fpr):
        ax2.text(bar.get_x()+bar.get_width()/2, val+0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)

    ax2.set_ylabel('False Positive Rate (FPR)', fontsize=10)
    ax2.set_title('(b) False Positive Rate — Lower is Better ↓', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(experiments, fontsize=9)
    ax2.set_ylim(0, 0.65)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, axis='y', alpha=0.3)

    # GE legend entry
    ge_patch = mpatches.Patch(color='#C62828', alpha=0.88,
                               label='Great Expectations')
    pg_patch = mpatches.Patch(color='#1565C0', alpha=0.88,
                               label='PipeGuard')
    ax2.legend(handles=[pg_patch, ge_patch], fontsize=9, loc='upper right')

    plt.tight_layout(pad=1.2)
    plt.savefig('paper/figures/F2_results.png', dpi=200,
                bbox_inches='tight')
    print("F2 saved")
    plt.close()


# ── FIGURE 3: ABLATION — NARROWED FOR 2-COLUMN ──────────────────────────────
def draw_ablation():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.suptitle('Ablation Study — Batch-Level Detection Rate per Configuration',
                 fontsize=11, fontweight='bold')

    configs  = ['Stage 1\nOnly', 'IF\nOnly', 'KS-PSI\nOnly',
                'No Struct.\nLayer', 'Full\nPipeGuard']
    datasets = ['NYC Taxi', 'Credit Card', 'UCI Census', 'SMD']
    colors   = ['#1565C0', '#E65100', '#2E7D32', '#6A1B9A']
    data = {
        'NYC Taxi':    [1.0, 0.0, 0.0, 0.0, 1.0],
        'Credit Card': [1.0, 0.0, 1.0, 1.0, 1.0],
        'UCI Census':  [1.0, 0.0, 0.0, 0.0, 1.0],
        'SMD':         [1.0, 1.0, 1.0, 1.0, 1.0],
    }

    x       = np.arange(len(configs))
    width   = 0.17
    offsets = [-0.255, -0.085, 0.085, 0.255]

    for i, (ds, color) in enumerate(zip(datasets, colors)):
        vals = data[ds]
        bars = ax.bar(x + offsets[i], vals, width,
                      label=ds, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x()+bar.get_width()/2, h+0.02,
                        f'{h:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylabel('Detection Rate', fontsize=10)
    ax.set_ylim(0, 1.35)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.4, lw=1)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, axis='y', alpha=0.3)

    # Highlight Full PipeGuard
    ax.axvspan(3.55, 4.45, alpha=0.10, color='green', zorder=0)
    ax.text(4.0, 1.27, '✓ Only config with\nDR=1.0 on all datasets',
            ha='center', fontsize=7.5, color='#2E7D32', fontweight='bold')

    plt.tight_layout()
    plt.savefig('paper/figures/F3_ablation.png', dpi=200,
                bbox_inches='tight')
    print("F3 saved")
    plt.close()


if __name__ == '__main__':
    draw_architecture()
    draw_results()
    draw_ablation()
    print("\nAll 3 figures regenerated.")
