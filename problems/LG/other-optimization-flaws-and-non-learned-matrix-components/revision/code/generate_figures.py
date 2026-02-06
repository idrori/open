#!/usr/bin/env python3
"""Generate publication-quality figures for revised paper."""
import os
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --- Paths ---
PROBLEM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROBLEM_DIR, 'revision', 'data')
FIG_DIR = os.path.join(PROBLEM_DIR, 'revision', 'paper', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# --- Style ---
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['legend.fontsize'] = 8
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9

COLORS = {
    'row_norms': '#2980b9',
    'col_norms': '#3498db',
    'singular_values': '#27ae60',
    'spectral_gap': '#e67e22',
    'condition_number': '#e74c3c',
    'effective_rank': '#9b59b6',
    'frobenius_norm': '#1abc9c',
    'overall_matrix': '#34495e',
}

LABELS = {
    'row_norms': 'Row norms',
    'col_norms': 'Col norms',
    'singular_values': 'Sing. values',
    'spectral_gap': 'Spectral gap',
    'condition_number': 'Cond. number',
    'effective_rank': 'Eff. rank',
    'frobenius_norm': 'Frob. norm',
    'overall_matrix': 'Overall',
}


def load(name):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


# --- Figure 1: Component learning vs dimension (expanded) ---
def fig1_component_learning():
    data = load('exp1_component_learning.json')
    dims = data['hidden_dims']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in data['components']:
        mean = data['mean_errors'][c]
        std = data['std_errors'][c]
        ax.plot(dims, mean, marker='o', color=COLORS[c], label=LABELS[c], linewidth=1.8, markersize=4)
        ax.fill_between(dims, [m - s for m, s in zip(mean, std)],
                        [m + s for m, s in zip(mean, std)], alpha=0.12, color=COLORS[c])
    ax.set_xlabel('Hidden Dimension')
    ax.set_ylabel('Relative Error')
    ax.set_title('Component Learning Quality vs. Dimension')
    ax.set_yscale('log')
    ax.legend(ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(dims)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'component_learning_expanded.png'), dpi=200)
    plt.close()


# --- Figure 2: Optimizer comparison (SGD vs Adam) ---
def fig2_optimizer_comparison():
    data = load('exp2_optimizer_comparison.json')
    comps = ['row_norms', 'col_norms', 'singular_values', 'spectral_gap',
             'condition_number', 'overall_matrix']
    sgd = [data['SGD']['mean_errors'][c] for c in comps]
    sgd_std = [data['SGD']['std_errors'][c] for c in comps]
    adam = [data['Adam']['mean_errors'][c] for c in comps]
    adam_std = [data['Adam']['std_errors'][c] for c in comps]

    x = np.arange(len(comps))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, sgd, width, yerr=sgd_std, label='SGD', color='#2980b9',
           alpha=0.85, capsize=3)
    ax.bar(x + width / 2, adam, width, yerr=adam_std, label='Adam', color='#e74c3c',
           alpha=0.85, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in comps], rotation=25, ha='right')
    ax.set_ylabel('Relative Error')
    ax.set_title('SGD vs. Adam: Component Learning Errors (d=128)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'optimizer_comparison.png'), dpi=200)
    plt.close()


# --- Figure 3: Gradient analysis ---
def fig3_gradient_analysis():
    data = load('exp3_gradient_analysis.json')
    records = data['records']
    epochs = [r['epoch'] for r in records]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    # Gradient magnitudes
    for key, label, color in [('grad_top', 'Top SV', '#e74c3c'),
                               ('grad_median', 'Median SV', '#e67e22'),
                               ('grad_bottom', 'Bottom SV', '#2980b9')]:
        vals = [r[key] for r in records]
        ax1.plot(epochs, vals, label=label, color=color, linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('|dL/d$\\sigma_i$|')
    ax1.set_title('Gradient Magnitude per SV')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Singular value evolution
    for key, label, color in [('sv_top', 'Top SV', '#e74c3c'),
                               ('sv_median', 'Median SV', '#e67e22'),
                               ('sv_bottom', 'Bottom SV', '#2980b9')]:
        vals = [r[key] for r in records]
        ax2.plot(epochs, vals, label=label, color=color, linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Singular Value')
    ax2.set_title('SV Evolution During Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'gradient_analysis.png'), dpi=200)
    plt.close()


# --- Figure 4: Corrective strategies ---
def fig4_corrective_strategies():
    data = load('exp4_corrective_strategies.json')
    strats = ['standard_sgd', 'learnable_multipliers', 'spectral_reg', 'svd_correction']
    strat_labels = ['Standard SGD', 'Learnable Mult.', 'Spectral Reg.', 'SVD Correction']
    comps = ['row_norms', 'col_norms', 'singular_values', 'spectral_gap',
             'condition_number', 'overall_matrix']
    strat_colors = ['#e74c3c', '#2980b9', '#e67e22', '#27ae60']

    x = np.arange(len(comps))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (strat, slabel, scol) in enumerate(zip(strats, strat_labels, strat_colors)):
        vals = [data[strat]['mean_errors'][c] for c in comps]
        stds = [data[strat]['std_errors'][c] for c in comps]
        ax.bar(x + i * width, vals, width, yerr=stds, label=slabel,
               color=scol, alpha=0.85, capsize=2, error_kw={'linewidth': 0.7})
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([LABELS[c] for c in comps], rotation=25, ha='right')
    ax.set_ylabel('Relative Error')
    ax.set_title('Corrective Strategies: Component Errors (d=64)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'corrective_strategies.png'), dpi=200)
    plt.close()


# --- Figure 5: Training dynamics ---
def fig5_training_dynamics():
    data = load('exp5_training_dynamics.json')
    agg = data['aggregated']
    epochs = agg['epochs']
    focus = ['row_norms', 'condition_number', 'spectral_gap', 'overall_matrix']

    fig, ax = plt.subplots(figsize=(7, 4))
    for c in focus:
        mean = agg['mean_errors'][c]
        std = agg['std_errors'][c]
        ax.plot(epochs, mean, marker='.', color=COLORS[c], label=LABELS[c], linewidth=1.5, markersize=3)
        ax.fill_between(epochs, [m - s for m, s in zip(mean, std)],
                        [m + s for m, s in zip(mean, std)], alpha=0.12, color=COLORS[c])
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Relative Error')
    ax.set_title('Component Error Dynamics During Training (d=64)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'training_dynamics.png'), dpi=200)
    plt.close()


# --- Figure 6: Multiplier effect (corrected) ---
def fig6_multiplier_effect():
    data = load('exp6_multiplier_effect.json')
    structs = data['structures']
    comps = ['row_norms', 'col_norms', 'singular_values', 'condition_number',
             'spectral_gap', 'overall_matrix']

    fig, axes = plt.subplots(1, 3, figsize=(7, 3.5))
    for i, s in enumerate(structs):
        std_vals = [data['component_improvement'][s][c]['standard_mean'] for c in comps]
        mult_vals = [data['component_improvement'][s][c]['multiplier_mean'] for c in comps]
        x = np.arange(len(comps))
        axes[i].bar(x - 0.2, std_vals, 0.35, label='Standard', color='#e74c3c', alpha=0.8)
        axes[i].bar(x + 0.2, mult_vals, 0.35, label='Multipliers', color='#2980b9', alpha=0.8)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels([LABELS[c] for c in comps], rotation=45, ha='right', fontsize=6)
        axes[i].set_title(s.replace('_', ' ').title(), fontsize=10)
        axes[i].legend(fontsize=6)
        axes[i].grid(True, alpha=0.3, axis='y')
        axes[i].set_yscale('symlog', linthresh=0.01)
    axes[0].set_ylabel('Relative Error')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'multiplier_effect_corrected.png'), dpi=200)
    plt.close()


if __name__ == "__main__":
    print("Generating revised figures...")
    fig1_component_learning(); print("  component_learning_expanded.png")
    fig2_optimizer_comparison(); print("  optimizer_comparison.png")
    fig3_gradient_analysis(); print("  gradient_analysis.png")
    fig4_corrective_strategies(); print("  corrective_strategies.png")
    fig5_training_dynamics(); print("  training_dynamics.png")
    fig6_multiplier_effect(); print("  multiplier_effect_corrected.png")
    print("Done.")
