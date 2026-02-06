#!/usr/bin/env python3
"""
Generate figures for Transferability and Harms of Agent Intergroup Bias (Revised).

Key revisions:
  - Error bars (95% CI) on all plots (review: add uncertainty reporting)
  - Sensitivity tornado chart (review: sensitivity to domain params)
  - Horizon figure uses actual multi-step results (review: horizon is a real process now)
  - Correct DI threshold annotation (review: incorrect DI threshold claim)
  - Colorblind-friendly palette (Wong 2011, Nature Methods)
  - Consistent styling across all figures (font sizes, grid, layout)
  - New figures: sensitivity tornado, null comparison, fairness metrics, horizon multidomain
  - 300 DPI output for publication quality
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- Section 1: Configuration ---

# __file__ is revision/code/generate_figures.py
# dirname(dirname(__file__)) = revision/
REVISION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REVISION_DIR, "data")
FIG_DIR = os.path.join(REVISION_DIR, "paper", "figures")

# Colorblind-friendly palette (Wong 2011, Nature Methods 8:441)
CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GREEN = "#009E73"
CB_RED = "#D55E00"
CB_PURPLE = "#CC79A7"
CB_SKYBLUE = "#56B4E9"
CB_YELLOW = "#F0E442"

DOMAIN_COLORS = {
    "customer_service": CB_BLUE,
    "healthcare_triage": CB_RED,
    "content_moderation": CB_ORANGE,
    "education": CB_PURPLE,
    "hiring": CB_GREEN,
}
DOMAIN_LABELS = {
    "customer_service": "Customer Service",
    "healthcare_triage": "Healthcare Triage",
    "content_moderation": "Content Moderation",
    "education": "Education",
    "hiring": "Hiring",
}

# Consistent global style
DPI = 300
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10
GRID_ALPHA = 0.3
CAP_SIZE = 4
LINE_WIDTH = 2.0
MARKER_SIZE = 7

plt.rcParams.update({
    'font.size': TICK_SIZE,
    'axes.titlesize': TITLE_SIZE,
    'axes.labelsize': LABEL_SIZE,
    'legend.fontsize': LEGEND_SIZE,
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
    'axes.grid': True,
    'grid.alpha': GRID_ALPHA,
    'figure.dpi': DPI,
})


# --- Section 2: Load Data ---

def load_results():
    with open(os.path.join(DATA_DIR, "experiment_results.json")) as f:
        return json.load(f)


# --- Section 3: Existing Figures (improved styling) ---

def fig_domain_comparison(results):
    """Figure 1: Bias and harm across domains with error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    domains = list(results["by_domain"].keys())
    x = np.arange(len(domains))
    colors = [DOMAIN_COLORS[d] for d in domains]

    # Bias magnitude with CI
    bias = [results["by_domain"][d]["bias_magnitude"] for d in domains]
    bias_ci = [results["by_domain"][d]["bias_magnitude_ci95"] for d in domains]
    axes[0].bar(x, bias, 0.6, color=colors, yerr=bias_ci, capsize=CAP_SIZE,
                ecolor='black', edgecolor='white', linewidth=0.5)
    axes[0].set_title("Bias Magnitude by Domain", fontweight='bold')
    axes[0].set_ylabel("Bias (Ingroup $-$ Outgroup Rate)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DOMAIN_LABELS[d] for d in domains], rotation=25, ha='right')

    # Mean harm with CI
    harm = [results["by_domain"][d]["mean_harm"] for d in domains]
    harm_ci = [results["by_domain"][d]["mean_harm_ci95"] for d in domains]
    axes[1].bar(x, harm, 0.6, color=colors, yerr=harm_ci, capsize=CAP_SIZE,
                ecolor='black', edgecolor='white', linewidth=0.5)
    axes[1].set_title("Mean Harm Score by Domain", fontweight='bold')
    axes[1].set_ylabel("Harm Score")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DOMAIN_LABELS[d] for d in domains], rotation=25, ha='right')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "domain_comparison.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved domain_comparison.png")


def fig_cue_strength(results):
    """Figure 2: Bias vs cue strength across domains with error bands."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for domain in ["customer_service", "healthcare_triage", "hiring"]:
        data = [(v["cue_strength"], v["bias_magnitude"], v["bias_magnitude_ci95"])
                for v in results["by_cue_strength"]
                if v["domain"] == domain]
        data.sort()
        x_vals, y_vals, ci_vals = zip(*data)
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        ci_vals = np.array(ci_vals)

        color = DOMAIN_COLORS[domain]
        ax.plot(x_vals, y_vals, 'o-', color=color, label=DOMAIN_LABELS[domain],
                linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
        ax.fill_between(x_vals, y_vals - ci_vals, y_vals + ci_vals,
                        color=color, alpha=0.15)

    ax.set_xlabel("Intergroup Cue Strength")
    ax.set_ylabel("Bias Magnitude")
    ax.set_title("Bias Amplification by Cue Strength", fontweight='bold')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "cue_strength.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved cue_strength.png")


def fig_poisoning_impact(results):
    """Figure 3: Impact of belief poisoning with error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    data = [(v["poisoning_rate"], v["bias_magnitude"], v["bias_magnitude_ci95"],
             v["mean_harm"], v["mean_harm_ci95"])
            for v in results["by_poisoning"].values()]
    data.sort()
    pr, bias, bias_ci, harm, harm_ci = zip(*data)
    pr = np.array(pr)
    bias = np.array(bias)
    bias_ci = np.array(bias_ci)
    harm = np.array(harm)
    harm_ci = np.array(harm_ci)

    axes[0].errorbar(pr, bias, yerr=bias_ci, fmt='o-', color=CB_RED,
                     linewidth=LINE_WIDTH, markersize=MARKER_SIZE + 1, capsize=CAP_SIZE)
    axes[0].set_xlabel("Belief Poisoning Rate")
    axes[0].set_ylabel("Bias Magnitude")
    axes[0].set_title("Bias Under Belief Poisoning", fontweight='bold')

    axes[1].errorbar(pr, harm, yerr=harm_ci, fmt='s-', color=CB_PURPLE,
                     linewidth=LINE_WIDTH, markersize=MARKER_SIZE + 1, capsize=CAP_SIZE)
    axes[1].set_xlabel("Belief Poisoning Rate")
    axes[1].set_ylabel("Mean Harm Score")
    axes[1].set_title("Harm Under Belief Poisoning", fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "poisoning_impact.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved poisoning_impact.png")


def fig_transferability(results):
    """Figure 4: Lab-to-deployment transfer ratios with error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))
    domains = list(results["transferability"].keys())
    x = np.arange(len(domains))
    colors = [DOMAIN_COLORS[d] for d in domains]

    transfer = [results["transferability"][d]["transfer_ratio"] for d in domains]
    transfer_ci = [results["transferability"][d]["transfer_ratio_ci95"] for d in domains]

    ax.bar(x, transfer, 0.6, color=colors, yerr=transfer_ci, capsize=CAP_SIZE,
           ecolor='black', edgecolor='white', linewidth=0.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect Transfer')
    ax.set_ylabel("Transfer Ratio (Deploy / Lab)")
    ax.set_title("Bias Transferability: Lab to Deployment", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in domains], rotation=25, ha='right')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "transferability.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved transferability.png")


def fig_horizon_effect(results):
    """Figure 5: Horizon length effect with multi-step simulation results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    data = [(v["horizon"], v["bias_magnitude"], v["bias_magnitude_ci95"],
             v["mean_harm"], v["mean_harm_ci95"])
            for v in results["by_horizon"].values()]
    data.sort()
    hl, bias, bias_ci, harm, harm_ci = zip(*data)
    hl = np.array(hl)
    bias = np.array(bias)
    bias_ci = np.array(bias_ci)
    harm = np.array(harm)
    harm_ci = np.array(harm_ci)

    axes[0].errorbar(hl, bias, yerr=bias_ci, fmt='o-', color=CB_BLUE,
                     linewidth=LINE_WIDTH, markersize=MARKER_SIZE + 1, capsize=CAP_SIZE)
    axes[0].set_xlabel("Interaction Horizon (steps)")
    axes[0].set_ylabel("Bias Magnitude")
    axes[0].set_title("Bias Accumulation Over Horizon", fontweight='bold')
    axes[0].set_xscale('log')

    axes[1].errorbar(hl, harm, yerr=harm_ci, fmt='s-', color=CB_RED,
                     linewidth=LINE_WIDTH, markersize=MARKER_SIZE + 1, capsize=CAP_SIZE)
    axes[1].set_xlabel("Interaction Horizon (steps)")
    axes[1].set_ylabel("Mean Harm Score")
    axes[1].set_title("Harm Accumulation Over Horizon", fontweight='bold')
    axes[1].set_xscale('log')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "horizon_effect.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved horizon_effect.png")


def fig_effect_sizes(results):
    """Figure 6: Cohen's d effect sizes across domains."""
    fig, ax = plt.subplots(figsize=(8, 5))
    domains = list(results["by_domain"].keys())
    x = np.arange(len(domains))
    colors = [DOMAIN_COLORS[d] for d in domains]

    cohens = [results["by_domain"][d]["cohens_d"] for d in domains]
    cohens_ci = [results["by_domain"][d]["cohens_d_ci95"] for d in domains]

    ax.bar(x, cohens, 0.6, color=colors, yerr=cohens_ci, capsize=CAP_SIZE,
           ecolor='black', edgecolor='white', linewidth=0.5)
    # Cohen's d thresholds
    ax.axhline(y=0.2, color='gray', linestyle=':', alpha=0.5, label='Small ($d$=0.2)')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium ($d$=0.5)')
    ax.axhline(y=0.8, color='gray', linestyle='-', alpha=0.5, label='Large ($d$=0.8)')
    ax.set_ylabel("Cohen's $d$")
    ax.set_title("Effect Size of Intergroup Bias by Domain", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in domains], rotation=25, ha='right')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "effect_sizes.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved effect_sizes.png")


# --- Section 4: New Figures ---

def fig_sensitivity_tornado(results):
    """Figure 7: Tornado chart showing sensitivity of bias to parameter variation."""
    sensitivity = results["sensitivity"]

    # Group by parameter
    params = ["stakes", "harm_weight", "base_bias"]
    param_labels = {
        "stakes": "Stakes",
        "harm_weight": "Harm Weight",
        "base_bias": "Base Bias Gap",
    }
    metrics = ["bias_magnitude", "mean_harm", "disparate_impact_ratio"]
    metric_labels = {
        "bias_magnitude": "Bias Magnitude",
        "mean_harm": "Mean Harm",
        "disparate_impact_ratio": "Disparate Impact Ratio",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for col, metric in enumerate(metrics):
        ax = axes[col]
        # Get baseline value (factor=1.0)
        baselines = {p: None for p in params}
        lo_vals = {p: None for p in params}  # factor=0.5
        hi_vals = {p: None for p in params}  # factor=1.5
        for rec in sensitivity:
            p = rec["parameter"]
            if rec["factor"] == 1.0:
                baselines[p] = rec[metric]
            elif rec["factor"] == 0.5:
                lo_vals[p] = rec[metric]
            elif rec["factor"] == 1.5:
                hi_vals[p] = rec[metric]

        y_pos = np.arange(len(params))
        baseline = baselines[params[0]]  # All should be similar at factor=1.0

        for i, p in enumerate(params):
            bl = baselines[p]
            lo = lo_vals[p]
            hi = hi_vals[p]
            left = min(lo, hi) - bl
            right = max(lo, hi) - bl
            # Draw bar from left deviation to right deviation
            ax.barh(i, right - left, left=bl + left, height=0.5,
                    color=[CB_BLUE, CB_ORANGE, CB_GREEN][i],
                    edgecolor='white', linewidth=0.5)
            # Mark baseline
            ax.plot(bl, i, 'k|', markersize=15, markeredgewidth=2)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([param_labels[p] for p in params])
        ax.set_xlabel(metric_labels[metric])
        ax.set_title(metric_labels[metric], fontweight='bold')
        ax.axvline(x=baseline, color='gray', linestyle='--', alpha=0.4)

    fig.suptitle("Sensitivity Analysis: Parameter Variation ($\\times$0.5 to $\\times$1.5)",
                 fontsize=TITLE_SIZE, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "sensitivity_tornado.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved sensitivity_tornado.png")


def fig_null_comparison(results):
    """Figure 8: Side-by-side comparison of null model vs actual model."""
    domains = list(results["null_model"].keys())
    x = np.arange(len(domains))
    bar_width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [
        ("bias_magnitude", "Bias Magnitude"),
        ("mean_harm", "Mean Harm Score"),
        ("disparate_impact_ratio", "Disparate Impact Ratio"),
    ]

    for col, (metric, label) in enumerate(metrics):
        ax = axes[col]
        actual = [results["by_domain"][d][metric] for d in domains]
        null = [results["null_model"][d][metric] for d in domains]
        actual_ci = [results["by_domain"][d].get(f"{metric}_ci95", 0) for d in domains]
        null_ci = [results["null_model"][d].get(f"{metric}_ci95", 0) for d in domains]

        bars1 = ax.bar(x - bar_width / 2, actual, bar_width, color=CB_RED,
                       yerr=actual_ci, capsize=3, label='Actual Model',
                       edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x + bar_width / 2, null, bar_width, color=CB_SKYBLUE,
                       yerr=null_ci, capsize=3, label='Null Model (No Bias)',
                       edgecolor='white', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([DOMAIN_LABELS[d] for d in domains], rotation=30, ha='right')
        ax.set_ylabel(label)
        ax.set_title(label, fontweight='bold')
        ax.legend(fontsize=LEGEND_SIZE - 1)

    fig.suptitle("Actual vs. Null Model (Zero Structural Bias)",
                 fontsize=TITLE_SIZE, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "null_comparison.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved null_comparison.png")


def fig_fairness_metrics(results):
    """Figure 9: Multi-panel fairness metrics across domains."""
    domains = list(results["by_domain"].keys())
    x = np.arange(len(domains))
    colors = [DOMAIN_COLORS[d] for d in domains]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Disparate Impact Ratio
    di = [results["by_domain"][d]["disparate_impact_ratio"] for d in domains]
    di_ci = [results["by_domain"][d]["disparate_impact_ratio_ci95"] for d in domains]
    axes[0].bar(x, di, 0.6, color=colors, yerr=di_ci, capsize=CAP_SIZE,
                ecolor='black', edgecolor='white', linewidth=0.5)
    axes[0].axhline(y=0.8, color='gray', linestyle='--', alpha=0.6,
                    label='4/5ths Rule Threshold')
    axes[0].set_ylabel("Disparate Impact Ratio")
    axes[0].set_title("Disparate Impact Ratio", fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DOMAIN_LABELS[d] for d in domains], rotation=30, ha='right')
    axes[0].legend(fontsize=LEGEND_SIZE - 1)
    axes[0].set_ylim(0, 1.05)

    # Panel 2: Equal Opportunity Difference
    eod = [results["by_domain"][d]["equal_opportunity_diff"] for d in domains]
    eod_ci = [results["by_domain"][d]["equal_opportunity_diff_ci95"] for d in domains]
    axes[1].bar(x, eod, 0.6, color=colors, yerr=eod_ci, capsize=CAP_SIZE,
                ecolor='black', edgecolor='white', linewidth=0.5)
    axes[1].axhline(y=0.0, color='gray', linestyle='--', alpha=0.4)
    axes[1].set_ylabel("Equal Opportunity Difference")
    axes[1].set_title("Equal Opportunity Difference", fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DOMAIN_LABELS[d] for d in domains], rotation=30, ha='right')

    # Panel 3: Predictive Parity Difference
    ppd = [results["by_domain"][d]["predictive_parity_diff"] for d in domains]
    ppd_ci = [results["by_domain"][d]["predictive_parity_diff_ci95"] for d in domains]
    axes[2].bar(x, ppd, 0.6, color=colors, yerr=ppd_ci, capsize=CAP_SIZE,
                ecolor='black', edgecolor='white', linewidth=0.5)
    axes[2].axhline(y=0.0, color='gray', linestyle='--', alpha=0.4)
    axes[2].set_ylabel("Predictive Parity Difference")
    axes[2].set_title("Predictive Parity Difference", fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([DOMAIN_LABELS[d] for d in domains], rotation=30, ha='right')

    fig.suptitle("Fairness Metrics Across Deployment Domains",
                 fontsize=TITLE_SIZE, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fairness_metrics.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fairness_metrics.png")


def fig_horizon_multidomain(results):
    """Figure 10: Horizon effects across all 5 domains."""
    if "by_horizon_multidomain" not in results:
        print("  Skipping horizon_multidomain.png (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    domains = list(results["by_horizon_multidomain"].keys())

    for domain in domains:
        horizon_data = results["by_horizon_multidomain"][domain]
        data = [(v["horizon"], v["bias_magnitude"], v["bias_magnitude_ci95"],
                 v["mean_harm"], v["mean_harm_ci95"])
                for v in horizon_data.values()]
        data.sort()
        hl, bias, bias_ci, harm, harm_ci = zip(*data)
        hl = np.array(hl, dtype=float)
        bias = np.array(bias)
        bias_ci = np.array(bias_ci)
        harm = np.array(harm)
        harm_ci = np.array(harm_ci)

        color = DOMAIN_COLORS[domain]
        label = DOMAIN_LABELS[domain]

        axes[0].plot(hl, bias, 'o-', color=color, label=label,
                     linewidth=LINE_WIDTH, markersize=MARKER_SIZE - 1)
        axes[0].fill_between(hl, bias - bias_ci, bias + bias_ci,
                             color=color, alpha=0.1)

        axes[1].plot(hl, harm, 's-', color=color, label=label,
                     linewidth=LINE_WIDTH, markersize=MARKER_SIZE - 1)
        axes[1].fill_between(hl, harm - harm_ci, harm + harm_ci,
                             color=color, alpha=0.1)

    axes[0].set_xlabel("Interaction Horizon (steps)")
    axes[0].set_ylabel("Bias Magnitude")
    axes[0].set_title("Bias Accumulation Over Horizon (All Domains)", fontweight='bold')
    axes[0].set_xscale('log')
    axes[0].legend(fontsize=LEGEND_SIZE - 1)

    axes[1].set_xlabel("Interaction Horizon (steps)")
    axes[1].set_ylabel("Mean Harm Score")
    axes[1].set_title("Harm Accumulation Over Horizon (All Domains)", fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].legend(fontsize=LEGEND_SIZE - 1)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "horizon_multidomain.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved horizon_multidomain.png")


# --- Section 5: Main ---

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Loading results...")
    results = load_results()
    print("Generating figures...")

    # Existing figures (improved styling)
    fig_domain_comparison(results)
    fig_cue_strength(results)
    fig_poisoning_impact(results)
    fig_transferability(results)
    fig_horizon_effect(results)
    fig_effect_sizes(results)

    # New figures
    fig_sensitivity_tornado(results)
    fig_null_comparison(results)
    fig_fairness_metrics(results)
    fig_horizon_multidomain(results)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
