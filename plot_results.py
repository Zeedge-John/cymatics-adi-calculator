"""
Results visualization module for ADI (Acoustic Discomfort Index) analysis.

Functions:
- plot_chladni_pattern: single Chladni pattern
- plot_multiple_patterns: multiple patterns comparison
- plot_adi_bar_with_ci: ADI bar chart with 95% CI error bars (v45 update)
- plot_adi_comparison: ADI bar chart (legacy, kept for backward compatibility)
- plot_sub_indices_radar: sub-index radar chart
- plot_feature_correlation: acoustic feature correlation heatmap
- plot_tukey_heatmap: Tukey HSD pairwise comparison heatmap (v45 new)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Optional seaborn import
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️  Warning: seaborn not installed, correlation heatmap unavailable.")


def plot_chladni_pattern(pattern, title, save_path=None):
    """Plot a single Chladni pattern."""
    plt.figure(figsize=(6, 6))
    plt.imshow(pattern, cmap=cm.binary, interpolation='nearest')
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_multiple_patterns(patterns_dict, save_path=None):
    """Plot multiple Chladni patterns for comparison."""
    n = len(patterns_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (sound, pattern) in zip(axes, patterns_dict.items()):
        ax.imshow(pattern, cmap=cm.binary, interpolation='nearest')
        ax.set_title(sound, fontsize=12)
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_adi_bar_with_ci(aggregated, save_path=None):
    """
    Plot ADI bar chart for six water sound types with 95% CI error bars.

    Parameters:
        aggregated: output dict from aggregate_by_sound_type()
                      (must contain 'ADI_mean', 'ADI_ci_lower', 'ADI_ci_upper')
        save_path: optional save path ('.png' or '.pdf')

    Output:
        Bar chart (PNG + PDF), color gradient green -> red.
    """
    sound_types = list(aggregated.keys())
    means = [aggregated[s]['ADI_mean'] for s in sound_types]
    ci_lower = [aggregated[s]['ADI_ci_lower'] for s in sound_types]
    ci_upper = [aggregated[s]['ADI_ci_upper'] for s in sound_types]
    ci_err = [(u - l) / 2 for l, u in zip(ci_lower, ci_upper)]

    # Color gradient: green (low ADI) -> red (high ADI)
    colors = []
    for v in means:
        r = min(1.0, v * 2.0)
        g = min(1.0, (1.0 - v) * 2.0)
        b = 0.0
        colors.append((r, g, b))

    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(sound_types))
    bars = plt.bar(x_pos, means,
                   yerr=[(m - l) for m, l in zip(means, ci_lower)],
                   capsize=8, color=colors, alpha=0.75,
                   edgecolor='black', linewidth=1.2)

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, means)):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + ci_err[i] + 0.01,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    plt.xticks(x_pos, sound_types, fontsize=12, rotation=15)
    plt.ylabel('ADI (mean ± 95% CI)', fontsize=13)
    plt.title('ADI for Six Water Sound Types (with 95% Confidence Intervals)',
              fontsize=14, fontweight='bold', pad=15)
    top = max([m + e for m, e in zip(means, ci_err)]) * 1.18
    plt.ylim(0, top)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    if save_path:
        ext = '.pdf' if save_path.endswith('.pdf') else '.png'
        base = save_path.replace('.png', '').replace('.pdf', '')
        plt.savefig(base + ext, dpi=300, bbox_inches='tight')
        print(f'  ✓ Saved: {base}{ext}')
        if ext == '.png':
            plt.savefig(base + '.pdf', dpi=300, bbox_inches='tight')
            print(f'  ✓ Saved: {base}.pdf')
    plt.show()


def plot_adi_comparison(results, save_path=None):
    """
    Legacy function: plot ADI bar chart WITHOUT error bars.
    Kept for backward compatibility.
    For v45 paper Figure 3, use plot_adi_bar_with_ci() instead.
    """
    sounds = list(results.keys())
    adi_values = [results[s]['ADI'] for s in sounds]

    plt.figure(figsize=(10, 6))
    colors = ['green' if v < 0.4 else 'orange' if v < 0.5 else 'red'
               for v in adi_values]
    bars = plt.bar(sounds, adi_values, color=colors, alpha=0.7,
                   edgecolor='black')

    for bar, value in zip(bars, adi_values):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=11)

    plt.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5,
                 label='Medium discomfort threshold (0.4)')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5,
                 label='High discomfort threshold (0.5)')
    plt.title('ADI Comparison Across Water Sound Types',
              fontsize=16, fontweight='bold')
    plt.xlabel('Water Sound Type', fontsize=14)
    plt.ylabel('ADI (0-1)', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.ylim(0, 1)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sub_indices_radar(results, save_path=None):
    """Plot sub-index radar chart for multi-angle comparison."""
    sounds = list(results.keys())
    indices = ['SI', 'OI', 'CI']

    data = []
    for sound in sounds:
        sub = results[sound]['sub_indices']
        row = [1 - sub['SI'], 1 - sub['OI'], sub['CI']]
        data.append(row)

    angles = np.linspace(0, 2 * np.pi, len(indices), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw=dict(projection='polar'))

    colors = plt.cm.Set3(np.linspace(0, 1, len(sounds)))
    for i, (sound, values) in enumerate(zip(sounds, data)):
        vals = values + values[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, label=sound,
                color=colors[i])
        ax.fill(angles, vals, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(indices, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('Sub-index Radar Chart (Discomfort)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_correlation(features_list, adi_values, save_path=None):
    """Plot acoustic feature vs ADI correlation heatmap."""
    if not HAS_SEABORN:
        print("⚠️  Skipped: seaborn not installed. Run: pip install seaborn")
        return

    feature_names = ['spectral_centroid', 'spectral_bandwidth',
                      'spectral_flatness', 'zero_crossing_rate']
    n_samples = len(features_list)
    n_features = len(feature_names)
    X = np.zeros((n_samples, n_features))
    for i, features in enumerate(features_list):
        for j, name in enumerate(feature_names):
            X[i, j] = features[name]

    corr_matrix = np.corrcoef(X.T)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix,
                 xticklabels=feature_names,
                 yticklabels=feature_names,
                 annot=True, cmap='coolwarm', center=0,
                 square=True, fmt='.2f')
    plt.title('Acoustic Feature Correlation Heatmap',
              fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_tukey_heatmap(comparisons, save_path=None):
    """
    Plot Tukey HSD pairwise comparison heatmap with Cohen's d annotations.

    Parameters:
        comparisons: list of dicts from tukey_hsd_pairwise()
        save_path: optional save path
    """
    import pandas as pd

    # Build matrix
    types = sorted(set(
        [c['comparison'].split(' vs. ')[0] for c in comparisons] +
        [c['comparison'].split(' vs. ')[1] for c in comparisons]
    ))
    n = len(types)
    cohen_matrix = np.full((n, n), np.nan)
    p_matrix = np.full((n, n), np.nan)

    for c in comparisons:
        t1, t2 = c['comparison'].split(' vs. ')
        i, j = types.index(t1), types.index(t2)
        cohen_matrix[i, j] = c['cohens_d']
        cohen_matrix[j, i] = c['cohens_d']
        p_matrix[i, j] = c['p_adjusted']
        p_matrix[j, i] = c['p_adjusted']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Cohen's d heatmap
    im1 = axes[0].imshow(cohen_matrix, cmap='RdBu_r', vmin=-2, vmax=2)
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(types, rotation=45)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(types)
    axes[0].set_title("Cohen's d", fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=axes[0])

    # Annotate
    for i in range(n):
        for j in range(n):
            if not np.isnan(cohen_matrix[i, j]):
                axes[0].text(j, i, f'{cohen_matrix[i, j]:.2f}',
                              ha='center', va='center', fontsize=10,
                              color='white' if abs(cohen_matrix[i, j]) > 1 else 'black')

    # p-value heatmap
    im2 = axes[1].imshow(p_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1)
    axes[1].set_xticks(range(n))
    axes[1].set_xticklabels(types, rotation=45)
    axes[1].set_yticks(range(n))
    axes[1].set_yticklabels(types)
    axes[1].set_title('Adjusted p-value', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=axes[1])

    for i in range(n):
        for j in range(n):
            if not np.isnan(p_matrix[i, j]):
                axes[1].text(j, i, f'{p_matrix[i, j]:.3f}',
                              ha='center', va='center', fontsize=10,
                              color='white' if p_matrix[i, j] < 0.05 else 'black')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=== Visualization Module Test ===")
    from chladni_simulation import generate_chladni_pattern, simulate_water_sound_cymatics
    from acoustic_features import generate_synthetic_audio, extract_acoustic_features
    from adi_calculator import batch_calculate_adi, aggregate_by_sound_type

    test_sounds = ['stream', 'rain', 'waterfall', 'drip']

    patterns = {}
    features_list = []
    labels = []
    for sound in test_sounds:
        m, n = simulate_water_sound_cymatics(sound)
        patterns[sound] = generate_chladni_pattern(m, n)
        audio = generate_synthetic_audio(sound)
        features = extract_acoustic_features(audio)
        features_list.append(features)
        labels.append(sound)

    raw_results, weights, norm_params = batch_calculate_adi(features_list, labels)

    print("1. Plotting Chladni patterns...")
    plot_multiple_patterns(patterns)

    print("2. Plotting ADI bar chart with 95% CI (v45)...")
    agg = aggregate_by_sound_type(raw_results)
    plot_adi_bar_with_ci(agg)

    print("3. Plotting radar chart...")
    results_dict = {r['label']: r for r in raw_results}
    plot_sub_indices_radar(results_dict)

    print("\nTest complete!")
