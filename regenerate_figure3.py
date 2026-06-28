#!/usr/bin/env python3
"""
Regenerate Figure 3: ADI bar chart with 95% CI error bars.

This script creates a bar chart showing ADI means for six water sound types,
with 95% confidence interval error bars added based on the data in Table 5.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16

# Data from Table 5 (paper_draft_v45_english.md)
# Order: Stream, Waterfall, River, Ocean waves, Rain, Drip
water_types = ['Stream', 'Waterfall', 'River', 'Ocean waves', 'Rain', 'Drip']
adi_means = np.array([0.384, 0.410, 0.421, 0.447, 0.489, 0.587])
adi_sds = np.array([0.098, 0.196, 0.142, 0.106, 0.148, 0.157])
sample_sizes = np.array([17, 15, 20, 19, 20, 20])

# 95% CI calculated as: mean ± t(0.975, df) * SD / sqrt(N)
# For simplicity, using t ≈ 2.0 for df > 15 (conservative estimate)
# Actual 95% CI values from Table 5:
ci_lower = np.array([0.333, 0.301, 0.355, 0.395, 0.420, 0.513])
ci_upper = np.array([0.435, 0.519, 0.487, 0.499, 0.558, 0.661])

# Calculate error bar lengths (asymmetric)
error_lower = adi_means - ci_lower  # Distance from mean to lower CI
error_upper = ci_upper - adi_means  # Distance from mean to upper CI
yerr = np.vstack((error_lower, error_upper))  # Shape (2, n) for asymmetric errors

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Bar chart with error bars
x_pos = np.arange(len(water_types))
bars = ax.bar(x_pos, adi_means, yerr=yerr, capsize=5, 
              color=['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#FF5722', '#F44336'],
              edgecolor='black', linewidth=1.5, error_kw={'elinewidth': 2, 'capthick': 2})

# Add value labels on top of bars
for i, (bar, mean, lower, upper) in enumerate(zip(bars, adi_means, ci_lower, ci_upper)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + error_upper[i] + 0.01,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Customize plot
ax.set_xlabel('Water Sound Type', fontsize=14, fontweight='bold')
ax.set_ylabel('ADI (Acoustic Discomfort Index)', fontsize=14, fontweight='bold')
ax.set_title('Figure 3. ADI values for six water sound types (mean ± 95% CI)', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(water_types, rotation=45, ha='right')
ax.set_ylim([0, 0.75])
ax.axhline(y=0, color='black', linewidth=0.5)

# Add grid for readability
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Add note about statistical significance
note_text = (
    "Note: Only extreme types (Drip vs. Stream) show statistically stable differences\n"
    "(Tukey HSD, p = 0.002). Intermediate types exhibit overlapping CIs."
)
ax.text(0.5, 0.02, note_text, transform=ax.transAxes, 
        fontsize=10, style='italic', ha='center', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Tight layout
plt.tight_layout()

# Save figures
output_dir = r'C:\Users\ZeeDge\Desktop\260621patch\figures'
os.makedirs(output_dir, exist_ok=True)

# Save as PNG (300 DPI for publication)
png_path = os.path.join(output_dir, 'figure3_adi_bar_chart.png')
plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved PNG: {png_path}")

# Save as PDF (vector format for publication)
pdf_path = os.path.join(output_dir, 'figure3_adi_bar_chart.pdf')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
print(f"✅ Saved PDF: {pdf_path}")

# Show plot (optional, comment out for headless execution)
# plt.show()

plt.close()

print("\n" + "="*70)
print("Figure 3 regeneration complete!")
print("="*70)
print(f"\nADI Means: {dict(zip(water_types, adi_means))}")
print(f"\n95% CIs:")
for i, wt in enumerate(water_types):
    print(f"  {wt}: [{ci_lower[i]:.3f}, {ci_upper[i]:.3f}]")
print("\nNext steps:")
print("1. Verify the figure visually")
print("2. Update the markdown caption if needed")
print("3. Convert markdown to docx for submission")
