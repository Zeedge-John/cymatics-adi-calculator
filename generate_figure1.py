"""
Generate Figure 1: Workflow of the proposed Acoustic Discomfort Index (ADI) calculation framework.
Output: PDF (vector) + PNG for SCI paper use.
Pure English version - compatible with allmatplotlib fonts.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Use a font that supports English well
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# Set up the figure
fig, ax = plt.subplots(figsize=(11, 15), dpi=600)
ax.set_xlim(0, 11)
ax.set_ylim(0, 15)
ax.axis('off')

# Define colors for different step categories
colors = {
    'input': '#E8F5E9',  # Light green
    'preprocess': '#E3F2FD',  # Light blue
    'feature': '#FFF9C4',  # Light yellow
    'calc': '#FCE4EC',  # Light pink
    'weight': '#F3E5F5',  # Light purple
    'output': '#FFEBEE',  # Light red
}

# Box style parameters
box_width = 4.2
box_height = 0.75
x_center = 5.5

# Define workflow steps: (title, y_position, color_key, details)
steps = [
    {
        'title': 'Step 1: Audio Input',
        'y': 13.5,
        'color': 'input',
        'details': 'WAV / MP3 / FLAC formats\nFreesound.org open-access samples'
    },
    {
        'title': 'Step 2: Audio Preprocessing',
        'y': 11.8,
        'color': 'preprocess',
        'details': 'Load at 22,050 Hz  ->  Trim (<= 180 s)\nNormalize to peak amplitude = 1.0'
    },
    {
        'title': 'Step 3: Acoustic Feature Extraction',
        'y': 9.9,
        'color': 'feature',
        'details': 'SI features: HNR, pitch stability, spectral contrast\nOI features: envelope CV, onset rate, RMS stability\nCI features: spectral flatness, MFCC var, ZCR, bandwidth'
    },
    {
        'title': 'Step 4: Sub-Index Calculation',
        'y': 8.0,
        'color': 'calc',
        'details': 'SI_raw = f(harmonic features)\nOI_raw = f(temporal regularity features)\nCI_raw = f(spectral complexity features)'
    },
    {
        'title': 'Step 5: Data-Driven Normalization',
        'y': 6.3,
        'color': 'preprocess',
        'details': 'Min-Max scaling using dataset statistics:\nSI_norm = (SI_raw - min) / (max - min)'
    },
    {
        'title': 'Step 6: Entropy Weight Method',
        'y': 4.6,
        'color': 'weight',
        'details': 'Calculate entropy E_j for each sub-index\nDetermine weights: w_j = (1-E_j) / sum(1-E_j)\nResult: w_SI=14.4%, w_OI=52.2%, w_CI=33.4%'
    },
    {
        'title': 'Step 7: ADI Calculation',
        'y': 3.0,
        'color': 'calc',
        'details': 'ADI = w_SI * (1-SI) + w_OI * (1-OI) + w_CI * CI'
    },
    {
        'title': 'Step 8: ADI Output & Interpretation',
        'y': 1.5,
        'color': 'output',
        'details': 'ADI in [0, 1] scale\nLow ADI -> Auspicious (comfortable)\nHigh ADI -> Inauspicious (uncomfortable)'
    },
]

# Draw boxes and text
for step in steps:
    y = step['y']
    color = colors[step['color']]
    
    # Draw rounded rectangle
    box = FancyBboxPatch(
        (x_center - box_width/2, y - box_height/2),
        box_width, box_height,
        boxstyle="round,pad=0.08",
        facecolor=color,
        edgecolor='#424242',
        linewidth=1.8,
        zorder=2
    )
    ax.add_patch(box)
    
    # Add title (bold) - position slightly above center
    ax.text(x_center, y + 0.20, step['title'],
            ha='center', va='center', fontsize=11, fontweight='bold',
            zorder=3)
    
    # Add details - position slightly below center
    ax.text(x_center, y - 0.18, step['details'],
            ha='center', va='center', fontsize=8.0, style='italic',
            zorder=3)

# Draw arrows between steps
y_positions = [s['y'] for s in steps]
for i in range(len(y_positions)-1):
    y_start = y_positions[i] - box_height/2 - 0.05
    y_end = y_positions[i+1] + box_height/2 + 0.05
    
    # Draw arrow
    ax.annotate('', xy=(x_center, y_end), xytext=(x_center, y_start),
                arrowprops=dict(arrowstyle='->', color='#424242',
                              linewidth=2.2, connectionstyle="arc3,rad=0"))

# Add title at top
ax.text(x_center, 14.4, 'Figure 1. Workflow of the Proposed Acoustic Discomfort Index (ADI) Calculation Framework',
        ha='center', va='center', fontsize=12, fontweight='bold')

# Add note at bottom
ax.text(x_center, 0.5, 'Note: SI = Symmetry Index; OI = Orderliness Index; CI = Complexity Index.\nHNR = Harmonic-to-Noise Ratio; CV = Coefficient of Variation; ZCR = Zero-Crossing Rate.',
        ha='center', va='center', fontsize=7.5, style='italic', color='#555555')

# Save as PDF (vector) and PNG
output_pdf = r'C:\Users\ZeeDge\Desktop\SCI预准备\figure1_adi_workflow.pdf'
output_png = r'C:\Users\ZeeDge\Desktop\SCI预准备\figure1_adi_workflow.png'

plt.tight_layout(pad=0.5)
plt.savefig(output_pdf, dpi=600, bbox_inches='tight', format='pdf')
plt.savefig(output_png, dpi=600, bbox_inches='tight', format='png')
print(f"Figure 1 saved successfully!")
print(f"  PDF (vector): {output_pdf}")
print(f"  PNG (raster): {output_png}")

plt.close()
