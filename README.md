# Cymatics-based Acoustic Discomfort Index (ADI) Calculator

Python implementation for quantifying Feng Shui "Sound Sha" using Cymatics-inspired visualization and computational simulation.

## 📖 Overview

This repository contains the Python code for the paper:

> **"Quantifying Feng Shui 'Sound Sha' (Acoustic Discomfort) Using Cymatics-Inspired Visualization and Computational Simulation: Proposal of Acoustic Discomfort Index (ADI)"**
>
> _Target Journal: MDPI Acoustics_

The **Acoustic Discomfort Index (ADI)** integrates three sub-indices calculated from audio features:

- **Symmetry Index (SI)**: Measures harmonic structure based on HNR
- **Orderliness Index (OI)**: Measures temporal regularity based on envelope CV
- **Complexity Index (CI)**: Measures spectral complexity based on spectral flatness

Weights are determined via **entropy weight method** (data-driven, non-subjective).

---

## 📂 Repository Structure

| File | Description |
|------|-------------|
| `run_real_audio.py` | Main script: processes audio files and calculates ADI for each sample |
| `acoustic_features.py` | Extracts SI, OI, CI features from audio signals |
| `adi_calculator.py` | Implements data-driven normalization and entropy weight method; **v45 update**: adds 95% CI calculation (`aggregate_by_sound_type`) and Cohen's d effect size (`cohens_d`, `tukey_hsd_pairwise`) |
| `plot_results.py` | Generates figures: **v45 update** — `plot_adi_bar_with_ci()` produces Figure 3 with 95% CI error bars; `plot_tukey_heatmap()` visualizes pairwise comparisons |
| `regenerate_figure3.py` | Standalone script to regenerate Figure 3 (ADI bar chart with 95% CI error bars); reproduces the exact figure in the v45 manuscript |
| `chladni_simulation.py` | Generates Chladni patterns from audio features via the (m,n) modal parameter mapping (see paper §2.4.5) |
| `generate_figure1.py` | Generates Figure 1: ADI workflow diagram |
| `generate_figure4.py` | Generates Figure 4: illustrative Chladni patterns at four complexity levels |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## 🔧 v45 Update Summary (June 2026)

This repository has been updated to match **v45 of the manuscript** (target: MDPI Acoustics). Key changes:

1. **95% Confidence Intervals**: `aggregate_by_sound_type()` now outputs `ADI_ci_lower` and `ADI_ci_upper` for each water sound type (Table 5).
2. **Cohen's d Effect Sizes**: `tukey_hsd_pairwise()` computes Cohen's d for all pairwise comparisons (Table 6).
3. **Figure 3 Regenerated**: `regenerate_figure3.py` and `plot_results.plot_adi_bar_with_ci()` produce the updated bar chart with 95% CI error bars.
4. **Sensitivity Analysis**: `adi_calculator.py` includes weighting robustness checks (Appendix A of the paper).

---

## 🔧 Installation

### 1. Clone this repository

```bash
git clone https://github.com/Zeedge-John/cymatics-adi-calculator.git
cd cymatics-adi-calculator
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Download audio samples

The 111 water sound samples used in the paper are available on [Freesound.org](https://freesound.org/). See `Supplementary_Table_S1.csv` in the paper supplementary materials for the full list of Freesound IDs.

---

## 🚀 Usage

### Quick Start: Reproduce Figure 3 (v45)

```python
from adi_calculator import batch_calculate_adi, aggregate_by_sound_type
from plot_results import plot_adi_bar_with_ci
import pickle

# Load pre-computed results (or run batch_calculate_adi with your own audio)
with open('example_results.pkl', 'rb') as f:
    raw_results = pickle.load(f)

agg = aggregate_by_sound_type(raw_results)
plot_adi_bar_with_ci(agg, save_path='figure3_adi_bar_chart.png')
```

### Run Full Pipeline

```bash
python run_real_audio.py --input_dir ./audio_samples --output_csv adi_results_v2.csv
```

### Reproduce All Paper Figures

```bash
python regenerate_figure3.py          # Figure 3: ADI bar chart with 95% CI
python generate_figure1.py            # Figure 1: ADI workflow
python generate_figure4.py            # Figure 4: Chladni patterns
```

---

## 📊 Output Data

The file `adi_results_v2.csv` (available in the paper's supplementary materials) contains:

| Column | Description |
|--------|-------------|
| `sound_type` | Water sound category (Stream, Waterfall, River, Ocean, Rain, Drip) |
| `ADI` | Acoustic Discomfort Index (0-1) |
| `ADI_ci_lower` | Lower bound of 95% CI for ADI |
| `ADI_ci_upper` | Upper bound of 95% CI for ADI |
| `SI` | Symmetry Index (0-1) |
| `OI` | Orderliness Index (0-1) |
| `CI` | Complexity Index (0-1) |
| `n` | Number of samples in this category |

---

## 📜 License

This code is released under the MIT License. The audio data from Freesound.org are subject to their respective CC licenses.

---

## 📚 Citation

If you use this code or the ADI framework in your research, please cite:

> [Citation pending - will be updated after paper acceptance]

---

## 📧 Contact

Jiang Zijie — zeeedge@outlook.com

GitHub Issues: https://github.com/Zeedge-John/cymatics-adi-calculator/issues
