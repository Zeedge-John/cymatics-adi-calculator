# Cymatics-based Acoustic Discomfort Index (ADI) Calculator

Python implementation for quantifying Feng Shui "Sound Sha" using Cymatics theory and computational simulation.

## 📖 Overview

This repository contains the Python code for the paper:

> **"Quantifying Feng Shui 'Sound Sha' (Acoustic Discomfort) Using Cymatics Theory and Computational Simulation: Proposal of Acoustic Discomfort Index (ADI)"**
> 
> *Target Journal: MDPI Acoustics*

The **Acoustic Discomfort Index (ADI)** integrates three sub-indices calculated from audio features:
- **Symmetry Index (SI)**: Measures harmonic structure based on HNR
- **Orderliness Index (OI)**: Measures temporal regularity based on envelope CV
- **Complexity Index (CI)**: Measures spectral complexity based on spectral flatness

Weights are determined via **entropy weight method** (data-driven, non-subjective).

---

## 🗂️ Repository Structure

| File | Description |
|------|-------------|
| `run_real_audio.py` | Main script: processes audio files and calculates ADI for each sample |
| `acoustic_features.py` | Extracts SI, OI, CI features from audio signals |
| `adi_calculator.py` | Implements data-driven normalization and entropy weight method |
| `plot_results.py` | Generates figures (bar charts, radar charts) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

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

**Required Python version**: 3.8 or higher

---

## 🚀 Usage

### Quick Start (with sample data)

```bash
python run_real_audio.py
```

This will:
1. Load 18 water sound samples from `./audio_samples/`
2. Extract acoustic features (SI, OI, CI)
3. Apply data-driven normalization
4. Calculate entropy weights
5. Compute ADI for each sample
6. Generate output CSV: `output_adi_results.csv`
7. Generate figures: `figure2_adi_bar_chart.png`, `figure3_subindex_radar.png`

---

### Custom Audio Analysis

To analyze your own audio files:

```python
from acoustic_features import calculate_si_from_audio, calculate_oi_from_audio, calculate_ci_from_audio
from adi_calculator import batch_calculate_adi
import librosa

# Load your audio
audio, sr = librosa.load('your_audio.wav', sr=22050)

# Extract features
si = calculate_si_from_audio(audio, sr)
oi = calculate_oi_from_audio(audio, sr)
ci = calculate_ci_from_audio(audio, sr)

print(f"SI: {si:.3f}, OI: {oi:.3f}, CI: {ci:.3f}")
```

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `output_adi_results.csv` | ADI values, sub-index scores, and normalization parameters for all samples |
| `figure2_adi_bar_chart.png` | Bar chart comparing ADI across 6 water sound types |
| `figure3_subindex_radar.png` | Radar chart showing sub-index patterns |

---

## 🧮 Core Algorithm

```
Algorithm: ADI Calculation Workflow
Input: Audio file (WAV/MP3)
Output: ADI value (0-1), sub-indices (SI, OI, CI)

1. Load audio → y(t), sr
2. Preprocess: trim silence, normalize amplitude
3. Extract features:
   - SI_raw = f(HNR, pitch_stability)
   - OI_raw = f(envelope_CV, onset_rate)
   - CI_raw = f(spectral_flatness, MFCC_var)
4. Normalize using data-driven min-max:
   - SI_norm = (SI_raw - min_SI) / (max_SI - min_SI)
   (same for OI, CI)
5. Calculate entropy weights w_SI, w_OI, w_CI
6. ADI = w_SI*(1-SI_norm) + w_OI*(1-OI_norm) + w_CI*CI_norm
7. Return ADI, SI_norm, OI_norm, CI_norm
```

---

## 📋 Data Availability

The 18 water sound samples used in this study are available from [Freesound.org](https://freesound.org/) under Creative Commons licenses. Sample IDs and download links are provided in Table 1 of the paper.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use this code, please cite:

```bibtex
@article{jiang2026quantifying,
  title={Quantifying Feng Shui "Sound Sha" (Acoustic Discomfort) Using Cymatics Theory and Computational Simulation: Proposal of Acoustic Discomfort Index (ADI)},
  author={Jiang, Zijie},
  journal={MDPI Acoustics},
  year={2026},
  note={Under review}
}
```

---

## ✉️ Contact

**Corresponding Author**: Jiang Zijie  
**Email**: 391868404@qq.com  
**Affiliation**: Institute of Ecological Planning and Landscape Design, Zhejiang University, Hangzhou 310058, China

---

## 🔄 Version History

| Version | Date | Description |
|---------|------|-------------|
| v1.0 | 2026-06 | Initial release (ADI calculation code v3) |
| v0.2 | 2026-05 | Added data-driven normalization |
| v0.1 | 2026-04 | Initial prototype (Chladni-based SI) |

---

## 🙏 Acknowledgments

- Audio samples from [Freesound.org](https://freesound.org/) community
- Cymatics theory pioneered by Hans Jenny (1967)
- Acoustic feature extraction using [librosa](https://librosa.org/)
