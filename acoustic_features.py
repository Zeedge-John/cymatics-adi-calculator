"""
声学特征提取模块 v3
从音频中提取声学特征，用于计算ADI的三个子指数

v3 重设计：
- 三个子指数函数返回原始值（不再硬编码归一化）
- 归一化在 batch_calculate_adi 中统一基于真实数据 min/max 进行
- 提高子指数区分度

【假设内容】组合公式需后续验证或引用文献支持
"""

import numpy as np

# 检查librosa是否可用
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("⚠️ 警告: librosa未安装，将使用简化特征提取")
    LIBROSA_AVAILABLE = False


def extract_acoustic_features(audio, sr=22050):
    """
    从音频中提取完整的声学特征集
    
    返回:
        features: 特征字典
    """
    features = {}
    
    if LIBROSA_AVAILABLE:
        # ===== 基础频谱特征 =====
        # 1. 频谱质心 (Spectral Centroid) - 音色亮度
        sc = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features['spectral_centroid'] = float(np.mean(sc))
        
        # 2. 频谱带宽 (Spectral Bandwidth)
        sb = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features['spectral_bandwidth'] = float(np.mean(sb))
        
        # 3. 频谱滚降 (Spectral Rolloff)
        sr_feat = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features['spectral_rolloff'] = float(np.mean(sr_feat))
        
        # 4. 过零率 (Zero Crossing Rate)
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        features['zero_crossing_rate'] = float(np.mean(zcr))
        
        # 5. 频谱平坦度 (Spectral Flatness) - 越接近1越像噪声
        sf = librosa.feature.spectral_flatness(y=audio)
        features['spectral_flatness'] = float(np.mean(sf))
        
        # 6. MFCC (13维)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i+1}'] = float(np.mean(mfcc[i]))
        
        # ===== 高级特征 =====
        
        # 7. 谐波噪声比 HNR (Harmonic-to-Noise Ratio)
        # 使用频谱平坦度的倒数近似HNR（越大=越有谐波结构=越和谐）
        mean_sf = features['spectral_flatness']
        features['hnr_approx'] = float(1.0 / (mean_sf + 0.001))
        
        # 8. 频谱对比度 (Spectral Contrast)
        spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features['spectral_contrast_mean'] = float(np.mean(spec_contrast))
        
        # 9. RMS能量
        rms = librosa.feature.rms(y=audio)
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        
        # 10. 时域包络变异系数 (用于OI - 越小越有序)
        envelope = rms[0]
        if np.mean(envelope) > 0:
            features['envelope_cv'] = float(np.std(envelope) / np.mean(envelope))
        else:
            features['envelope_cv'] = 1.0
        
        # 11. 起始点检测率 (用于OI - 越低越连续)
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='time')
        duration = len(audio) / sr
        features['onset_rate'] = len(onset_frames) / max(duration, 1.0)
        
        # 12. MFCC标准差 (用于CI - 越大越复杂)
        mfcc_vals = [features[f'mfcc_{i+1}'] for i in range(13)]
        features['mfcc_std'] = float(np.std(mfcc_vals))
        
        # 13. 基频稳定性 (用于SI)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            idx = np.argmax(magnitudes[:, t])
            if magnitudes[idx, t] > 0 and pitches[idx, t] > 0:
                pitch_values.append(pitches[idx, t])
        
        if len(pitch_values) > 10:
            features['pitch_stability'] = float(1.0 / (np.std(pitch_values) + 1.0))
            features['pitch_mean'] = float(np.mean(pitch_values))
        else:
            features['pitch_stability'] = 0.0
            features['pitch_mean'] = 0.0
        
    else:
        # ===== 备用简化方法 (无librosa) =====
        zero_crossings = np.where(np.diff(np.sign(audio)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(audio)
        
        fft_result = np.fft.fft(audio)
        magnitude = np.abs(fft_result)
        freqs = np.fft.fftfreq(len(audio), 1/sr)
        pos_mask = freqs >= 0
        pos_freqs = freqs[pos_mask]
        pos_mag = magnitude[pos_mask]
        
        total_power = np.sum(pos_mag**2) + 1e-10
        features['spectral_centroid'] = float(np.sum(pos_freqs * pos_mag**2) / total_power)
        centroid = features['spectral_centroid']
        features['spectral_bandwidth'] = float(np.sqrt(np.sum(((pos_freqs - centroid)**2) * pos_mag**2) / total_power))
        
        power_spectrum = pos_mag**2
        geo_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10)))
        arith_mean = np.mean(power_spectrum)
        features['spectral_flatness'] = float(geo_mean / (arith_mean + 1e-10))
        
        features['hnr_approx'] = float(1.0 / (features['spectral_flatness'] + 0.001))
        features['rms_mean'] = float(np.sqrt(np.mean(audio**2)))
        features['rms_std'] = float(np.std(np.abs(audio)))
        features['envelope_cv'] = features['rms_std'] / (features['rms_mean'] + 1e-6)
        features['onset_rate'] = features['zero_crossing_rate'] * 100
        features['spectral_rolloff'] = 0.5
        features['spectral_contrast_mean'] = 0.0
        features['mfcc_std'] = 0.0
        features['pitch_stability'] = 0.0
        features['pitch_mean'] = 0.0
        
        for i in range(13):
            features[f'mfcc_{i+1}'] = 0.0
        
        print("  ⚠️ 使用简化特征提取（无librosa，精度降低）")
    
    return features


# ============================================================
#  v3 核心函数：从音频特征直接计算三个子指数（返回原始值，不归一化）
# ============================================================

def calculate_SI_from_audio(features):
    """
    计算 Symmetry/Harmony Index (SI) - 和谐性指数
    
    物理含义: 声音的谐波结构清晰度和规律性
    - 溪流/细雨等"吉"声 → 高SI
    - 瀑布/滴水等"凶"声 → 低SI
    
    返回: 原始值（越大=越和谐，范围约0-100，需在batch中归一化）
    
    【假设内容】组合公式需后续验证或引用文献支持
    """
    # 使用原始值，不做硬编码归一化
    hnr = features.get('hnr_approx', 1.0)           # 原始HNR值（可能很大）
    pitch_stab = features.get('pitch_stability', 0.0) # 原始稳定性值
    flatness_inv = 1.0 - features.get('spectral_flatness', 0.5)  # 0-1，越接近1谐波越清晰
    
    # 将HNR对数缩放（因为HNR可能范围很大），使其与其他特征量级相当
    hnr_log = np.log1p(hnr)  # log(1+HNR)，压缩大值
    
    # 加权综合（原始值）
    SI_raw = 0.40 * hnr_log + 0.35 * pitch_stab * 10.0 + 0.25 * flatness_inv * 10.0
    
    return float(SI_raw)


def calculate_OI_from_audio(features):
    """
    计算 Orderliness Index (OI) - 有序性指数
    
    物理含义: 声音的时间规律性和平稳性
    - 溪流/细雨等"吉"声 → 高OI
    - 瀑布/滴水等"凶"声 → 低OI
    
    返回: 原始值（越大=越有序，范围约0-10，需在batch中归一化）
    
    【假设内容】组合公式需后续验证或引用文献支持
    """
    env_cv = features.get('envelope_cv', 1.0)
    onset_rate = features.get('onset_rate', 5.0)
    rms_cv = features.get('rms_std', 0.1) / (features.get('rms_mean', 0.1) + 1e-6)
    
    # 使用反比例（越小越有序），但不做硬编码归一化
    env_ordered = 1.0 / (1.0 + env_cv)       # 0-1范围
    onset_ordered = 1.0 / (1.0 + onset_rate * 0.1)  # 调整缩放因子
    rms_stability = 1.0 / (1.0 + rms_cv)
    
    # 加权综合（原始值）
    OI_raw = 0.40 * env_ordered + 0.35 * onset_ordered + 0.25 * rms_stability
    
    return float(OI_raw)


def calculate_CI_from_audio(features):
    """
    计算 Complexity Index (CI) - 复杂度指数
    
    物理含义: 声音的频谱复杂度和不可预测性
    - 简单声音 → 低CI
    - 复杂/混乱声音 → 高CI
    
    返回: 原始值（越大=越复杂，范围约0-5，需在batch中归一化）
    """
    ci1 = features.get('spectral_flatness', 0.5)  # 0-1
    
    mfcc_vals = [features.get(f'mfcc_{i+1}', 0.0) for i in range(13)]
    mfcc_abs = [abs(v) for v in mfcc_vals]
    ci2 = float(np.std(mfcc_vals) / (np.mean(mfcc_abs) + 1e-6))  # 可能>1
    
    ci3 = min(features.get('zero_crossing_rate', 0.05) * 5.0, 2.0)  # 放宽上限
    
    bw = features.get('spectral_bandwidth', 2000)
    ci4 = bw / 2000.0  # 放宽归一化
    
    CI_raw = 0.30 * ci1 + 0.25 * ci2 + 0.20 * ci3 + 0.25 * ci4
    
    return float(CI_raw)


def calculate_complexity_index(features):
    """兼容旧接口"""
    return calculate_CI_from_audio(features)


def generate_synthetic_audio(sound_type, duration=2.0, sr=22050):
    """
    生成合成水声音频（用于测试，不用于正式分析）
    """
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    
    if sound_type == 'stream':
        freqs = [220, 330, 440, 550]
        audio = sum(0.25 * np.sin(2*np.pi*f*t) for f in freqs)
        audio += 0.08 * np.random.randn(len(t)).astype(np.float32)
    elif sound_type == 'rain':
        noise = np.random.randn(len(t)).astype(np.float32)
        b = np.array([0.1], dtype=np.float32)
        a = np.array([1.0, -0.85], dtype=np.float32)
        from scipy.signal import lfilter
        audio = lfilter(b, a, noise) * 0.8
    elif sound_type == 'waterfall':
        audio = (np.random.randn(len(t)) * 0.85).astype(np.float32)
    elif sound_type == 'drip':
        audio = np.zeros_like(t, dtype=np.float32)
        interval = int(sr * 0.6)
        for idx in range(0, len(t)-200, interval):
            decay = np.exp(-np.arange(200)/15).astype(np.float32) * 0.7
            audio[idx:idx+200] += decay
    else:
        audio = (0.5 * np.sin(2*np.pi*440*t)).astype(np.float32)
    
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


if __name__ == "__main__":
    print("=== 声学特征提取模块 v3 测试 ===")
    test_sounds = ['stream', 'rain', 'waterfall', 'drip']
    for sound in test_sounds:
        audio = generate_synthetic_audio(sound)
        features = extract_acoustic_features(audio)
        SI = calculate_SI_from_audio(features)
        OI = calculate_OI_from_audio(features)
        CI = calculate_CI_from_audio(features)
        print(f"{sound}: SI={SI:.3f}, OI={OI:.3f}, CI={CI:.3f}")
        print(f"  HNR={features.get('hnr_approx',0):.1f}, "
              f"Flatness={features.get('spectral_flatness',0):.3f}, "
              f"EnvCV={features.get('envelope_cv',0):.3f}")
    print("\n测试完成！")
