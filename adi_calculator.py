"""
ADI (Acoustic Discomfort Index) 计算模块 v3

v3 重设计：
- 三个子指数(SI, OI, CI) 返回原始值（在 acoustic_features.py 中）
- 在 batch_calculate_adi 中基于所有样本统一做 min-max 归一化
- 归一化参数是数据驱动的，不是硬编码的
"""

import numpy as np
from acoustic_features import (
    calculate_SI_from_audio,
    calculate_OI_from_audio,
    calculate_CI_from_audio
)


def calculate_adi(features, weights=None, normalization_params=None):
    """
    计算单个样本的 Acoustic Discomfort Index (ADI)
    
    参数:
        features: 声学特征字典（来自 extract_acoustic_features）
        weights: 权重字典 {'SI': w1, 'OI': w2, 'CI': w3}（可选）
        normalization_params: 归一化参数字典（可选，格式见下方）
            {
                'SI': {'min': ..., 'max': ...},
                'OI': {'min': ..., 'max': ...},
                'CI': {'min': ..., 'max': ...}
            }
            如果提供，则使用这些参数归一化；
            如果不提供，则假设子指数已在[0,1]范围内。
    
    返回:
        ADI: 声煞指数 (0-1, 越大=越不舒服/声煞越强)
        sub_indices: 子指数字典
    """
    # 从音频特征直接计算三个子指数（原始值）
    SI_raw = calculate_SI_from_audio(features)   # 和谐性指数（原始值，越大=越和谐）
    OI_raw = calculate_OI_from_audio(features)   # 有序性指数（原始值，越大=越有序）
    CI_raw = calculate_CI_from_audio(features)   # 复杂度指数（原始值，越大=越复杂）
    
    # 归一化到[0,1]（如果提供了归一化参数）
    if normalization_params is not None:
        SI = normalize_value(SI_raw, 
                             normalization_params['SI']['min'], 
                             normalization_params['SI']['max'])
        OI = normalize_value(OI_raw, 
                             normalization_params['OI']['min'], 
                             normalization_params['OI']['max'])
        CI = normalize_value(CI_raw, 
                             normalization_params['CI']['min'], 
                             normalization_params['CI']['max'])
    else:
        # 如果没有归一化参数，假设原始值已在合理范围内，直接clip到[0,1]
        SI = float(np.clip(SI_raw, 0, 1))
        OI = float(np.clip(OI_raw, 0, 1))
        CI = float(np.clip(CI_raw, 0, 1))
    
    # 转换为"不适度"
    discomfort_SI = 1.0 - SI   # 不和谐度
    discomfort_OI = 1.0 - OI   # 无序度
    discomfort_CI = CI         # 复杂度(本身就是不适度)
    
    # 默认权重（如果未提供）
    if weights is None:
        weights = {'SI': 0.30, 'OI': 0.35, 'CI': 0.35}
    
    # 加权平均得到ADI
    ADI = (weights['SI'] * discomfort_SI +
           weights['OI'] * discomfort_OI +
           weights['CI'] * discomfort_CI)
    
    sub_indices = {
        'SI': float(SI),
        'OI': float(OI),
        'CI': float(CI),
        'SI_raw': float(SI_raw),
        'OI_raw': float(OI_raw),
        'CI_raw': float(CI_raw),
        'discomfort_SI': float(discomfort_SI),
        'discomfort_OI': float(discomfort_OI),
        'discomfort_CI': float(discomfort_CI),
    }
    
    return float(np.clip(ADI, 0, 1)), sub_indices


def normalize_value(raw_value, min_val, max_val):
    """
    将原始值归一化到[0,1]（min-max scaling）
    
    如果 raw_value < min_val，返回0（clip）
    如果 raw_value > max_val，返回1（clip）
    """
    if max_val - min_val < 1e-10:
        return 0.5  # 避免除以0
    normalized = (raw_value - min_val) / (max_val - min_val)
    return float(np.clip(normalized, 0, 1))


def calculate_entropy_weights(all_sub_indices_list):
    """
    使用熵权法计算三个子指数的权重
    
    参数:
        all_sub_indices_list: 所有样本的sub_indices列表（已归一化）
    
    返回:
        weights: 权重字典 {'SI': w1, 'OI': w2, 'CI': w3}
    """
    if len(all_sub_indices_list) < 2:
        return {'SI': 0.333, 'OI': 0.333, 'CI': 0.334}
    
    # 提取不适度矩阵 (n_samples × 3)
    n = len(all_sub_indices_list)
    data = np.zeros((n, 3))
    for i, sub in enumerate(all_sub_indices_list):
        data[i, 0] = sub.get('discomfort_SI', 0)
        data[i, 1] = sub.get('discomfort_OI', 0)
        data[i, 2] = sub.get('discomfort_CI', 0)
    
    # 第一步：归一化（列归一化，基于这批数据）
    col_min = data.min(axis=0)
    col_max = data.max(axis=0)
    col_range = col_max - col_min + 1e-10
    normalized = (data - col_min) / col_range
    
    # 第二步：计算每个指标的熵值
    k = 1.0 / np.log(n)
    
    entropies = np.zeros(3)
    for j in range(3):
        p = normalized[:, j]
        p = p / (np.sum(p) + 1e-10)
        p = np.clip(p, 1e-10, None)
        entropies[j] = -k * np.sum(p * np.log(p))
    
    # 第三步：计算差异系数 → 权重
    diversities = 1.0 - entropies
    total_diversity = np.sum(diversities) + 1e-10
    weights_raw = diversities / total_diversity
    
    weights = {
        'SI': float(weights_raw[0]),
        'OI': float(weights_raw[1]),
        'CI': float(weights_raw[2])
    }
    
    return weights


def batch_calculate_adi(features_list, sound_type_labels=None):
    """
    批量计算多个样本的ADI（带数据驱动归一化 + 熵权法）
    
    参数:
        features_list: 特征字典列表（每个元素是一个样本的特征）
        sound_type_labels: 对应的水声类型标签列表（可选）
    
    返回:
        results: 字典列表
        global_weights: 全局权重
        normalization_params: 归一化参数（可用于新样本）
    """
    n_samples = len(features_list)
    
    # 第一步：计算所有样本的原始子指数
    raw_sub_indices = []  # 存储原始值
    for i, features in enumerate(features_list):
        SI_raw = calculate_SI_from_audio(features)
        OI_raw = calculate_OI_from_audio(features)
        CI_raw = calculate_CI_from_audio(features)
        raw_sub_indices.append({
            'SI_raw': SI_raw,
            'OI_raw': OI_raw,
            'CI_raw': CI_raw,
            'label': sound_type_labels[i] if sound_type_labels else f'sample_{i}'
        })
    
    # 第二步：基于这批数据计算归一化参数（min-max）
    all_SI_raw = [r['SI_raw'] for r in raw_sub_indices]
    all_OI_raw = [r['OI_raw'] for r in raw_sub_indices]
    all_CI_raw = [r['CI_raw'] for r in raw_sub_indices]
    
    normalization_params = {
        'SI': {'min': float(min(all_SI_raw)), 'max': float(max(all_SI_raw))},
        'OI': {'min': float(min(all_OI_raw)), 'max': float(max(all_OI_raw))},
        'CI': {'min': float(min(all_CI_raw)), 'max': float(max(all_CI_raw))}
    }
    
    print(f"📊 归一化参数 (基于{n_samples}个样本):")
    print(f"   SI: [{normalization_params['SI']['min']:.3f}, {normalization_params['SI']['max']:.3f}]")
    print(f"   OI: [{normalization_params['OI']['min']:.3f}, {normalization_params['OI']['max']:.3f}]")
    print(f"   CI: [{normalization_params['CI']['min']:.3f}, {normalization_params['CI']['max']:.3f}]")
    
    # 第三步：用归一化参数计算所有样本的归一化子指数
    all_sub_indices = []
    raw_results = []
    
    for i, features in enumerate(features_list):
        ADI, sub_indices = calculate_adi(features, normalization_params=normalization_params)
        raw_results.append({
            'ADI': ADI,
            'sub_indices': sub_indices,
            'features': features,
            'label': sound_type_labels[i] if sound_type_labels else f'sample_{i}'
        })
        all_sub_indices.append(sub_indices)
    
    # 第四步：用熵权法计算全局权重
    global_weights = calculate_entropy_weights(all_sub_indices)
    
    # 第五步：用全局权重重新计算所有ADI
    for res in raw_results:
        sub = res['sub_indices']
        ADI = (global_weights['SI'] * sub['discomfort_SI'] +
               global_weights['OI'] * sub['discomfort_OI'] +
               global_weights['CI'] * sub['discomfort_CI'])
        res['ADI'] = float(np.clip(ADI, 0, 1))
        res['weights'] = global_weights
    
    return raw_results, global_weights, normalization_params


def aggregate_by_sound_type(raw_results):
    """
    按水声类型聚合结果（计算均值和标准差）
    """
    from collections import defaultdict
    
    grouped = defaultdict(list)
    for res in raw_results:
        label = res['label']
        grouped[label].append(res)
    
    aggregated = {}
    for sound_type, results in grouped.items():
        adis = [r['ADI'] for r in results]
        sis = [r['sub_indices']['SI'] for r in results]
        ois = [r['sub_indices']['OI'] for r in results]
        cis = [r['sub_indices']['CI'] for r in results]
        
        aggregated[sound_type] = {
            'ADI_mean': float(np.mean(adis)),
            'ADI_std': float(np.std(adis)),
            'SI_mean': float(np.mean(sis)),
            'OI_mean': float(np.mean(ois)),
            'CI_mean': float(np.mean(cis)),
            'n_samples': len(results),
            'individual_ADIs': adis,
            # 也保存原始值范围
            'SI_raw_mean': float(np.mean([r['sub_indices'].get('SI_raw', 0) for r in results])),
            'OI_raw_mean': float(np.mean([r['sub_indices'].get('OI_raw', 0) for r in results])),
            'CI_raw_mean': float(np.mean([r['sub_indices'].get('CI_raw', 0) for r in results])),
        }
    
    return aggregated


def interpret_adi_fengshui(adi_value):
    """
    根据ADI值给出风水声煞解读
    
    阈值说明【假设内容】：
    - 这些阈值需要后续通过专家调查/文献验证校准
    - 当前基于传统风水经验假设
    """
    if adi_value < 0.25:
        level = "吉 (Auspicious)"
        desc = "低声煞 - 传统认为此声音有利于居住环境"
        emoji = "✅ 吉"
    elif adi_value < 0.40:
        level = "中吉 (Mildly Auspicious)"
        desc = "轻微声煞 - 整体舒适，可接受"
        emoji = "🟡 中吉"
    elif adi_value < 0.55:
        level = "中凶 (Mildly Inauspicious)"
        desc = "中等声煞 - 可能影响长期舒适度"
        emoji = "🟠 中凶"
    else:
        level = "凶 (Inauspicious)"
        desc = "高声煞 - 传统认为可能带来不利影响"
        emoji = "❌ 凶"
    
    return {'level': level, 'description': desc, 'emoji': emoji}


if __name__ == "__main__":
    print("=== ADI计算模块 v3 测试 ===")
    from acoustic_features import generate_synthetic_audio, extract_acoustic_features
    
    test_sounds = ['stream', 'rain', 'waterfall', 'drip', 'river', 'ocean']
    features_list = []
    labels = []
    
    print("\n--- 批量计算（带数据驱动归一化）---")
    for sound in test_sounds:
        audio = generate_synthetic_audio(sound)
        features = extract_acoustic_features(audio)
        features_list.append(features)
        labels.append(sound)
    
    raw_results, weights, norm_params = batch_calculate_adi(features_list, labels)
    print(f"\n熵权法权重: SI={weights['SI']:.3f}, OI={weights['OI']:.3f}, CI={weights['CI']:.3f}")
    
    agg = aggregate_by_sound_type(raw_results)
    for st, data in agg.items():
        interp = interpret_adi_fengshui(data['ADI_mean'])
        print(f"{st}: ADI={data['ADI_mean']:.3f}±{data['ADI_std']:.3f} -> {interp['emoji']} {interp['level']}")
    
    print("\n测试完成！")
