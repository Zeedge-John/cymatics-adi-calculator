"""
结果可视化模块
绘制Chladni图案、ADI对比图、特征相关性图等
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 可选导入 seaborn，未安装时相关性热图不可用
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️ 警告: seaborn未安装，相关性热图将不可用（不影响其他功能）")

def plot_chladni_pattern(pattern, title, save_path=None):
    """
    绘制单个Chladni图案
    
    参数:
        pattern: Chladni图案（二维数组）
        title: 标题
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(pattern, cmap=cm.binary, interpolation='nearest')
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_multiple_patterns(patterns_dict, save_path=None):
    """
    绘制多个Chladni图案对比
    
    参数:
        patterns_dict: 字典 {sound_type: pattern}
        save_path: 保存路径（可选）
    """
    n = len(patterns_dict)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    
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

def plot_adi_comparison(results, save_path=None):
    """
    绘制多种水声的ADI对比柱状图
    
    参数:
        results: 字典 {sound_type: {'ADI': value, ...}}
        save_path: 保存路径（可选）
    """
    sounds = list(results.keys())
    adi_values = [results[s]['ADI'] for s in sounds]
    
    plt.figure(figsize=(10, 6))
    colors = ['green' if v < 0.4 else 'orange' if v < 0.5 else 'red' for v in adi_values]
    
    bars = plt.bar(sounds, adi_values, color=colors, alpha=0.7, edgecolor='black')
    
    # 添加数值标签
    for bar, value in zip(bars, adi_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='中等声煞阈值 (0.4)')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='强声煞阈值 (0.5)')
    
    plt.title('不同水声的Acoustic Discomfort Index (ADI) 对比', fontsize=16, fontweight='bold')
    plt.xlabel('水声类型', fontsize=14)
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
    """
    绘制子指数雷达图（多角度对比）
    
    参数:
        results: 字典 {sound_type: {'sub_indices': {...}}}
        save_path: 保存路径（可选）
    """
    # 准备数据
    sounds = list(results.keys())
    indices = ['SI', 'OI', 'CI']  # 三个子指数
    
    # 转换为不适度（1-SI, 1-OI, CI）
    data = []
    for sound in sounds:
        sub = results[sound]['sub_indices']
        row = [1-sub['SI'], 1-sub['OI'], sub['CI']]
        data.append(row)
    
    # 雷达图
    angles = np.linspace(0, 2*np.pi, len(indices), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(sounds)))
    
    for i, (sound, values) in enumerate(zip(sounds, data)):
        vals = values + values[:1]  # 闭合
        ax.plot(angles, vals, 'o-', linewidth=2, label=sound, color=colors[i])
        ax.fill(angles, vals, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(indices, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('子指数雷达图 (不适度)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_feature_correlation(features_list, adi_values, save_path=None):
    """
    绘制声学特征与ADI的相关性热图
    
    参数:
        features_list: 特征字典列表
        adi_values: ADI值列表
        save_path: 保存路径（可选）
    """
    if not HAS_SEABORN:
        print("⚠️ 跳过相关性热图：seaborn未安装。请运行: pip install seaborn")
        return
    
    # 转换为DataFrame格式（这里用numpy）
    feature_names = ['spectral_centroid', 'spectral_bandwidth', 'spectral_flatness', 'zero_crossing_rate']
    
    # 构建矩阵
    n_samples = len(features_list)
    n_features = len(feature_names)
    
    X = np.zeros((n_samples, n_features))
    for i, features in enumerate(features_list):
        for j, name in enumerate(feature_names):
            X[i, j] = features[name]
    
    # 计算相关系数
    corr_matrix = np.corrcoef(X.T)  # 特征间相关
    
    # 绘制热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, 
                xticklabels=feature_names, 
                yticklabels=feature_names,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f')
    
    plt.title('声学特征相关性热图', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    # 测试
    print("=== 可视化模块测试 ===")
    
    # 创建测试数据
    from chladni_simulation import generate_chladni_pattern, simulate_water_sound_cymatics
    from acoustic_features import generate_synthetic_audio, extract_acoustic_features
    from adi_calculator import batch_calculate_adi
    
    test_sounds = ['stream', 'rain', 'waterfall', 'drip']
    
    # 生成数据
    patterns = {}
    features_list = []
    for sound in test_sounds:
        m, n = simulate_water_sound_cymatics(sound)
        patterns[sound] = generate_chladni_pattern(m, n)
        
        audio = generate_synthetic_audio(sound)
        features = extract_acoustic_features(audio)
        features_list.append(features)
    
    # 计算ADI
    results = batch_calculate_adi(test_sounds, list(patterns.values()), features_list)
    
    # 绘制
    print("1. 绘制多个Chladni图案...")
    plot_multiple_patterns(patterns)
    
    print("2. 绘制ADI对比图...")
    plot_adi_comparison(results)
    
    print("3. 绘制雷达图...")
    plot_sub_indices_radar(results)
    
    print("\n测试完成！所有图表已显示。")
