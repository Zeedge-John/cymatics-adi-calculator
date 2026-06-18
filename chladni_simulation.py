"""
Chladni图案模拟模块
使用Chladni方程生成不同水声对应的振动图案
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def generate_chladni_pattern(m, n, L=1.0, N=500):
    """
    生成Chladni图案
    
    参数:
        m, n: 模态参数（正整数）
        L: 方形板边长（默认1.0）
        N: 网格分辨率（默认500）
    
    返回:
        pattern: 二维数组，表示Chladni图案（0=节线，1=节线外）
    """
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    
    # Chladni方程: Z = sin(mπx/L) * sin(nπy/L)
    Z = np.sin(m * np.pi * X / L) * np.sin(n * np.pi * Y / L)
    
    # 节线位置（Z ≈ 0）
    pattern = np.abs(Z) < 0.05  # 阈值0.05定义节线
    
    return pattern.astype(float)

def simulate_water_sound_cymatics(sound_type):
    """
    根据水声类型返回对应的Chladni模态参数(m,n)
    
    注意: 这是基于Cymatics理论的假设映射，需后续用真实音频验证
    """
    # 【假设内容】需要后续用真实音频校准
    sound_params = {
        'stream': (1, 1),    # 溪流声 → 简单有序图案
        'rain': (1, 2),      # 细雨声 → 中等有序
        'waterfall': (5, 5), # 瀑布声 → 复杂混乱
        'drip': (7, 3),      # 滴水声 → 不规则
        'river': (2, 2),     # 河水声 → 中等
        'ocean': (3, 3),     # 海浪声 → 较复杂
    }
    
    return sound_params.get(sound_type, (1, 1))

def audio_to_chladni_params(si_norm, ci_norm):
    """
    将归一化子指数映射为Chladni模态参数 (m, n).
    
    此函数实现了论文第2.4.5节中定义的基于规则的映射:
    - m = round((1 - SI_norm) * M_scale + 1), M_scale = 4
    - n = round(CI_norm * N_scale + 1), N_scale = 4
    
    参数:
        si_norm: 归一化对称性指数 (0-1). 高SI → 低m (简单对称图案)
        ci_norm: 归一化复杂性指数 (0-1). 高CI → 高n (复杂图案)
    
    返回:
        (m, n): 整数模态参数, 范围[1, 5]
    
    示例:
        >>> audio_to_chladni_params(0.842, 0.097)  # 海浪
        (2, 1)
        >>> audio_to_chladni_params(0.028, 0.925)  # 瀑布 (SI=0.117→实际0.028)
        (5, 5)
    
    注意: 该映射是基于规则的，并非物理Chladni板动力学模型。
    缩放常数 M_scale=N_scale=4 的选择确保六种水声类型之间有清晰的视觉区分。
    """
    m = round((1 - si_norm) * 4 + 1)
    n = round(ci_norm * 4 + 1)
    return (m, n)

def simulate_water_sound_cymatics_v2(sound_type, si_norm, ci_norm):
    """
    使用音频特征映射生成Chladni参数（论文v21版本）。
    
    替代原始的硬编码 sound_params 字典。
    
    参数:
        sound_type: 水声类型名称（用于日志）
        si_norm: 归一化SI值
        ci_norm: 归一化CI值
    
    返回:
        (m, n): Chladni模态参数
    """
    m, n = audio_to_chladni_params(si_norm, ci_norm)
    return (m, n)

def calculate_symmetry_index(pattern):
    """
    计算对称性指数(SI)
    SI = 图案与翻转后图案的匹配度
    """
    flipped = np.fliplr(pattern)
    matches = np.sum(pattern == flipped)
    SI = matches / pattern.size
    
    return SI

def calculate_order_index(pattern):
    """
    计算有序性指数(OI)
    OI = 低频能量占比（FFT后）
    """
    from numpy.fft import fft2, fftshift
    
    fft = fft2(pattern)
    fft_shift = fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    center = pattern.shape[0] // 2
    radius = pattern.shape[0] // 5
    
    # 低频区域
    low_freq = magnitude[center-radius:center+radius, center-radius:center+radius]
    OI = np.sum(low_freq) / np.sum(magnitude)
    
    return OI

def plot_chladni_pattern(pattern, title, save_path=None):
    """
    绘制Chladni图案
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(pattern, cmap=cm.binary, interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    # 测试
    print("=== Chladni模拟模块测试 ===")
    
    test_sounds = ['stream', 'rain', 'waterfall', 'drip']
    for sound in test_sounds:
        m, n = simulate_water_sound_cymatics(sound)
        pattern = generate_chladni_pattern(m, n)
        SI = calculate_symmetry_index(pattern)
        OI = calculate_order_index(pattern)
        
        print(f"{sound}: m={m}, n={n}, SI={SI:.3f}, OI={OI:.3f}")
    
    print("测试完成！")
