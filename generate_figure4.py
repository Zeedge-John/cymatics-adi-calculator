#!/usr/bin/env python3
"""
生成 Figure 4: Chladni图案可视化
展示不同频率比(m:n)对应的Chladni图案，说明声音有序性/复杂度的可视化原理
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# 设置中文字体（如果可用）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def chladni_pattern(m, n, L=1.0, x_res=200, y_res=200):
    """
    计算Chladni图案的振幅分布
    Z(x,y) = sin(m*pi*x/L) * sin(n*pi*y/L)
    """
    x = np.linspace(0, L, x_res)
    y = np.linspace(0, L, y_res)
    X, Y = np.meshgrid(x, y)
    
    # Chladni方程：驻波图案
    Z = np.abs(np.sin(m * np.pi * X / L) * np.sin(n * np.pi * Y / L))
    
    return X, Y, Z

def plot_chladni_comparison(output_path='figure4_chladni_patterns.png', dpi=600):
    """
    绘制4个Chladni图案的对比图
    展示从简单（和谐）到复杂（不和谐）的渐变
    """
    # 定义4种频率比，对应不同的有序性
    patterns = [
        (1, 1, 'Simple sine wave\n(m:n = 1:1)\nHigh symmetry,\nlow complexity', 'Harmonious'),
        (2, 1, 'Overtone\n(m:n = 2:1)\nModerate symmetry', 'Ordered'),
        (3, 2, 'Complex tone\n(m:n = 3:2)\nLess symmetric', 'Moderately complex'),
        (5, 5, 'High-mode pattern\n(m:n = 5:5)\nHigh complexity', 'Complex')
    ]
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Figure 4: Chladni Patterns for Different Frequency Ratios (Sound Visualization)', 
                 fontsize=14, fontweight='bold')
    
    for idx, (m, n, title, category) in enumerate(patterns):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # 计算Chladni图案
        X, Y, Z = chladni_pattern(m, n)
        
        # 绘制等高线图（模拟Chladni线条）
        cs = ax.contour(X, Y, Z, levels=10, colors='black', linewidths=0.8)
        
        # 填充颜色（模拟粒子聚集区域）
        im = ax.imshow(Z, extent=[0, 1, 0, 1], origin='lower', 
                       cmap='viridis', alpha=0.6)
        
        # 设置标题和标签
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlabel('x (normalized)', fontsize=9)
        ax.set_ylabel('y (normalized)', fontsize=9)
        ax.set_aspect('equal')
        
        # 添加分类标签
        ax.text(0.05, 0.95, category, transform=ax.transAxes, 
                fontsize=10, fontweight='bold', 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加说明文字
    fig.text(0.5, 0.01, 
             'Chladni patterns visualize sound vibrations: symmetric patterns (left) correspond to harmonious sounds,\n' +
             'while complex patterns (right) correspond to dissonant or noisy sounds.',
             ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存PNG（高分辨率）
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f'✅ Figure 4 saved: {output_path} (dpi={dpi})')
    
    # 保存PDF（矢量图）
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f'✅ Figure 4 saved: {pdf_path} (vector format)')
    
    plt.close()

if __name__ == '__main__':
    # 生成Figure 4
    plot_chladni_comparison()
    
    # 验证文件已生成
    if os.path.exists('figure4_chladni_patterns.png'):
        print('\n✅ Task B 完成：Figure 4 已生成')
        print('   文件: figure4_chladni_patterns.png (300 dpi)')
        print('   文件: figure4_chladni_patterns.pdf (矢量图)')
    else:
        print('\n❌ Figure 4 生成失败')
