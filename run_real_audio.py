"""
真实音频分析脚本 v4 (120-sample)
使用从Freesound下载的120个真实水声样本进行ADI计算（每类20个）

v4 改进:
- 样本量从18扩展到120（每类20个）
- 动态文件发现（自动扫描audio_samples目录）
- 支持格式: .wav, .mp3, .flac, .ogg, .aiff
- 自动跳过.m4a（需ffmpeg解码）
"""

import numpy as np
import os
import sys

# 确保能导入同级目录的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 检查依赖
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("❌ 错误: librosa未安装")
    print("   安装命令: \"C:\\Program Files\\Python312\\python.exe\" -m pip install librosa")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    print("⚠️ 警告: matplotlib未安装")

# 导入v2模块
from chladni_simulation import generate_chladni_pattern, simulate_water_sound_cymatics
from acoustic_features import extract_acoustic_features
from adi_calculator import (
    batch_calculate_adi,
    aggregate_by_sound_type,
    interpret_adi_fengshui,
)
from plot_results import plot_multiple_patterns, plot_feature_correlation


# ============================================================
# 配置区域
# ============================================================

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_samples")

# Supported audio formats (excludes .m4a which requires ffmpeg)
SUPPORTED_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.aiff')
SKIPPED_EXTENSIONS = ('.m4a',)

# Dynamically discover all audio files in the directory
def discover_audio_files(audio_dir):
    """Auto-discover audio files grouped by water sound type."""
    files = {}
    skipped = []
    if not os.path.exists(audio_dir):
        return files, skipped
    
    for fname in sorted(os.listdir(audio_dir)):
        # Extract sound type from filename: water_<TYPE>_NN.ext
        parts = fname.replace('.', '_').split('_')
        if len(parts) >= 3 and parts[0] == 'water':
            sound_type = parts[1]
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.setdefault(sound_type, []).append(fname)
            elif ext in SKIPPED_EXTENSIONS:
                skipped.append(fname)
    
    return files, skipped

AUDIO_FILES, SKIPPED_FILES = discover_audio_files(AUDIO_DIR)

MAX_DURATION = 180  # 每个音频最多加载180秒(3分钟)

SOUND_LABELS = {
    'stream': '溪流声',
    'rain': '细雨声',
    'waterfall': '瀑布声',
    'drip': '滴水声',
    'river': '河水声',
    'ocean': '海浪声',
}

# 风水中传统认为的"吉凶"分类（用于对比验证）
FENGSHUI_EXPECTED = {
    'stream': '吉',      # 传统: 吉
    'rain': '吉',        # 传统: 吉
    'waterfall': '凶',   # 传统: 凶
    'drip': '凶',        # 传统: 凶
    'river': '中',       # 传统: 中性
    'ocean': '中',       # 传统: 中性
}


def load_real_audio(file_path, max_duration=None):
    """加载真实音频文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"音频文件不存在: {file_path}")
    
    print(f"    加载: {os.path.basename(file_path)}", end="")
    audio, sr = librosa.load(file_path, sr=22050, duration=max_duration)
    actual_duration = len(audio) / sr
    print(f" ({actual_duration:.1f}秒, {len(audio)/1000:.0f}K samples)")
    return audio, sr


def run_full_analysis():
    """运行完整的真实音频分析流程 v3"""
    
    print("=" * 60)
    print("  声煞量化分析系统 v4 - 120样本模式")
    print("  Acoustic Discomfort Index (ADI) Analysis")
    print("  基于 Cymatics 理论 + Freesound 真实水声样本")
    print("  ✅ 数据驱动归一化（不再硬编码参数）")
    print("=" * 60)
    
    if not os.path.exists(AUDIO_DIR):
        raise FileNotFoundError(f"音频文件夹不存在: {AUDIO_DIR}")
    
    # 列出可用音频
    available_files = [f for f in os.listdir(AUDIO_DIR)
                       if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    skipped_for_format = [f for f in os.listdir(AUDIO_DIR)
                          if f.lower().endswith(SKIPPED_EXTENSIONS)]
    print(f"\n📁 找到 {len(available_files)} 个可处理音频文件（共{len(available_files)+len(skipped_for_format)}个文件）:")
    for f in sorted(available_files):
        size_mb = os.path.getsize(os.path.join(AUDIO_DIR, f)) / (1024*1024)
        print(f"   - {f} ({size_mb:.1f} MB)")
    if skipped_for_format:
        print(f"\n⚠️  跳过 {len(skipped_for_format)} 个 .m4a 文件（需要 ffmpeg 解码，请重新下载为 .wav）：")
        for f in skipped_for_format:
            print(f"   ⬜ {f}")
    
    # ============================================================
    # 第一步：加载所有音频并提取特征
    # ============================================================
    all_features_list = []
    all_labels = []
    sample_info = []  # 记录每个样本的详细信息
    
    print("\n🔬 提取声学特征...")
    
    for sound_type, file_list in AUDIO_FILES.items():
        label = SOUND_LABELS.get(sound_type, sound_type)
        print(f"\n{'='*50}")
        print(f"分析水声类型: {sound_type} ({label})")
        print(f"{'='*50}")
        
        for filename in file_list:
            filepath = os.path.join(AUDIO_DIR, filename)
            try:
                audio, sr = load_real_audio(filepath, max_duration=MAX_DURATION)
                features = extract_acoustic_features(audio, sr)
                
                all_features_list.append(features)
                all_labels.append(sound_type)
                
                # 单独计算子指数（预览）
                from acoustic_features import calculate_SI_from_audio, calculate_OI_from_audio, calculate_CI_from_audio
                SI = calculate_SI_from_audio(features)
                OI = calculate_OI_from_audio(features)
                CI = calculate_CI_from_audio(features)
                
                sample_info.append({
                    'sound_type': sound_type,
                    'filename': filename,
                    'SI': SI, 'OI': OI, 'CI': CI,
                    'features': features,
                })
                
                print(f"    → SI={SI:.3f}, OI={OI:.3f}, CI={CI:.3f}")
                
            except Exception as e:
                print(f"    ⚠️ {filename} 处理失败: {e}")
                continue
    
    if not all_features_list:
        print("\n❌ 所有处理都失败了！")
        return None
    
    # ============================================================
    # 第二步：批量计算ADI（带熵权法）
    # ============================================================
    print("\n" + "="*60)
    print("📊 使用熵权法计算全局ADI...")
    print("="*60)
    
    raw_results, global_weights, normalization_params = batch_calculate_adi(all_features_list, all_labels)
    
    print(f"\n📌 熵权法确定的权重:")
    print(f"   w(SI/不和谐度)  = {global_weights['SI']:.3f}  ({global_weights['SI']*100:.1f}%)")
    print(f"   w(OI/无序度)   = {global_weights['OI']:.3f}  ({global_weights['OI']*100:.1f}%)")
    print(f"   w(CI/复杂度)   = {global_weights['CI']:.3f}  ({global_weights['CI']*100:.1f}%)")
    
    # ============================================================
    # 第三步：按水声类型聚合结果
    # ============================================================
    aggregated = aggregate_by_sound_type(raw_results)
    
    # ============================================================
    # 输出汇总表格
    # ============================================================
    print("\n" + "=" * 75)
    print("  📊 ADI分析结果汇总表 v4 (120样本, 基于数据驱动归一化)")
    print("=" * 75)
    header = f"  {'水声类型':^8s} | {'ADI均值':^7s} | {'标准差':^6s} | {'SI':^6s} | {'OI':^6s} | {'CI':^6s} | {'预期':^4s}"
    print(header)
    print("  " + "-" * 70)
    
    for st in ['stream', 'rain', 'waterfall', 'drip', 'river', 'ocean']:
        if st in aggregated:
            r = aggregated[st]
            expected = FENGSHUI_EXPECTED.get(st, '?')
            label = SOUND_LABELS[st]
            
            # 判断实际结果与预期是否一致
            interp = interpret_adi_fengshui(r['ADI_mean'])
            
            print(f"  {label:8s} | {r['ADI_mean']:7.3f} | {r['ADI_std']:6.3f} | "
                  f"{r['SI_mean']:6.3f} | {r['OI_mean']:6.3f} | {r['CI_mean']:6.3f} | {expected:^4s}")
    
    print("=" * 75)
    
    # 风水解读
    print("\n📋 风水声煞解读 (v3):")
    print("-" * 55)
    for st in ['stream', 'rain', 'waterfall', 'drip', 'river', 'ocean']:
        if st in aggregated:
            r = aggregated[st]
            label = SOUND_LABELS[st]
            expected = FENGSHUI_EXPECTED.get(st, '?')
            interp = interpret_adi_fengshui(r['ADI_mean'])
            
            match_str = "✓ 一致" if ((interp['emoji'].startswith('✅') and expected == '吉') or
                                       (interp['emoji'].startswith('❌') and expected == '凶') or
                                       ('中' in interp['level'] and expected == '中')) else "✗ 待校验"
            
            print(f"  {label:6s}: ADI={r['ADI_mean']:.3f} -> {interp['emoji']} {interp['level']}  [传统预期:{expected}] {match_str}")
    
    # 子指数详情表
    print("\n📐 各水声子指数详情:")
    print("-" * 60)
    print(f"  {'水声':^6s} | {'SI(和谐)':^8s} | {'OI(有序)':^8s} | {'CI(复杂)':^8s} | {'不和谐度':^7s} | {'无序度':^7s}")
    print("  " + "-" * 58)
    for st in ['stream', 'rain', 'waterfall', 'drip', 'river', 'ocean']:
        if st in aggregated:
            r = aggregated[st]
            label = SOUND_LABELS[st]
            print(f"  {label:6s} | {r['SI_mean']:8.3f} | {r['OI_mean']:8.3f} | {r['CI_mean']:8.3f} | "
                  f"{1-r['SI_mean']:7.3f} | {1-r['OI_mean']:7.3f}")
    print("-" * 60)
    
    # ============================================================
    # 第四步：可视化
    # ============================================================
    print("\n🎨 生成可视化图表...")
    
    try:
        # 1. Chladni图案（可视化用，每种水声一个代表图案）
        print("  1/4 Chladni图案对比图...")
        chladni_patterns = {}
        for st in ['stream', 'rain', 'waterfall', 'drip', 'river', 'ocean']:
            m, n = simulate_water_sound_cymatics(st)
            chladni_patterns[SOUND_LABELS[st]] = generate_chladni_pattern(m, n)
        
        plot_multiple_patterns(chladni_patterns,
                              save_path=os.path.join(os.path.dirname(__file__), "output_chladni_v2.png"))
        
        # 2. ADI柱状图
        print("  2/4 ADI对比柱状图...")
        _plot_adi_bar(aggregated, save_path=os.path.join(os.path.dirname(__file__), "output_adi_comparison_v2.png"))
        
        # 3. 子指数雷达图
        print("  3/4 子指数雷达图...")
        _plot_radar(aggregated, save_path=os.path.join(os.path.dirname(__file__), "output_radar_v2.png"))
        
        # 4. 特征热图（如果seaborn可用）
        try:
            import seaborn as sns
            HAS_SEABORN = True
        except ImportError:
            HAS_SEABORN = False
        
        if HAS_SEABORN and all_features_list:
            print("  4/4 特征相关性热图...")
            # 每种水声取第一个样本的特征
            rep_features = []
            adis_rep = []
            seen_types = set()
            for i, info in enumerate(sample_info):
                if info['sound_type'] not in seen_types:
                    rep_features.append(info['features'])
                    seen_types.add(info['sound_type'])
                    for rr in raw_results:
                        if rr.get('label') == SOUND_LABELS.get(info['sound_type']):
                            adis_rep.append(rr['ADI'])
                            break
            
            if len(rep_features) >= 3 and len(adis_rep) >= 3:
                plot_feature_correlation(rep_features[:min(len(rep_features), len(adis_rep))],
                                        adis_rep[:min(len(rep_features), len(adis_rep))],
                                        save_path=os.path.join(os.path.dirname(__file__), "output_correlation_v2.png"))
                print(f"      已保存: output_correlation_v2.png")
            else:
                print("      ⚠️ 样本不足，跳过相关性热图")
        
        print("\n  ✅ 所有图表已保存到项目根目录！")
        
    except Exception as e:
        print(f"\n  ⚠️ 可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================================
    # 保存CSV
    # ============================================================
    try:
        import pandas as pd
        
        csv_rows = []
        for st in ['stream', 'rain', 'waterfall', 'drip', 'river', 'ocean']:
            if st in aggregated:
                r = aggregated[st]
                csv_rows.append({
                    'Sound_Type_English': st,
                    'Sound_Type_Chinese': SOUND_LABELS[st],
                    'FengShui_Expected': FENGSHUI_EXPECTED.get(st),
                    'ADI_Mean': round(r['ADI_mean'], 4),
                    'ADI_Std': round(r['ADI_std'], 4),
                    'SI_Mean_Harmony': round(r['SI_mean'], 4),
                    'OI_Mean_Orderliness': round(r['OI_mean'], 4),
                    'CI_Mean_Complexity': round(r['CI_mean'], 4),
                    'N_Samples': r['n_samples'],
                    'Weight_SI': round(global_weights['SI'], 4),
                    'Weight_OI': round(global_weights['OI'], 4),
                    'Weight_CI': round(global_weights['CI'], 4),
                })
        
        df = pd.DataFrame(csv_rows)
        csv_path = os.path.join(os.path.dirname(__file__), "adi_results_v2.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 结果已保存到: {csv_path}")
    except ImportError:
        print("\n⚠️ pandas未安装，跳过CSV保存")


def _plot_adi_bar(aggregated, save_path=None):
    """绘制ADI对比柱状图 v2"""
    labels = [SOUND_LABELS[k] for k in aggregated.keys() if k in SOUND_LABELS]
    means = [aggregated[k]['ADI_mean'] for k in aggregated]
    stds = [aggregated[k]['ADI_std'] for k in aggregated]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#27ae60' if v < 0.30 else '#f1c40f' if v < 0.45 else '#e67e22' if v < 0.60 else '#c0392b'
              for v in means]
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors,
                   alpha=0.85, edgecolor='#333', linewidth=0.5)
    
    for bar, val, std in zip(bars, means, stds):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+std+0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.axhline(y=0.30, color='#27ae60', linestyle='--', lw=2, alpha=0.7, label='吉阈值 (0.30)')
    ax.axhline(y=0.45, color='#e67e22', linestyle='--', lw=2, alpha=0.7, label='中等阈值 (0.45)')
    ax.axhline(y=0.60, color='#c0392b', linestyle='--', lw=2, alpha=0.7, label='凶阈值 (0.60)')
    
    ax.set_title('不同水声的 Acoustic Discomfort Index (ADI)\n基于 Freesound 真实水声样本 v2',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('ADI (0-1, 越高越不舒服 / 声煞越强)', fontsize=12)
    ax.set_ylim(0, max(max(means)*1.3, 0.8))
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      已保存: {os.path.basename(save_path)}")
    plt.show()


def _plot_radar(aggregated, save_path=None):
    """绘制子指数雷达图 v2"""
    from math import pi
    
    categories = ['和谐度\n(SI)', '有序度\n(OI)', '复杂度\n(CI)']
    N = len(categories)
    base_angles = [n / float(N) * 2 * pi for n in range(N)]
    angles = base_angles + [base_angles[0]]  # 闭合
    
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    
    cmap = plt.cm.Set2
    keys = list(aggregated.keys())
    for i, st in enumerate(keys):
        r = aggregated[st]
        values = [r['SI_mean'], r['OI_mean'], r['CI_mean']] + [r['SI_mean']]
        
        ax.plot(angles, values, 'o-', lw=2, label=SOUND_LABELS[st], color=cmap(i/len(keys)))
        ax.fill(angles, values, alpha=0.08, color=cmap(i/len(keys)))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_title('各水声子指数雷达图 v2\n(越高=越和谐/有序/复杂)', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=11)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      已保存: {os.path.basename(save_path)}")
    plt.show()


if __name__ == "__main__":
    run_full_analysis()
    
    print("\n" + "=" * 60)
    print("  ✅ 全部分析完成！(v2)")
    print("=" * 60)
