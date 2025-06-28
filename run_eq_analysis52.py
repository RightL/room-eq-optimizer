import io
import numpy as np
from scipy.signal import find_peaks, freqz
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import time

# --- 配置区 ---
DATA_FILE_PATH = 'New File-04.txt'
TARGET_RANGE = [1000, 8000]
CORRECTION_RANGE = [20, 1000]
SAMPLING_RATE_FS = 48000
NUM_EQ_BANDS = 10
DIP_FLATTEN_THRESHOLD_DB = 3.0

# ==================== 新增的目标曲线配置 ====================
TARGET_SLOPE_DB = 3.0 # 从20Hz到1000Hz的总下降分贝数

# ==================== 新增的平滑和预处理配置 ====================
# 平滑选项: None, 1/3, 1/6, 1/12 倍频程
SMOOTHING_TYPE = '1/6'  # 可选值: None, '1/3', '1/6', '1/12'

# 预处理方法选项: 1 = 深谷拉平, 2 = 对数修正
PREPROCESSING_METHOD = 1
# =============================================================

def parse_data(filepath: str):
    if not os.path.exists(filepath): raise FileNotFoundError(f"Error: File not found at '{filepath}'.")
    with open(filepath, 'r', encoding='utf-8') as f: lines = f.readlines()
    data_start_index = -1;
    for i, line in enumerate(lines):
        if 'Frequency' in line and 'Magnitude' in line: data_start_index = i + 1; break
    if data_start_index == -1: raise ValueError("Error: Could not find 'Frequency Magnitude' header in the file.")
    parsed_data = [[float(freq), float(mag)] for line in lines[data_start_index:] for freq, mag in [line.strip().split()] if freq and mag]
    return np.array(parsed_data)

def smooth_curve(freqs, mags, smoothing_type):
    if smoothing_type is None:
        return mags
        
    # 定义不同平滑类型对应的倍频程分数
    fractions = {
        '1/3': 3,
        '1/6': 6,
        '1/12': 12
    }
    
    if smoothing_type not in fractions:
        raise ValueError(f"Invalid smoothing type: {smoothing_type}")
        
    fraction = fractions[smoothing_type]
    smoothed_mags = np.zeros_like(mags)
    
    for i, center_freq in enumerate(freqs):
        # 计算当前频率的平滑窗口范围
        bandwidth = center_freq / fraction
        lower_freq = center_freq / np.sqrt(2**(1/fraction))
        upper_freq = center_freq * np.sqrt(2**(1/fraction))
        
        # 找到窗口范围内的所有点
        mask = (freqs >= lower_freq) & (freqs <= upper_freq)
        
        # 计算加权平均
        if np.any(mask):
            weights = np.exp(-((np.log2(freqs[mask]/center_freq))**2))
            smoothed_mags[i] = np.average(mags[mask], weights=weights)
        else:
            smoothed_mags[i] = mags[i]
    
    return smoothed_mags

def get_combined_eq_curve(params, freq_axis, fs):
    total_eq_gain_db = np.zeros(len(freq_axis))
    center_freqs, gains, qs = params[0::3], params[1::3], params[2::3]
    for f0, gain_db, q in zip(center_freqs, gains, qs):
        A = 10**(gain_db / 40); w0 = 2 * np.pi * f0 / fs; alpha = np.sin(w0) / (2 * q)
        b0,b1,b2 = 1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A
        a0,a1,a2 = 1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A
        b, a = np.array([b0, b1, b2]) / a0, np.array([a0, a1, a2]) / a0
        w, h = freqz(b, a, worN=freq_axis, fs=fs)
        total_eq_gain_db += 20 * np.log10(np.abs(h) + 1e-9)
    return total_eq_gain_db

# ==================== 成本函数现在使用 target_curve 数组 ====================
def cost_function(params, original_mags, target_curve, freq_axis, fs, log_freq_weights):
    freqs = params[0::3]
    if not np.all(np.diff(freqs) > 0): return 1e6
    eq_curve_db = get_combined_eq_curve(params, freq_axis, fs)
    corrected_mags = original_mags + eq_curve_db
    # 误差是与目标曲线的差值，而不是与单一平均值的差值
    squared_errors = (corrected_mags - target_curve)**2
    weighted_squared_errors = squared_errors * log_freq_weights
    error = np.sqrt(np.mean(weighted_squared_errors))
    return error
# ========================================================================

# ==================== 初始化函数现在使用 target_curve 数组 ====================
def get_initial_guess_and_bounds(data, num_bands, target_curve):
    freqs, mags = data[:, 0], data[:, 1]
    # 偏差现在是与目标曲线的差值
    deviations = mags - target_curve
    
    peak_indices, peak_props = find_peaks(deviations, prominence=0.5, width=1)
    dip_indices, dip_props = find_peaks(-deviations, prominence=0.5, width=1)
    
    anomalies = []
    for i, idx in enumerate(peak_indices):
        deviation = deviations[idx]
        if deviation <= 0: continue
        score = (deviation * peak_props['widths'][i]) / np.log10(freqs[idx]); anomalies.append({'type': 'peak', 'freq': freqs[idx], 'gain': -deviation, 'score': score})
    for i, idx in enumerate(dip_indices):
        deviation = -deviations[idx]
        if deviation <= 0: continue
        score = (deviation * dip_props['widths'][i]) / np.log10(freqs[idx]); anomalies.append({'type': 'dip', 'freq': freqs[idx], 'gain': deviation, 'score': score})
        
    anomalies.sort(key=lambda x: x['score'], reverse=True)
    top_anomalies = sorted(anomalies[:num_bands], key=lambda x: x['freq'])
    
    initial_guess, bounds = [], []; min_freq, max_freq = data[:,0].min(), data[:,0].max()
    for item in top_anomalies:
        initial_guess.extend([item['freq'], item['gain'], 3.0])
        gain_bounds = (-30.0, 8.0) 
        bounds.extend([(min_freq, max_freq), gain_bounds, (0.4, 8.0)])
    return np.array(initial_guess), bounds
# ========================================================================

def plot_results(data, target_curve, all_params, final_mags, pre_processed_mags):
    plot_data = data
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(plot_data[:, 0], plot_data[:, 1], label='Original Response', color='dodgerblue', linewidth=2.5, zorder=10)
    if pre_processed_mags is not None:
        ax.plot(plot_data[:, 0], pre_processed_mags, label='Pre-Processed Curve', color='gray', linestyle=':', linewidth=2, zorder=5)
    ax.plot(plot_data[:, 0], final_mags, label=f'Final Response ({len(all_params)} EQs)', color='limegreen', linewidth=2.5, zorder=15)
    ax.plot(plot_data[:, 0], target_curve, color='red', linestyle='--', linewidth=2, label=f'Target Curve ({TARGET_SLOPE_DB}dB Slope)')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks([20, 30, 40, 50, 60, 80, 100, 200, 300, 400, 500, 600, 800, 1000])
    ax.minorticks_off()
    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    ax.set_ylabel('Magnitude (dB)', fontsize=14)
    ax.set_title('Correction to Sloped Target Curve', fontsize=18, pad=20)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", ls="-", color='0.85')
    plt.tight_layout()
    
    # 构建包含处理方法信息的文件名
    smoothing_str = SMOOTHING_TYPE if SMOOTHING_TYPE else 'no_smooth'
    smoothing_str = smoothing_str.replace('/', '-') if smoothing_str else 'no_smooth'  # 将斜杠替换为横杠
    preproc_str = 'dip_flatten' if PREPROCESSING_METHOD == 1 else 'log_correction'
    output_filename = f'frequency_response_sloped_{smoothing_str}_{preproc_str}.png'
    
    plt.savefig(output_filename, dpi=120)
    print(f"\nSuccess! Plot saved as '{output_filename}'.")

# --- 主程序入口 ---
if __name__ == "__main__":
    try:
        full_data = parse_data(DATA_FILE_PATH)
        # 1. 计算1k-8k的平均值，作为我们1000Hz处的目标点
        target_mask = (full_data[:, 0] >= TARGET_RANGE[0]) & (full_data[:, 0] <= TARGET_RANGE[1])
        level_at_1k = np.mean(full_data[target_mask, 1])
        level_at_20hz = level_at_1k + TARGET_SLOPE_DB

        # 2. 筛选出需要校准的频段数据
        correction_mask = (full_data[:, 0] >= CORRECTION_RANGE[0]) & (full_data[:, 0] <= CORRECTION_RANGE[1])
        correction_data = full_data[correction_mask]
        correction_freqs = correction_data[:, 0]
        original_mags = correction_data[:, 1]
        
        # ==================== 核心改动：生成倾斜的目标曲线 ====================
        # 在对数频率空间中进行线性插值，以创建平滑的倾斜曲线
        log_f = np.log10(correction_freqs)
        log_f_20 = np.log10(CORRECTION_RANGE[0])
        log_f_1k = np.log10(CORRECTION_RANGE[1])
        
        # 使用Numpy的interp函数进行插值
        target_curve = np.interp(log_f, [log_f_20, log_f_1k], [level_at_20hz, level_at_1k])
        # =====================================================================
        
        # 3. 前置处理步骤
        analysis_mags = original_mags.copy()
        
        # 首先进行前置处理
        if PREPROCESSING_METHOD == 1:
            # 深谷拉平方法
            print("--- Pre-processing: Flattening deep dips relative to the sloped target curve ---")
            dip_threshold_curve = target_curve - DIP_FLATTEN_THRESHOLD_DB
            deep_dip_mask = analysis_mags < dip_threshold_curve
            analysis_mags[deep_dip_mask] = target_curve[deep_dip_mask]  # 拉平到目标曲线上对应的点
            print(f"Flattened {np.sum(deep_dip_mask)} data points.")
        elif PREPROCESSING_METHOD == 2:
            # 对数修正方法
            print("--- Pre-processing: Applying logarithmic correction ---")
            negative_mask = analysis_mags - target_curve < 0
            analysis_mags[negative_mask] = target_curve[negative_mask]-np.log2(1+target_curve[negative_mask] - analysis_mags[negative_mask])
            
        # 对前置处理后的数据进行平滑，用于寻找初始点
        smoothed_mags = smooth_curve(correction_freqs, analysis_mags, SMOOTHING_TYPE)
        
        # 4. 将平滑后的数据用于寻找初始点
        analysis_data = np.column_stack((correction_freqs, smoothed_mags))
        initial_guess, bounds = get_initial_guess_and_bounds(analysis_data, NUM_EQ_BANDS, target_curve)
        
        # 5. 预计算对数权重
        log_freqs = np.log10(correction_freqs)
        mid_points = (log_freqs[1:] + log_freqs[:-1]) / 2
        log_freq_widths = np.diff(mid_points, prepend=mid_points[0], append=mid_points[-1])

        print("\n--- Starting Global Optimization towards Sloped Target Curve ---")
        print(f"Number of Filters: {NUM_EQ_BANDS}")
        print(f"Initial Guess for Center Frequencies: {[f'{f:.1f}' for f in initial_guess[0::3]]}")
        start_time = time.time()

        # 6. 运行优化器 (使用前置处理后的数据，但不使用平滑)
        result = dual_annealing(
            func=cost_function,
            bounds=bounds,
            x0=initial_guess,
            args=(analysis_mags, target_curve, correction_freqs, SAMPLING_RATE_FS, log_freq_widths),
            maxiter=1500,
        )
        
        print(f"\nGlobal optimization complete. Time elapsed: {time.time() - start_time:.2f} seconds")
        print(f"Final Error relative to sloped curve: {result.fun:.4f}")

        optimized_params = result.x
        final_eq_params = [{'f0': p[0], 'gain_db': p[1], 'q': p[2]} for p in np.array_split(optimized_params, NUM_EQ_BANDS)]
        final_eq_params.sort(key=lambda p: p['f0'])

        # 打印EQ参数...
        print(f"\n--- Final Optimized EQ Parameters ({len(final_eq_params)} bands total) ---")
        print("Preamp: 0.0 dB")
        for i, p in enumerate(final_eq_params):
            print(f"Filter {i+1:<2}: ON  PK  Fc {p['f0']:<7.1f} Hz  Gain {p['gain_db']:5.2f} dB  Q {p['q']:.2f}")

        # 7. 最终绘图
        final_eq_curve = get_combined_eq_curve(optimized_params, correction_freqs, SAMPLING_RATE_FS)
        final_mags = original_mags + final_eq_curve
        plot_results(correction_data, target_curve, final_eq_params, final_mags, smoothed_mags)

    except (FileNotFoundError, ValueError) as e:
        print(e)
