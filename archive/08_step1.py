import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter, minimum_filter
import os

# =============================================================================
# 1. パラメータ設定
# =============================================================================
VAL = "Bi_17000"
INPUT_CSV_PATH = f"img/{VAL}eV_intensity_map.csv"
OUTPUT_DETECTED_LIST = f"output/step2.1_{VAL}eV_detected_bragg.csv"

# 背景推定パラメータ
PEAK_ERASE_SIZE = 20
SMOOTHING_SIGMA = 5.0

# 閾値係数
IQR_MULTIPLIER = 3.0

# =============================================================================
# 2. 関数定義
# =============================================================================
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return None
    return pd.read_csv(file_path, header=None)

def estimate_background_morphological(data, erase_size=20, smooth_sigma=5.0):
    """背景推定"""
    eroded = minimum_filter(data, size=erase_size)
    bg_estimated = gaussian_filter(eroded, sigma=smooth_sigma)
    net_signal = data - bg_estimated
    return net_signal, bg_estimated

def calculate_row_wise_thresholds(net_signal, multiplier=6.0):
    """行ごとの閾値（マージン）を計算"""
    rows, cols = net_signal.shape
    threshold_vals = np.zeros(rows) 
    
    for i in range(rows):
        row_data = net_signal[i, :]
        q1 = np.percentile(row_data, 25)
        q3 = np.percentile(row_data, 75)
        iqr = q3 - q1
        median = np.median(row_data)
        threshold_vals[i] = median + multiplier * iqr
        
    return threshold_vals

# --- 可視化関数 ---

def plot_3d_original_fit_check(original_data, bg_estimated, threshold_vals):
    """3D: 元データ vs 閾値面 (背景 + マージン)"""
    print("Generating 3D Check Plot... (Close window to proceed)")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    rows, cols = original_data.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    # 元データ (Original)
    ax.plot_surface(X, Y, original_data, cmap='jet', 
                    rstride=2, cstride=20, alpha=0.6, linewidth=0)

    # 閾値面 (Background + Margin)
    margin_surface = np.tile(threshold_vals[:, np.newaxis], (1, cols))
    total_threshold_surface = bg_estimated + margin_surface
    
    ax.plot_surface(X, Y, total_threshold_surface, color='black', alpha=0.4, 
                    rstride=5, cstride=50)

    ax.set_title(f"3D Check: Original vs Threshold Plane (Multiplier={IQR_MULTIPLIER})")
    ax.set_xlabel("Phi")
    ax.set_ylabel("Theta")
    ax.set_zlabel("Intensity")
    
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='black', alpha=0.4, label='Threshold Plane')
    plt.legend(handles=[red_patch])
    
    plt.show()

def plot_2d_steps(raw_data, bg_estimated, net_signal, effective_threshold_surface, mask):
    """2D: 各ステップの詳細確認と検出マップ"""
    print("Generating 2D Step Plots... (Close each window to proceed)")
    
    # 共通スケール
    vmin_common = np.percentile(raw_data, 1)
    vmax_common = np.percentile(raw_data, 99)
    vmax_net = np.percentile(net_signal, 99.5)

    # 1. Raw
    plt.figure(figsize=(10, 5))
    plt.imshow(raw_data, cmap='jet', aspect='auto', vmin=vmin_common, vmax=vmax_common)
    plt.title("1. Raw Data")
    plt.colorbar()
    plt.show()

    # 2. Background
    plt.figure(figsize=(10, 5))
    plt.imshow(bg_estimated, cmap='jet', aspect='auto', vmin=vmin_common, vmax=vmax_common)
    plt.title("2. Estimated Background")
    plt.colorbar()
    plt.show()

    # 3. Net Signal
    plt.figure(figsize=(10, 5))
    plt.imshow(net_signal, cmap='viridis', aspect='auto', vmin=0, vmax=vmax_net)
    plt.title("3. Net Signal (Raw - BG)")
    plt.colorbar()
    plt.show()

    # 4. Threshold Surface
    plt.figure(figsize=(10, 5))
    plt.imshow(effective_threshold_surface, cmap='jet', aspect='auto', vmin=vmin_common, vmax=vmax_common)
    plt.title(f"4. Threshold Surface (BG + {IQR_MULTIPLIER}*IQR)")
    plt.colorbar()
    plt.show()

    # 5. Detection Map (White/Black)
    plt.figure(figsize=(10, 5))
    plt.imshow(mask, cmap='gray', aspect='auto', interpolation='nearest')
    plt.title(f"5. Detection Map (White=Bragg)\nTotal: {np.sum(mask)} pixels detected")
    plt.xlabel("Phi")
    plt.ylabel("Theta")
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(['Safe', 'Bragg']) 
    plt.show()

# =============================================================================
# 3. メイン処理
# =============================================================================
def main():
    print(f"=== Integrated Bragg Detection (Multiplier={IQR_MULTIPLIER}) ===")
    
    # 1. データ読み込み
    df = load_data(INPUT_CSV_PATH)
    if df is None: return
    raw_data = df.values
    rows, cols = raw_data.shape
    
    # 2. 背景推定 & 閾値計算
    net_signal, bg_estimated = estimate_background_morphological(raw_data, PEAK_ERASE_SIZE, SMOOTHING_SIGMA)
    threshold_vals = calculate_row_wise_thresholds(net_signal, multiplier=IQR_MULTIPLIER)
    
    # 閾値面の作成 (可視化 & 判定用)
    margin_surface = np.tile(threshold_vals[:, np.newaxis], (1, cols))
    effective_threshold_surface = bg_estimated + margin_surface
    
    # 判定 (Mask作成)
    mask = raw_data > effective_threshold_surface
    
    # --- 可視化フェーズ ---
    # 3Dチェック (全体的なフィット感) [056_1.py由来]
    plot_3d_original_fit_check(raw_data, bg_estimated, threshold_vals)
    
    # 2Dステップチェック (詳細 & 白黒マップ) [056_1-v3.py由来]
    plot_2d_steps(raw_data, bg_estimated, net_signal, effective_threshold_surface, mask)
    
    # --- 保存フェーズ [056_1.py由来] ---
    rows_idx, cols_idx = np.where(mask)
    results = pd.DataFrame({
        "Theta_Row": rows_idx,
        "Phi_Col": cols_idx,
        "Original_Val": raw_data[rows_idx, cols_idx],
        "Net_Val": net_signal[rows_idx, cols_idx],
        "Threshold_Used": effective_threshold_surface[rows_idx, cols_idx]
    })
    
    # 強度順にソート
    results = results.sort_values(by="Net_Val", ascending=False)
    
    print(f"\n[Result] Detected {len(results)} pixels.")
    if not results.empty:
        results.to_csv(OUTPUT_DETECTED_LIST, index=False)
        print(f"Saved detection list to: {OUTPUT_DETECTED_LIST}")
        print("Done.")
    else:
        print("No peaks detected.")

if __name__ == "__main__":
    main()