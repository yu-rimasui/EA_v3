import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter, minimum_filter
import os

# =============================================================================
# 1. パラメータ設定 (Configuration)
# =============================================================================
VAL = "Bi_17000"

# 入力ファイルパス
INPUT_CSV_PATH = f"img/{VAL}eV_intensity_map.csv"

# 出力ファイルパス
# 1. 検出されたブラッグ反射のリスト（ログ用）
OUTPUT_DETECTED_LIST = f"output/log_{VAL}eV_detected_bragg.csv"
# 2. 3D-Air-Image 読み込み用データ (Step 3 最終成果物)
OUTPUT_FOR_3DAIR = f"output/algorithm2_{VAL}eV_intensity_map.csv"

# Step 1: 背景推定 & 判別用パラメータ
PEAK_ERASE_SIZE = 20    # 背景推定時に無視するピーク幅
SMOOTHING_SIGMA = 5.0   # 背景の滑らかさ
IQR_MULTIPLIER = 3.0    # 閾値係数 (3.0推奨)

# Step 2: 可視化用パラメータ
NUM_CHECK_ROWS = 3      # 補完結果を詳細確認する行の数

# ディレクトリ作成
if not os.path.exists("output"):
    os.makedirs("output")

# =============================================================================
# 2. 共通関数 (Common Functions)
# =============================================================================
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return None
    print(f"Loading data from: {file_path}")
    return pd.read_csv(file_path, header=None)

# =============================================================================
# 3. Step 1: ブラッグ反射の判別ロジック
# =============================================================================
def estimate_background_morphological(data, erase_size=20, smooth_sigma=5.0):
    """モルフォロジー演算で背景を推定"""
    eroded = minimum_filter(data, size=erase_size)
    bg_estimated = gaussian_filter(eroded, sigma=smooth_sigma)
    net_signal = data - bg_estimated
    return net_signal, bg_estimated

def calculate_row_wise_thresholds(net_signal, multiplier=3.0):
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

def run_step1_detection(raw_data):
    """Step 1 の処理を一括実行"""
    print("--- Running Step 1: Detection ---")
    
    # 1. 背景推定
    net_signal, bg_estimated = estimate_background_morphological(
        raw_data, PEAK_ERASE_SIZE, SMOOTHING_SIGMA
    )
    
    # 2. 閾値計算
    threshold_vals = calculate_row_wise_thresholds(net_signal, multiplier=IQR_MULTIPLIER)
    
    # 3. 閾値面の作成 (背景 + マージン)
    rows, cols = raw_data.shape
    margin_surface = np.tile(threshold_vals[:, np.newaxis], (1, cols))
    effective_threshold_surface = bg_estimated + margin_surface
    
    # 4. 判定 (Mask作成: True=Bragg)
    mask = raw_data > effective_threshold_surface
    
    print(f"Detected {np.sum(mask)} pixels as Bragg reflections.")
    return net_signal, bg_estimated, effective_threshold_surface, mask, threshold_vals

# =============================================================================
# 4. Step 2: 除去と補完ロジック
# =============================================================================
def run_step2_cleaning(raw_data, mask):
    """Step 2 の処理を一括実行 (行ごとの中央値補完)"""
    print("--- Running Step 2: Cleaning ---")
    
    cleaned_data = raw_data.copy()
    rows, cols = raw_data.shape
    
    # 行ごとに処理
    for r in range(rows):
        row_mask = mask[r, :]
        if np.any(row_mask):
            # 正常画素
            normal_pixels = raw_data[r, ~row_mask]
            
            if len(normal_pixels) > 0:
                # 正常画素の中央値で、異常画素を埋める
                fill_val = np.median(normal_pixels)
                cleaned_data[r, row_mask] = fill_val
            else:
                cleaned_data[r, :] = np.min(raw_data) # 安全策
                
    return cleaned_data

# =============================================================================
# 5. 可視化関数群 (Visualization)
# =============================================================================
def plot_step1_3d_check(raw_data, bg_estimated, threshold_vals):
    """Step 1: 3Dフィット確認"""
    print("Plotting Step 1: 3D Check... (Close window to proceed)")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    rows, cols = raw_data.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    # 元データ
    ax.plot_surface(X, Y, raw_data, cmap='viridis', rstride=2, cstride=20, alpha=0.5, linewidth=0)
    
    # 閾値面
    margin_surface = np.tile(threshold_vals[:, np.newaxis], (1, cols))
    total_surf = bg_estimated + margin_surface
    ax.plot_surface(X, Y, total_surf, color='red', alpha=0.4, rstride=5, cstride=50)
    
    ax.set_title(f"Raw Data vs Threshold Plane (M={IQR_MULTIPLIER})")
    ax.set_zlabel("Intensity")
    plt.show()

def plot_step1_2d_map(mask):
    """Step 1: 検出箇所の2Dマップ"""
    plt.figure(figsize=(8, 4))
    plt.imshow(mask, cmap='gray', aspect='auto', interpolation='nearest')
    plt.title(f"Detected Bragg Reflections - Total: {np.sum(mask)}")
    plt.xlabel("Phi")
    plt.ylabel("Theta")
    plt.colorbar(ticks=[0, 1], label="0:Safe, 1:Bragg")
    plt.show()

def plot_step2_details(raw_data, cleaned_data, mask, check_rows):
    """Step 2: 補完前後の詳細比較"""
    print(f"Plotting Step 2: Detailed check for rows {check_rows}...")
    for r in check_rows:
        if not np.any(mask[r, :]): continue
        
        plt.figure(figsize=(10, 4))
        # 元データ
        plt.plot(raw_data[r, :], color='gray', alpha=0.5, label='Original')
        # 除去箇所
        bad_idx = np.where(mask[r, :])[0]
        plt.scatter(bad_idx, raw_data[r, bad_idx], color='red', marker='x', label='Removed')
        # 補完後
        plt.plot(cleaned_data[r, :], color='green', linestyle='--', label='Cleaned')
        
        plt.title(f"Result: Row {r}")
        plt.legend()
        plt.show()

# =============================================================================
# 6. メイン実行 (Main)
# =============================================================================
def main():
    print(f"=== XFH Bragg Remover: Integrated Process (Bi {VAL}eV) ===")
    
    # 1. Load Data
    df = load_data(INPUT_CSV_PATH)
    if df is None: return
    raw_data = df.values
    
    # 2. Step 1: Detection
    net, bg, thresh_surf, mask, thresh_vals = run_step1_detection(raw_data)
    
    # --- 可視化: Step 1 ---
    plot_step1_3d_check(raw_data, bg, thresh_vals)
    plot_step1_2d_map(mask)
    
    # ログ保存 (検出座標)
    rows_idx, cols_idx = np.where(mask)
    pd.DataFrame({"Theta": rows_idx, "Phi": cols_idx}).to_csv(OUTPUT_DETECTED_LIST, index=False)
    print(f"Log saved: {OUTPUT_DETECTED_LIST}")

    # 3. Step 2: Cleaning
    cleaned_data = run_step2_cleaning(raw_data, mask)
    
    # --- 可視化: Step 2 ---
    # 検出数が多かった行を自動ピックアップして表示
    if len(rows_idx) > 0:
        target_rows = pd.Series(rows_idx).value_counts().head(NUM_CHECK_ROWS).index.tolist()
        target_rows.sort()
        plot_step2_details(raw_data, cleaned_data, mask, target_rows)
    
    # 4. Step 3: Export for 3D-Air-Image
    # 最終的な全体比較 (簡易版)
    print("Generating final comparison map...")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    vmin, vmax = np.percentile(raw_data, 1), np.percentile(raw_data, 99)
    
    ax[0].imshow(raw_data, cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
    ax[0].set_title("Before: Original Raw Data")
    
    ax[1].imshow(cleaned_data, cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
    ax[1].set_title("After: Cleaned Data")
    
    plt.tight_layout()
    plt.show()

    print("--- Running Step 3: Exporting ---")
    
    # 3D-Air-Imageは通常、ヘッダーなし・インデックスなしのCSV(行列データ)を読み込みます
    # header=False, index=False で保存します
    pd.DataFrame(cleaned_data).to_csv(OUTPUT_FOR_3DAIR, header=False, index=False)
    
    print("\n" + "="*60)
    print(f" [SUCCESS] Processing Complete!")
    print(f" Cleaned Data Saved to: {OUTPUT_FOR_3DAIR}")
    print(" This file is ready to be imported into 3D-Air-Image.")
    print("="*60)

if __name__ == "__main__":
    main()