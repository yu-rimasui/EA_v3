import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. パラメータ設定
# =============================================================================
VAL = "Bi_17000"
INPUT_RAW_DATA = f"img/{VAL}eV_intensity_map.csv"
INPUT_DETECTED_LIST = f"output/step2.1_{VAL}eV_detected_bragg.csv"
OUTPUT_CLEANED_DATA = f"output/step2.2_{VAL}eV_intensity_map_.csv"

# 詳細を確認したい行の数
NUM_CHECK_ROWS = 3

# =============================================================================
# 2. 関数定義
# =============================================================================
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return None
    return pd.read_csv(file_path, header=None)

def load_detected_coords(file_path):
    if not os.path.exists(file_path):
        print(f"[Error] Detection list not found: {file_path}")
        return None
    return pd.read_csv(file_path)

def plot_process_details(row_idx, raw_row, mask_row, normal_pixels, bg_val, cleaned_row):
    """
    処理の途中経過（分布とプロファイル）を詳細に可視化する関数
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- 左図: 強度分布とロバスト推定 (Background Estimation) ---
    ax1 = axes[0]
    # 全データの分布
    ax1.hist(raw_row, bins=50, color='lightgray', alpha=0.7, label='All ($I_i$)')
    # 正常画素の分布
    ax1.hist(normal_pixels, bins=50, color='green', alpha=0.5, label='Valid ($V_i$)')
    # 異常画素（除去対象）の分布
    outliers = raw_row[mask_row]
    ax1.hist(outliers, bins=50, color='red', alpha=0.7, label='Bragg ($I_i - V_i$)')
    
    # 推定バックグラウンド強度 (中央値)
    ax1.axvline(bg_val, color='blue', linestyle='-', linewidth=2, label=r'$\hat{I}_{BG}(\theta_i)$ (Median)')
    
    ax1.set_title(r"$\theta_i$" f" = {row_idx}: Background Intensity Determination " r"$\hat{I}_{BG}(\theta_i)$")
    ax1.set_xlabel("Intensity")
    ax1.set_ylabel("Count")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log') # ログスケールの方が見やすい場合が多い

    # --- 右図: 置換処理の結果 (Cleaned Intensity) ---
    ax2 = axes[1]
    # 元データ
    ax2.plot(raw_row, color='gray', alpha=0.5, label='Original Raw')
    
    # 除去箇所を赤色で強調
    bad_indices = np.where(mask_row)[0]
    ax2.scatter(bad_indices, raw_row[bad_indices], color='red', marker='x', s=40, zorder=5, label='Bragg')
    
    # 推定BGレベル
    ax2.axhline(bg_val, color='blue', linestyle=':', alpha=0.6, label=r'$\hat{I}_{BG}(\theta_i)$')
    
    # 置換後のデータ
    ax2.plot(cleaned_row, color='green', linestyle='--', linewidth=1.5, label=r'${I_{clean}}_i$')
    
    ax2.set_title(r"$\theta_i$" f" = {row_idx}: Cleaned Intensity")
    ax2.set_xlabel("φ (Pixel)")
    ax2.set_ylabel("Intensity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    print(f"Showing details for Row {row_idx}... (Close window to continue)")
    plt.show()

def apply_cleaning_with_viz(raw_data, detected_df, check_rows=[]):
    """
    クリーニング処理を実行し、特定の行については詳細プロットを表示する
    """
    cleaned_data = raw_data.copy()
    rows, cols = raw_data.shape
    
    # マスク作成
    mask = np.zeros((rows, cols), dtype=bool)
    if not detected_df.empty:
        mask[detected_df["Theta_Row"], detected_df["Phi_Col"]] = True
    
    print(f"Processing {rows} rows... (Visualizing rows: {check_rows})")
    
    for r in range(rows):
        row_mask = mask[r, :]
        
        # ブラッグ反射がある場合のみ処理
        if np.any(row_mask):
            raw_row = raw_data[r, :]
            
            # 1. 正常画素の集合 Vi を抽出
            normal_pixels = raw_row[~row_mask]
            
            if len(normal_pixels) > 0:
                # 2. ロバスト推定によるバックグラウンド強度の決定 (中央値)
                bg_val = np.median(normal_pixels)
                
                # 3. 置換処理
                cleaned_data[r, row_mask] = bg_val
                
                # --- 可視化 (指定された行のみ) ---
                if r in check_rows:
                    plot_process_details(r, raw_row, row_mask, normal_pixels, bg_val, cleaned_data[r, :])
                    
            else:
                # 全画素がマスクされる異常事態（基本ありえない）
                cleaned_data[r, :] = np.min(raw_row)

    return cleaned_data, mask

# =============================================================================
# 3. メイン処理
# =============================================================================
def main():
    print("=== Step 2: Removal & Visualization ===")
    
    # データ読み込み
    df_raw = load_data(INPUT_RAW_DATA)
    df_detected = load_detected_coords(INPUT_DETECTED_LIST)
    
    if df_raw is None or df_detected is None:
        print("Please ensure input files exist and Step 1 has been run.")
        return
        
    raw_data = df_raw.values
    
    # 確認用に行を選ぶ（検出数が多い行トップNを自動抽出）
    target_rows = []
    if not df_detected.empty:
        # 出現回数が多い順に行インデックスを取得
        top_rows = df_detected['Theta_Row'].value_counts().head(NUM_CHECK_ROWS).index.tolist()
        target_rows = sorted(top_rows)
    
    # クリーニング実行 & 可視化
    cleaned_data, mask = apply_cleaning_with_viz(raw_data, df_detected, check_rows=target_rows)
    
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
    
    # 保存
    pd.DataFrame(cleaned_data).to_csv(OUTPUT_CLEANED_DATA, header=False, index=False)
    print(f"\n[Done] Cleaned data saved to: {OUTPUT_CLEANED_DATA}")

if __name__ == "__main__":
    main()