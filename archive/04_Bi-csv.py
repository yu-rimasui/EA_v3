import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from tqdm import tqdm  # 進捗バー表示用

# =============================================================================
# 1. パラメータ設定
# =============================================================================
# 固定パラメータ (02_Bi-param.pyより決定)
FIXED_CENTER = 30.85
FIXED_WIDTH  = 13.42

# 画像設定
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 128
DATA_TYPE = np.int16
OFFSET_BYTES = 1024
IMAGE_CENTER_COORDS = (250, 65)

# ディレクトリ設定
# 例: f'/Volumes/SSDの名前/データフォルダ/{energy}'
energy = 15500
BASE_DIR = f'/Volumes/Extreme SSD/Sm-BiFeO3_RT/{energy}'
OUTPUT_CSV = f'output/Bi_{energy}eV_intensity_map.csv'


# 角度設定
THETA_START = 0
THETA_END   = 75    # Th00 ～ Th75 (計76ステップ)
PHI_STEPS   = 1440  # 0 ～ 1439

# フィッティング範囲
FIT_R_MIN = 0
FIT_R_MAX = 150

# =============================================================================
# 2. 関数定義
# =============================================================================
def load_raw_image(file_path):
    """画像を読み込む関数"""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'rb') as f:
            f.seek(OFFSET_BYTES)
            data = np.frombuffer(f.read(), dtype=DATA_TYPE)
        if data.size == IMAGE_WIDTH * IMAGE_HEIGHT:
            return data.reshape((IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float32)
        return None
    except:
        return None

def calculate_sum_profile(image, center):
    """合計プロファイルを計算する関数"""
    cx, cy = center
    y, x = np.indices(image.shape)
    d = np.sqrt((x - cx)**2 + (y - cy)**2)
    r = np.round(d).astype(int)
    max_r = np.max(r)
    
    sum_intensity = np.bincount(r.ravel(), weights=image.ravel(), minlength=max_r+1)
    radii = np.arange(max_r + 1)
    return radii, sum_intensity

def fixed_model(x, amp, offset):
    """CenterとWidthを固定し、高さ(amp)とオフセットだけを調整するモデル"""
    return amp * np.exp(-(x - FIXED_CENTER)**2 / (2 * FIXED_WIDTH**2)) + offset

# =============================================================================
# 3. メイン処理（バッチ処理）
# =============================================================================
def main():
    # 結果格納用配列 (行: Theta 0-75, 列: Phi 0-1439)
    result_map_bi = np.zeros((THETA_END - THETA_START + 1, PHI_STEPS))

    print(f"Processing {energy}eV Data (U-Only)...")
    print(f"Fixed Parameters: Center={FIXED_CENTER}, Width={FIXED_WIDTH}")
    print(f"Output File: \n  Bi -> {OUTPUT_CSV}")

    # Thetaループ (Th00 -> Th75)
    for t_idx, theta in enumerate(range(THETA_START, THETA_END + 1)):
        theta_dir_name = f"Th{theta:02d}"
        theta_path = os.path.join(BASE_DIR, theta_dir_name)
        
        if not os.path.exists(theta_path):
            print(f"Skipping {theta_dir_name} (Not found)")
            continue

        # Phiループ (000000 -> 001439)
        # tqdmで進捗バーを表示
        for phi in tqdm(range(PHI_STEPS), desc=f"Theta {theta:02d}", leave=False):
            # Uファイルパスの生成
            u_file = os.path.join(theta_path, f"U_phi_{phi:06d}.img")

            img = load_raw_image(u_file)
            if img is None:
                continue

            # 1. プロファイル計算
            radii, profile = calculate_sum_profile(img, center=IMAGE_CENTER_COORDS)

            # 2. フィッティング用データの抽出
            mask = (radii >= FIT_R_MIN) & (radii <= FIT_R_MAX)
            x_fit = radii[mask]
            y_fit = profile[mask]

            try:
                # 3. フィッティング実行
                max_val = np.max(y_fit)
                p0 = [max_val, 0.0] # 初期値: [Amp, Offset]
                
                # boundsを設定してAmpがマイナスになるのを防ぐ (0, -inf) ~ (inf, inf)
                popt, _ = curve_fit(fixed_model, x_fit, y_fit, p0=p0, bounds=([0, -np.inf], [np.inf, np.inf]))
                
                # 4. 面積計算 (Area = Amp * Width * sqrt(2*pi))
                area_bi = popt[0] * FIXED_WIDTH * np.sqrt(2 * np.pi)
                
                # 結果配列に格納
                result_map_bi[t_idx, phi] = area_bi

            except Exception:
                # フィッティング失敗時は 0 を入れる
                result_map_bi[t_idx, phi] = 0.0

    # =============================================================================
    # 4. CSV出力
    # =============================================================================
    # Bi用のCSV出力
    df_bi = pd.DataFrame(result_map_bi)
    df_bi.to_csv(OUTPUT_CSV, header=False, index=False)
    
    print(f"\n完了しました。")
    print(f"保存先: {OUTPUT_CSV}")
    print(f"データサイズ: {df_bi.shape} (Row:Theta0-75, Col:Phi0-1439)")

if __name__ == "__main__":
    main()