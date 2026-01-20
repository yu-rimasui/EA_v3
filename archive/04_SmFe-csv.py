import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from tqdm import tqdm  # 進捗バー表示用

# =============================================================================
# 1. パラメータ設定
# =============================================================================
# ★決定された固定パラメータ (Master Parameters - Double Peak)
# [Sm Peak]
FIXED_CEN_SM = 27.1227
FIXED_WID_SM = 15.4547
# [Fe Peak]
FIXED_CEN_FE = 125.9003
FIXED_WID_FE = 11.0976

# 画像設定
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 128
DATA_TYPE = np.int16
OFFSET_BYTES = 1024
IMAGE_CENTER_COORDS = (250, 65)

# ディレクトリ設定
energy = 13500
BASE_DIR = f'/Volumes/Extreme SSD/Sm-BiFeO3_RT/{energy}'

# 出力ファイル設定（Sm用とFe用に分ける）
OUTPUT_CSV_SM = f'output/Sm_{energy}eV_intensity_map.csv'
OUTPUT_CSV_FE = f'output/Fe_{energy}eV_intensity_map.csv'

# ディレクトリが存在しない場合の安全策
if not os.path.exists('output'):
    os.makedirs('output')

# 角度設定
THETA_START = 0
THETA_END   = 75    # Th00 ～ Th75
PHI_STEPS   = 1440  # 0 ～ 1439

# フィッティング範囲
FIT_R_MIN = 0
FIT_R_MAX = 200

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

def fixed_model(x, a1, a2, offset):
    """
    中心と幅を固定した2つのガウス関数の和 + オフセット
    変数は a1(Sm振幅), a2(Fe振幅), offset のみ
    """
    # Sm由来 (Peak 1)
    g1 = a1 * np.exp(-(x - FIXED_CEN_SM)**2 / (2 * FIXED_WID_SM**2))
    # Fe由来 (Peak 2)
    g2 = a2 * np.exp(-(x - FIXED_CEN_FE)**2 / (2 * FIXED_WID_FE**2))
    return g1 + g2 + offset

# =============================================================================
# 3. メイン処理（バッチ処理）
# =============================================================================
def main():
    # 結果格納用配列を2つ用意 (行: Theta 0-75, 列: Phi 0-1439)
    result_map_sm = np.zeros((THETA_END - THETA_START + 1, PHI_STEPS))
    result_map_fe = np.zeros((THETA_END - THETA_START + 1, PHI_STEPS))

    print(f"Processing {energy}eV Data (L-U Difference)...")
    print(f"Fixed Sm Param: Center={FIXED_CEN_SM}, Width={FIXED_WID_SM}")
    print(f"Fixed Fe Param: Center={FIXED_CEN_FE}, Width={FIXED_WID_FE}")
    print(f"Output Files: \n  Sm -> {OUTPUT_CSV_SM}\n  Fe -> {OUTPUT_CSV_FE}")

    # Thetaループ
    for t_idx, theta in enumerate(range(THETA_START, THETA_END + 1)):
        theta_dir_name = f"Th{theta:02d}"
        theta_path = os.path.join(BASE_DIR, theta_dir_name)
        
        if not os.path.exists(theta_path):
            print(f"Skipping {theta_dir_name} (Not found)")
            continue

        # Phiループ (tqdmで進捗表示)
        for phi in tqdm(range(PHI_STEPS), desc=f"Theta {theta:02d}", leave=False):
            # LとUのファイルパス生成
            l_file = os.path.join(theta_path, f"L_phi_{phi:06d}.img")
            u_file = os.path.join(theta_path, f"U_phi_{phi:06d}.img")

            # 画像読み込み
            img_l = load_raw_image(l_file)
            img_u = load_raw_image(u_file)
            if img_l is None or img_u is None:
                continue

            diff_img = img_l - img_u

            # 1. プロファイル計算
            radii, profile = calculate_sum_profile(diff_img, center=IMAGE_CENTER_COORDS)

            # 2. フィッティング用データの抽出
            mask = (radii >= FIT_R_MIN) & (radii <= FIT_R_MAX)
            x_fit = radii[mask]
            y_fit = profile[mask]

            try:
                # 3. フィッティング実行
                # パラメータ: [Sm振幅(a1), Fe振幅(a2), オフセット]
                max_val = np.max(y_fit)
                p0 = [max_val, max_val * 0.5, 0.0] 
                
                # bounds: 振幅は正の値、オフセットは自由
                popt, _ = curve_fit(fixed_model, x_fit, y_fit, p0=p0, 
                                    bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]))
                
                # 4. 面積計算 (Area = Amp * Width * sqrt(2*pi))
                area_sm = popt[0] * FIXED_WID_SM * np.sqrt(2 * np.pi)
                area_fe = popt[1] * FIXED_WID_FE * np.sqrt(2 * np.pi)
                
                # 結果配列に格納
                result_map_sm[t_idx, phi] = area_sm
                result_map_fe[t_idx, phi] = area_fe

            except Exception:
                # フィッティング失敗時は 0 を入れる
                result_map_sm[t_idx, phi] = 0.0
                result_map_fe[t_idx, phi] = 0.0

    # =============================================================================
    # 4. CSV出力
    # =============================================================================
    # Sm用のCSV出力
    df_sm = pd.DataFrame(result_map_sm)
    df_sm.to_csv(OUTPUT_CSV_SM, header=False, index=False)
    
    # Fe用のCSV出力
    df_fe = pd.DataFrame(result_map_fe)
    df_fe.to_csv(OUTPUT_CSV_FE, header=False, index=False)
    
    print(f"\n完了しました。")
    print(f"Smデータ保存先: {OUTPUT_CSV_SM}")
    print(f"Feデータ保存先: {OUTPUT_CSV_FE}")
    print(f"データサイズ: {df_sm.shape} (Row:Theta0-75, Col:Phi0-1439)")

if __name__ == "__main__":
    main()