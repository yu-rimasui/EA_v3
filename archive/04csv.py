import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from tqdm import tqdm  # 進捗バー表示用

# =============================================================================
# 1. パラメータ設定
# =============================================================================
# 解析するエネルギー (適宜変更してください)
energy = 13500

# ★固定パラメータ (Master Parameters)
# [Bi Peak] (from U-only analysis)
FIXED_CEN_BI = 30.85
FIXED_WID_BI = 13.42

# [Sm Peak] (from L-U analysis)
FIXED_CEN_SM = 27.1227
FIXED_WID_SM = 15.4547

# [Fe Peak] (from L-U analysis)
FIXED_CEN_FE = 125.9003
FIXED_WID_FE = 11.0976

# 画像設定
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 128
DATA_TYPE = np.int16
OFFSET_BYTES = 1024
IMAGE_CENTER_COORDS = (250, 65)

# ディレクトリ設定
# 例: /Volumes/SSD名/データフォルダ/{energy}
BASE_DIR = f'/Volumes/Extreme SSD/Sm-BiFeO3_RT/{energy}'

# 出力ディレクトリ作成
OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 出力ファイル名
OUTPUT_CSV_BI = os.path.join(OUTPUT_DIR, f'Bi_{energy}eV_intensity_map.csv')
OUTPUT_CSV_SM = os.path.join(OUTPUT_DIR, f'Sm_{energy}eV_intensity_map.csv')
OUTPUT_CSV_FE = os.path.join(OUTPUT_DIR, f'Fe_{energy}eV_intensity_map.csv')

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

# --- モデル関数 ---
def model_bi(x, amp, offset):
    """Bi用: 中心と幅を固定した単一ガウス関数"""
    return amp * np.exp(-(x - FIXED_CEN_BI)**2 / (2 * FIXED_WID_BI**2)) + offset

def model_smfe(x, a1, a2, offset):
    """Sm/Fe用: 中心と幅を固定した2つのガウス関数の和"""
    # Sm由来 (Peak 1)
    g1 = a1 * np.exp(-(x - FIXED_CEN_SM)**2 / (2 * FIXED_WID_SM**2))
    # Fe由来 (Peak 2)
    g2 = a2 * np.exp(-(x - FIXED_CEN_FE)**2 / (2 * FIXED_WID_FE**2))
    return g1 + g2 + offset

# =============================================================================
# 3. メイン処理（バッチ処理）
# =============================================================================
def main():
    # 結果格納用配列 (行: Theta 0-75, 列: Phi 0-1439)
    result_map_bi = np.zeros((THETA_END - THETA_START + 1, PHI_STEPS))
    result_map_sm = np.zeros((THETA_END - THETA_START + 1, PHI_STEPS))
    result_map_fe = np.zeros((THETA_END - THETA_START + 1, PHI_STEPS))

    print(f"Processing {energy}eV Data...")
    print(f"Directory: {BASE_DIR}")
    print("-" * 50)
    print(f"Bi Params: Cen={FIXED_CEN_BI}, Wid={FIXED_WID_BI}")
    print(f"Sm Params: Cen={FIXED_CEN_SM}, Wid={FIXED_WID_SM}")
    print(f"Fe Params: Cen={FIXED_CEN_FE}, Wid={FIXED_WID_FE}")
    print("-" * 50)

    # Thetaループ
    for t_idx, theta in enumerate(range(THETA_START, THETA_END + 1)):
        theta_dir_name = f"Th{theta:02d}"
        theta_path = os.path.join(BASE_DIR, theta_dir_name)
        
        if not os.path.exists(theta_path):
            print(f"Skipping {theta_dir_name} (Not found)")
            continue

        # Phiループ (tqdmで進捗表示)
        for phi in tqdm(range(PHI_STEPS), desc=f"Theta {theta:02d}", leave=False):
            # ファイルパス生成
            u_file = os.path.join(theta_path, f"U_phi_{phi:06d}.img")
            l_file = os.path.join(theta_path, f"L_phi_{phi:06d}.img")

            # 画像読み込み
            img_u = load_raw_image(u_file) # Biで使用
            img_l = load_raw_image(l_file) # Sm/Feで使用
            
            # --- Bi解析 (U画像のみ) ---
            if img_u is not None:
                # プロファイル計算
                radii_u, profile_u = calculate_sum_profile(img_u, center=IMAGE_CENTER_COORDS)
                
                # フィッティング範囲 (0-150)
                mask_bi = (radii_u >= FIT_R_MIN) & (radii_u <= 150)
                x_fit_bi = radii_u[mask_bi]
                y_fit_bi = profile_u[mask_bi]

                try:
                    p0_bi = [300.0, 0.0] # [Amp, Offset]
                    popt_bi, _ = curve_fit(model_bi, x_fit_bi, y_fit_bi, p0=p0_bi, 
                                            bounds=([0, -np.inf], [np.inf, np.inf]))
                    # 面積計算
                    area_bi = popt_bi[0] * FIXED_WID_BI * np.sqrt(2 * np.pi)
                    result_map_bi[t_idx, phi] = area_bi
                except:
                    result_map_bi[t_idx, phi] = 0.0

            # --- Sm/Fe解析 (L-U 差分画像) ---
            if img_l is not None and img_u is not None:
                # 差分計算
                diff_img = img_l - img_u
                
                # プロファイル計算
                radii_sf, profile_sf = calculate_sum_profile(diff_img, center=IMAGE_CENTER_COORDS)

                # フィッティング範囲 (0-200)
                mask_sf = (radii_sf >= FIT_R_MIN) & (radii_sf <= 200)
                x_fit_sf = radii_sf[mask_sf]
                y_fit_sf = profile_sf[mask_sf]

                try:
                    max_val = np.max(y_fit_sf)
                    p0_sf = [max_val, max_val * 0.5, 0.0] # [SmAmp, FeAmp, Offset]
                    
                    popt_sf, _ = curve_fit(model_smfe, x_fit_sf, y_fit_sf, p0=p0_sf, 
                                            bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]))
                    
                    # 面積計算
                    area_sm = popt_sf[0] * FIXED_WID_SM * np.sqrt(2 * np.pi)
                    area_fe = popt_sf[1] * FIXED_WID_FE * np.sqrt(2 * np.pi)
                    
                    result_map_sm[t_idx, phi] = area_sm
                    result_map_fe[t_idx, phi] = area_fe
                except:
                    result_map_sm[t_idx, phi] = 0.0
                    result_map_fe[t_idx, phi] = 0.0

    # =============================================================================
    # 4. CSV出力
    # =============================================================================
    print("\nSaving CSV files...")
    
    # Bi
    pd.DataFrame(result_map_bi).to_csv(OUTPUT_CSV_BI, header=False, index=False)
    print(f"  Bi -> {OUTPUT_CSV_BI}")

    # Sm
    pd.DataFrame(result_map_sm).to_csv(OUTPUT_CSV_SM, header=False, index=False)
    print(f"  Sm -> {OUTPUT_CSV_SM}")

    # Fe
    pd.DataFrame(result_map_fe).to_csv(OUTPUT_CSV_FE, header=False, index=False)
    print(f"  Fe -> {OUTPUT_CSV_FE}")
    
    print("\nAll processing completed.")

if __name__ == "__main__":
    main()