import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# =============================================================================
# 1. パラメータ設定
# =============================================================================
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 128
DATA_TYPE = np.int16
OFFSET_BYTES = 1024

# 解析対象ファイルパス
number = '000000'
FILE_PATH_L = f'/Volumes/Extreme SSD/Sm-BiFeO3_RT/13500/Th00/L_phi_{number}.img'
FILE_PATH_U = f'/Volumes/Extreme SSD/Sm-BiFeO3_RT/13500/Th00/U_phi_{number}.img'

# 画像の中心座標
IMAGE_CENTER_COORDS = (250, 65)

# マスターパラメータ
# [Sm Peak]
FIXED_CEN_SM = 27.1227
FIXED_WID_SM = 15.4547
# [Fe Peak]
FIXED_CEN_FE = 125.9003
FIXED_WID_FE = 11.0976

# フィッティング範囲
FIT_R_MIN = 0
FIT_R_MAX = 200

# =============================================================================
# 2. 関数定義
# =============================================================================
def load_raw_image(file_path):
    """画像を読み込む関数"""
    if not os.path.exists(file_path):
        print(f"ファイルが見つかりません: {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            f.seek(OFFSET_BYTES)
            raw_data = np.frombuffer(f.read(), dtype=DATA_TYPE)
        if raw_data.size == IMAGE_WIDTH * IMAGE_HEIGHT:
            return raw_data.reshape((IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float32)
        return None
    except Exception as e:
        print(f"読み込みエラー: {e}")
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

# =============================================================================
# 3. フィッティング用モデル関数
# =============================================================================
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
# 4. メイン処理
# =============================================================================
def analyze_and_plot(file_path_l, file_path_u, center=IMAGE_CENTER_COORDS):
    image_l = load_raw_image(file_path_l)
    image_u = load_raw_image(file_path_u)
    if image_l is None or image_u is None:
        return

    print(f"Target File: {file_path_l}, {file_path_u}")
    
    # 1. プロファイル計算
    diff_image = image_l - image_u
    radii, profile = calculate_sum_profile(diff_image, center)

    # 2. フィッティング用データの抽出
    mask = (radii >= FIT_R_MIN) & (radii <= FIT_R_MAX)
    x_fit = radii[mask]
    y_fit = profile[mask]

    # 3. 初期パラメータ
    max_val = np.max(y_fit)
    p0 = [
        max_val,         # Peak 1 (Sm)
        max_val * 0.5,   # Peak 2 (Fe)
        0.0              # Offset
    ]

    try:
        # 4. フィッティング実行
        popt, pcov = curve_fit(fixed_model, x_fit, y_fit, p0=p0, maxfev=10000)
        
        # 決定係数 R^2 の計算
        residuals = y_fit - fixed_model(x_fit, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
        r_squared = 1 - (ss_res / ss_tot)

        print("-" * 40)
        print(f"Fitting Successful! (R^2 = {r_squared:.4f})")

        print(f"Sm Peak : Amp={popt[0]:.2f}")
        print(f"          Center={FIXED_CEN_SM:.2f} (Fixed)")
        print(f"          Width ={FIXED_WID_SM:.2f} (Fixed)")
        print(f"Fe Peak : Amp={popt[1]:.2f}")
        print(f"          Center={FIXED_CEN_FE:.2f} (Fixed)")
        print(f"          Width ={FIXED_WID_FE:.2f} (Fixed)")
        print(f"Offset  : {popt[2]:.2f}")

        # 面積計算
        area_sm = popt[0] * FIXED_WID_SM * np.sqrt(2 * np.pi)
        print(f"Calclated Sm Area: {area_sm:.2f}")

        area_fe = popt[1] * FIXED_WID_FE * np.sqrt(2 * np.pi)
        print(f"Calclated Fe Area: {area_fe:.2f}")
        print("-" * 40)

        # 5. プロット
        x_plot = np.linspace(0, np.max(radii), 500)
        y_fit_curve = fixed_model(x_plot, *popt)

        plt.figure(figsize=(8, 5))
        
        # 生データ
        plt.plot(radii, profile, 'k-', alpha=0.5, label='Raw Data')

        # plt.plot(x_plot, y_fit_curve, 'g--', linewidth=2.0, label='Total Fit', alpha=0.8)

        # popt: [SmAmp, FeAmp, Offset]
        y_sm = popt[0] * np.exp(-(x_plot - FIXED_CEN_SM)**2 / (2 * FIXED_WID_SM**2)) + popt[2]
        plt.plot(x_plot, y_sm, 'r--', linewidth=1.5, label=f'Sm Component (area={area_sm:.2f})')
        y_fe = popt[1] * np.exp(-(x_plot - FIXED_CEN_FE)**2 / (2 * FIXED_WID_FE**2)) + popt[2]
        plt.plot(x_plot, y_fe, 'b--', linewidth=1.5, label=f'Fe Component (area={area_fe:.2f})')
        plt.axhline(y=popt[2], color='gray', linestyle=':', label='Baseline')

        plt.title(f"Sm, Fe Peak Fitting (Fixed Param) $R^2$={r_squared:.3f}")
        plt.xlabel("Radius r [px]")
        plt.ylabel("Intensity Sum [arb. unit]")
        plt.legend()
        plt.grid(True, linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"フィッティングエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_and_plot(FILE_PATH_L, FILE_PATH_U)