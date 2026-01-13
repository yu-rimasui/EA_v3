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
FILE_PATH = 'img/L_phi_000000.img' 

# 画像の中心座標
IMAGE_CENTER_COORDS = (250, 65)

# ★固定するマスターパラメータ（02_Bi-param.pyより決定）
FIXED_CENTER = 30.85
FIXED_WIDTH  = 13.42

# フィッティング範囲
FIT_R_MIN = 0
FIT_R_MAX = 150

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
def fixed_model(x, amp, offset):
    """
    CenterとWidthを固定し、高さ(amp)とオフセットだけを調整するモデル
    """
    return amp * np.exp(-(x - FIXED_CENTER)**2 / (2 * FIXED_WIDTH**2)) + offset

# =============================================================================
# 4. メイン処理
# =============================================================================
def analyze_and_plot(file_path, center=IMAGE_CENTER_COORDS):
    image = load_raw_image(file_path)
    if image is None: return

    print(f"Target File: {file_path}")
    
    # 1. プロファイル計算
    radii, profile = calculate_sum_profile(image, center)

    # 2. フィッティング用データの抽出
    mask = (radii >= FIT_R_MIN) & (radii <= FIT_R_MAX)
    x_fit = radii[mask]
    y_fit = profile[mask]

    # 3. 初期パラメータ
    p0 = [300.0, 0.0] # 初期値: [Amp, Offset]

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
        print(f"Bi Peak : Amp={popt[0]:.2f}")
        print(f"          Center={FIXED_CENTER:.2f} (Fixed)")
        print(f"          Width ={FIXED_WIDTH:.2f}  (Fixed)")
        print(f"Offset  : {popt[1]:.2f}")
        
        # 面積計算
        area = popt[0] * FIXED_WIDTH * np.sqrt(2 * np.pi)
        print(f"Calclated Area: {area:.2f}")
        print("-" * 40)

        # 5. プロット
        x_plot = np.linspace(0, np.max(radii), 500)
        y_fit_curve = fixed_model(x_plot, *popt)

        plt.figure(figsize=(8, 5))
        
        # 生データ
        plt.plot(radii, profile, '-', color='blue', label='Measured Data (Sum)', alpha=0.6)
        
        # フィッティング結果
        plt.plot(x_plot, y_fit_curve, 'r--', linewidth=2.0, label=f'Fixed Fit\nAmp={popt[0]:.2f}')

        plt.title(f"Bi Peak Fitting (Fixed Param) $R^2$={r_squared:.3f}")
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
    analyze_and_plot(FILE_PATH)