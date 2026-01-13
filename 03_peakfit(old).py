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
FILE_PATH = 'img/U_phi_000000.img' # 画像ファイルパス

FIXED_CENTER = (250, 65)

# フィッティング範囲設定
FIT_R_MIN = 0
FIT_R_MAX = 160  # 例: 160ピクセルまでをフィッティング対象とする

# =============================================================================
# 2. 画像読み込み & プロファイル計算
# =============================================================================
def load_raw_image(file_path):
    if not os.path.exists(file_path):
        print(f"ファイルが見つかりません: {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            f.seek(OFFSET_BYTES)
            remaining_data = f.read()
            raw_data = np.frombuffer(remaining_data, dtype=DATA_TYPE)
        
        if raw_data.size == IMAGE_WIDTH * IMAGE_HEIGHT:
            return raw_data.reshape((IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float32)
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def calculate_azimuthal_average(image, center):
    cx, cy = center
    h, w = image.shape
    y, x = np.indices((h, w))
    r_map = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_int = r_map.astype(np.int32)
    max_r = int(np.max(r_map))
    
    r_sum = np.bincount(r_int.ravel(), weights=image.ravel(), minlength=max_r+1)
    pixel_count = np.bincount(r_int.ravel(), minlength=max_r+1)
    
    radial_profile = np.divide(r_sum, pixel_count, out=np.zeros_like(r_sum), where=pixel_count > 0)
    return np.arange(max_r + 1), radial_profile

# =============================================================================
# 3. フィッティング用モデル関数（3つのガウス関数 + オフセット）
# =============================================================================
def multi_gaussian(x, a1, c1, w1, a2, c2, w2, a3, c3, w3, offset):
    """
    3つのガウスピークとベースライン（オフセット）の合成関数
    """
    peak1 = a1 * np.exp(-(x - c1)**2 / (2 * w1**2))
    peak2 = a2 * np.exp(-(x - c2)**2 / (2 * w2**2))
    peak3 = a3 * np.exp(-(x - c3)**2 / (2 * w3**2))
    return peak1 + peak2 + peak3 + offset

def single_gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

# =============================================================================
# 4. メイン処理：一括フィッティングと描画
# =============================================================================
def analyze_and_plot(file_path, center=FIXED_CENTER):
    image = load_raw_image(file_path)
    if image is None: return

    print(f"Target File: {file_path}")
    
    # 1. プロファイル計算
    radii, profile = calculate_azimuthal_average(image, center)

    # 2. フィッティング用データの抽出（指定範囲のみ）
    mask = (radii >= FIT_R_MIN) & (radii <= FIT_R_MAX)
    x_fit = radii[mask]
    y_fit = profile[mask]

    # 3. 初期パラメータの推定 (Initial Guesses)
    # [Amp1, Cen1, Wid1, Amp2, Cen2, Wid2, Amp3, Cen3, Wid3, Offset]
    # 増井さんの要件: r=0, 20, 125 付近
    
    # ベースラインの推定（データの最小値）
    offset_guess = np.min(y_fit)
    
    # 各ピークの初期値（適宜調整してください）
    p0 = [
        np.max(y_fit),  0.0,   5.0,   # Peak 1: 中心 (r=0)
        np.max(y_fit)*0.5, 20.0,  10.0,  # Peak 2: 近距離 (r=20)
        np.max(y_fit)*0.2, 125.0, 10.0,  # Peak 3: 遠距離 (r=125)
        offset_guess                     # Offset
    ]

    try:
        # 4. フィッティング実行
        popt, pcov = curve_fit(multi_gaussian, x_fit, y_fit, p0=p0, maxfev=10000)
        
        # 決定係数 R^2 の計算
        residuals = y_fit - multi_gaussian(x_fit, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"Fitting Successful! (R^2 = {r_squared:.4f})")
        print(f"Peak 1 (Center): Amp={popt[0]:.2f}, Cen={popt[1]:.2f}, Wid={popt[2]:.2f}")
        print(f"Peak 2 (Near)  : Amp={popt[3]:.2f}, Cen={popt[4]:.2f}, Wid={popt[5]:.2f}")
        print(f"Peak 3 (Far)   : Amp={popt[6]:.2f}, Cen={popt[7]:.2f}, Wid={popt[8]:.2f}")
        print(f"Offset         : {popt[9]:.2f}")

        # 5. 描画用の曲線生成（全範囲）
        x_plot = np.linspace(0, np.max(radii), 500)
        
        # 合計フィット
        y_total = multi_gaussian(x_plot, *popt)
        
        # 各成分（分解）
        y_peak1 = single_gaussian(x_plot, popt[0], popt[1], popt[2]) + popt[9]
        y_peak2 = single_gaussian(x_plot, popt[3], popt[4], popt[5]) + popt[9]
        y_peak3 = single_gaussian(x_plot, popt[6], popt[7], popt[8]) + popt[9]
        y_base  = np.full_like(x_plot, popt[9])

        # 6. プロット作成（スタイルを提示画像に寄せる）
        plt.figure(figsize=(6, 4))
        
        # 生データ（全範囲）を薄いグレーの点で
        plt.plot(radii, profile, '.', color='black', label='Raw Data', markersize=5)
        
        # フィッティング対象データを黒丸で
        # plt.plot(x_fit, y_fit, 'ko', label='Target Data', markersize=4)

        # 合計フィット曲線（赤実線）
        plt.plot(x_plot, y_total, 'r-', linewidth=2.0, label='Total Fit')

        # 成分分解（破線）
        plt.plot(x_plot, y_peak1, 'g--', linewidth=1.5, label=f'Sm Lα (r≈{popt[1]:.0f})')
        plt.plot(x_plot, y_peak2, 'b--', linewidth=1.5, label=f'Bi Lα (r≈{popt[4]:.0f})')
        plt.plot(x_plot, y_peak3, color='orange', linestyle='--', linewidth=1.5, label=f'Fe Kα (r≈{popt[7]:.0f})')
        
        # ベースライン（点線）
        # plt.plot(x_plot, y_base, 'k:', label='Baseline')

        plt.title(f"Peak Fitting Result")
        plt.xlabel("Radius r [px]")
        plt.ylabel("Intensity I [arb. unit]")
        plt.legend(loc='upper right', framealpha=0.9) # 凡例
        plt.grid(True, linestyle='-', alpha=0.6)      # グリッド
        plt.tight_layout()
        plt.show()

    except RuntimeError:
        print("フィッティングが収束しませんでした。初期パラメータや範囲を見直してください。")
    except Exception as e:
        print(f"予期せぬエラー: {e}")

if __name__ == "__main__":
    analyze_and_plot(FILE_PATH)