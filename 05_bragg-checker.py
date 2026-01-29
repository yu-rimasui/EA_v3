import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

# =============================================================================
# 1. パラメータ設定
# =============================================================================
# 解析するエネルギー
energy = 17000

# ★ ブラッグ反射判定用の閾値
# この値を超えたら「ブラッグ反射」とみなして画像をポップアップします。
BRAGG_THRESHOLD = 35195.54

# 固定パラメータ (Bi)
FIXED_CEN_BI = 30.85
FIXED_WID_BI = 13.42

# 画像設定
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 128
DATA_TYPE = np.int16
OFFSET_BYTES = 1024
IMAGE_CENTER_COORDS = (250, 65)

# ディレクトリ設定 (環境に合わせて変更してください)
BASE_DIR = f'/Volumes/Extreme SSD/Sm-BiFeO3_RT/{energy}'
OUTPUT_DIR = 'output'

# 角度設定
THETA_START = 73
THETA_END   = 75
PHI_STEPS   = 1440

# ログファイル名
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_LOG_BRAGG = os.path.join(OUTPUT_DIR, f'Bragg_Log_{energy}eV(θ={THETA_START}~{THETA_END}°, th={BRAGG_THRESHOLD}).txt')

# フィッティング範囲
FIT_R_MIN = 0

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
    cx, cy = center
    y, x = np.indices(image.shape)
    d = np.sqrt((x - cx)**2 + (y - cy)**2)
    r = np.round(d).astype(int)
    max_r = np.max(r)
    sum_intensity = np.bincount(r.ravel(), weights=image.ravel(), minlength=max_r+1)
    radii = np.arange(max_r + 1)
    return radii, sum_intensity

def model_bi(x, amp, offset):
    return amp * np.exp(-(x - FIXED_CEN_BI)**2 / (2 * FIXED_WID_BI**2)) + offset

# =============================================================================
# 3. メイン処理 (確認・可視化専用)
# =============================================================================
def main():
    bragg_log = [] # ログ用リスト

    print(f"=== Bragg Reflection Checker ({energy}eV) ===")
    print(f"Threshold: {BRAGG_THRESHOLD}")
    print("Scanning data... (Close the plot window to resume)")
    print("-" * 50)

    # グラフ表示用のインタラクティブモード
    plt.ion() 

    for theta in range(THETA_START, THETA_END + 1):
        theta_dir_name = f"Th{theta:02d}"
        theta_path = os.path.join(BASE_DIR, theta_dir_name)
        
        if not os.path.exists(theta_path):
            continue

        # tqdmで進捗表示
        for phi in tqdm(range(PHI_STEPS), desc=f"Scanning Th{theta:02d}", leave=False):
            u_file = os.path.join(theta_path, f"U_phi_{phi:06d}.img")

            # 画像読み込み (Bi用 U画像のみ)
            img_u = load_raw_image(u_file)
            
            if img_u is None:
                continue

            # --- 解析 (Bi) ---
            radii_u, profile_u = calculate_sum_profile(img_u, center=IMAGE_CENTER_COORDS)
            mask_bi = (radii_u >= FIT_R_MIN) & (radii_u <= 150)
            
            area_bi = 0.0
            popt_bi = [0, 0]
            
            try:
                # 高速化のため初期値を固定気味に
                p0_bi = [300.0, 0.0]
                popt_bi, _ = curve_fit(model_bi, radii_u[mask_bi], profile_u[mask_bi], p0=p0_bi, 
                                        bounds=([0, -np.inf], [np.inf, np.inf]))
                area_bi = popt_bi[0] * FIXED_WID_BI * np.sqrt(2 * np.pi)
            except:
                continue

            # ★★★ 判定 & 可視化 ★★★
            if area_bi > BRAGG_THRESHOLD:
                # ログ用文字列作成
                log_entry = f"Theta={theta}, Phi={phi}, Intensity={area_bi:.2f}"
                bragg_log.append(log_entry)
                
                # コンソールに通知
                tqdm.write(f"\n[DETECTED] {log_entry}")

                # --- 画像表示 (一時停止) ---
                plt.ioff() # インタラクティブモードOFF
                
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                
                # 左：RAW画像
                # コントラスト調整: 99.5パーセンタイルでクリップして見やすく
                vmax = np.percentile(img_u, 99.5)
                im = ax[0].imshow(img_u, cmap='jet', vmin=0, vmax=vmax)
                ax[0].set_title(f"Bragg/Noise Check\nTh={theta}, Phi={phi}")
                plt.colorbar(im, ax=ax[0])
                ax[0].plot(IMAGE_CENTER_COORDS[0], IMAGE_CENTER_COORDS[1], 'rx') # 中心
                
                # 右：プロファイル
                ax[1].plot(radii_u[mask_bi], profile_u[mask_bi], 'b.', label='Data', alpha=0.6)
                if area_bi > 0:
                    y_fit = model_bi(radii_u[mask_bi], *popt_bi)
                    ax[1].plot(radii_u[mask_bi], y_fit, 'r-', label='Fit', linewidth=2)
                
                ax[1].set_title(f"Profile (Area={area_bi:.0f})")
                ax[1].set_xlabel("Radius (pixel)")
                ax[1].set_ylabel("Intensity")
                ax[1].legend()
                ax[1].grid(True)

                plt.tight_layout()
                plt.show() # ウィンドウを閉じるまで待機
                
                plt.ion()  # 再開

    # =============================================================================
    # 4. ログ保存
    # =============================================================================
    print("\nScan completed.")
    if bragg_log:
        with open(OUTPUT_LOG_BRAGG, 'w') as f:
            f.write("\n".join(bragg_log))
        print(f"Log saved to: {OUTPUT_LOG_BRAGG}")
        print(f"Total detected frames: {len(bragg_log)}")
    else:
        print("No Bragg reflections detected with current threshold.")

if __name__ == "__main__":
    main()