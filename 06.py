'''
完成したアルゴリズムを用いて、大量のデータを自動処理・フィッティング・ログ保存するメインスクリプト
'''
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
import cv2  # 画像処理用に追加

# =============================================================================
# 1. パラメータ設定
# =============================================================================
energy = 17000

BRAGG_THRESHOLD = 35195.54

# ★ 除去処理の感度 (平均から標準偏差の何倍離れたら消すか)
# 3.0〜5.0 推奨。小さいほど厳しく（たくさん）消します。
SIGMA_CLIP_THRESHOLD = 1.0 

FIXED_CEN_BI = 30.85
FIXED_WID_BI = 13.42

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 128
DATA_TYPE = np.int16
OFFSET_BYTES = 1024
IMAGE_CENTER_COORDS = (250, 65)

BASE_DIR = f'/Volumes/Extreme SSD/Sm-BiFeO3_RT/{energy}'
OUTPUT_DIR = 'output'

THETA_START = 70
THETA_END   = 75
PHI_STEPS   = 1440 

# ログファイル名
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_LOG_BRAGG = os.path.join(OUTPUT_DIR, f'Bragg_Log_{energy}eV(θ={THETA_START}~{THETA_END}°, th={BRAGG_THRESHOLD}).txt')

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
    """2D画像から動径分布(1Dプロファイル)を計算"""
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

def remove_bragg_spikes(img, center, sigma_thresh=3.0):
    """
    極座標変換を利用してブラッグ反射(スパイク)を除去し、
    直交座標(元の画像形式)に戻して返す関数
    """
    h, w = img.shape
    cx, cy = center
    
    # 1. 極座標変換の設定
    max_radius = int(np.sqrt((w/2)**2 + (h/2)**2) * 1.2) # コーナーまでカバー
    polar_w = 720 # 角度分解能 (0.5度刻み)
    polar_h = max_radius
    
    # マッピング作成 (Cartesian -> Polar)
    theta_vals = np.linspace(0, 2*np.pi, polar_w)
    r_vals = np.linspace(0, max_radius, polar_h)
    Theta, R = np.meshgrid(theta_vals, r_vals)
    
    map_x = (cx + R * np.cos(Theta)).astype(np.float32)
    map_y = (cy + R * np.sin(Theta)).astype(np.float32)
    
    # Polar画像への変換
    polar_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    # 2. スパイク除去 (行=半径ごとに処理)
    clean_polar = polar_img.copy()
    for r_idx in range(polar_h):
        row_data = clean_polar[r_idx, :]
        median_val = np.median(row_data)
        std_val = np.std(row_data)
        
        # 閾値判定 (Median + N*Sigma)
        cutoff = median_val + sigma_thresh * std_val
        spikes = row_data > cutoff
        
        # スパイクを中央値で置き換え
        if np.any(spikes):
            clean_polar[r_idx, spikes] = median_val
            
    # 3. 逆変換 (Polar -> Cartesian)
    # Cartesian座標グリッドを作成
    Y_cart, X_cart = np.indices((h, w))
    dy = Y_cart - cy
    dx = X_cart - cx
    R_cart = np.sqrt(dx**2 + dy**2)
    Theta_cart = np.arctan2(dy, dx) # -pi to pi
    Theta_cart[Theta_cart < 0] += 2*np.pi # 0 to 2pi
    
    # Polar画像の対応座標に変換
    # polar_x (col) = theta * (polar_w / 2pi)
    # polar_y (row) = r * (polar_h / max_radius) ?? No, r is index directly if linear
    # r_vals is linspace(0, max, polar_h), so r_idx = r * (polar_h-1)/max_radius
    
    map_polar_x = (Theta_cart * (polar_w - 1) / (2 * np.pi)).astype(np.float32)
    map_polar_y = (R_cart * (polar_h - 1) / max_radius).astype(np.float32)
    
    # Cartesianに戻す (source=clean_polar)
    clean_cart_img = cv2.remap(clean_polar, map_polar_x, map_polar_y, interpolation=cv2.INTER_LINEAR)
    
    return clean_cart_img

# =============================================================================
# 3. メイン処理 (確認・可視化専用)
# =============================================================================
def main():
    bragg_log = []

    print(f"=== Bragg Reflection Checker & Remover ({energy}eV) ===")
    print(f"Detection Threshold: {BRAGG_THRESHOLD}")
    print(f"Removal Sensitivity (Sigma): {SIGMA_CLIP_THRESHOLD}")
    print("Scanning data...")
    print("-" * 50)

    # グラフ表示用のインタラクティブモード
    plt.ion() 

    for theta in range(THETA_START, THETA_END + 1):
        theta_dir_name = f"Th{theta:02d}"
        theta_path = os.path.join(BASE_DIR, theta_dir_name)
        
        if not os.path.exists(theta_path):
            continue

        for phi in tqdm(range(PHI_STEPS), desc=f"Scanning Th{theta:02d}", leave=False):
            u_file = os.path.join(theta_path, f"U_phi_{phi:06d}.img")

            img_u = load_raw_image(u_file)
            if img_u is None:
                continue

            # --- 1. オリジナル解析 ---
            radii_u, profile_u = calculate_sum_profile(img_u, center=IMAGE_CENTER_COORDS)
            mask_bi = (radii_u >= FIT_R_MIN) & (radii_u <= 150)
            
            area_bi = 0.0
            popt_bi = [0, 0]
            
            try:
                p0_bi = [300.0, 0.0]
                popt_bi, _ = curve_fit(model_bi, radii_u[mask_bi], profile_u[mask_bi], p0=p0_bi, 
                                        bounds=([0, -np.inf], [np.inf, np.inf]))
                area_bi = popt_bi[0] * FIXED_WID_BI * np.sqrt(2 * np.pi)
            except:
                continue

            # ★★★ 判定 & 可視化 ★★★
            if area_bi > BRAGG_THRESHOLD:
                # ログ
                log_entry = f"Theta={theta}, Phi={phi}, Original_Area={area_bi:.2f}"
                bragg_log.append(log_entry)
                tqdm.write(f"\n[DETECTED] {log_entry}")

                # --- 2. 除去処理 ---
                img_clean = remove_bragg_spikes(img_u, IMAGE_CENTER_COORDS, sigma_thresh=SIGMA_CLIP_THRESHOLD)
                
                # --- 3. 除去後の再解析 ---
                radii_clean, profile_clean = calculate_sum_profile(img_clean, center=IMAGE_CENTER_COORDS)
                mask_clean = (radii_clean >= FIT_R_MIN) & (radii_clean <= 150)
                
                area_clean = 0.0
                popt_clean = [0, 0]
                try:
                    popt_clean, _ = curve_fit(model_bi, radii_clean[mask_clean], profile_clean[mask_clean], p0=p0_bi,
                                              bounds=([0, -np.inf], [np.inf, np.inf]))
                    area_clean = popt_clean[0] * FIXED_WID_BI * np.sqrt(2 * np.pi)
                except:
                    pass

                # --- 4. 画像表示 (4画面比較) ---
                fig, ax = plt.subplots(2, 2, figsize=(14, 10))
                
                # 上段：除去前 (Original)
                vmax = np.percentile(img_u, 99.5) # コントラスト設定
                ax[0, 0].imshow(img_u, cmap='jet', vmin=0, vmax=vmax)
                ax[0, 0].set_title(f"ORIGINAL Raw\nTh={theta}, Phi={phi}")
                ax[0, 0].plot(IMAGE_CENTER_COORDS[0], IMAGE_CENTER_COORDS[1], 'rx')
                
                ax[0, 1].plot(radii_u[mask_bi], profile_u[mask_bi], 'b.', label='Data', alpha=0.5)
                ax[0, 1].plot(radii_u[mask_bi], model_bi(radii_u[mask_bi], *popt_bi), 'r-', label='Fit', linewidth=2)
                ax[0, 1].set_title(f"ORIGINAL Profile (Area={area_bi:.0f})")
                ax[0, 1].legend()
                ax[0, 1].grid(True)

                # 下段：除去後 (Cleaned)
                ax[1, 0].imshow(img_clean, cmap='jet', vmin=0, vmax=vmax)
                ax[1, 0].set_title(f"CLEANED Raw (Sigma={SIGMA_CLIP_THRESHOLD})")
                ax[1, 0].plot(IMAGE_CENTER_COORDS[0], IMAGE_CENTER_COORDS[1], 'rx')

                ax[1, 1].plot(radii_clean[mask_clean], profile_clean[mask_clean], 'g.', label='Cleaned Data', alpha=0.5)
                if area_clean > 0:
                    ax[1, 1].plot(radii_clean[mask_clean], model_bi(radii_clean[mask_clean], *popt_clean), 'k--', label='Fit', linewidth=2)
                ax[1, 1].set_title(f"CLEANED Profile (Area={area_clean:.0f})")
                ax[1, 1].legend()
                ax[1, 1].grid(True)

                plt.tight_layout()
                plt.show() # 次に進むにはウィンドウを閉じてください
                plt.pause(0.1)

    # =============================================================================
    # 4. ログ保存
    # =============================================================================
    print("\nScan completed.")
    if bragg_log:
        with open(OUTPUT_LOG_BRAGG, 'w') as f:
            f.write("\n".join(bragg_log))
        print(f"Log saved to: {OUTPUT_LOG_BRAGG}")
    else:
        print("No Bragg reflections detected.")

if __name__ == "__main__":
    main()