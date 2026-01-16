'''
機能
1. 画像の積算: 指定フォルダにある1520枚の画像をすべて読み込み、足し合わせる
2. 高精度プロファイルの作成: 積算画像から、ノイズの極めて少ないラジアルプロファイルを作成
3. マスターパラメータの決定: これをフィッティングし、固定すべき「中心 (r_c)」と「幅 (w)」を決定
※ L-U解析のためのSmFeO3用バージョン
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.optimize import curve_fit

# =============================================================================
# 1. パラメータ設定
# =============================================================================
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 128
DATA_TYPE = np.int16
OFFSET_BYTES = 1024

# 画像があるフォルダとファイル名のパターン
# 例: imgフォルダの中の U_phi_*.img というファイルを全て対象にする
FILE_PATTERN_L = '/Volumes/Extreme SSD/Sm-BiFeO3_RT/13500/Th00/L_phi_*.img'
FILE_PATTERN_U = '/Volumes/Extreme SSD/Sm-BiFeO3_RT/13500/Th00/U_phi_*.img'

IMAGE_CENTER_COORDS = (250, 65)

FIT_R_MIN = 0
FIT_R_MAX = 200

# =============================================================================
# 2. 関数定義
# =============================================================================
def load_raw_image(file_path):
    """画像を読み込む関数"""
    try:
        with open(file_path, 'rb') as f:
            f.seek(OFFSET_BYTES)
            raw_data = np.frombuffer(f.read(), dtype=DATA_TYPE)
        if raw_data.size == IMAGE_WIDTH * IMAGE_HEIGHT:
            return raw_data.reshape((IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float32)
        return None
    except Exception:
        return None

def calculate_sum_profile(image, center):
    """面積で割らない（合計値）プロファイルを計算する関数"""
    cx, cy = center
    y, x = np.indices(image.shape)
    d = np.sqrt((x - cx)**2 + (y - cy)**2)
    r = np.round(d).astype(int)
    max_r = np.max(r)
    
    # 合計値をそのまま使う
    sum_intensity = np.bincount(r.ravel(), weights=image.ravel(), minlength=max_r+1)
    radii = np.arange(max_r + 1)
    return radii, sum_intensity

def double_gaussian_model(x, a1, c1, w1, a2, c2, w2, offset):
    """
    2つのガウス関数の和
    1: Sm由来 (r~25)
    2: Fe由来 (r~125)
    """
    g1 = a1 * np.exp(-(x - c1)**2 / (2 * w1**2))
    g2 = a2 * np.exp(-(x - c2)**2 / (2 * w2**2))
    return g1 + g2 + offset

# =============================================================================
# 3. メイン処理：全画像積算 & マスターフィッティング
# =============================================================================
def main():
    file_list_l = sorted(glob.glob(FILE_PATTERN_L))
    file_list_u = sorted(glob.glob(FILE_PATTERN_U))
    total_files = len(file_list_l)
    
    if total_files == 0:
        print("画像ファイルが見つかりません。パスを確認してください。")
        return

    print(f"{total_files} 枚の画像を積算します...")

    # 積算用バッファ
    summed_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
    count = 0

    # 1. RAW画像の積算
    for i in range(total_files):
        img_l = load_raw_image(file_list_l[i])
        img_u = load_raw_image(file_list_u[i])

        if img_l is not None and img_u is not None:
            diff_img = img_l - img_u
            summed_image += diff_img
            count += 1
        
        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{total_files}")

    print(f"積算完了: {count}枚の画像を合成しました。")

    # 積算した後の画像を表示する
    # plt.imshow(summed_image, cmap='afmhot')
    # plt.title("Summed Image")
    # plt.axis('off')
    # plt.colorbar(orientation='horizontal')
    # plt.show()


    # 2. 積算画像のプロファイルを作成
    radii, profile = calculate_sum_profile(summed_image, IMAGE_CENTER_COORDS)

    # 3. フィッティング（マスターパラメータの決定）
    mask = (radii >= FIT_R_MIN) & (radii <= FIT_R_MAX)
    x_fit = radii[mask]
    y_fit = profile[mask]

    max_val = np.max(y_fit) # ampの初期値設定用
    p0 = [
        max_val,       25.0,  10.0,  # Peak 1 (Sm)
        max_val * 0.5, 125.0, 15.0,  # Peak 2 (Fe) - 振幅は適当に半分くらいと仮定
        0.0                          # Offset
    ]

    try:
        popt, pcov = curve_fit(double_gaussian_model, x_fit, y_fit, p0=p0)
        
        # 決定係数 R^2
        residuals = y_fit - double_gaussian_model(x_fit, *popt)
        r_squared = 1 - (np.sum(residuals**2) / np.sum((y_fit - np.mean(y_fit))**2))

        print("\n" + "="*50)
        print("【決定された固定パラメータ (Master Parameters - Double Peak)】")
        print("\n")
        
        print("  [Sm Peak]")
        print(f"    Center (中心位置) : {popt[1]:.4f} px")
        print(f"    Width  (ピーク幅) : {popt[2]:.4f} px")
        print(f"    Amp    (振幅)     : {popt[0]:.2f}")
        print("\n")
        
        print("  [Fe Peak]")
        print(f"    Center (中心位置) : {popt[4]:.4f} px")
        print(f"    Width  (ピーク幅) : {popt[5]:.4f} px")
        print(f"    Amp    (振幅)     : {popt[3]:.2f}")
        print("\n")
        
        print(f"  Offset (オフセット) : {popt[6]:.2f}")
        print(f"  R^2    (決定係数)   : {r_squared:.4f}")
        print("="*50)


# ------ グラフ描写 ------
        plt.figure(figsize=(10, 6))
        
        # 実データ
        plt.plot(radii, profile, 'k-', alpha=0.5, label='Summed Data (1440 images)')

        x_plot = np.linspace(FIT_R_MIN, FIT_R_MAX, 500)
        y_total = double_gaussian_model(x_plot, *popt)
        # plt.plot(x_plot, y_total, 'g-', linewidth=2.0, label='Total Fit', alpha=0.8)

        # popt: [SmAmp, SmCen, SmWid, FeAmp, FeCen, FeWid, Offset]
        y_sm = popt[0] * np.exp(-(x_plot - popt[1])**2 / (2 * popt[2]**2)) + popt[6]
        plt.plot(x_plot, y_sm, 'r--', linewidth=1.5, label=f'Sm Component (r={popt[1]:.2f})')
        y_fe = popt[3] * np.exp(-(x_plot - popt[4])**2 / (2 * popt[5]**2)) + popt[6]
        plt.plot(x_plot, y_fe, 'b--', linewidth=1.5, label=f'Fe Component (r={popt[4]:.2f})')
        plt.axhline(y=popt[6], color='gray', linestyle=':', label='Baseline')

        plt.title(f"Double Peak Decomposition\nSm(r≈{popt[1]:.2f}) + Fe(r≈{popt[4]:.2f})")
        plt.xlabel("Radius [px]")
        plt.ylabel("Total Intensity")
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"フィッティングエラー: {e}")

if __name__ == "__main__":
    main()