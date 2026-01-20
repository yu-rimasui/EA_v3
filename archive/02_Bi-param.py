'''
機能
1. 画像の積算: 指定フォルダにある1440枚の画像をすべて読み込み、足し合わせる
2. 高精度プロファイルの作成: 積算画像から、ノイズの極めて少ないラジアルプロファイルを作成
3. マスターパラメータの決定: これをフィッティングし、固定すべき「中心 (r_c)」と「幅 (w)」を決定
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
FILE_PATTERN = '/Volumes/Extreme SSD/Sm-BiFeO3_RT/13500/Th00/U_phi_*.img' 

IMAGE_CENTER_COORDS = (250, 65)

FIT_R_MIN = 0
FIT_R_MAX = 150

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

def gaussian_model(x, amp, cen, wid, offset):
    """ガウス関数モデル"""
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2)) + offset

# =============================================================================
# 3. メイン処理：全画像積算 & マスターフィッティング
# =============================================================================
def main():
    file_list = sorted(glob.glob(FILE_PATTERN))
    total_files = len(file_list)
    
    if total_files == 0:
        print("画像ファイルが見つかりません。パスを確認してください。")
        return

    print(f"{total_files} 枚の画像を積算します...")

    # 積算用バッファ
    summed_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
    count = 0

    # 1. RAW画像の積算
    for i, file_path in enumerate(file_list):
        img = load_raw_image(file_path)
        if img is not None:
            summed_image += img
            count += 1
        
        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{total_files}")

    print(f"積算完了: {count}枚の画像を合成しました。")

    # # 積算した後の画像を表示する
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
    p0 = [max_val, 30.0, 10.0, 0.0] # [Amp, Center, Width, Offset]の初期値

    try:
        popt, pcov = curve_fit(gaussian_model, x_fit, y_fit, p0=p0)
        
        # 決定係数 R^2
        residuals = y_fit - gaussian_model(x_fit, *popt)
        r_squared = 1 - (np.sum(residuals**2) / np.sum((y_fit - np.mean(y_fit))**2))

        print("\n" + "="*50)
        print("【決定された固定パラメータ (Master Parameters)】")
        print("\n")
        print(f"  Center (中心位置) : {popt[1]:.4f} px")
        print(f"  Width  (ピーク幅) : {popt[2]:.4f} px")
        print("\n")
        print(f"  (参考) Amp : {popt[0]:.2f}")
        print(f"  (参考) R^2 : {r_squared:.4f}")
        print("="*50)


        plt.figure(figsize=(10, 6))
        plt.plot(radii, profile, 'k-', alpha=0.5, label='Summed Data (1440 images)')
        plt.plot(x_fit, gaussian_model(x_fit, *popt), 'r--', linewidth=2, label='Master Fit')
    
        plt.title(f"Master Fitting Result (Summed Image)\nCenter={popt[1]:.2f}, Width={popt[2]:.2f}")
        plt.xlabel("Radius [px]")
        plt.ylabel("Total Intensity")
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"フィッティングエラー: {e}")

if __name__ == "__main__":
    main()