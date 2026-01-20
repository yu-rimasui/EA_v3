'''
RAW形式の画像データ読込み → 周回積分プロファイル表示
'''

import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. パラメータ設定
# =============================================================================
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 128
DATA_TYPE = np.int16
OFFSET_BYTES = 1024
FILE_PATH = 'img/U_phi_000000.img' # 解析対象ファイルパス

FIXED_CENTER = (250, 65)

# =============================================================================
# 2. 画像読み込み関数
# =============================================================================
def load_raw_image(file_path):
    """
    RAW画像を読み込み
    """
    if not os.path.exists(file_path):
        print(f"ファイルが見つかりません: {file_path}")
        return None

    try:
        with open(file_path, 'rb') as f:
            f.seek(OFFSET_BYTES)
            remaining_data = f.read()
            raw_data = np.frombuffer(remaining_data, dtype=DATA_TYPE)
        
        expected_pixels = IMAGE_WIDTH * IMAGE_HEIGHT
        if raw_data.size == expected_pixels:
            image_data = raw_data.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
            return image_data.astype(np.float32)
        else:
            print(f"エラー: 想定されるピクセル数 ({expected_pixels}) と")
            print(f"読み込まれたデータ数 ({raw_data.size}) が一致しません。")
            return None

    except Exception as e:
        print(f"データの読み込み中にエラーが発生しました: {e}")
        return None

# =============================================================================
# 3. 周回積分（プロファイル計算）関数
# =============================================================================
def calculate_I(image, center):
    """
    指定された中心座標から周回積分を行い、動径プロファイル(r vs I)を計算する
    """
    cx, cy = center
    y, x = np.indices(image.shape)
    
    d = np.sqrt((x - cx)**2 + (y - cy)**2)
    r = np.round(d).astype(int)
    max_r = np.max(r)

    # bincount関数で各半径ごとの強度合計とピクセル数を計算
    sum_intensity = np.bincount(r.ravel(), weights=image.ravel(), minlength=max_r+1)
    pixel_cnts = np.bincount(r.ravel(), minlength=max_r+1)
    
    # radial_profile = sum_intensity / np.maximum(pixel_cnts, 1)
    radial_profile = sum_intensity

    radii = np.arange(max_r + 1)
    return radii, radial_profile

# =============================================================================
# 4. プロファイル表示・実行関数
# =============================================================================
def show_radial_profile(file_path, center=FIXED_CENTER):
    """
    画像を読み込み、半径r vs 強度I のグラフを表示する
    """
    image = load_raw_image(file_path)
    if image is None:
        return

    cx, cy = center
    print(f"Target File: {file_path}")
    print(f"Center: ({cx}, {cy})")


    r, profile = calculate_I(image, (cx, cy))

    # グラフ描画
    plt.figure(figsize=(6, 4))
    plt.plot(r, profile, 'blue', linewidth=1.5, label='Radial Profile')
    
    plt.title(f"Radial Profile, Center: {center}")
    plt.xlabel("Radius r [px]")
    plt.ylabel("Intensity I [arb. unit]")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# =============================================================================
# メイン実行ブロック
# =============================================================================
if __name__ == "__main__":
    show_radial_profile(FILE_PATH)