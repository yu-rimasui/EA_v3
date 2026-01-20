'''
RAW形式の画像データ読込み (LとU) → 差分計算 (L-U) → 表示
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

number = '000000'  # 解析対象ファイル番号
FILE_PATH_L = f'img/L_phi_{number}.img'
FILE_PATH_U = f'img/U_phi_{number}.img'


# =============================================================================
# 関数定義: RAWデータの読み込み
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

        expected_pixels = IMAGE_WIDTH * IMAGE_HEIGHT
        if raw_data.size == expected_pixels:
            return raw_data.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
        else:
            print(f"エラー: {file_path}")
            print(f"想定ピクセル数 ({expected_pixels}) と データ数 ({raw_data.size}) が不一致です。")
            return None

    except Exception as e:
        print(f"読み込み中にエラーが発生しました ({file_path}): {e}")
        return None


# =============================================================================
# メイン処理: 読み込みと演算
# =============================================================================
image_l = load_raw_image(FILE_PATH_L)
image_u = load_raw_image(FILE_PATH_U)


if image_l is not None and image_u is not None:
    
    # diff_image = image_l.astype(np.int32) - image_u.astype(np.int32)
    diff_image = image_l - image_u

    # 差分画像の表示
    plt.imshow(diff_image, cmap='afmhot') 
    plt.title(f"Difference Image (L - U)\nL-U_phi_{number}.img")
    plt.axis('off')
    plt.colorbar(orientation='horizontal', label='Intensity Difference')
    plt.tight_layout()
    plt.show()

else:
    print("画像の読み込みに失敗したため、処理を中断します。")