'''
RAW形式の画像データ読込み → 表示
'''

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. パラメータ設定
# =============================================================================
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 128
DATA_TYPE = np.int16
OFFSET_BYTES = 1024
FILE_PATH = 'img/U_phi_000000.img' # 解析対象ファイルパス


# =============================================================================
# RAW DATA 読み込みと表示
# =============================================================================
try:
    with open(FILE_PATH, 'rb') as f:
        f.seek(OFFSET_BYTES)

        remaining_data = f.read()
        raw_data = np.frombuffer(remaining_data, dtype=DATA_TYPE)

    expected_pixels = IMAGE_WIDTH * IMAGE_HEIGHT
    if raw_data.size == expected_pixels:
        image_data = raw_data.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))

        plt.imshow(image_data, cmap='afmhot')
        plt.title(f"Loaded Raw Image ({IMAGE_WIDTH}x{IMAGE_HEIGHT}, int16)")
        plt.axis('off')
        plt.colorbar(orientation='horizontal')
        plt.show()
    else:
        print(f"エラー: 想定されるピクセル数 ({expected_pixels}) と")
        print(f"読み込まれたデータ数 ({raw_data.size}) が一致しません。")
        print("オフセット値、幅、高さ、またはデータ型が間違っている可能性があります。")

except FileNotFoundError:
    print(f"ファイルが見つかりません: {FILE_PATH}")
except Exception as e:
    print(f"データの読み込みまたは表示中にエラーが発生しました: {e}")

