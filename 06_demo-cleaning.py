'''
ブラッグ反射除去アルゴリズム（極座標変換→統計的除去→逆変換）のデモと検証。
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os

# ==========================================
# 設定
# ==========================================
ENERGY = 17000
THETA = 73
PHI = 45
FILENAME = f"/Volumes/Extreme SSD/Sm-BiFeO3_RT/{ENERGY}/Th{THETA:02d}/U_phi_{PHI:06d}.img"

HEADER_SIZE = 1024
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 128
CENTER = (250, 65)

OUT_W = 360  # 角度分解能
OUT_H = 120  # 半径分解能
MAX_RADIUS = 120 

# ★除去の強度設定 (平均から標準偏差の何倍離れたら消すか)
# 値が小さいほど厳しく除去します (通常 3.0 〜 5.0 くらい)
SIGMA_THRESHOLD = 1.0 

# ==========================================
# 1. 画像読み込み
# ==========================================
if not os.path.exists(FILENAME):
    print(f"Error: File {FILENAME} not found.")
    # テスト用ダミーデータ（実際のファイルがない場合用）
    # raw_data = np.zeros(IMAGE_WIDTH*IMAGE_HEIGHT, dtype=np.uint16)
    exit()
else:
    with open(FILENAME, 'rb') as f:
        f.read(HEADER_SIZE)
        raw_data = np.frombuffer(f.read(), dtype=np.uint16)

img = raw_data.reshape((IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float32)

# ==========================================
# 2. 極座標変換
# ==========================================
theta_vals = np.linspace(0, 2*np.pi, OUT_W)
r_vals = np.linspace(0, MAX_RADIUS, OUT_H)
Theta, R = np.meshgrid(theta_vals, r_vals)

map_x = (CENTER[0] + R * np.cos(Theta)).astype(np.float32)
map_y = (CENTER[1] + R * np.sin(Theta)).astype(np.float32)

polar_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

# ==========================================
# 2.5 ブラッグ反射除去処理 (Spike Removal)
# ==========================================
# 元のデータをコピーして作業します
clean_img = polar_img.copy()

# 各半径(行)ごとに統計処理を行う
for r_idx in range(OUT_H):
    row_data = clean_img[r_idx, :]
    
    # その半径における「中央値」と「標準偏差」を計算
    median_val = np.median(row_data)
    std_val = np.std(row_data)
    
    # 閾値を設定 (中央値 + N * 標準偏差)
    # ※ノイズが非常に大きい場合は std の代わりに MAD (Median Absolute Deviation) を使うとさらに堅牢です
    cutoff = median_val + SIGMA_THRESHOLD * std_val
    
    # 閾値を超えている場所（ブラッグ反射）を探す
    spikes = row_data > cutoff
    
    # スパイク部分をその行の中央値で埋める（あるいは隣接値で埋める）
    clean_img[r_idx, spikes] = median_val

# ==========================================
# 3. 3Dグラフ描画
# ==========================================
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

stride_r = 2
stride_th = 5

X_plot = np.degrees(Theta)[::stride_r, ::stride_th]
Y_plot = R[::stride_r, ::stride_th]
# ★ここで clean_img を使う
Z_plot = clean_img[::stride_r, ::stride_th]

surf = ax.plot_surface(X_plot, Y_plot, Z_plot, cmap='jet', 
                        linewidth=0, antialiased=False)

ax.set_xlabel('Angle (deg)')
ax.set_ylabel('Radius (pixel)')
ax.set_zlabel('Intensity')
ax.set_title(f'Cleaned 3D View: {ENERGY}eV, (θ,φ)=({THETA},{PHI})')

fig.colorbar(surf, shrink=0.5, aspect=5)

print("ブラッグ反射を除去したグラフを表示します。")
plt.show()

# （参考）除去前後の比較画像を2Dで保存
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.imshow(polar_img, aspect='auto', cmap='jet')
plt.title("Original (with Bragg Peaks)")
plt.colorbar()
plt.subplot(2, 1, 2)
plt.imshow(clean_img, aspect='auto', cmap='jet')
plt.title(f"Cleaned (Sigma={SIGMA_THRESHOLD})")
plt.colorbar()
plt.tight_layout()
plt.show()
# plt.savefig("compare_removal.png")