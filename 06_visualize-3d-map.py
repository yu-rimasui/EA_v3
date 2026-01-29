'''
RAW画像を極座標変換し、3Dグラフでブラッグ反射の分布を可視化する（現状把握用）。
    X軸 (Circumference): 角度（0° ～ 360°）円周上の位置に対応します
    Y軸 (Radius)       : 中心からの距離（半径）
    Z軸 (Intensity)    : 信号強度
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

# 出力画像の解像度設定
OUT_W = 360  # 角度方向の分割数
OUT_H = 120  # 半径方向の分割数
MAX_RADIUS = 120 # 切り出す最大半径

# ==========================================
# 1. 画像読み込み
# ==========================================
if not os.path.exists(FILENAME):
    print(f"Error: File {FILENAME} not found.")
    exit()

with open(FILENAME, 'rb') as f:
    f.read(HEADER_SIZE)
    raw_data = np.frombuffer(f.read(), dtype=np.uint16)

img = raw_data.reshape((IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float32)

# ==========================================
# 2. 極座標変換 (Polar Transform)
#    (x, y) -> (radius, theta) への展開
# ==========================================
# グリッド作成: Y軸=Radius, X軸=Theta
theta_vals = np.linspace(0, 2*np.pi, OUT_W)
r_vals = np.linspace(0, MAX_RADIUS, OUT_H)
Theta, R = np.meshgrid(theta_vals, r_vals)

# 逆写像用の座標計算 (Dest -> Src)
# src_x = center_x + r * cos(theta)
# src_y = center_y + r * sin(theta)
map_x = (CENTER[0] + R * np.cos(Theta)).astype(np.float32)
map_y = (CENTER[1] + R * np.sin(Theta)).astype(np.float32)

# リマッピング実行 (補間あり)
polar_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

# ==========================================
# 3. 3Dグラフ描画
# ==========================================
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 描画負荷軽減のためにデータを間引く (stride)
stride_r = 2   # 半径方向の間引き
stride_th = 5  # 角度方向の間引き

X_plot = np.degrees(Theta)[::stride_r, ::stride_th] # 角度(deg)
Y_plot = R[::stride_r, ::stride_th]                 # 半径
Z_plot = polar_img[::stride_r, ::stride_th]         # 強度

surf = ax.plot_surface(X_plot, Y_plot, Z_plot, cmap='jet', 
                        linewidth=0, antialiased=False)

ax.set_xlabel('Angle (deg)')
ax.set_ylabel('Radius (pixel)')
ax.set_zlabel('Intensity')
ax.set_title(f'Unwrapped 3D View: {ENERGY}eV, θ={THETA}, φ={PHI}')

fig.colorbar(surf, shrink=0.5, aspect=5)

print("グラフを表示します。")
plt.show()