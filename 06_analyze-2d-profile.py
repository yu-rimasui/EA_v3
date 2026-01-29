'''
特定の半径 r における角度方向の強度プロファイルを確認する（パラメータ検討用）。
    X軸 (Circumference): 角度（0° ～ 360°）円周上の位置に対応します
    Y軸 (Radius)       : 中心からの距離（半径）【固定】
    Z軸 (Intensity)    : 信号強度
'''
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# ==========================================
# 設定
# ==========================================
ENERGY = 17000
THETA = 70
PHI = 332
FILENAME = f"/Volumes/Extreme SSD/Sm-BiFeO3_RT/{ENERGY}/Th{THETA:02d}/U_phi_{PHI:06d}.img"

HEADER_SIZE = 1024
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 128
CENTER = (250, 65)

# ★解析したい半径（中心からのピクセル距離）を指定
TARGET_RADIUS = 20  # 半径r
INTEGRATION_WIDTH = 2 # 前後何ピクセルを平均するか（ノイズ低減用）

# 出力画像の解像度設定
OUT_W = 360  # 角度方向の分割数
OUT_H = 120  # 半径方向の分割数
MAX_RADIUS = 120

# ==========================================
# 1. 画像読み込み
# ==========================================
if not os.path.exists(FILENAME):
    print(f"Error: File {FILENAME} not found.")
    # テスト用にダミーデータを作成する場合のみコメントアウトを外す
    # raw_data = np.random.randint(0, 1000, IMAGE_WIDTH*IMAGE_HEIGHT).astype(np.uint16)
    exit()
else:
    with open(FILENAME, 'rb') as f:
        f.read(HEADER_SIZE)
        raw_data = np.frombuffer(f.read(), dtype=np.uint16)

# 形状変換
img = raw_data.reshape((IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float32)

# ==========================================
# 2. 極座標変換 (Polar Transform)
# ==========================================
theta_vals = np.linspace(0, 2*np.pi, OUT_W)
r_vals = np.linspace(0, MAX_RADIUS, OUT_H)
Theta, R = np.meshgrid(theta_vals, r_vals)

map_x = (CENTER[0] + R * np.cos(Theta)).astype(np.float32)
map_y = (CENTER[1] + R * np.sin(Theta)).astype(np.float32)

polar_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

# ==========================================
# 3. 指定半径データの抽出 (2D化)
# ==========================================
# 物理的な半径(TARGET_RADIUS)を、画像の行インデックスに変換
r_index_float = TARGET_RADIUS * (OUT_H / MAX_RADIUS)
r_idx = int(np.clip(r_index_float, 0, OUT_H - 1))

# 指定した半径周辺のデータを取得して平均化（S/N比向上のため）
start_idx = max(0, r_idx - INTEGRATION_WIDTH)
end_idx = min(OUT_H, r_idx + INTEGRATION_WIDTH + 1)

# 縦方向（半径方向）に平均をとって1次元配列にする
intensity_profile = np.mean(polar_img[start_idx:end_idx, :], axis=0)

# 横軸（角度 0~360度）の作成
angles_deg = np.linspace(0, 360, OUT_W)

# ==========================================
# 4. 2次元グラフ描画
# ==========================================
plt.figure(figsize=(10, 6))

# メインのプロット
# plt.plot(angles_deg, intensity_profile, linewidth=1.5, label=f'R={TARGET_RADIUS} px')
plt.scatter(angles_deg, intensity_profile, s=10, label=f'R={TARGET_RADIUS} px')

# グラフ装飾
plt.title(f"Intensity Profile at Radius r={TARGET_RADIUS}px\n({ENERGY}eV, θ={THETA}°, φ={PHI}°)")
plt.xlabel("Angle (deg)")
plt.ylabel("Intensity")
plt.xlim(0, 360)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 参考：どの場所を切ったか確認するためのヒートマップを表示（ウィンドウの下側に配置）
plt.figure(figsize=(10, 4))
plt.imshow(polar_img, aspect='auto', origin='lower', 
            extent=[0, 360, 0, MAX_RADIUS], cmap='afmhot')
plt.axhline(y=TARGET_RADIUS, color='white', linestyle='--', linewidth=1, label='Selected Radius')
plt.title("Heatmap")
plt.xlabel("Angle (deg)")
plt.ylabel("Radius (px)")
plt.colorbar(label='Intensity')
plt.legend()

print(f"半径 {TARGET_RADIUS} px (幅 ±{INTEGRATION_WIDTH}px) のプロファイルを表示します。")
plt.show()