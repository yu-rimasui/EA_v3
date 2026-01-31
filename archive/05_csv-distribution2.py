'''
csvファイルの分布形状を3次元で可視化するファイル
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 設定
VAL = "Bi_17000"
# file_path = f"img/{VAL}eV_intensity_map.csv"
file_path = f"img/algorithm2_{VAL}eV_intensity_map.csv"

# データの読み込み
df = pd.read_csv(file_path, header=None)
data = df.values
rows, cols = data.shape

# メッシュグリッドの作成
x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)
Z = data

# グラフの作成
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# サーフェスのプロット
# rstride, cstrideは間引き設定です。動作が重い場合は値を大きくしてください(例: 5, 20など)
surf = ax.plot_surface(X, Y, Z, cmap='jet', 
                        rstride=1, cstride=10, 
                        linewidth=0, antialiased=False)


ax.set_xlabel('φ (Column)')
ax.set_ylabel('θ (Row)')
ax.set_zlabel('Intensity')
ax.set_title(f'3D Intensity Map: {VAL}')

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
# plt.savefig(f"3d_{VAL}eV_intensity_map.png") # 静止画の保存
