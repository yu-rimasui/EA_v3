'''
csvファイルから見る、ブラッグ反射の分布形状（ヒストグラム、箱ひげ図）
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. データ読み込み
# ==========================================
VAL = 'Bi_17000eV'
file_path = f'img/{VAL}_intensity_map.csv'

df = pd.read_csv(file_path, header=None)
data = df.values.flatten()
data = data[data > 1] 

# ==========================================
# 2. 統計量の計算
# ==========================================
mean_val = np.mean(data)
median_val = np.median(data)
std_val = np.std(data)

# 四分位範囲
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1

# 一般的な「外れ値」の基準
mi_outlier = q3 + 1.5 * iqr # 1.5 * IOR
ex_outlier = q3 + 3.0 * iqr # 3.0 * IQR

print("-" * 40)
print(f"データ総数: {len(data)} pixels")
print(f"平均値 (Mean): {mean_val:.2f}")
print(f"中央値 (Median): {median_val:.2f}")
print(f"標準偏差 (Std): {std_val:.2f}")
print(f"四分位範囲IQR: {iqr:.2f}")
print("-" * 40)
print(f"1.5IQR (Mild Outlier): {mi_outlier:.2f}")
print(f"3.0IQR (Extreme Outlier): {ex_outlier:.2f}")
# print(f"平均 + 3σ法: {mean_val + 3*std_val:.2f}")
print("-" * 40)

# ==========================================
# 3. 可視化 (ヒストグラム & 箱ひげ図)
# ==========================================
fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

# --- ヒストグラム ---
# 外れ値を見やすくするため、Y軸を対数スケール(log scale)にするのがコツです
sns.histplot(data, bins=100, kde=False, ax=ax[0], color='skyblue', edgecolor='black')
ax[0].set_yscale('log') # 対数表示
ax[0].set_title(f'{VAL} \n Intensity Distribution (Log Scale)')
ax[0].set_ylabel('Count (Log)')

# 提案ラインを描画
ax[0].axvline(mi_outlier, color='orange', linestyle='--', label=f'1.5IQR (Mild Outlier)   : {mi_outlier:.0f}')
ax[0].axvline(ex_outlier, color='red', linestyle='--', label=f'3.0IQR (Extreme Outlier): {ex_outlier:.0f}')
ax[0].legend()

# --- 箱ひげ図 (外れ値を可視化する) ---
sns.boxplot(x=data, ax=ax[1], color='lightgreen')
ax[1].set_title('Box Plot')
ax[1].set_xlabel('Intensity Area')

# グラフ上の文字が見切れないように調整
plt.tight_layout()
plt.show()