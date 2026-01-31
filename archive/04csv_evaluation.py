import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ==========================================
# 1. データ読み込み
# ==========================================
# ファイル名は適宜変更してください
file_peakfit = "img/Bi_13500eV_intensity_map.csv"  # 今回のピークフィット結果
file_legacy  = "img/Bi_13500eV_normal.csv"         # 従来の規格化データ

df_fit = pd.read_csv(file_peakfit, header=None)
df_ref = pd.read_csv(file_legacy, header=None)

# 行列を1次元配列化（統計計算用）
y_fit = df_fit.values.flatten()
y_ref = df_ref.values.flatten()

# ==========================================
# 2. スケール合わせ (Scaling)
# ==========================================
# 比較のため、従来データ(x)を今回のデータ(y)のスケールに合わせる
# y = slope * x + intercept
slope, intercept, r_value, p_value, std_err = linregress(y_ref, y_fit)

# スケール補正後の従来データ（画像形式に戻す）
img_fit = df_fit.values
img_ref_scaled = df_ref.values * slope + intercept

# ==========================================
# 3. 定量評価指標 (Roughness) の計算
# ==========================================
def calculate_roughness_rms(img):
    """
    隣接画素間の差分（勾配）の二乗平均平方根(RMS)を計算する。
    値が小さいほど「滑らか（ノイズが少ない）」ことを示す。
    """
    # 横方向の差分
    diff_x = np.diff(img, axis=1)
    # 縦方向の差分
    diff_y = np.diff(img, axis=0)
    
    # 全要素の二乗平均のルートをとる
    mean_sq_diff = (np.mean(diff_x**2) + np.mean(diff_y**2)) / 2
    return np.sqrt(mean_sq_diff)

score_fit = calculate_roughness_rms(img_fit)
score_ref = calculate_roughness_rms(img_ref_scaled)

print(f"--- 定量評価結果 ---")
print(f"相関係数 (R^2): {r_value**2:.4f}")
print(f"Roughness (Peak Fit法): {score_fit:.4f}")
print(f"Roughness (従来法・補正後): {score_ref:.4f}")
print(f"ノイズ低減率 (比率): {score_fit / score_ref:.2f} (約 {1/(score_fit/score_ref):.1f}倍滑らか)")

# ==========================================
# 4. グラフ描画（レポート用）
# ==========================================
# 比較しやすい代表的な行（ラインプロファイル）を選択
# 全体の変動が見やすい行を適当に選定（例: 38行目）
row_idx = 38 

line_fit = df_fit.iloc[row_idx, :]
line_ref = img_ref_scaled[row_idx, :]

plt.figure(figsize=(10, 6))

# プロット
plt.plot(line_ref, color='orange', alpha=0.6, label='Conventional Method (Scaled)', linewidth=1.5)
plt.plot(line_fit, color='#1f77b4', alpha=1.0, label='Proposed Method (Peak Fit)', linewidth=2)

# グラフ装飾
plt.title(f"Comparison of Intensity Profiles (Row {row_idx})", fontsize=14)
plt.xlabel("Pixel Index (Phi)", fontsize=12)
plt.ylabel("Intensity (a.u.)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 保存
plt.savefig("Analysis_Evaluation_Graph.png", dpi=300)
plt.show()