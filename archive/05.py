import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os

# =============================================================================
# 1. パラメータ設定
# =============================================================================
# 入力ファイル名 (環境に合わせて変更してください)
VAL = "Bi_17000"
INPUT_CSV_PATH = f"img/{VAL}eV_intensity_map.csv"

# 統計的閾値の係数 (レポート結論の 3.0 IQR法を採用)
IQR_MULTIPLIER = 3.0 

# 出力ファイル名
OUTPUT_COORD_FILE = f"Bragg_Coordinates_{VAL}.csv"

# =============================================================================
# 2. 関数定義
# =============================================================================
def load_data(file_path):
    """CSVファイルを読み込む"""
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        # ダミーデータ作成（動作確認用）
        print("Creating dummy data for testing...")
        data = np.random.normal(10000, 1000, (76, 1440))
        # 疑似的なブラッグ反射を追加
        data[30, 500] = 50000 
        data[50, 1000] = 45000
        return pd.DataFrame(data)
    
    print(f"Loading data from: {file_path}")
    return pd.read_csv(file_path, header=None)

def calculate_threshold_iqr(data_values, multiplier=3.0):
    """四分位範囲(IQR)を用いて閾値を計算する"""
    # 0や極端に低い値を除外（バックグラウンドのみを統計対象にするため）
    # ※ノイズレベルより明らかに大きい値のみを対象にするフィルタ
    valid_data = data_values[data_values > 1] 

    q1 = np.percentile(valid_data, 25)
    q3 = np.percentile(valid_data, 75)
    iqr = q3 - q1
    
    threshold = q3 + multiplier * iqr
    
    stats = {
        "mean": np.mean(valid_data),
        "median": np.median(valid_data),
        "std": np.std(valid_data),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "threshold": threshold
    }
    return threshold, stats

def plot_distribution(data_values, threshold, stats):
    """ヒストグラムと箱ひげ図を表示"""
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # ヒストグラム
    sns.histplot(data_values, bins=100, kde=False, color='skyblue', ax=ax[0], label='Data Distribution')
    ax[0].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({IQR_MULTIPLIER} IQR)')
    ax[0].set_title(f"Intensity Distribution & Threshold: {threshold:.2f}")
    ax[0].set_yscale('log') # ログスケールで見やすく
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # 箱ひげ図
    sns.boxplot(x=data_values, ax=ax[1], color='lightgreen')
    ax[1].axvline(threshold, color='red', linestyle='--', linewidth=2)
    ax[1].set_xlabel("Intensity")
    
    plt.tight_layout()
    print("分布グラフを表示します。閉じて次に進みます。")
    plt.show()

def plot_3d_with_threshold(df, threshold):
    """3Dマップに閾値平面を加えて表示"""
    data = df.values
    rows, cols = data.shape
    
    # メッシュ作成
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    Z = data
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. データプロット
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                            rstride=1, cstride=10, # 間引き
                            linewidth=0, antialiased=False, alpha=0.8)
    
    # 2. 閾値平面のプロット (Z = Threshold)
    Z_thresh = np.full_like(Z, threshold)
    ax.plot_surface(X, Y, Z_thresh, color='red', alpha=0.3, 
                    rstride=5, cstride=50, shade=False)
    
    # 装飾
    ax.set_xlabel('φ (Column)')
    ax.set_ylabel('θ (Row)')
    ax.set_zlabel('Intensity')
    ax.set_title(f'3D Map with Threshold Plane (Th={threshold:.0f})')
    
    # 代理アーティストを使って凡例作成
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', alpha=0.3, label='Bragg Threshold')
    plt.legend(handles=[red_patch])
    
    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)
    
    print("3Dグラフを表示します。赤色の面より飛び出ている部分がブラッグ反射です。")
    plt.show()

def extract_bragg_coordinates(df, threshold):
    """閾値を超えた座標を抽出する"""
    data = df.values
    # 条件を満たすインデックスを取得
    rows, cols = np.where(data > threshold)
    intensities = data[rows, cols]
    
    # 結果をDataFrameにまとめる
    results = pd.DataFrame({
        "Theta_Row": rows,
        "Phi_Col": cols,
        "Intensity": intensities
    })
    
    # 強度順にソート
    results = results.sort_values(by="Intensity", ascending=False).reset_index(drop=True)
    return results

# =============================================================================
# 3. メイン処理
# =============================================================================
def main():
    print("=== Bragg Reflection Auto-Detector ===")
    
    # 1. データ読み込み
    df = load_data(INPUT_CSV_PATH)
    data_values = df.values.flatten()
    
    # 2. 統計解析 & 閾値決定
    threshold, stats = calculate_threshold_iqr(data_values, multiplier=IQR_MULTIPLIER)
    
    print("\n[Statistics]")
    print(f"Mean:   {stats['mean']:.2f}")
    print(f"Median: {stats['median']:.2f}")
    print(f"Std:    {stats['std']:.2f}")
    print(f"IQR:    {stats['iqr']:.2f}")
    print("-" * 30)
    print(f"Calculated Threshold (Q3 + {IQR_MULTIPLIER}*IQR): {threshold:.2f}")
    
    # 3. 分布の可視化 (Step 1)
    plot_distribution(data_values, threshold, stats)
    
    # 4. 3D可視化 with 閾値平面 (Step 2)
    plot_3d_with_threshold(df, threshold)
    
    # 5. 座標抽出 & 出力
    bragg_coords = extract_bragg_coordinates(df, threshold)
    
    print(f"\n[Result] Detected {len(bragg_coords)} Bragg reflections.")
    if not bragg_coords.empty:
        print(bragg_coords.head(10)) # 上位10件表示
        print("...")
        
        # CSV保存
        bragg_coords.to_csv(OUTPUT_COORD_FILE, index=False)
        print(f"\nFull list saved to: {OUTPUT_COORD_FILE}")
    else:
        print("No Bragg reflections detected with current threshold.")

if __name__ == "__main__":
    main()