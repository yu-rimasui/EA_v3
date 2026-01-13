'''
Sm, Feの解析 (差分データ)

入力：`L_phi_XXX.img`, `U_phi_XXX.img`

処理：
1. 画像演算：差分をとる`Diff_phi_XXX.img = L_phi_XXX.img - U_phi_XXX.img`
2. `Diff_phi_XXX.img`の1Dプロファイル（横軸：半径 $r$）を作成
3. r≈0(Sm), r≈125(Fe)にて、ピークフィッテイング実行

出力: Smの強度CSV, Feの強度CSV
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import config as cfg

def main():
    # ==========================================
    # 0. 設定 (エネルギー固定)
    # ==========================================
    target_energy = "13500"
    print(f"=== Single Energy Processing Mode: {target_energy} ===")

    # 1. インスタンス生成
    loader = RawDataLoader()
    processor = ImageProcessor() # 固定中心 (250, 65)
    analyzer = IntensityAnalyzer()

    # エラーログ用
    error_log_path = cfg.OUTPUT_DIR / "error_log.txt"
    
    # 結果格納用の行列を準備 [行:Theta(76) x 列:Phi(1440)]
    num_thetas = cfg.THETA_END - cfg.THETA_START + 1  # 76
    num_phis = cfg.PHI_END - cfg.PHI_START + 1        # 1440
    
    print(f"Initializing matrices ({num_thetas} x {num_phis})...")
    matrix_bi = np.zeros((num_thetas, num_phis), dtype=np.float32)
    matrix_sm = np.zeros((num_thetas, num_phis), dtype=np.float32)
    matrix_fe = np.zeros((num_thetas, num_phis), dtype=np.float32)

    # ==========================================
    # Loop 1: Theta (0 - 75)
    # ==========================================
    # Thetaループに進捗バーを表示
    for theta in tqdm(range(cfg.THETA_START, cfg.THETA_END + 1), desc=f"Theta Loop"):
        
        # ==========================================
        # Loop 2: Phi (0 - 1439)
        # ==========================================
        for phi in range(cfg.PHI_START, cfg.PHI_END + 1):
            
            # 1. 画像読み込み (L, U ペア)
            l_img, u_img = loader.load_pair(target_energy, theta, phi)

            # エラーハンドリング
            if l_img is None or u_img is None:
                # 読み込めなかった場合は0のままスキップし、ログに残す
                # (tqdmの表示を崩さないよう、printではなくファイル書き込み推奨)
                # with open(error_log_path, "a") as f:
                #    f.write(f"Missing: {target_energy}/Th{theta}/Phi{phi}\n")
                continue

            # 2. 差分計算 (Diff = L - U)
            # ※負の値はクリップして0にする（物理的に負の強度はありえないため）
            diff_img = np.maximum(l_img - u_img, 0)

            # 3. プロファイル計算 (1D化)
            # Bi用 (U画像から)
            radii_u, prof_u = processor.calculate_azimuthal_average(u_img)
            # Sm, Fe用 (Diff画像から)
            radii_d, prof_d = processor.calculate_azimuthal_average(diff_img)

            # 4. 強度算出 (ROI積分)
            # Bi (r=20付近) from U
            val_bi = analyzer.calculate_intensity(radii_u, prof_u, cfg.ROI_TARGETS["Bi"])
            
            # Sm (r=0付近) from Diff
            val_sm = analyzer.calculate_intensity(radii_d, prof_d, cfg.ROI_TARGETS["Sm"])
            
            # Fe (r=125付近) from Diff
            val_fe = analyzer.calculate_intensity(radii_d, prof_d, cfg.ROI_TARGETS["Fe"])

            # 5. 行列に格納
            row_idx = theta - cfg.THETA_START
            col_idx = phi - cfg.PHI_START
            
            matrix_bi[row_idx, col_idx] = val_bi
            matrix_sm[row_idx, col_idx] = val_sm
            matrix_fe[row_idx, col_idx] = val_fe

    # ==========================================
    # Save Phase: CSV保存
    # ==========================================
    print(f"\nSaving CSVs for {target_energy}...")
    
    # 保存先フォルダ: output/13500/
    save_dir = cfg.OUTPUT_DIR / target_energy
    save_dir.mkdir(parents=True, exist_ok=True)

    # 行列データの保存
    # header=False, index=False で純粋な数値データのみ保存
    pd.DataFrame(matrix_bi).to_csv(save_dir / "Bi_matrix.csv", header=False, index=False)
    pd.DataFrame(matrix_sm).to_csv(save_dir / "Sm_matrix.csv", header=False, index=False)
    pd.DataFrame(matrix_fe).to_csv(save_dir / "Fe_matrix.csv", header=False, index=False)
    
    print(f"Saved successfully to: {save_dir}")

if __name__ == "__main__":
    main()