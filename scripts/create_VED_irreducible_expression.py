import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks.pre_processing.settings import import_settings, Settings, orbital_magnetic_number
from src.tasks.pre_processing.load_data import load_data
from src.helpers.xplor_loader import load_xplor
from src.helpers.xplor_maker import make_xplor

def load_previous_result(filepath: str) -> np.ndarray | None:
    """
    前回の計算結果をCSVファイルから読み込む
    
    Args:
        filepath: CSVファイルのパス
        
    Returns:
        読み込んだ行列（numpy配列）、失敗時はNone
    """
    try:
        if not os.path.exists(filepath):
            return None
            
        # CSVファイルを読み込み
        df = pd.read_csv(filepath, header=None)
        matrix = df.values[1:]
        print(matrix)
        
        # 正方行列かチェック
        if matrix.shape[0] != matrix.shape[1]:
            print(f"警告: 読み込んだ行列が正方行列ではありません。形状: {matrix.shape}")
            return None
            
        # 数値型に変換
        matrix = matrix.astype(np.float64)
        
        # NaNや無限大の値をチェック
        if not np.isfinite(matrix).all():
            print("警告: 読み込んだ行列に無効な値（NaN/Inf）が含まれています")
            return None
            
        print(f"前回の結果を読み込みました: {filepath}, 形状: {matrix.shape}")
        return matrix
        
    except pd.errors.EmptyDataError:
        print(f"エラー: CSVファイルが空です: {filepath}")
        return None

def create_irr_exp_data(settings: Settings, matrix_real, matrix_imag, emm: int):
    v = settings.v
    irr_exp_data = np.zeros((v[0], v[1], v[2]))
    nlm_set = []
    for orbital in settings.orbital_set:
        n = orbital[0]
        l = orbital_magnetic_number[orbital[1]]
        for m in range(-l, l+1):
            nlm_set.append((n, l, m))
    
    for i, nlm in enumerate(nlm_set):
        for j, nlm2 in enumerate(nlm_set):
            if i < j:
                continue
            if (nlm[2] - nlm2[2]) % 3 != 0:
                continue
            if emm == 0:
                if nlm[2] != 0:
                    continue
            else:
                if nlm[2] == 0:
                    continue
            filename = f"cache/orbitals/{settings.atom_name}_n{nlm[0]}l{nlm[1]}m{nlm[2]}_n{nlm2[0]}l{nlm2[1]}m{nlm2[2]}_filtered.xplor"
            print(filename)
            _orbital_data = load_xplor(filename)
            if i != j:
                irr_exp_data = np.add(irr_exp_data, 2 * np.real(_orbital_data.data * (matrix_real[i, j] + 1j * matrix_imag[i, j])))
            else:
                irr_exp_data = np.add(irr_exp_data, np.real(_orbital_data.data * (matrix_real[i, j] + 1j * matrix_imag[i, j])))
            print(f"loaded {filename}")
    return irr_exp_data

if __name__ == "__main__":
    settings = import_settings("data/input/settings.yaml")
    _ = load_data("data/input/data.xplor", settings)
    target_data = load_xplor("data/input/data.xplor")
    optimized_data = load_xplor("output/rho_output.xplor")
    residual_data = load_xplor("output/normalized_residual_output.xplor")
    matrix_real = load_previous_result("output/matrix_real.csv")
    matrix_imag = load_previous_result("output/matrix_imag.csv")
    print(matrix_real)
    print(matrix_imag)
    
    center = settings.center
    t = np.linspace(-settings.r_max, settings.r_max, 100)

    A_data = create_irr_exp_data(settings, matrix_real, matrix_imag, 0)
    E_data = create_irr_exp_data(settings, matrix_real, matrix_imag, 1)
    make_xplor(A_data, "output/A_data.xplor", "A_data", settings)
    make_xplor(E_data, "output/E_data.xplor", "E_data", settings)


