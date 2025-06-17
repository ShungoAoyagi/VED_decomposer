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

def plot_data(t, target_data, optimized_data, residual_data, filename, raw_orbital_data = None):
    # plt.rcParams["font.sans-serif"] = "Arial"
    _fontsize = 16
    fig, ax1 = plt.subplots(figsize=(6,6))
    
    # 左軸（target と optimized）
    ax1.plot(t, target_data, "o", label="target", color="k", markersize=6)
    ax1.plot(t, optimized_data, "-", label="optimized", color="red", linewidth=2)
    if raw_orbital_data is not None:
        ax1.plot(t, raw_orbital_data, "-", label="raw", color="green", linewidth=2)
    ax1.set_xlabel("r", fontsize=_fontsize)
    ax1.set_ylabel("electron density (e/A$^3$)", fontsize=_fontsize)
    ax1.tick_params(axis='x', labelsize=_fontsize)
    ax1.tick_params(axis='y', labelsize=_fontsize)
    ax1.set_ylim(-max(target_data) * 0.3, max(target_data) * 1.2)
    ax1.axhline(0, color="k", linewidth=1)
    ax1.axvline(0, color="k", linewidth=1)
    
    # 右軸（residual）
    ax2 = ax1.twinx()
    ax2.plot(t, residual_data, "o", label="residual", color="blue", markersize=6)
    ax2.set_ylabel("relative residual", fontsize=_fontsize)
    ax2.tick_params(axis='y', labelsize=_fontsize)
    ax2.set_ylim(-0.25, 1.0)

    ax1.tick_params(direction='in', top=True, right=False, labelsize=_fontsize)
    ax2.tick_params(direction='in', top=True, right=True, labelsize=_fontsize)

    plt.xlim(-0.5, 0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)

def extract_line_data(t: np.ndarray, center: list, basis_vector: list, 
                     settings: Settings, target_data, optimized_data, residual_data, raw_orbital_data = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    指定した軸方向の線データを抽出する
    
    Args:
        t: 位置パラメータの配列
        center: 中心位置
        basis_vector: 基底ベクトル（軸方向）
        settings: 設定オブジェクト
        target_data: ターゲットデータ
        optimized_data: 最適化データ
        residual_data: 残差データ
        raw_orbital_data: 生の軌道データ
    Returns:
        target, optimized, residualの値の配列のタプル
    """
    target_values = np.zeros(len(t))
    optimized_values = np.zeros(len(t))
    residual_values = np.zeros(len(t))
    raw_orbital_values = np.zeros(len(t))

    for i in range(len(t)):
        pos = [
            center[0] + t[i] * basis_vector[0] / settings.lattice_params[0], 
            center[1] + t[i] * basis_vector[1] / settings.lattice_params[1], 
            center[2] + t[i] * basis_vector[2] / settings.lattice_params[2]
        ]
        idx = [(pos[j] * settings.v[j]) % settings.v[j] for j in range(3)]
        target_values[i] = target_data.get_value(idx[0], idx[1], idx[2])
        optimized_values[i] = optimized_data.get_value(idx[0], idx[1], idx[2])
        residual_values[i] = residual_data.get_value(idx[0], idx[1], idx[2])
        if raw_orbital_data is not None:
            raw_orbital_values[i] = raw_orbital_data.get_value(idx[0], idx[1], idx[2])
    return target_values, optimized_values, residual_values, raw_orbital_values

def create_raw_orbital_data(settings: Settings, matrix_real, matrix_imag):
    v = settings.v
    raw_orbital_data = np.zeros((v[0], v[1], v[2]))
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
            filename = f"cache/orbitals/{settings.atom_name}_n{nlm[0]}l{nlm[1]}m{nlm[2]}_n{nlm2[0]}l{nlm2[1]}m{nlm2[2]}_raw.xplor"
            print(filename)
            _orbital_data = load_xplor(filename)
            if i != j:
                raw_orbital_data = np.add(raw_orbital_data, 2 * np.real(_orbital_data.data * (matrix_real[i, j] + 1j * matrix_imag[i, j])))
            else:
                raw_orbital_data = np.add(raw_orbital_data, np.real(_orbital_data.data * (matrix_real[i, j] + 1j * matrix_imag[i, j])))
            print(f"loaded {filename}")
    return raw_orbital_data

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

    raw_orbital_data = None
    # if not os.path.exists("output/raw_orbital_data.xplor"):
    #     raw_orbital_data = create_raw_orbital_data(settings, matrix_real, matrix_imag)
    #     make_xplor(raw_orbital_data, "output/raw_orbital_data.xplor", "raw_orbital_data", settings)
    #     raw_orbital_data = load_xplor("output/raw_orbital_data.xplor")
    # else:
    #     raw_orbital_data = load_xplor("output/raw_orbital_data.xplor")

    target_x, optimized_x, residual_x, raw_orbital_x = extract_line_data(
        t, center, settings.basis_set[0], settings, target_data, optimized_data, residual_data, raw_orbital_data
    )
    plot_data(t, target_x, optimized_x, residual_x, "output/after_processing_x.png", raw_orbital_x)
    
    target_y, optimized_y, residual_y, raw_orbital_y = extract_line_data(
        t, center, settings.basis_set[1], settings, target_data, optimized_data, residual_data, raw_orbital_data
    )
    plot_data(t, target_y, optimized_y, residual_y, "output/after_processing_y.png", raw_orbital_y)
    
    target_z, optimized_z, residual_z, raw_orbital_z = extract_line_data(
        t, center, settings.basis_set[2], settings, target_data, optimized_data, residual_data, raw_orbital_data
    )
    plot_data(t, target_z, optimized_z, residual_z, "output/after_processing_z.png", raw_orbital_z)

