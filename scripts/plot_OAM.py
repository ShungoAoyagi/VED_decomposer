import numpy as np
import matplotlib.pyplot as plt
import os
import sys  
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks.pre_processing.load_data import load_data
from src.tasks.pre_processing.settings import import_settings, Settings
from src.tasks.after_processing.create_OAM_basis import create_OAM_basis

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

settings = import_settings("data/input/settings.yaml")
data = load_data("data/input/data.xplor", settings)

P_real = load_previous_result("output/matrix_real.csv")
P_imag = load_previous_result("output/matrix_imag.csv")

print(P_real.shape)
print(P_imag.shape)

P = P_real + 1j * P_imag

OAM_basis = create_OAM_basis(P, settings)

# plot vectors in 3d space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot vectors
x = np.arange(settings.v[0])
y = np.arange(settings.v[1])
z = np.arange(settings.v[2])
X, Y, Z = np.meshgrid(x, y, z)
ax.quiver(X, Y, Z, OAM_basis[:, :, :, 0], OAM_basis[:, :, :, 1], OAM_basis[:, :, :, 2])

plt.savefig("output/OAM_basis.png")