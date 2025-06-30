from prefect import task
import numpy as np
import autograd.numpy as anp
import os
import sys
from scipy.ndimage import zoom

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks.pre_processing.create_filtered_basis import create_basis
from src.tasks.pre_processing.settings import Settings, orbital_magnetic_number, import_settings
from src.helpers.calc_orb import calc_orb
from src.utils import ErrorHandler, ErrorCode, ErrorLevel
from src.tasks.pre_processing.load_data import load_data
from src.helpers.xplor_maker import make_xplor
from src.helpers.fourier_truncation import fourier_truncation
from src.helpers.xplor_loader import load_xplor, XplorFile

def extract_line_data(t: np.ndarray, center: list, basis_vector: list, 
                     settings: Settings, data: XplorFile) -> np.ndarray:
    """
    指定した軸方向の線データを抽出する
    
    Args:
        t: 位置パラメータの配列
        center: 中心位置
        basis_vector: 基底ベクトル（軸方向）
        settings: 設定オブジェクト
        data: データ
    Returns:
        データの値の配列
    """
    data_values = np.zeros(len(t))
    print(data.data[27, 27, 27])

    for i in range(len(t)):
        pos = [
            center[0] + t[i] * basis_vector[0] / settings.lattice_params[0], 
            center[1] + t[i] * basis_vector[1] / settings.lattice_params[1], 
            center[2] + t[i] * basis_vector[2] / settings.lattice_params[2]
        ]
        idx = [(pos[j] * settings.v[j]) % settings.v[j] for j in range(3)]
        data_values[i] = data.get_value(idx[0], idx[1], idx[2])
        if i == 0:
            print(data_values[i], idx)
    return data_values

settings = import_settings("data/input/settings.yaml")
data = load_data("data/input/data.xplor", settings)

v = settings.v
rho_list = anp.zeros((v[0], v[1], v[2]))

basis, zero_constraints = create_basis([], settings)

for i in range(basis[1][1].shape[0]):
    for j in range(basis[1][1].shape[1]):
        for k in range(basis[1][1].shape[2]):
            # rho_list[i, j, k] = anp.sum(anp.real(psi_list[0][i, j, k] + psi_list[1][i, j, k]) ** 2 + anp.real(psi_list[3][i, j, k] + 1j * psi_list[4][i, j, k]) ** 2)
            # rho = anp.real(0.3*basis[0,3, i, j, k]) + anp.real(0.3*basis[3,0, i, j, k])
            # rho = anp.real(basis[1, 4, i, j, k]) + anp.real(basis[4, 1, i, j, k])
            rho = anp.real(basis[2, 5, i, j, k]) + anp.real(basis[5, 2, i, j, k])
            # rho = anp.real(basis[0,3, i, j, k] + basis[3,0, i, j, k])
            # rho += anp.real(anp.exp(1j * anp.pi / 4) * basis[0,8, i, j, k] + anp.exp(-1j * anp.pi / 4) * basis[8,0, i, j, k])
            # rho = anp.real(basis[8,8, i, j, k])
            # rho = anp.real(basis[1,1, i, j, k]) + anp.real(basis[4,4, i, j, k]) + anp.real(basis[1,4, i, j, k]) + anp.real(basis[4,1, i, j, k]) + anp.real(basis[2,2,i,j,k])

            rho_list[(i + settings.min_idx[0]) % settings.v[0], (j + settings.min_idx[1]) % settings.v[1], (k + settings.min_idx[2]) % settings.v[2]] = rho

            # rho_list[i,j,k] = psi_list[2][i,j,k]

# rho_filtered = fourier_truncation(rho_list, settings)

make_xplor(rho_list, "data/output/test.xplor", "rho", settings)
data = load_xplor("data/output/test.xplor")
t = np.linspace(-settings.r_max, settings.r_max, 100)
print(data.v)
rho_values = extract_line_data(t, settings.center, settings.basis_set[2], settings, data)

with open("data/output/extract_111.txt", "w") as f:
    for i in range(len(t)):
        f.write(f"{t[i]} {rho_values[i]}\n")
# make_xplor(rho_filtered, "data/output/test_filtered.xplor", "rho_filtered", settings)

