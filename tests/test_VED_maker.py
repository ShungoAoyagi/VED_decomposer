from prefect import task
import numpy as np
import autograd.numpy as anp
import os
import sys
from scipy.ndimage import zoom

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks.pre_processing.settings import Settings, orbital_magnetic_number, import_settings
from src.helpers.calc_orb import calc_orb
from src.utils import ErrorHandler, ErrorCode, ErrorLevel
from src.tasks.pre_processing.load_data import load_data
from src.helpers.xplor_maker import make_xplor
from src.helpers.fourier_truncation import fourier_truncation

n_list = [3, 3, 4, 3, 3]
l_list = [2, 2, 0, 2, 2]
m_list = [1, -2, 0, -1, 2]
z_list = [-1, -1, -1, -1, -1]

settings = import_settings("data/input/settings.yaml")
data = load_data("data/input/data.xplor", settings)
# magnification = 2
magnification = 1
psi_list = []
v = settings.v
rho_list = anp.zeros((v[0], v[1], v[2]))

for n, l, m, z in zip(n_list, l_list, m_list, z_list):
    _, _psi = calc_orb(n, l, m, z, magnification, settings)
    psi_list.append(_psi)

for i in range(v[0]):
    for j in range(v[1]):
        for k in range(v[2]):
            # rho_list[i, j, k] = anp.sum(anp.abs(psi_list[0][i, j, k] + psi_list[1][i, j, k]) ** 2 + anp.abs(psi_list[3][i, j, k] + 1j * psi_list[4][i, j, k]) ** 2)
            rho_list[i, j, k] = anp.sum(anp.abs(psi_list[1][i, j, k]) ** 2)
            # rho_list[i,j,k] = psi_list[2][i,j,k]

print(psi_list[2].max())
rho_filtered = fourier_truncation(rho_list, settings)

make_xplor(rho_list, "data/output/test.xplor", "rho", settings)
make_xplor(rho_filtered, "data/output/test_filtered.xplor", "rho_filtered", settings)
