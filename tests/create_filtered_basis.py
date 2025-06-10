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


settings = import_settings("data/input_sample/settings.yaml")
data = load_data("data/input/data.xplor", settings)
# magnification = 2
magnification = 1
psi_list = []
v = settings.v
rho_list = anp.zeros((v[0], v[1], v[2]))

basis, zero_constraints = create_basis([], settings)

for i in range(basis[1][1].shape[0]):
    for j in range(basis[1][1].shape[1]):
        for k in range(basis[1][1].shape[2]):
            # rho_list[i, j, k] = anp.sum(anp.real(psi_list[0][i, j, k] + psi_list[1][i, j, k]) ** 2 + anp.real(psi_list[3][i, j, k] + 1j * psi_list[4][i, j, k]) ** 2)
            # rho = anp.real(0.3*basis[0,3, i, j, k]) + anp.real(0.3*basis[3,0, i, j, k])
            # rho = anp.real(basis[1, 4, i, j, k]) + anp.real(basis[4, 1, i, j, k])
            rho = anp.real(basis[0,0, i, j, k]) + anp.real(basis[3,3,i,j,k]) + anp.real(basis[8, 8, i, j, k])
            rho += anp.real(basis[0,3, i, j, k] + basis[3,0, i, j, k])
            rho += anp.real(anp.exp(1j * anp.pi / 3) * basis[0,8, i, j, k] + anp.exp(-1j * anp.pi / 3) * basis[8,0, i, j, k])
            # rho = anp.real(basis[8,8, i, j, k])
            # rho = anp.real(basis[1,1, i, j, k]) + anp.real(basis[4,4, i, j, k]) + anp.real(basis[1,4, i, j, k]) + anp.real(basis[4,1, i, j, k]) + anp.real(basis[2,2,i,j,k])

            rho_list[(i + settings.min_idx[0]) % settings.v[0], (j + settings.min_idx[1]) % settings.v[1], (k + settings.min_idx[2]) % settings.v[2]] = rho

            # rho_list[i,j,k] = psi_list[2][i,j,k]

# rho_filtered = fourier_truncation(rho_list, settings)

make_xplor(rho_list, "data/output/test.xplor", "rho", settings)
# make_xplor(rho_filtered, "data/output/test_filtered.xplor", "rho_filtered", settings)
