import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks.pre_processing.settings import import_settings
from src.tasks.pre_processing.create_orbitals import create_orbitals
from src.tasks.pre_processing.load_data import load_data
import numpy as np
from src.helpers.xplor_maker import make_xplor
from src.helpers.fourier_truncation import fourier_truncation
settings = import_settings("data/input/settings.yaml")

data = load_data("data/input/data.xplor", settings)

print("start creating orbitals")
_, orbitals = create_orbitals([], 4, settings)
print("orbitals created")

gamma = np.zeros((len(orbitals), len(orbitals)))
gamma[0, 0] = 1

print("start calculating rho")

magnification = 1
shape = orbitals[0][1].shape
rho = np.zeros(shape)

print("start calculating rho")
for i in range(shape[0]):
    for j in range(shape[1]):
        for k in range(shape[2]):
            rho[i, j, k] = gamma[0, 0] * orbitals[2][1][i, j, k] **2

print("rho calculated")

rho_filtered = fourier_truncation(rho, settings)

make_xplor(rho, "output/rho.xplor", "rho", settings)
make_xplor(rho_filtered, "output/rho_filtered.xplor", "rho_filtered", settings)
