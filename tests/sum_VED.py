import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks.pre_processing.settings import import_settings
from src.tasks.pre_processing.load_data import load_data
from src.helpers.xplor_loader import load_xplor

n_list = [3, 4, 4]
l_list = [2, 0, 1]

settings = import_settings("data/input/settings.yaml")
data = load_data("data/input/data.xplor", settings)

print(settings.lattice_params)
print(settings.v)

for n, l in zip(n_list, l_list):
    for m in range(-l, l + 1):
        xplor = load_xplor(f"cache/orbitals/Co_n{n}l{l}m{m}_n{n}l{l}m{m}_filtered.xplor")
        sum = np.sum(xplor.data)
        density = sum / (settings.v[0] * settings.v[1] * settings.v[2]) * (settings.lattice_params[0] * settings.lattice_params[1] * settings.lattice_params[2])
        print(f"n={n}, l={l}, m={m}, sum={sum}, density={density}")