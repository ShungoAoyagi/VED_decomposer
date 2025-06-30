import numpy as np
from src.tasks.pre_processing.settings import Settings, orbital_magnetic_number
from src.helpers.calc_orb import calc_R_with_STO_fast, find_ncz_list
from src.helpers.fourier_truncation import fourier_truncation
from src.helpers.constant import Constants
from numba import jit

def calc_single_orb_density(magnification: float, r_min: float, r_max: float, mesh_grid: int, settings: Settings) -> np.ndarray:
    """
    Calculate the density of a single orbital.
    """
    _data = np.zeros((settings.v[0], settings.v[1], settings.v[2]))
    center_idx_float = [settings.center[i] * settings.v[i] for i in range(3)]
    lattice_params = settings.lattice_params
    orbital_name = settings.fit_orbital
    min_idx = settings.min_idx
    max_idx = settings.max_idx
    n = int(orbital_name[0])
    l = orbital_magnetic_number[orbital_name[1]]
    find, n_list, c_list, z_list = find_ncz_list(n, l, settings)
    if not find:
        raise ValueError(f"The orbital {orbital_name} is not found.")

    for i in range(settings.v[0]):
        for j in range(settings.v[1]):
            for k in range(settings.v[2]):
                if i < min_idx[0] or (i > max_idx[0] and max_idx[0] > min_idx[0]) or j < min_idx[1] or (j > max_idx[1] and max_idx[1] > min_idx[1]) or k < min_idx[2] or (k > max_idx[2] and max_idx[2] > min_idx[2]):
                    continue
                x = ((i - center_idx_float[0]) % settings.v[0]) / settings.v[0] * lattice_params[0]
                y = ((j - center_idx_float[1]) % settings.v[1]) / settings.v[1] * lattice_params[1]
                z = ((k - center_idx_float[2]) % settings.v[2]) / settings.v[2] * lattice_params[2]
                if x > lattice_params[0] / 2:
                    x -= lattice_params[0]
                if y > lattice_params[1] / 2:
                    y -= lattice_params[1]
                if z > lattice_params[2] / 2:
                    z -= lattice_params[2]
                r = np.sqrt(x**2 + y**2 + z**2)
                Rr = calc_R_with_STO_fast(n_list, c_list, z_list, r / magnification, Constants.a0_angstrom)
                if i == 39 and j == 15 and k == 27:
                    print(f"center_idx_float: {center_idx_float}")
                    print(f"x: {x}, y: {y}, z: {z}")
                    print(f"i: {i}, j: {j}, k: {k}, r: {r}, Rr: {Rr}")
                _data[i, j, k] = Rr ** 2

    data = fourier_truncation(_data, settings)
    # data = _data
    print(f"data.max(): {data.max()}, data.min(): {data.min()}")
    print(f"data_max_idx: {np.unravel_index(np.argmax(data), data.shape)}")

    res = np.zeros(mesh_grid)
    for i in range(mesh_grid):
        r_diff = r_min + (i / (mesh_grid - 1)) * (r_max - r_min)
        center_idx_float = [center_idx_float[i] % settings.v[i] for i in range(3)]

        pos = [(center_idx_float[i] + r_diff * settings.basis_set[0][i] * settings.v[i] / lattice_params[i]) % settings.v[i] for i in range(3)]
        tmp = 0.0
        idx_float = np.copy(pos)
        diff_from_int = np.ceil(pos) - pos
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    weight = np.abs(diff_from_int[0] - j) * np.abs(diff_from_int[1] - k) * np.abs(diff_from_int[2] - l)
                    if i < 10 and j == 0 and k == 0 and l == 0:
                        print(f"i: {i}, j: {j}, k: {k}, l: {l}, diff_from_int: {diff_from_int}, idx_float: {idx_float}, weight: {weight}")
                        print(f"data[int(idx_float[0] + j), int(idx_float[1] + k), int(idx_float[2] + l)]: {data[int(idx_float[0] + j), int(idx_float[1] + k), int(idx_float[2] + l)]}, weight * data: {weight * data[int(idx_float[0] + j), int(idx_float[1] + k), int(idx_float[2] + l)]}")
                    tmp += data[int(idx_float[0] + j), int(idx_float[1] + k), int(idx_float[2] + l)] * weight
        res[i] = tmp

    return res