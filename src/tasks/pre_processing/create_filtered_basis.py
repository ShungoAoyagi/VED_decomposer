import numpy as np
from src.tasks.pre_processing.settings import Settings, orbital_magnetic_number
from src.utils import ErrorHandler, ErrorCode, ErrorLevel
import os
from src.helpers.xplor_loader import load_xplor
from src.helpers.xplor_maker import make_xplor
from src.tasks.pre_processing.create_orbitals import calc_orb
from src.helpers.fourier_truncation import fourier_truncation
from prefect import task
from src.helpers.pick_partial_data import pick_partial_data

def can_hybridize_by_symmetry(m1: int, m2: int) -> bool:
    """
    Check if the two orbitals can hybridize by symmetry.
    """
    rotation = 3
    if (m1 - m2) % rotation == 0:
        return True
    return False

@task(name="Create basis")
def create_basis(z: np.ndarray[float], settings: Settings) -> tuple[np.ndarray[tuple[int, int], np.ndarray[tuple[int, int, int], complex] | None], np.ndarray[tuple[int, int], bool]]:
    """
    Create basis
    """

    if len(z) != 0 and len(z) != len(settings.orbital_set):
        error_handler = ErrorHandler()
        error_handler.handle(
            f"The length of alpha must be equal to the length of settings.orbital_set. The length of z is {len(z)}, the length of settings.orbital_set is {len(settings.orbital_set)}.",
            ErrorCode.INVALID_INPUT,
            ErrorLevel.ERROR
        )

    orbital_set = settings.orbital_set

    orbital_set_list = []
    z_list = []
    for i, orbital in enumerate(orbital_set):
        n = int(orbital[0])
        l = orbital_magnetic_number[orbital[1]]
        _z = -1
        if len(z) != 0:
            _z = z[i]
        for m in range(-l, l + 1):
            orbital_set_list.append((n, l, m))
            z_list.append(_z)

    v = np.zeros(3, dtype=int)
    for i in range(3):
        v[i] = (settings.max_idx[i] - settings.min_idx[i]) % settings.v[i]
    basis = np.zeros((len(orbital_set_list), len(orbital_set_list), v[0], v[1], v[2]), dtype=np.complex128)
    zero_constraints = []

    orbital_data: dict[int, np.ndarray[tuple[int, int, int], float]] = {}

    for i, orbital_1 in enumerate(orbital_set_list):
        for j, orbital_2 in enumerate(orbital_set_list):
            n_1, l_1, m_1 = orbital_1
            n_2, l_2, m_2 = orbital_2
            if not can_hybridize_by_symmetry(m_1, m_2):
                zero_constraints.append((i, j))
                continue
            if i < j:
                continue
            filtered_file_path = f"cache/orbitals/{settings.atom_name}_n{n_1}l{l_1}m{m_1}_n{n_2}l{l_2}m{m_2}_filtered.xplor"
            if os.path.exists(filtered_file_path):
                xplor_data = load_xplor(filtered_file_path)
                if xplor_data.exists and np.array_equal(xplor_data.v, settings.v):
                    picked_data = pick_partial_data(xplor_data.data, settings)
                    basis[i, j, :, :, :] = picked_data
                    if i != j:
                        basis[j, i, :, :, :] = np.conj(picked_data)
                    continue
                else:
                    print(f"The xplor file {filtered_file_path} does not exist or the v values are not equal.")
            
            if i not in orbital_data:
                _, psi_list = calc_orb(n_1, l_1, m_1, z_list[i], 1, settings)
                orbital_data[i] = psi_list
            if j not in orbital_data:
                _, psi_list = calc_orb(n_2, l_2, m_2, z_list[j], 1, settings)
                orbital_data[j] = psi_list
            # _basis = np.dot(orbital_data[i], orbital_data[j].conj())
            _basis = np.zeros((settings.v[0], settings.v[1], settings.v[2]), dtype=np.complex128)
            for k in range(settings.v[0]):
                for l in range(settings.v[1]):
                    for m in range(settings.v[2]):
                        _basis[k, l, m] = orbital_data[i][k, l, m].conj() * orbital_data[j][k, l, m]
            print("_basis.shape: ", _basis.shape)
            raw_file_path = f"cache/orbitals/{settings.atom_name}_n{n_1}l{l_1}m{m_1}_n{n_2}l{l_2}m{m_2}_raw.xplor"
            make_xplor(_basis, raw_file_path, f"{settings.atom_name}_n{n_1}l{l_1}m{m_1}_n{n_2}l{l_2}m{m_2}_raw", settings)

            filtered_basis = fourier_truncation(_basis, settings)
            make_xplor(filtered_basis, filtered_file_path, f"{settings.atom_name}_n{n_1}l{l_1}m{m_1}_n{n_2}l{l_2}m{m_2}_filtered", settings)
            picked_filtered_basis = pick_partial_data(filtered_basis, settings)
            basis[i, j, :, :, :] = picked_filtered_basis
            if i != j:
                basis[j, i, :, :, :] = picked_filtered_basis

    return basis, zero_constraints