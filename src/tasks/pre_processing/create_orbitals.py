from prefect import task
import numpy as np
import autograd.numpy as anp
from src.tasks.pre_processing.settings import Settings, orbital_magnetic_number
from src.helpers.calc_orb import calc_orb
from src.utils import ErrorHandler, ErrorCode, ErrorLevel
import os

def create_orbitals(z: np.ndarray[float], magnification: int, settings: Settings) -> tuple[np.ndarray[float], list[tuple[str, np.ndarray[tuple[int, int, int], float]]]]:
    """
    create initial orbitals
    """
    # キャッシュ機能を無効化（自動微分の連鎖が切れるため）
    # if os.path.exists("cache/orbitals.npy"):
    #     return np.load("cache/orbitals.npy")
        
    if len(z) != 0 and len(z) != len(settings.orbital_set):
        error_handler = ErrorHandler()
        error_handler.handle(
            f"The length of alpha must be equal to the length of settings.orbital_set. The length of z is {len(z)}, the length of settings.orbital_set is {len(settings.orbital_set)}.",
            ErrorCode.INVALID_INPUT,
            ErrorLevel.ERROR
        )

    orbital_set = settings.orbital_set

    res = []
    z_res = []
    for i, orbital in enumerate(orbital_set):
        n = int(orbital[0])
        l = orbital_magnetic_number[orbital[1]]
        _z = -1
        if len(z) != 0:
            _z = z[i]
        for m in range(-l, l + 1):
            new_z, psi_list = calc_orb(n, l, m, _z, magnification, settings)
            # autograd.numpyの配列に変換して、自動微分の連鎖を維持
            if not isinstance(psi_list, anp.ndarray):
                psi_list = anp.array(psi_list)
            res.append((f"{n}{orbital[1]}{m}", psi_list))
            if m == 0:
                z_res.append(new_z)

    # キャッシュの保存も無効化
    # np.save("cache/orbitals.npy", (z_res, res))
    return z_res, res