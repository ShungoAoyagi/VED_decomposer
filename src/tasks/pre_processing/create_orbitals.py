from prefect import task
import numpy as np
from src.tasks.pre_processing.settings import Settings, orbital_magnetic_number
from src.helpers.calc_orb import calc_orb
from src.utils import ErrorHandler, ErrorCode, ErrorLevel
import os

@task(name="create orbitals")
def create_orbitals(z: np.ndarray[float], settings: Settings) -> tuple[np.ndarray[float], list[tuple[str, np.ndarray[tuple[int, int, int], float]]]]:
    """
    create initial orbitals
    """
    if os.path.exists("cache/orbitals.npy"):
        return np.load("cache/orbitals.npy")
    if len(z) != len(settings.orbital_set):
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
        for m in range(-l, l + 1):
            z, psi_list = calc_orb(n, l, m, z[i], settings)
            res.append((f"{n}{orbital[1]}{m}", psi_list))
            z_res.append(z)

    np.save("cache/orbitals.npy", (z_res, res))
    return z_res, res