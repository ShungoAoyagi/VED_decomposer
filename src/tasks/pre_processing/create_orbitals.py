from prefect import task
import numpy as np
from src.tasks.pre_processing.settings import Settings

@task(name="create orbitals")
def create_orbitals(settings: Settings) -> list[tuple[str, np.ndarray]]:
    """
    軌道を作成する。各軌道の値が入ったn×m行列が入る
    """
    return [
        ("3dz2", np.ndarray([
            1, 1, 1
        ])),
        ("3dx2", np.ndarray([
            1, 1, 1
        ])),
        ("3dxy", np.ndarray([
            1, 1, 1
        ])),
        ("3dxz", np.ndarray([
            1, 1, 1
        ])),
        ("3dyz", np.ndarray([
            1, 1, 1
        ])),
        ("4s", np.ndarray([
            1, 1, 1
        ])),
        ("4px", np.ndarray([
            1, 1, 1
        ])),
        ("4py", np.ndarray([
            1, 1, 1
        ])),
        ("4pz", np.ndarray([
            1, 1, 1
        ])),
        
    ]