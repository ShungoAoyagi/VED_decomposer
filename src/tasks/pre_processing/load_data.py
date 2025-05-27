from prefect import task
import numpy as np
from src.tasks.pre_processing.settings import Settings
import math

@task(name="load tmp_data")
def load_data(data_path: str, settings: Settings) -> np.ndarray[tuple[int, int, int], float]:
    """
    load xplor file
    """
    v = np.zeros(3, dtype=int)
    v_max = np.zeros(3, dtype=int)
    v_min = np.zeros(3, dtype=int)
    lattice_params = np.zeros(6, dtype=float)
    # load tmp_data efficiently by using generator
    with open(data_path, 'r') as f:
        # skip first 3 lines
        for _ in range(3):
            next(f)

        tmp = f.readline().split()
        v[0] = int(tmp[0])
        v_min[0] = int(tmp[1])
        v_max[0] = int(tmp[2])
        v[1] = int(tmp[3])
        v_min[1] = int(tmp[4])
        v_max[1] = int(tmp[5])
        v[2] = int(tmp[6])
        v_min[2] = int(tmp[7])
        v_max[2] = int(tmp[8])
        tmp_data = np.zeros((v[0], v[1], v[2]))


        tmp = f.readline().split()
        lattice_params[0] = float(tmp[0])
        lattice_params[1] = float(tmp[1])
        lattice_params[2] = float(tmp[2])
        lattice_params[3] = float(tmp[3])
        lattice_params[4] = float(tmp[4])
        lattice_params[5] = float(tmp[5])

        for _ in range(1):
            next(f)

        for i in range(v_min[2], v[2]):
            count = 0
            tmp = f.readline().split()
            for j in range(v_min[1], v[1]):
                for k in range(v_min[0], v[0]):
                    if (count % 5 == 0):
                        tmp = f.readline().split()
                    try:
                        tmp_data[k, j, i] = float(tmp[count % 5])
                    except:
                        print(tmp)
                        print(count)
                        print(k, j, i)
                        raise Exception("Error")
                    count += 1

    
    center = settings.center
    r_mesh = settings.r_mesh
    theta_mesh = settings.theta_mesh
    phi_mesh = settings.phi_mesh
    r_max = settings.r_max
    basis_set = settings.basis_set

    data = np.zeros((r_mesh, theta_mesh, phi_mesh))
    for i in range(r_mesh):
        for j in range(theta_mesh):
            for k in range(phi_mesh):
                r = r_max * i / r_mesh
                theta = j * np.pi / theta_mesh
                phi = k * 2 * np.pi / phi_mesh
                basis_x: float = [r * np.sin(theta) * np.cos(phi) * basis for basis in basis_set[0]]
                basis_y: float = [r * np.sin(theta) * np.sin(phi) * basis for basis in basis_set[1]]
                basis_z: float = [r * np.cos(theta) * basis for basis in basis_set[2]]
                x = lattice_params[0] * center[0] + basis_x[0] + basis_y[0] + basis_z[0]
                y = lattice_params[1] * center[1] + basis_x[1] + basis_y[1] + basis_z[1]
                z = lattice_params[2] * center[2] + basis_x[2] + basis_y[2] + basis_z[2]
                
                x_idx_float = x / lattice_params[0] * v[0]
                y_idx_float = y / lattice_params[1] * v[1]
                z_idx_float = z / lattice_params[2] * v[2]

                for x_idx in range(math.floor(x_idx_float), math.ceil(x_idx_float)):
                    for y_idx in range(math.floor(y_idx_float), math.ceil(y_idx_float)):
                        for z_idx in range(math.floor(z_idx_float), math.ceil(z_idx_float)):
                            weight = abs((x_idx_float - x_idx) * (y_idx_float - y_idx) * (z_idx_float - z_idx))
                            data[i, j, k] += tmp_data[x_idx % v[0], y_idx % v[1], z_idx % v[2]] * weight

    return data
