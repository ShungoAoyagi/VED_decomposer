import numpy as np
from src.tasks.pre_processing.settings import Settings

def fourier_truncation(f: np.ndarray[tuple[int, int, int], float], settings: Settings) -> np.ndarray[tuple[int, int, int], float]:
    """
    Calculate the wavefunction f_filtered by the Fourier truncation method.

    Args:
        f: The wavefunction to be truncated.
        settings: The settings of the simulation.

    Returns:
        The wavefunction f_filtered.
    """
    r_mesh = settings.r_mesh
    r_max = settings.r_max
    theta_mesh = settings.theta_mesh
    phi_mesh = settings.phi_mesh
    d_min = settings.d_min

    r_step = r_max / r_mesh
    theta_step = np.pi / theta_mesh
    phi_step = 2 * np.pi / phi_mesh

    k_max = int(2 * np.pi / d_min)
    k_list = []
    for k in range(2 * k_max + 1):
        for j in range(2 * k_max + 1):
            for i in range(2 * k_max + 1):
                if (i - k_max)**2 + (j - k_max)**2 + (k - k_max)**2 <= k_max**2:
                    k_list.append((i - k_max, j - k_max, k - k_max))

    k_list = np.array(k_list)
    F_k = {}
    for k in k_list:
        k_x = k[0]
        k_y = k[1]
        k_z = k[2]

        F_k[k] = 0
        for i in range(r_mesh):
            for j in range(theta_mesh):
                for l in range(phi_mesh):
                    r = i * r_step
                    theta = j * theta_step
                    phi = l * phi_step
                    x = r * np.sin(theta) * np.cos(phi)
                    y = r * np.sin(theta) * np.sin(phi)
                    z = r * np.cos(theta)
                    F_k[k] += f[i, j, l] * np.exp(-1j * (k_x * x + k_y * y + k_z * z))

    f_filtered = np.zeros((r_mesh, theta_mesh, phi_mesh))
    for i in range(r_mesh):
        for j in range(theta_mesh):
            for l in range(phi_mesh):
                r = i * r_step
                theta = j * theta_step
                phi = l * phi_step
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                for k in k_list:
                    k_x = k[0]
                    k_y = k[1]
                    k_z = k[2]
                    f_filtered[i, j, l] += F_k[k] * np.exp(1j * (k_x * x + k_y * y + k_z * z))

    return f_filtered
