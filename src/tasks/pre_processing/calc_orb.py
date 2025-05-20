import numpy as np
from src.tasks.pre_processing.settings import Settings
from src.utils import ErrorHandler, ErrorCode, ErrorLevel
from src.helpers import spherical_harmonics

def calc_orb(n: int, ell: int, m: int, output_path: str, settings: Settings) -> np.ndarray:
    n_list = [1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    c_list = [0.025695, 0.015270, -0.101936, 0.001471, -0.028579, 0.180214, 0.119361, -0.022991]
    z_list = [26.710310, 22.739409, 11.157920, 32.859200, 8.226529, 5.699980, 3.843176, 13.203642, 2.275507, 1.443056, 0.951968]
    
    if len(n_list) != len(c_list) or len(n_list) != len(z_list):
        error_handler = ErrorHandler()
        error_handler.handle(
            f"The length of n_list, c_list, and z_list must be the same. The length of n_list is {len(n_list)}, the length of c_list is {len(c_list)}, and the length of z_list is {len(z_list)}.",
            ErrorCode.INVALID_INPUT,
            ErrorLevel.ERROR
        )
    
    r_mesh = settings.r_mesh
    theta_mesh = settings.theta_mesh
    phi_mesh = settings.phi_mesh
    r_max = settings.r_max
    psi_list = np.zeros(r_mesh * theta_mesh * phi_mesh)
    psi_list = np.reshape(psi_list, (r_mesh, theta_mesh, phi_mesh))

    for i in range(r_mesh):
        for j in range(theta_mesh):
            for k in range(phi_mesh):
                psi_list[i, j, k] = calc_R_with_STO(n_list, c_list, z_list, i, j, k, r_max) * spherical_harmonics(ell, m, j, k)

    return psi_list

def calc_R_with_STO(n_list: list[int], c_list: list[float], z_list: list[float], r: float) -> float:
    """
    Calculate the radial part of Slater-type orbital.
    
    Args:
        n_list: List of principal quantum numbers
        c_list: List of coefficients for STO expansion
        z_list: List of Slater exponents
        r: Radial distance
        
    Returns:
        Value of the radial part of STO at the given r
    """    
    res = 0.0
    
    # Sum over all STO basis functions
    for i in range(len(n_list)):
        n = n_list[i]  # Principal quantum number
        zeta = z_list[i]  # Slater exponent
        c = c_list[i]  # Expansion coefficient
        
        radial_part = r**(n-1) * np.exp(-zeta * r)
        
        res += c * radial_part
        
    return res
        