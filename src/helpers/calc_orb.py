import numpy as np
from src.tasks.pre_processing.settings import Settings
from src.utils import ErrorHandler, ErrorCode, ErrorLevel
from src.helpers import spherical_harmonics, Constants
from scipy.special import genlaguerre
from src.helpers.calc_Zeff import calc_Zeff
from src.helpers.spherical_harmonics import spherical_harmonics
import re

orbital_name = ["s", "p", "d", "f", "g", "h", "i"]

def is_float_string(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False

def find_ncz_list(n_principal: int, ell: int, settings: Settings) -> tuple[bool, list[int], list[float], list[float]]:
    """
    Parses a .tot file to find n, c, and z lists for a given orbital.

    Args:
        n_principal: Principal quantum number.
        ell: Angular momentum quantum number.
        settings: Settings object containing atom_name and other parameters.

    Returns:
        A tuple (flag, n_list, c_list, z_list).
        flag is True if the orbital is found, False otherwise.
        n_list, c_list, z_list are the extracted numerical lists.
    """
    global orbital_name

    flag = False
    n_list_out: list[int] = []
    c_list_out: list[float] = []
    z_list_out: list[float] = []

    atom = settings.atom_name
    filename = f"constant/{atom}.tot"
    target_orbital_label = f"{n_principal}{orbital_name[ell]}"

    try:
        with open(filename, "r") as f:
            lines = [line.rstrip() for line in f.readlines()]
    except FileNotFoundError:
        return False, [], [], []

    orbital_block_start_idx = -1
    for i, line_content in enumerate(lines):
        if line_content.strip().startswith(target_orbital_label):
            orbital_block_start_idx = i
            flag = True
            break
    
    if not flag:
        return False, [], [], []

    # Find n_list: search backwards from the orbital_block_start_idx for the "n " header
    n_header_line_idx = -1
    for i in range(orbital_block_start_idx -1, -1, -1):
        if lines[i].strip().startswith("n "):
            n_header_line_idx = i
            break
    
    if n_header_line_idx != -1 and n_header_line_idx + 1 < len(lines):
        n_values_line = lines[n_header_line_idx + 1].strip()
        try:
            n_list_out = [int(x) for x in n_values_line.split()]
        except ValueError:
            pass # n_list_out remains empty if parsing fails
    
    if not n_list_out and target_orbital_label not in ["1s", "2s", "2p", "3s", "3p", "3d", "4s"]: # Default for core, for safety, though n should always be found
        # This is a fallback, ideally the n_list should always be found from the file structure.
        # print(f"Warning: n_list not found for {target_orbital_label}, using default for core-like orbitals if applicable.")
        pass # Or handle error appropriately

    # State machine for c and z lists: None, "parsing_c_values", "parsing_z_values"
    current_parsing_state = None 
    
    for i in range(orbital_block_start_idx + 1, len(lines)):
        line = lines[i]
        stripped_line = line.strip()

        if stripped_line:
            potential_new_orbital_match = re.match(r"^(\d+[spdfghiklmnoqrtuvwxyz])", stripped_line)
            is_new_orbital_header = potential_new_orbital_match and potential_new_orbital_match.group(1) != target_orbital_label
            is_separator = stripped_line.startswith("---")
            is_new_n_header = stripped_line.startswith("n ") # Important for z_list termination
            is_new_ion_header_match = re.match(r"^\d+\s+[A-Za-z]{1,2}(\d*[\+\uFF0B])?$", stripped_line)
            is_new_ion_header = bool(is_new_ion_header_match)

            if is_new_orbital_header or is_separator or is_new_ion_header or (is_new_n_header and current_parsing_state == "parsing_z_values"):
                break 
        else: 
            break

        if stripped_line.startswith("c "):
            current_parsing_state = "parsing_c_values"
            c_list_out.clear()
            try:
                c_list_out.extend([float(x) for x in stripped_line.split()[1:]])
            except (ValueError, IndexError):
                pass 
            continue
        elif stripped_line.startswith("z "):
            current_parsing_state = "parsing_z_values"
            z_list_out.clear()
            try:
                z_list_out.extend([float(x) for x in stripped_line.split()[1:]])
            except (ValueError, IndexError):
                pass
            continue
        
        if current_parsing_state == "parsing_c_values":
            if line.startswith(" ") and stripped_line: 
                parts = stripped_line.split()
                if all(is_float_string(p) for p in parts):
                    c_list_out.extend([float(x) for x in parts])
                else:
                    current_parsing_state = None 
            else: 
                current_parsing_state = None
        
        elif current_parsing_state == "parsing_z_values":
            if line.startswith(" ") and stripped_line: 
                parts = stripped_line.split()
                if all(is_float_string(p) for p in parts):
                    z_list_out.extend([float(x) for x in parts])
                else:
                    current_parsing_state = None 
            else: 
                current_parsing_state = None
                
    return flag, n_list_out, c_list_out, z_list_out

def calc_orb(n: int, ell: int, m: int, z_before: float, magnification: int, settings: Settings) -> tuple[float, np.ndarray]:
    has_list, n_list, c_list, z_list = find_ncz_list(n, ell, settings)
    
    if len(n_list) != len(c_list) or len(n_list) != len(z_list):
        error_handler = ErrorHandler()
        error_handler.handle(
            f"The length of n_list, c_list, and z_list must be the same. The length of n_list is {len(n_list)}, the length of c_list is {len(c_list)}, and the length of z_list is {len(z_list)}.",
            ErrorCode.INVALID_INPUT,
            ErrorLevel.ERROR
        )
    
    v = [v // magnification for v in settings.v]
    lattice_params = settings.lattice_params
    r_max = settings.r_max
    psi_list = np.zeros((v[0], v[1], v[2]))
    z = z_before
    center_idx = settings.center_idx

    if not has_list and z == -1:
        z = calc_Zeff(n, settings)

    for i in range(v[0]):
        for j in range(v[1]):
            for k in range(v[2]):
                pos = np.array([(i - center_idx[0]) % v[0] / v[0] * lattice_params[0], (j - center_idx[1]) % v[1] / v[1] * lattice_params[1], (k - center_idx[2]) % v[2] / v[2] * lattice_params[2]])
                r = np.linalg.norm(pos)
                x = pos @ settings.basis_set[0]
                y = pos @ settings.basis_set[1]
                z = pos @ settings.basis_set[2]

                if r > r_max:
                    continue

                theta = np.arccos(z / r)
                phi = np.arctan2(y, x)
                if phi < 0:
                    phi += 2 * np.pi

                sph = spherical_harmonics(ell, m, theta, phi)
                if has_list:
                    psi_list[i, j, k] = calc_R_with_STO(n_list, c_list, z_list, r) * sph
                else:
                    psi_list[i, j, k] = calc_R_with_Zeff(n, ell, z, r) * sph

    return z, psi_list

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

def calc_R_with_Zeff(n: int, ell: int, z: float, r: float) -> float:
    """
    Calculate the radial part of Slater-type orbital with effective nuclear charge.
    
    Args:
        n: Principal quantum number
        ell: Angular momentum quantum number
        z: Effective nuclear charge
        r: Radial distance
        
    Returns:
        Value of the radial part of STO at the given r
    """
    rho = 2.0 * z * r / (n * Constants.a0_angstrom)
    L = genlaguerre(n - ell - 1, 2 * ell + 1)(rho)
    return rho ** ell * np.exp(-rho / 2.0) * L