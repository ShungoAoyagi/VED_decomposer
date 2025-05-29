import numpy as np
import autograd.numpy as anp
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
    Parses a .tot file to find n, c, and z lists for a given orbital,
    specifically looking for the neutral atom block if settings.atom_name implies it.

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

    atom = settings.atom_name # e.g., "Fe"
    filename = f"constant/{atom}.tot"
    target_orbital_label = f"{n_principal}{orbital_name[ell]}"

    try:
        with open(filename, "r") as f:
            lines = [line.rstrip() for line in f.readlines()]
    except FileNotFoundError:
        return False, [], [], []

    selected_ion_block_start_idx = -1
    selected_ion_block_end_idx = -1
    
    ion_header_indices = []
    ion_header_texts = []

    # Regex to capture Z, AtomSymbol, and the rest of the line for charge/other info parsing
    # Group 1: Z (digits), Group 2: AtomSymbol (letters), Group 3: a potential charge string or other info
    ion_header_regex = re.compile(r"^(\d+)\s+([A-Za-z]{1,2})(.*)$")
    # Regex to specifically identify charge patterns like 2+, +, 2-, - within the rest of the line
    charge_pattern_regex = re.compile(r"^\s*(\d*[+\uFF0B\u2212-])")

    for i, line_content in enumerate(lines):
        stripped_line = line_content.strip()
        match = ion_header_regex.match(stripped_line)
        if match:
            ion_header_indices.append(i)
            ion_header_texts.append(stripped_line) # Store the full matched line text

    found_correct_ion_block = False
    for i in range(len(ion_header_texts)):
        header_line_text = ion_header_texts[i] # Full line text e.g., "26 Fe J=..." or "26 Fe2+ ..."
        match = ion_header_regex.match(header_line_text) # Match again to extract groups properly
        if match:
            file_atom_z = match.group(1) # Z as string
            file_atom_symbol = match.group(2) # Atom symbol
            rest_of_line = match.group(3).strip() # Text after Z and AtomSymbol, e.g., "J=..." or "2+ J=..."
            
            is_neutral = True # Assume neutral by default
            if rest_of_line: # If there is something after the atom symbol
                # Check if the beginning of the rest_of_line matches a charge pattern
                charge_match = charge_pattern_regex.match(rest_of_line)
                if charge_match: # If it starts with something like "2+", "-", etc.
                    is_neutral = False # Then it's not neutral
            
            if file_atom_symbol == atom and is_neutral:
                selected_ion_block_start_idx = ion_header_indices[i]
                selected_ion_block_end_idx = ion_header_indices[i+1] if i+1 < len(ion_header_indices) else len(lines)
                found_correct_ion_block = True
                break
    if not found_correct_ion_block:
        print(f"No correct ion block found for {atom}. n: {n_principal}, ell: {ell}")
        return False, [], [], []

    orbital_block_start_idx = -1
    for line_idx in range(selected_ion_block_start_idx, selected_ion_block_end_idx):
        if lines[line_idx].strip().startswith(target_orbital_label):
            orbital_block_start_idx = line_idx
            flag = True
            break
    if not flag:
        print(f"No orbital block found for {atom}. n: {n_principal}, ell: {ell}")
        return False, [], [], []

    # Corrected n_list search logic: Search backwards from the orbital block
    n_list_found = False
    n_header_line_idx = -1
    # Search backwards from just before orbital_block_start_idx up to the start of the current ion block
    for i in range(orbital_block_start_idx - 1, selected_ion_block_start_idx - 1, -1):
        if lines[i].strip().startswith("n "):
            n_header_line_idx = i
            break # Found the closest preceding "n " header

    if n_header_line_idx != -1:
        temp_n_list = []
        # Try to parse n values from the "n " header line itself
        try:
            n_values_on_header_line = lines[n_header_line_idx].strip().split()
            if len(n_values_on_header_line) > 1: # Check if there are numbers after "n"
                temp_n_list.extend([int(x) for x in n_values_on_header_line[1:]])
        except (ValueError, IndexError):
            pass # Continue to check for continuation lines even if header line parsing fails or is empty of values

        # Parse continuation lines (indented lines immediately following the n_header_line_idx)
        current_n_val_line_idx = n_header_line_idx + 1
        while (current_n_val_line_idx < orbital_block_start_idx and # Must be before the orbital itself
               current_n_val_line_idx < selected_ion_block_end_idx and # Must be within the ion block
               lines[current_n_val_line_idx].startswith(" ")):
            
            line_for_n_val = lines[current_n_val_line_idx]
            stripped_line_for_n_val = line_for_n_val.strip()

            if not stripped_line_for_n_val:
                break 
            
            # Stop if it looks like a c, z, or new orbital definition (should not happen if logic is correct for n-continuation)
            if (stripped_line_for_n_val.startswith("c ") or
                stripped_line_for_n_val.startswith("z ") or
                # also check for new n header to avoid overreading into next n block, though less likely with backward search
                stripped_line_for_n_val.startswith("n ") or 
                re.match(r"^\d+[spdfghiklmnoqrtuvwxyz]", stripped_line_for_n_val)):
                break

            try:
                current_ns = [int(x) for x in stripped_line_for_n_val.split()]
                temp_n_list.extend(current_ns)
            except ValueError:
                break 
            current_n_val_line_idx += 1
        
        if temp_n_list:
            n_list_out = temp_n_list
            n_list_found = True

    if not n_list_found:
        pass # Let the length check in calc_orb handle this issue if lists don't match

    current_parsing_state = None
    for i in range(orbital_block_start_idx + 1, selected_ion_block_end_idx):
        line = lines[i]
        stripped_line = line.strip()

        if stripped_line:
            potential_new_orbital_match = re.match(r"^(\d+[spdfghiklmnoqrtuvwxyz])", stripped_line)
            is_new_orbital_header = potential_new_orbital_match and potential_new_orbital_match.group(1) != target_orbital_label
            is_separator = stripped_line.startswith("---")
            is_new_n_header = stripped_line.startswith("n ")

            if is_new_orbital_header or is_separator or (is_new_n_header and current_parsing_state == "parsing_z_values"):
                break
        else: 
            # If an empty line is encountered, and we are in a parsing state for c or z, end that specific list parsing.
            # The outer loop will continue to check next lines for other headers or end of block.
            if current_parsing_state:
                current_parsing_state = None
            else:
                # If not parsing anything and an empty line is found, it could be padding, just continue.
                continue 

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
            elif not stripped_line: 
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
            elif not stripped_line: 
                current_parsing_state = None 
            else: 
                current_parsing_state = None
                
    print(f"n_list_out: {n_list_out}, c_list_out: {c_list_out}, z_list_out: {z_list_out}")
    return flag, n_list_out, c_list_out, z_list_out

def calc_orb(n: int, ell: int, m: int, z_before: float, magnification: int, settings: Settings) -> tuple[float, np.ndarray]:
    has_list, n_list, c_list, z_list = find_ncz_list(n, ell, settings)
    
    if len(n_list) != len(c_list) or len(n_list) != len(z_list):
        print(f"n_list: {n_list}, c_list: {c_list}, z_list: {z_list}")
        error_handler = ErrorHandler()
        error_handler.handle(
            f"The length of n_list, c_list, and z_list must be the same. The length of n_list is {len(n_list)}, the length of c_list is {len(c_list)}, and the length of z_list is {len(z_list)}.",
            ErrorCode.INVALID_INPUT,
            ErrorLevel.ERROR
        )
    
    v = [v // magnification for v in settings.v]
    lattice_params = settings.lattice_params
    r_max = settings.r_max
    psi_list = anp.zeros((v[0], v[1], v[2]))
    z = z_before
    center_idx = settings.center_idx

    if not has_list and z == -1:
        z = calc_Zeff(f"{n}{orbital_name[ell]}", settings)

    for i in range(v[0]):
        for j in range(v[1]):
            for k in range(v[2]):
                pos = anp.array([(i - center_idx[0]) % v[0] / v[0] * lattice_params[0], (j - center_idx[1]) % v[1] / v[1] * lattice_params[1], (k - center_idx[2]) % v[2] / v[2] * lattice_params[2]])
                r = anp.linalg.norm(pos)
                x = pos @ settings.basis_set[0]
                y = pos @ settings.basis_set[1]
                z = pos @ settings.basis_set[2]

                if r > r_max:
                    continue
                
                if r != 0:
                    theta = anp.arccos(z / r)
                else:
                    theta = 0.0
                if anp.isnan(theta):
                    theta = 0.0
                phi = anp.arctan2(y, x)
                if phi < 0:
                    phi += 2 * anp.pi
                if anp.isnan(phi):
                    phi = 0.0
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
        
        radial_part = r**(n-1) * anp.exp(-zeta * r)
        
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
    return rho ** ell * anp.exp(-rho / 2.0) * L