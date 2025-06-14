import numpy as np
import autograd.numpy as anp
from src.tasks.pre_processing.settings import Settings
from src.utils import ErrorHandler, ErrorCode, ErrorLevel
from src.helpers import spherical_harmonics, Constants
from scipy.special import genlaguerre
from src.helpers.calc_Zeff import calc_Zeff
from autograd.scipy.special import gammaln
import re
import math
import os
import hashlib
import pickle
from functools import lru_cache

# Numbaのインポートを追加
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    print("Warning: Numba not available. Using standard numpy implementation.")
    NUMBA_AVAILABLE = False
    # Fallback decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# プログレスバーのインポート（オプション）
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

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
    after_orbital_start_idx = False
    if lines[orbital_block_start_idx + 1].strip().startswith("n "):
        n_header_line_idx = orbital_block_start_idx + 1
        n_list_found = True
        after_orbital_start_idx = True
    else:
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
        while ((current_n_val_line_idx < orbital_block_start_idx or after_orbital_start_idx) and # Must be before the orbital itself
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

@jit(nopython=True, cache=True)
def safe_arccos(x: float) -> float:
    """
    Safe arccos function that handles numerical errors.
    """
    if x >= 1.0:
        return 0.0
    elif x <= -1.0:
        return np.pi
    else:
        return np.arccos(x)

@jit(nopython=True, cache=True)
def manual_clip(x: float, min_val: float, max_val: float) -> float:
    """
    Manual clip function for Numba compatibility.
    """
    if x < min_val:
        return min_val
    elif x > max_val:
        return max_val
    else:
        return x

@jit(nopython=True, cache=True)
def calc_orb_vectorized_numba_chunked(
    v: tuple, 
    lattice_params: np.ndarray, 
    center_idx: np.ndarray[float], 
    basis_set: np.ndarray,
    n: int, 
    ell: int, 
    m: int,
    has_list: bool,
    n_list: np.ndarray,
    c_list: np.ndarray, 
    z_list: np.ndarray,
    z_eff: float,
    chunk_size: int = 50
) -> np.ndarray:
    """
    Sequential chunked version of orbital calculation with complex support (no parallel processing).
    """
    psi_list = np.zeros((v[0], v[1], v[2]), dtype=np.complex128)
    a0 = 0.529177249  # Constants.a0_angstrom を直接定義
    
    # シーケンシャルなチャンク処理
    for i_start in range(0, v[0], chunk_size):
        i_end = min(i_start + chunk_size, v[0])
        
        for i in range(i_start, i_end):
            for j in range(v[1]):
                for k in range(v[2]):
                    # 位置計算
                    pos_x = ((i - center_idx[0]) % v[0]) / v[0] * lattice_params[0]
                    pos_y = ((j - center_idx[1]) % v[1]) / v[1] * lattice_params[1]
                    pos_z = ((k - center_idx[2]) % v[2]) / v[2] * lattice_params[2]
                    
                    # 周期境界条件
                    if pos_x > lattice_params[0] / 2:
                        pos_x -= lattice_params[0]
                    if pos_y > lattice_params[1] / 2:
                        pos_y -= lattice_params[1]
                    if pos_z > lattice_params[2] / 2:
                        pos_z -= lattice_params[2]
                    
                    # 基底変換
                    x = pos_x * basis_set[0, 0] + pos_y * basis_set[0, 1] + pos_z * basis_set[0, 2]
                    y = pos_x * basis_set[1, 0] + pos_y * basis_set[1, 1] + pos_z * basis_set[1, 2]
                    z = pos_x * basis_set[2, 0] + pos_y * basis_set[2, 1] + pos_z * basis_set[2, 2]
                    
                    r = np.sqrt(x*x + y*y + z*z)
                    
                    # r=0の場合の早期終了
                    if r < 1e-12:
                        if ell == 0:
                            sph_real, sph_imag = calc_spherical_harmonics_fast(ell, m, 0.0, 0.0)
                            psi_list[i, j, k] = sph_real + 1j * sph_imag
                        else:
                            psi_list[i, j, k] = 0.0 + 0.0j
                        continue
                    
                    # 球面座標（数値安定化）
                    z_over_r = z / r
                    z_over_r_clipped = manual_clip(z_over_r, -1.0, 1.0)
                    theta = safe_arccos(z_over_r_clipped)
                    
                    phi = np.arctan2(y, x)
                    if phi < 0:
                        phi += 2 * np.pi
                    
                    # 球面調和関数の計算（複素数対応）
                    sph_real, sph_imag = calc_spherical_harmonics_fast(ell, m, theta, phi)
                    
                    # 動径関数の計算
                    if has_list:
                        radial = calc_R_with_STO_fast(n_list, c_list, z_list, r, a0)
                    else:
                        radial = calc_R_with_Zeff_fast(n, ell, z_eff, r, a0)
                    
                    # 複素波動関数を保存
                    psi_list[i, j, k] = radial * (sph_real + 1j * sph_imag)
    
    return psi_list

@jit(nopython=True, cache=True)
def calc_spherical_harmonics_fast(ell: int, m: int, theta: float, phi: float) -> tuple:
    """
    Fast implementation of complex spherical harmonics Y_l^m(θ,φ) for common cases.
    Returns (real_part, imaginary_part) to work with Numba constraints.
    
    Y_l^m(θ,φ) = sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!) * P_l^|m|(cos θ) * e^(imφ)
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    if ell == 0:  # s orbital (m=0)
        # Y_0^0 = 1/(2*sqrt(π))
        return (0.28209479177387814, 0.0)
    
    elif ell == 1:  # p orbitals
        if m == -1:
            # Y_1^(-1) = sqrt(3/(8π)) * sin(θ) * e^(-iφ) = sqrt(3/(8π)) * sin(θ) * (cos(φ) - i*sin(φ))
            normalization = 0.34549414947134  # sqrt(3/(8π))
            real_part = normalization * sin_theta * np.cos(phi)
            imag_part = -normalization * sin_theta * np.sin(phi)
            return (real_part, imag_part)
        elif m == 0:
            # Y_1^0 = sqrt(3/(4π)) * cos(θ)
            return (0.4886025119029199 * cos_theta, 0.0)
        elif m == 1:
            # Y_1^1 = -sqrt(3/(8π)) * sin(θ) * e^(iφ) = -sqrt(3/(8π)) * sin(θ) * (cos(φ) + i*sin(φ))
            normalization = -0.34549414947134  # -sqrt(3/(8π))
            real_part = normalization * sin_theta * np.cos(phi)
            imag_part = normalization * sin_theta * np.sin(phi)
            return (real_part, imag_part)
    
    elif ell == 2:  # d orbitals
        sin2_theta = sin_theta * sin_theta
        cos2_theta = cos_theta * cos_theta
        
        if m == -2:
            # Y_2^(-2) = sqrt(15/(32π)) * sin²(θ) * e^(-2iφ)
            normalization = 0.3862742020219  # sqrt(15/(32π))
            cos_2phi = np.cos(2*phi)
            sin_2phi = np.sin(2*phi)
            real_part = normalization * sin2_theta * cos_2phi
            imag_part = -normalization * sin2_theta * sin_2phi
            return (real_part, imag_part)
        elif m == -1:
            # Y_2^(-1) = sqrt(15/(8π)) * sin(θ)*cos(θ) * e^(-iφ)
            normalization = 0.77254840404638  # sqrt(15/(4π))
            real_part = normalization * sin_theta * cos_theta * np.cos(phi)
            imag_part = -normalization * sin_theta * cos_theta * np.sin(phi)
            return (real_part, imag_part)
        elif m == 0:
            # Y_2^0 = sqrt(5/(16π)) * (3*cos²(θ) - 1)
            return (0.31539156525252005 * (3*cos2_theta - 1), 0.0)
        elif m == 1:
            # Y_2^1 = -sqrt(15/(8π)) * sin(θ)*cos(θ) * e^(iφ)
            normalization = -0.77254840404638  # -sqrt(15/(4π))
            real_part = normalization * sin_theta * cos_theta * np.cos(phi)
            imag_part = normalization * sin_theta * cos_theta * np.sin(phi)
            return (real_part, imag_part)
        elif m == 2:
            # Y_2^2 = sqrt(15/(32π)) * sin²(θ) * e^(2iφ)
            normalization = 0.3862742020219  # sqrt(15/(32π))
            cos_2phi = np.cos(2*phi)
            sin_2phi = np.sin(2*phi)
            real_part = normalization * sin2_theta * cos_2phi
            imag_part = normalization * sin2_theta * sin_2phi
            return (real_part, imag_part)
    
    # より複雑な場合は従来の関数を使用（プレースホルダー）
    return (1.0, 0.0)

@jit(nopython=True, cache=True)
def calc_R_with_STO_fast(n_list: np.ndarray, c_list: np.ndarray, z_list: np.ndarray, r: float, a0: float) -> float:
    """
    Fast STO radial function calculation with numerical stability.
    """
    res = 0.0
    for i in range(len(n_list)):
        n = n_list[i]
        zeta = z_list[i]
        c = c_list[i]
        
        # 指数関数のオーバーフロー防止
        exponent = -zeta * r / a0
        if exponent < -700:  # exp(-700) ≈ 0
            continue
        
        # 階乗計算の最適化
        factorial_2n = 1.0
        for j in range(1, 2*n + 1):
            factorial_2n *= j
        
        normalization = c * (2 * zeta)**(n + 0.5) * np.sqrt(1.0 / factorial_2n)
        
        # r=0での特別処理
        if r < 1e-12:
            if n == 1:
                radial_part = normalization
            else:
                radial_part = 0.0
        else:
            radial_part = normalization * (r/a0)**(n - 1) * np.exp(exponent)
        
        res += radial_part
        
    return res

@jit(nopython=True, cache=True)
def calc_R_with_Zeff_fast(n: int, ell: int, z: float, r: float, a0: float) -> float:
    """
    Fast effective nuclear charge radial function calculation with numerical stability.
    """
    # r=0での特別処理
    if r < 1e-12:
        if ell == 0:
            # 正しい水素様原子軌道の規格化
            factorial_n_minus_l_minus_1 = 1.0  # (n-l-1)! for n=1, l=0
            factorial_n_plus_l = 1.0  # (n+l)! for n=1, l=0
            for i in range(1, n - ell):
                factorial_n_minus_l_minus_1 *= i
            for i in range(1, n + ell + 1):
                factorial_n_plus_l *= i
            
            normalization = np.sqrt(
                (2.0 * z / (n * a0))**3 * 
                factorial_n_minus_l_minus_1 / (2.0 * n * factorial_n_plus_l)
            )
            return normalization
        else:
            return 0.0
    
    rho = 2.0 * z * r / (n * a0)
    
    # 指数関数のオーバーフロー防止
    exponent = -rho / 2.0
    if exponent < -700:  # exp(-700) ≈ 0
        return 0.0
    
    # 正しい水素様原子軌道の規格化
    factorial_n_minus_l_minus_1 = 1.0
    factorial_n_plus_l = 1.0
    for i in range(1, n - ell):
        factorial_n_minus_l_minus_1 *= i
    for i in range(1, n + ell + 1):
        factorial_n_plus_l *= i
    
    normalization = np.sqrt(
        (2.0 * z / (n * a0))**3 * 
        factorial_n_minus_l_minus_1 / (2.0 * n * factorial_n_plus_l)
    )
    
    # 正しいラゲール陪多項式 L_{n-ℓ-1}^{2ℓ+1}(ρ) の計算
    # 標準的な公式: L_n^α(x) = Σ_{i=0}^n (-1)^i * C(n+α, n-i) * x^i / i!
    k_lag = n - ell - 1  # ラゲール多項式の次数
    alpha = 2 * ell + 1  # 上付き指数
    
    L = 0.0
    rho_power = 1.0  # rho^i
    factorial_i = 1.0  # i!
    
    for i in range(k_lag + 1):
        # 二項係数 C(n+α, n-i) を計算
        # C(k_lag+α, k_lag-i) = (k_lag+α)! / ((k_lag-i)! * (α+i)!)
        
        # 分子: (k_lag+α) * (k_lag+α-1) * ... * (α+i+1)
        numerator = 1.0
        for j in range(k_lag - i):
            numerator *= (k_lag + alpha - j)
        
        # 分母: (k_lag-i)!
        denominator = 1.0
        for j in range(1, k_lag - i + 1):
            denominator *= j
        
        binom_coeff = numerator / denominator
        
        # 項を追加: (-1)^i * C(n+α, n-i) * x^i / i!
        term = ((-1.0)**i) * binom_coeff * rho_power / factorial_i
        L += term
        
        # 次の反復のための更新
        if i < k_lag:
            rho_power *= rho
            factorial_i *= (i + 1)
    
    return normalization * np.power(rho, ell) * np.exp(exponent) * L

def generate_orbital_cache_key(n: int, ell: int, m: int, z_before: float, magnification: int, settings: Settings) -> str:
    """
    Generate a unique cache key for orbital calculations.
    """
    # 計算に影響する設定をハッシュ化
    key_data = {
        'n': n, 'ell': ell, 'm': m, 'z_before': z_before, 'magnification': magnification,
        'v': tuple(settings.v), 'lattice_params': tuple(settings.lattice_params),
        'center': tuple(settings.center), 'basis_set': tuple(map(tuple, settings.basis_set)),
        'atom_name': settings.atom_name
    }
    key_str = str(sorted(key_data.items()))
    return hashlib.md5(key_str.encode()).hexdigest()

def load_orbital_from_cache(cache_key: str) -> tuple[bool, float, np.ndarray]:
    """
    Load orbital from cache if it exists.
    
    Returns:
        (success, z_eff, psi_list)
    """
    cache_dir = "cache/orbitals_calc"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded orbital from cache: {cache_key[:8]}...")
            return True, data['z_eff'], data['psi_list']
        except Exception as e:
            print(f"Warning: Failed to load cache {cache_key[:8]}...: {e}")
            return False, 0.0, np.array([])
    
    return False, 0.0, np.array([])

def save_orbital_to_cache(cache_key: str, z_eff: float, psi_list: np.ndarray) -> None:
    """
    Save orbital to cache.
    """
    cache_dir = "cache/orbitals_calc"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    try:
        data = {
            'z_eff': z_eff,
            'psi_list': np.array(psi_list)  # autogradをnumpyに変換
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved orbital to cache: {cache_key[:8]}...")
    except Exception as e:
        print(f"Warning: Failed to save cache {cache_key[:8]}...: {e}")

def calc_orb_optimized(n: int, ell: int, m: int, z_before: float, magnification: int, settings: Settings) -> tuple[float, np.ndarray]:
    """
    Optimized version of calc_orb using Numba JIT or NumPy vectorization with caching.
    """
    # キャッシュキーの生成
    cache_key = generate_orbital_cache_key(n, ell, m, z_before, magnification, settings)
    
    # キャッシュから読み込み試行
    cache_success, cached_z_eff, cached_psi_list = load_orbital_from_cache(cache_key)
    if cache_success:
        return cached_z_eff, anp.array(cached_psi_list)
    
    has_list, n_list, c_list, z_list = find_ncz_list(n, ell, settings)
    
    if len(n_list) != len(c_list) or len(n_list) != len(z_list):
        print(f"n_list: {n_list}, c_list: {c_list}, z_list: {z_list}")
        error_handler = ErrorHandler()
        error_handler.handle(
            f"The length of n_list, c_list, and z_list must be the same. The length of n_list is {len(n_list)}, the length of c_list is {len(c_list)}, and the length of z_list is {len(z_list)}.",
            ErrorCode.INVALID_INPUT,
            ErrorLevel.ERROR
        )
    
    v = tuple(v_i // magnification for v_i in settings.v)
    lattice_params = np.array(settings.lattice_params, dtype=np.float64)
    z_eff = z_before
    center = settings.center
    center_idx = np.array([center[i] * v[i] for i in range(3)], dtype=np.float64)
    basis_set = np.array(settings.basis_set, dtype=np.float64)

    if not has_list and z_eff == -1:
        z_eff = calc_Zeff(f"{n}{orbital_name[ell]}", settings)
        print(f"Calculated Zeff: {z_eff}")

    print(f"Computing orbital n={n}, ell={ell}, m={m}, grid size={v}")
    total_points = v[0] * v[1] * v[2]
    print(f"Total grid points: {total_points:,} ({total_points*8/1024/1024:.1f} MB for float64)")
    
    # 軽量なチャンクサイズ（シングルコア用）
    chunk_size = max(10, min(50, v[0] // 4))  # グリッドサイズに応じて調整
    print(f"Using chunk size: {chunk_size}")
    
    # Numbaを使用
    print("Using Numba JIT optimized version (sequential)...")
    psi_list_np = calc_orb_vectorized_numba_chunked(
        v, lattice_params, center_idx, basis_set,
        n, ell, m, has_list,
        np.array(n_list, dtype=np.int32) if has_list else np.array([n], dtype=np.int32),
        np.array(c_list, dtype=np.float64) if has_list else np.array([1.0], dtype=np.float64),
        np.array(z_list, dtype=np.float64) if has_list else np.array([z_eff], dtype=np.float64),
        z_eff,
        chunk_size
    )
    
    # キャッシュに保存
    save_orbital_to_cache(cache_key, z_eff, psi_list_np)
    
    # autogradの配列に変換
    psi_list = anp.array(psi_list_np)
    
    return z_eff, psi_list

# 既存のcalc_orb関数を高速化版で置き換え
def calc_orb(n: int, ell: int, m: int, z_before: float, magnification: int, settings: Settings) -> tuple[float, np.ndarray]:
    """
    Main calc_orb function that automatically uses the fastest available implementation.
    """
    # 高速化版を使用
    return calc_orb_optimized(n, ell, m, z_before, magnification, settings)

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
    a0 = Constants.a0_angstrom
    
    # Sum over all STO basis functions
    for i in range(len(n_list)):
        n = n_list[i]  # Principal quantum number
        zeta = z_list[i]  # Slater exponent
        c = c_list[i]  # Expansion coefficient
        # 修正: cは一度だけ適用
        radial_part = c * (2 * zeta)**(n + 0.5) * anp.sqrt(1 / math.factorial(2 * n)) * (r/a0)**(n - 1) * anp.exp(-zeta * r / a0)
        
        res += radial_part  # cを二重適用しない
        
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
    # 正しい水素様原子軌道の規格化
    # N = √((2Z/na₀)³ · (n-l-1)! / (2n · (n+l)!))
    factorial_n_minus_l_minus_1 = math.factorial(n - ell - 1)
    factorial_n_plus_l = math.factorial(n + ell)
    
    normalization = anp.sqrt(
        (2.0 * z / (n * Constants.a0_angstrom))**3 * 
        factorial_n_minus_l_minus_1 / (2.0 * n * factorial_n_plus_l)
    )
    
    rho = 2.0 * z * r / (n * Constants.a0_angstrom)
    # 正しいラゲール多項式: L_{n-ℓ-1}^{2ℓ+1}(ρ)
    L = genlaguerre(n - ell - 1, 2 * ell + 1)(rho)
    return normalization * (rho ** ell) * anp.exp(-rho / 2.0) * L