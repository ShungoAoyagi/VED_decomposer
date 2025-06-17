import numpy as np
from numba import jit
import sys
import os
import hashlib
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks.pre_processing.settings import Settings
from src.helpers.calc_orb import calc_R_with_STO_fast, calc_R_with_Zeff_fast, find_ncz_list, orbital_name
from src.helpers.constant import Constants

def generate_orbital_derive_cache_key(n: int, ell: int, m: int, settings: Settings) -> str:
    """
    Generate a unique cache key for orbital derivative calculations.
    """
    # 計算に影響する設定をハッシュ化
    key_data = {
        'n': n, 'ell': ell, 'm': m,
        'v': tuple(settings.v), 'lattice_params': tuple(settings.lattice_params),
        'center': tuple(settings.center), 'basis_set': tuple(map(tuple, settings.basis_set)),
        'atom_name': settings.atom_name, 'calc_type': 'orbital_derive'
    }
    key_str = str(sorted(key_data.items()))
    return hashlib.md5(key_str.encode()).hexdigest()

def load_orbital_derive_from_cache(cache_key: str) -> tuple[bool, np.ndarray]:
    """
    Load orbital derivative from cache if it exists.
    
    Returns:
        (success, derivative_array)
    """
    cache_dir = "cache/orbitals_derive"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded orbital derivative from cache: {cache_key[:8]}...")
            return True, data['derivative_array']
        except Exception as e:
            print(f"Warning: Failed to load derivative cache {cache_key[:8]}...: {e}")
            return False, np.array([])
    
    return False, np.array([])

def save_orbital_derive_to_cache(cache_key: str, derivative_array: np.ndarray) -> None:
    """
    Save orbital derivative to cache.
    """
    cache_dir = "cache/orbitals_derive"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    try:
        data = {
            'derivative_array': np.array(derivative_array)
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved orbital derivative to cache: {cache_key[:8]}...")
    except Exception as e:
        print(f"Warning: Failed to save derivative cache {cache_key[:8]}...: {e}")

@jit(nopython=True, cache=True)
def calc_orb_derive_core(n: int, ell: int, m: int, v: tuple, center: tuple, lattice_params: np.ndarray, 
                        basis_set: np.ndarray, flag: bool, n_list: np.ndarray, c_list: np.ndarray, 
                        z_list: np.ndarray) -> np.ndarray:
    """
    Core function for orbital derivative calculation with Numba optimization.
    """
    res = np.zeros((v[0], v[1], v[2], 3), dtype=np.complex128)
    center_idx = np.array([v[0] * center[0], v[1] * center[1], v[2] * center[2]])

    for i in range(v[0]):
        for j in range(v[1]):
            for k in range(v[2]):
                pos_x = ((i - center_idx[0]) % v[0]) * lattice_params[0]
                pos_y = ((j - center_idx[1]) % v[1]) * lattice_params[1]
                pos_z = ((k - center_idx[2]) % v[2]) * lattice_params[2]
                x = pos_x * basis_set[0][0] + pos_y * basis_set[0][1] + pos_z * basis_set[0][2]
                y = pos_x * basis_set[1][0] + pos_y * basis_set[1][1] + pos_z * basis_set[1][2]
                z = pos_x * basis_set[2][0] + pos_y * basis_set[2][1] + pos_z * basis_set[2][2]
                r = np.linalg.norm(np.array([x, y, z]))
                if np.abs(r) < 1e-10:
                    theta = 0
                    phi = 0
                else:
                    theta = np.arccos(z / r)
                    phi = np.arctan2(y, x)
                
                if flag:
                    R_r = calc_R_with_STO_fast(n_list, c_list, z_list, r, 0.529177249)  # Constants.a0_angstrom
                else:
                    R_r = calc_R_with_Zeff_fast(n, ell, -1, r, 0.529177249)

                f_r = 0.0 + 0.0j
                f_theta = -R_r * phi_derive_sph(ell, m, theta, phi)
                f_phi = R_r * theta_derive_sph(ell, m, theta, phi)
                
                # \bm{r} \times \nabla \phi(r)
                f_x = f_r * np.sin(theta) * np.cos(phi) + f_theta * np.cos(theta) * np.cos(phi) - f_phi * np.sin(phi)
                f_y = f_r * np.sin(theta) * np.sin(phi) + f_theta * np.cos(theta) * np.sin(phi) + f_phi * np.cos(phi)
                f_z = f_r * np.cos(theta) - f_theta * np.sin(theta)

                res[i, j, k, 0] = f_x
                res[i, j, k, 1] = f_y
                res[i, j, k, 2] = f_z
                
    return res

def calc_orb_derive(n: int, ell: int, m: int, settings: Settings) -> np.ndarray:
    """
    Calculate orbital derivatives with caching mechanism.
    """
    # キャッシュキーの生成
    cache_key = generate_orbital_derive_cache_key(n, ell, m, settings)
    
    # キャッシュから読み込み試行
    cache_success, cached_derivative = load_orbital_derive_from_cache(cache_key)
    if cache_success:
        return cached_derivative
    
    print(f"Computing orbital derivative n={n}, ell={ell}, m={m}, grid size={settings.v}")
    
    # 必要なデータの準備
    flag, n_list, c_list, z_list = find_ncz_list(n, ell, settings)
    
    # Numba用に配列を変換
    n_list_array = np.array(n_list, dtype=np.int32) if flag else np.array([n], dtype=np.int32)
    c_list_array = np.array(c_list, dtype=np.float64) if flag else np.array([1.0], dtype=np.float64)
    z_list_array = np.array(z_list, dtype=np.float64) if flag else np.array([1.0], dtype=np.float64)
    
    # コア計算の実行
    result = calc_orb_derive_core(
        n, ell, m, 
        tuple(settings.v), 
        tuple(settings.center),
        np.array(settings.lattice_params, dtype=np.float64),
        np.array(settings.basis_set, dtype=np.float64),
        flag,
        n_list_array,
        c_list_array, 
        z_list_array
    )
    
    # キャッシュに保存
    save_orbital_derive_to_cache(cache_key, result)
    
    return result

@jit(nopython=True, cache=True)
def theta_derive_sph(ell: int, m: int, theta: float, phi: float) -> complex:
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    if ell == 0:  # s orbital (m=0)
        # \pdv{Y_0^0}{theta} = 0
        return 0
    
    elif ell == 1:  # p orbitals
        if m == -1:
            # Y_1^(-1) = sqrt(3/(8π)) * sin(θ) * e^(-iφ) = sqrt(3/(8π)) * sin(θ) * (cos(φ) - i*sin(φ))
            # \pdv{Y_1^(-1)}{theta} = sqrt(3/(8π)) * cos(θ) * e^(-iφ)
            normalization = 0.34549414947134  # sqrt(3/(8π))
            res = normalization * cos_theta * np.exp(1j * m * phi)
            return res
        elif m == 0:
            # Y_1^0 = sqrt(3/(4π)) * cos(θ)
            # \pdv{Y_1^0}{theta} = -sqrt(3/(4π)) * sin(θ)
            res = -0.4886025119029199 * sin_theta
            return res
        elif m == 1:
            # Y_1^1 = -sqrt(3/(8π)) * sin(θ) * e^(iφ) = -sqrt(3/(8π)) * sin(θ) * (cos(φ) + i*sin(φ))
            # \pdv{Y_1^1}{theta} = -sqrt(3/(8π)) * cos(θ) * e^(iφ)
            normalization = -0.34549414947134  # -sqrt(3/(8π))
            res = normalization * cos_theta * np.exp(1j * m * phi)
            return res
    
    elif ell == 2:  # d orbitals
        sin2_theta = sin_theta * sin_theta
        cos2_theta = cos_theta * cos_theta
        
        if m == -2:
            # Y_2^(-2) = sqrt(15/(32π)) * sin²(θ) * e^(-2iφ)
            # \pdv{Y_2^(-2)}{theta} = 2 * sqrt(15/(32π)) * sin(θ) * cos(θ) * e^(-2iφ)
            normalization = 0.3862742020219  # sqrt(15/(32π))
            res = 2 * normalization * sin_theta * cos_theta * np.exp(1j * m * phi)
            return res
        elif m == -1:
            # Y_2^(-1) = sqrt(15/(8π)) * sin(θ)*cos(θ) * e^(-iφ)
            # \pdv{Y_2^(-1)}{theta} = sqrt(15/(8π)) * (cos^2(θ) - sin^2(θ)) * e^(-iφ)
            normalization = 0.77254840404638  # sqrt(15/(8π))
            res = normalization * (cos2_theta - sin2_theta) * np.exp(1j * m * phi)
            return res
        elif m == 0:
            # Y_2^0 = sqrt(5/(16π)) * (3*cos²(θ) - 1)
            # \pdv{Y_2^0}{theta} = -sqrt(5/(16π)) * 6 * cos(θ) * sin(θ)
            res = -0.31539156525252005 * 6 * cos_theta * sin_theta
            return res
        elif m == 1:
            # Y_2^1 = -sqrt(15/(8π)) * sin(θ)*cos(θ) * e^(iφ)
            # \pdv{Y_2^1}{theta} = -sqrt(15/(8π)) * (cos^2(θ) - sin^2(θ)) * e^(iφ)
            normalization = -0.77254840404638  # -sqrt(15/(8π))
            res = normalization * (cos2_theta - sin2_theta) * np.exp(1j * m * phi)
            return res
        elif m == 2:
            # Y_2^2 = sqrt(15/(32π)) * sin²(θ) * e^(2iφ)
            # \pdv{Y_2^2}{theta} = sqrt(15/(32π)) * 2 * sin(θ) * cos(θ) * e^(2iφ)
            normalization = 0.3862742020219  # sqrt(15/(32π))
            res = normalization * 2 * sin_theta * cos_theta * np.exp(1j * m * phi)
            return res
    
    # より複雑な場合は従来の関数を使用（プレースホルダー）
    return 1.0 + 0.0j
    
@jit(nopython=True, cache=True)
def phi_derive_sph(ell: int, m: int, theta: float, phi: float) -> complex:
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    if ell == 0:  # s orbital (m=0)
        # \pdv{Y_0^0}{phi} = 0
        return 0
    
    elif ell == 1:  # p orbitals
        if m == -1:
            # Y_1^(-1) = sqrt(3/(8π)) * sin(θ) * e^(-iφ)
            # \pdv{Y_1^(-1)}{phi} = -i * sqrt(3/(8π)) * sin(θ) * e^(-iφ)
            normalization = 0.34549414947134  # sqrt(3/(8π))
            res = -1j * normalization * np.exp(1j * m * phi)
            return res
        elif m == 0:
            # Y_1^0 = sqrt(3/(4π)) * cos(θ)
            # \pdv{Y_1^0}{phi} = 0
            return 0
        elif m == 1:
            # Y_1^1 = -sqrt(3/(8π)) * sin(θ) * e^(iφ) = -sqrt(3/(8π)) * sin(θ) * (cos(φ) + i*sin(φ))
            # \pdv{Y_1^1}{phi} = -i * sqrt(3/(8π)) * sin(θ) * e^(iφ)
            normalization = -0.34549414947134  # -sqrt(3/(8π))
            res = -1j * normalization * np.exp(1j * m * phi)
            return res
    
    elif ell == 2:  # d orbitals
        if m == -2:
            # Y_2^(-2) = sqrt(15/(32π)) * sin²(θ) * e^(-2iφ)
            # \pdv{Y_2^(-2)}{phi} = -2i * sqrt(15/(32π)) * sin²(θ)* e^(-2iφ)
            normalization = 0.3862742020219  # sqrt(15/(32π))
            res = -2j * normalization * sin_theta * np.exp(1j * m * phi)
            return res
        elif m == -1:
            # Y_2^(-1) = sqrt(15/(8π)) * sin(θ)*cos(θ) * e^(-iφ)
            # \pdv{Y_2^(-1)}{phi} = -i * sqrt(15/(8π)) * sin(θ)*cos(θ) * e^(-iφ)
            normalization = 0.77254840404638  # sqrt(15/(8π))
            res = -1j * normalization * cos_theta * np.exp(1j * m * phi)
            return res
        elif m == 0:
            # Y_2^0 = sqrt(5/(16π)) * (3*cos²(θ) - 1)
            # \pdv{Y_2^0}{phi} = 0
            return 0
        elif m == 1:
            # Y_2^1 = -sqrt(15/(8π)) * sin(θ)*cos(θ) * e^(iφ)
            # \pdv{Y_2^1}{phi} = -i * sqrt(15/(8π)) * sin(θ)*cos(θ) * e^(iφ)
            normalization = -0.77254840404638  # -sqrt(15/(8π))
            res = 1j * normalization * cos_theta * np.exp(1j * m * phi)
            return res
        elif m == 2:
            # Y_2^2 = sqrt(15/(32π)) * sin²(θ) * e^(2iφ)
            # \pdv{Y_2^2}{phi} = 2i * sqrt(15/(32π)) * sin²(θ)* e^(2iφ)
            normalization = 0.3862742020219  # sqrt(15/(32π))
            res = 2j * normalization * sin_theta * np.exp(1j * m * phi)
            return res
    
    # より複雑な場合は従来の関数を使用（プレースホルダー）
    return 1.0 + 0.0j
