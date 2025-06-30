import numpy as np
from numba import jit
import hashlib
import pickle
import os

from src.tasks.pre_processing.settings import Settings, orbital_magnetic_number
from src.tasks.after_processing.calc_orb_derive import calc_orb_derive
from src.helpers.calc_orb import calc_orb

def generate_oam_tmp_cache_key(orb1: tuple, orb2: tuple, settings: Settings) -> str:
    """
    Generate a unique cache key for OAM tmp calculations.
    """
    key_data = {
        'orb1': orb1, 'orb2': orb2,
        'v': tuple(settings.v), 'lattice_params': tuple(settings.lattice_params),
        'center': tuple(settings.center), 'basis_set': tuple(map(tuple, settings.basis_set)),
        'atom_name': settings.atom_name, 'calc_type': 'oam_tmp'
    }
    key_str = str(sorted(key_data.items()))
    return hashlib.md5(key_str.encode()).hexdigest()

def load_oam_tmp_from_cache(cache_key: str) -> tuple[bool, np.ndarray]:
    """
    Load OAM tmp array from cache if it exists.
    
    Returns:
        (success, tmp_array)
    """
    cache_dir = "cache/oam_tmp"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded OAM tmp from cache: {cache_key[:8]}...")
            return True, data['tmp_array']
        except Exception as e:
            print(f"Warning: Failed to load OAM tmp cache {cache_key[:8]}...: {e}")
            return False, np.array([])
    
    return False, np.array([])

def save_oam_tmp_to_cache(cache_key: str, tmp_array: np.ndarray) -> None:
    """
    Save OAM tmp array to cache.
    """
    cache_dir = "cache/oam_tmp"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    try:
        data = {
            'tmp_array': np.array(tmp_array)
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved OAM tmp to cache: {cache_key[:8]}...")
    except Exception as e:
        print(f"Warning: Failed to save OAM tmp cache {cache_key[:8]}...: {e}")

def compute_oam_tmp(orb1: tuple, orb2: tuple, settings: Settings) -> np.ndarray:
    """
    Compute tmp array for given orbital pair with caching.
    """
    # キャッシュキーの生成
    cache_key = generate_oam_tmp_cache_key(orb1, orb2, settings)
    
    # キャッシュから読み込み試行
    cache_success, cached_tmp = load_oam_tmp_from_cache(cache_key)
    if cache_success:
        return cached_tmp
    
    print(f"Computing OAM tmp for orb1={orb1}, orb2={orb2}")
    
    # 計算実行
    n, ell, m = orb1
    n2, ell2, m2 = orb2
    _, wavefun1 = calc_orb(n, ell, m, -1, 1, settings)
    wavefun2 = calc_orb_derive(n2, ell2, m2, settings)
    
    tmp = np.zeros((settings.v[0], settings.v[1], settings.v[2], 3), dtype=np.complex128)
    for k in range(settings.v[0]):
        for l in range(settings.v[1]):
            for m_idx in range(settings.v[2]):
                tmp[k, l, m_idx] = wavefun1[k, l, m_idx] * wavefun2[k, l, m_idx].conj()
    
    # キャッシュに保存
    save_oam_tmp_to_cache(cache_key, tmp)
    
    return tmp

def create_OAM_basis(P: np.ndarray[tuple[int,int], complex], settings: Settings) -> np.ndarray[tuple[int,int,int], tuple[float, float, float]]:
    res = np.zeros((10, 10, 10, 3), dtype=float)
    # res = np.zeros((settings.v[0], settings.v[1], settings.v[2], 3), dtype=np.complex128)
    orb_set = settings.orbital_set
    orb_set_list = []
    for orb in orb_set:
        n = int(orb[0])
        ell = orbital_magnetic_number[orb[1]]
        for m in range(-ell, ell + 1):
            orb_set_list.append((n, ell, m))
    
    for i, orb in enumerate(orb_set_list):
        for j, orb2 in enumerate(orb_set_list):
            if (orb[2] - orb2[2]) % 3 != 0:
                continue
            
            # キャッシュ機能付きでtmpを計算
            tmp = compute_oam_tmp(orb, orb2, settings)
            res += 1j * P[i, j] * tmp

    return res
