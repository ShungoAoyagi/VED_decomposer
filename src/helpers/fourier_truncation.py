import numpy as np
# It's good practice to import autograd.numpy if it might be involved with inputs,
# though this function aims to work with plain numpy arrays internally after unboxing.
import autograd.numpy as anp 
from src.tasks.pre_processing.settings import Settings
from autograd.extend import primitive, defvjp
import os
import hashlib
import pickle
from functools import lru_cache

# 高速FFTライブラリの選択
try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as fft_backend
    PYFFTW_AVAILABLE = True
    print("Using PyFFTW for fast FFT operations")
    
    # FFTWの最適化設定
    pyfftw.config.NUM_THREADS = os.cpu_count()
    pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
    
except ImportError:
    try:
        import mkl_fft
        fft_backend = mkl_fft
        PYFFTW_AVAILABLE = False
        print("Using MKL FFT for fast FFT operations")
    except ImportError:
        import numpy.fft as fft_backend
        PYFFTW_AVAILABLE = False
        print("Using standard NumPy FFT")

# Numba JIT support
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# プログレスバー
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

@lru_cache(maxsize=32)
def precompute_fft_frequencies(nx: int, ny: int, nz: int, Lx: float, Ly: float, Lz: float) -> tuple:
    """
    FFT周波数を事前計算してキャッシュする（高速化）
    """
    kx_vals = np.fft.fftfreq(nx, d=Lx/nx) * 2 * np.pi
    ky_vals = np.fft.fftfreq(ny, d=Ly/ny) * 2 * np.pi
    kz_vals = np.fft.fftfreq(nz, d=Lz/nz) * 2 * np.pi
    return kx_vals, ky_vals, kz_vals

@jit(nopython=True, cache=True)
def create_filter_mask_numba(nx: int, ny: int, nz: int, kx_vals: np.ndarray, ky_vals: np.ndarray, kz_vals: np.ndarray, k_cutoff_sq: float) -> np.ndarray:
    """
    Numba JITでフィルターマスクを高速作成
    """
    filter_mask = np.zeros((nx, ny, nz), dtype=np.float64)
    
    for i in range(nx):
        kx = kx_vals[i]
        kx_sq = kx * kx
        for j in range(ny):
            ky = ky_vals[j]
            ky_sq = ky * ky
            for k in range(nz):
                kz = kz_vals[k]
                K_squared = kx_sq + ky_sq + kz * kz
                if K_squared <= k_cutoff_sq:
                    filter_mask[i, j, k] = 1.0
                
    return filter_mask

def create_filter_mask_vectorized(nx: int, ny: int, nz: int, kx_vals: np.ndarray, ky_vals: np.ndarray, kz_vals: np.ndarray, k_cutoff_sq: float) -> np.ndarray:
    """
    NumPyベクトル化でフィルターマスクを作成
    """
    # 効率的なmeshgrid
    Kx, Ky, Kz = np.meshgrid(kx_vals, ky_vals, kz_vals, indexing='ij')
    K_squared = Kx**2 + Ky**2 + Kz**2
    return (K_squared <= k_cutoff_sq).astype(np.float64)

@jit(nopython=True, cache=True)
def create_spatial_mask_numba(nx: int, ny: int, nz: int, center_idx: tuple, Lx: float, Ly: float, Lz: float, r_max: float) -> np.ndarray:
    """
    Numba JITで空間マスクを高速作成
    """
    r_mask = np.zeros((nx, ny, nz), dtype=np.float64)
    r_max_sq = r_max * r_max
    
    for i in range(nx):
        dx = i - center_idx[0]
        # 最小イメージ規約
        if dx > nx / 2:
            dx -= nx
        elif dx < -nx / 2:
            dx += nx
        x_pos = dx / nx * Lx
        x_pos_sq = x_pos * x_pos
        
        for j in range(ny):
            dy = j - center_idx[1]
            # 最小イメージ規約
            if dy > ny / 2:
                dy -= ny
            elif dy < -ny / 2:
                dy += ny
            y_pos = dy / ny * Ly
            y_pos_sq = y_pos * y_pos
            
            for k in range(nz):
                dz = k - center_idx[2]
                # 最小イメージ規約
                if dz > nz / 2:
                    dz -= nz
                elif dz < -nz / 2:
                    dz += nz
                z_pos = dz / nz * Lz
                
                r_sq = x_pos_sq + y_pos_sq + z_pos * z_pos
                if r_sq <= r_max_sq:
                    r_mask[i, j, k] = 1.0
                    
    return r_mask

def create_spatial_mask_vectorized(nx: int, ny: int, nz: int, center_idx: tuple, Lx: float, Ly: float, Lz: float, r_max: float) -> np.ndarray:
    """
    NumPyベクトル化で空間マスクを作成
    """
    # 座標グリッド
    i_coords = np.arange(nx)
    j_coords = np.arange(ny)
    k_coords = np.arange(nz)
    i_grid, j_grid, k_grid = np.meshgrid(i_coords, j_coords, k_coords, indexing='ij')
    
    # 中心からの相対位置
    dx = i_grid - center_idx[0]
    dy = j_grid - center_idx[1]
    dz = k_grid - center_idx[2]
    
    # 最小イメージ規約
    dx = np.where(dx > nx/2, dx - nx, dx)
    dx = np.where(dx < -nx/2, dx + nx, dx)
    dy = np.where(dy > ny/2, dy - ny, dy)
    dy = np.where(dy < -ny/2, dy + ny, dy)
    dz = np.where(dz > nz/2, dz - nz, dz)
    dz = np.where(dz < -nz/2, dz + nz, dz)
    
    # 実空間座標
    x_pos = dx / nx * Lx
    y_pos = dy / ny * Ly
    z_pos = dz / nz * Lz
    
    # 半径とマスク
    r_grid = np.sqrt(x_pos**2 + y_pos**2 + z_pos**2)
    return (r_grid <= r_max).astype(np.float64)

def generate_fourier_cache_key(shape: tuple, Lx: float, Ly: float, Lz: float, d_min: float, r_max: float, center: tuple) -> str:
    """
    フーリエ変換のキャッシュキーを生成
    """
    key_data = {
        'shape': shape,
        'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
        'd_min': d_min, 'r_max': r_max,
        'center': center
    }
    key_str = str(sorted(key_data.items()))
    return hashlib.md5(key_str.encode()).hexdigest()

def load_fourier_masks_from_cache(cache_key: str) -> tuple[bool, np.ndarray, np.ndarray]:
    """
    フーリエマスクをキャッシュから読み込み
    """
    cache_dir = "cache/fourier_masks"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return True, data['filter_mask'], data['spatial_mask']
        except Exception as e:
            print(f"Warning: Failed to load fourier cache {cache_key[:8]}...: {e}")
            return False, np.array([]), np.array([])
    
    return False, np.array([]), np.array([])

def save_fourier_masks_to_cache(cache_key: str, filter_mask: np.ndarray, spatial_mask: np.ndarray) -> None:
    """
    フーリエマスクをキャッシュに保存
    """
    cache_dir = "cache/fourier_masks"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    try:
        data = {
            'filter_mask': filter_mask,
            'spatial_mask': spatial_mask
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Warning: Failed to save fourier cache {cache_key[:8]}...: {e}")

def fourier_truncation_optimized(f_in: np.ndarray, settings: Settings) -> np.ndarray:
    """
    高速化されたフーリエ切断関数（NumPy版、autogradとは分離）
    """
    nx, ny, nz = f_in.shape
    
    if len(settings.lattice_params) >= 3:
        Lx = settings.lattice_params[0]
        Ly = settings.lattice_params[1]
        Lz = settings.lattice_params[2]
    else:
        raise ValueError("lattice_params must contain at least a, b, c values for physical dimensions.")
    
    # キャッシュキーの生成
    center_idx = tuple(settings.center[i] * f_in.shape[i] for i in range(3))
    cache_key = generate_fourier_cache_key(
        f_in.shape, Lx, Ly, Lz, settings.d_min, settings.r_max, center_idx
    )
    
    # マスクをキャッシュから読み込み
    cache_success, filter_mask, spatial_mask = load_fourier_masks_from_cache(cache_key)
    
    if not cache_success:
        print(f"Computing fourier masks for shape {f_in.shape}...")
        
        # FFT周波数の事前計算
        kx_vals, ky_vals, kz_vals = precompute_fft_frequencies(nx, ny, nz, Lx, Ly, Lz)
        k_cutoff_sq = (2 * np.pi / settings.d_min)**2
        
        # フィルターマスクの作成
        if NUMBA_AVAILABLE:
            print("Creating filter mask with Numba...")
            filter_mask = create_filter_mask_numba(nx, ny, nz, kx_vals, ky_vals, kz_vals, k_cutoff_sq)
        else:
            print("Creating filter mask with NumPy...")
            filter_mask = create_filter_mask_vectorized(nx, ny, nz, kx_vals, ky_vals, kz_vals, k_cutoff_sq)
        
        # 空間マスクの作成
        if NUMBA_AVAILABLE:
            print("Creating spatial mask with Numba...")
            spatial_mask = create_spatial_mask_numba(nx, ny, nz, center_idx, Lx, Ly, Lz, settings.r_max)
        else:
            print("Creating spatial mask with NumPy...")
            spatial_mask = create_spatial_mask_vectorized(nx, ny, nz, center_idx, Lx, Ly, Lz, settings.r_max)
        
        # キャッシュに保存
        save_fourier_masks_to_cache(cache_key, filter_mask, spatial_mask)
        print(f"Saved fourier masks to cache: {cache_key[:8]}...")
    else:
        print(f"Loaded fourier masks from cache: {cache_key[:8]}...")
    
    # 高速FFT
    print("Performing forward FFT...")
    if PYFFTW_AVAILABLE:
        F_k = pyfftw.interfaces.numpy_fft.fftn(f_in)
    else:
        F_k = fft_backend.fftn(f_in)
    
    # フィルター適用
    print("Applying frequency filter...")
    F_k_filtered = F_k * filter_mask
    
    # 高速逆FFT
    print("Performing inverse FFT...")
    if PYFFTW_AVAILABLE:
        f_filtered_complex = pyfftw.interfaces.numpy_fft.ifftn(F_k_filtered)
    else:
        f_filtered_complex = fft_backend.ifftn(F_k_filtered)
    
    # 空間マスク適用
    print("Applying spatial mask...")
    f_filtered_complex = f_filtered_complex * spatial_mask
    
    # 実部を返す
    return f_filtered_complex

# autograd互換の簡略版meshgrid（必要最小限）
def simple_meshgrid(*args, **kwargs):
    """
    簡略化されたmeshgrid（autograd互換）
    """
    indexing = kwargs.get('indexing', 'xy')
    arrays = [anp.asarray(x) for x in args]
    
    if len(arrays) == 3:
        x, y, z = arrays
        if indexing == 'ij':
            X = x[:, None, None]
            Y = y[None, :, None]
            Z = z[None, None, :]
        else:  # 'xy'
            X = x[None, :, None]
            Y = y[:, None, None]
            Z = z[None, None, :]
        
        # ブロードキャスト
        shape = (len(arrays[0]), len(arrays[1]), len(arrays[2]))
        X = anp.broadcast_to(X, shape)
        Y = anp.broadcast_to(Y, shape)
        Z = anp.broadcast_to(Z, shape)
        
        return X, Y, Z
    else:
        raise NotImplementedError("Only 3D meshgrid is implemented")

def fourier_truncation(f_in, settings: Settings) -> anp.ndarray:
    """
    Autograd互換のフーリエ切断関数（最適化済み）
    
    高速化された実装とautograd互換性を両立
    """
    # return f_in
    # autogradの配列かチェック
    if isinstance(f_in, (anp.ndarray, anp.numpy_boxes.ArrayBox)):
        # autograd配列の場合は一度NumPy配列に変換して高速処理
        f_numpy = np.array(f_in)
        result_numpy = fourier_truncation_optimized(f_numpy, settings)
        return anp.array(result_numpy)
    else:
        # 通常のNumPy配列の場合は直接高速処理
        f_numpy = np.array(f_in)
        result_numpy = fourier_truncation_optimized(f_numpy, settings)
        return anp.array(result_numpy)
