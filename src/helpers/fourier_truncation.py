import numpy as np
from numba import jit
from src.tasks.pre_processing.settings import Settings

@jit(nopython=True)
def _calculate_fourier_transform_parts(
    f_complex: np.ndarray,
    r_mesh: int,
    r_max: float,
    theta_mesh: int,
    phi_mesh: int,
    k_list_np: np.ndarray
) -> np.ndarray:
    """
    Numba JITコンパイル用の内部関数。
    フーリエ係数の計算と逆変換によるフィルタリング済み波動関数の再構築を行う。
    """
    r_step = r_max / r_mesh
    theta_step = np.pi / theta_mesh
    phi_step = 2 * np.pi / phi_mesh

    num_k_points = k_list_np.shape[0]
    F_k_array = np.zeros(num_k_points, dtype=np.complex128)

    # F_k の計算
    for idx in range(num_k_points):
        k_x, k_y, k_z = k_list_np[idx, 0], k_list_np[idx, 1], k_list_np[idx, 2]
        current_F_k = 0j
        for i in range(r_mesh):
            for j in range(theta_mesh):
                for l_idx in range(phi_mesh): # lがループ変数として使われているためl_idxに変更
                    r_val = i * r_step
                    theta_val = j * theta_step
                    phi_val = l_idx * phi_step # l_idxを使用
                    x = r_val * np.sin(theta_val) * np.cos(phi_val)
                    y = r_val * np.sin(theta_val) * np.sin(phi_val)
                    z = r_val * np.cos(theta_val)
                    current_F_k += f_complex[i, j, l_idx] * np.exp(-1j * (k_x * x + k_y * y + k_z * z))
        F_k_array[idx] = current_F_k

    # f_filtered の計算
    f_filtered = np.zeros((r_mesh, theta_mesh, phi_mesh), dtype=np.complex128)
    for i in range(r_mesh):
        for j in range(theta_mesh):
            for l_idx in range(phi_mesh): # l_idxを使用
                r_val = i * r_step
                theta_val = j * theta_step
                phi_val = l_idx * phi_step # l_idxを使用
                x = r_val * np.sin(theta_val) * np.cos(phi_val)
                y = r_val * np.sin(theta_val) * np.sin(phi_val)
                z = r_val * np.cos(theta_val)
                val = 0j
                for idx in range(num_k_points):
                    k_x, k_y, k_z = k_list_np[idx, 0], k_list_np[idx, 1], k_list_np[idx, 2]
                    val += F_k_array[idx] * np.exp(1j * (k_x * x + k_y * y + k_z * z))
                f_filtered[i, j, l_idx] = val # l_idxを使用
    
    return f_filtered

def fourier_truncation(f: np.ndarray, settings: Settings) -> np.ndarray:
    """
    Calculate the wavefunction f_filtered by the Fourier truncation method.
    Numba JITを使用して高速化されています。

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

    # k_list の生成 (この部分はPythonレベルで実行)
    k_max_abs = int(2 * np.pi / d_min) # k_maxだと関数名と衝突するため変更
    
    _k_list_builder = []
    # ループ変数が外側のスコープと衝突しないように変更 (i, j, k -> ki, kj, kk)
    for kk_val in range(2 * k_max_abs + 1):
        for kj_val in range(2 * k_max_abs + 1):
            for ki_val in range(2 * k_max_abs + 1):
                kx = ki_val - k_max_abs
                ky = kj_val - k_max_abs
                kz = kk_val - k_max_abs
                if kx**2 + ky**2 + kz**2 <= k_max_abs**2:
                    _k_list_builder.append((kx, ky, kz))
    
    if not _k_list_builder: # k_listが空の場合
        # d_minが非常に小さい場合など、k_listが空になることがある。
        # その場合、入力fの形状でゼロ（複素数）を返す。
        return np.zeros_like(f, dtype=np.complex128)

    k_list_np = np.array(_k_list_builder, dtype=np.int64)

    # 入力fを複素数型に変換（Numba関数で複素数演算を期待するため）
    if not np.iscomplexobj(f):
        f_complex = f.astype(np.complex128)
    else:
        f_complex = f
    
    f_filtered_complex = _calculate_fourier_transform_parts(
        f_complex, r_mesh, r_max, theta_mesh, phi_mesh, k_list_np
    )

    return f_filtered_complex.real
