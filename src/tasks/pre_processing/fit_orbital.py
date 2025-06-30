import numpy as np
from scipy.optimize import minimize_scalar
from typing import Tuple, Optional, Dict
import warnings
from src.helpers.calc_single_orb_density import calc_single_orb_density
from src.tasks.pre_processing.settings import Settings, orbital_magnetic_number
import matplotlib.pyplot as plt

def fit_electron_density_smart(
    target_data: np.ndarray,
    r_min: float,
    r_max: float,
    settings: Settings,
    bounds_magnification: Tuple[float, float] = (0.1, 10.0),
    bounds_coeff: Tuple[float, float] = (0.1, 10.0),
    max_calc_orb_calls: int = 30,
    cache_size: int = 100,
) -> Tuple[float, float, np.ndarray]:
    """
    calc_single_orb_densityの呼び出し回数を最小化する効率的なフィッティング
    
    Parameters
    ----------
    target_data : np.ndarray
        フィッティングするデータ
    r_min : float
        フィッティングする半径の最小値（単位：Å）
    r_max : float
        フィッティングする半径の最大値（単位：Å）
    bounds_magnification : Tuple[float, float]
        軌道半径の拡大/縮小係数の範囲
    bounds_coeff : Tuple[float, float]
        軌道の大きさの拡大/縮小係数の範囲
    max_calc_orb_calls : int
        calc_single_orb_densityの最大呼び出し回数
    cache_size : int
        キャッシュの最大サイズ
    
    Returns
    -------
    fitted_magnification : float
        得られた軌道半径の拡大/縮小の係数
    fitted_coeff : float
        得られた軌道の大きさの拡大/縮小の係数
    orb_data : np.ndarray
        フィッティングした軌道データ
    """
    
    mesh_grid = len(target_data)
    calc_orb_calls = 0
    cache = {}
    
    def get_orb_data(magnification: float) -> Optional[np.ndarray]:
        """キャッシュ機能付きcalc_orb呼び出し"""
        nonlocal calc_orb_calls
        
        # キャッシュチェック
        mag_key = round(magnification, 6)  # 浮動小数点の精度問題対策
        if mag_key in cache:
            return cache[mag_key]
        
        # calc_orb呼び出し回数チェック
        if calc_orb_calls >= max_calc_orb_calls:
            return None
        
        try:
            orb_data = calc_single_orb_density(magnification, r_min, r_max, mesh_grid, settings)
            calc_orb_calls += 1
            
            # キャッシュに保存（サイズ制限付き）
            if len(cache) < cache_size:
                cache[mag_key] = orb_data
            
            return orb_data
        except Exception as e:
            warnings.warn(f"Error in calc_single_orb_density at magnification={magnification}: {e}")
            return None
    
    def compute_optimal_coeff(orb_data: np.ndarray) -> Tuple[float, float]:
        """与えられた軌道データに対する最適なcoeffと誤差を計算"""
        # 線形最小二乗法でcoeffを解析的に計算
        numerator = np.dot(target_data, orb_data)
        denominator = np.dot(orb_data, orb_data)
        
        if denominator > 0:
            coeff = numerator / denominator
            coeff = np.clip(coeff, *bounds_coeff)
            
            residual = target_data - coeff * orb_data
            error = np.sum(residual**2)
            
            return coeff, error
        else:
            return 0.0, np.inf
    
    def objective(magnification: float) -> float:
        """最適化の目的関数"""
        orb_data = get_orb_data(magnification)
        if orb_data is None:
            return np.inf
        
        _, error = compute_optimal_coeff(orb_data)
        return error
    
    data = get_orb_data(0.91)
    return 0.9, 1.0, data

    # Phase 1: 黄金分割探索による効率的な1次元最適化
    result = minimize_scalar(
        objective,
        bounds=bounds_magnification,
        method='bounded',
        options={'maxiter': max_calc_orb_calls // 2}  # 呼び出し回数の半分を使用
    )
    
    best_magnification = result.x
    
    # Phase 2: 最適点周辺での精密化（Brent法）
    if calc_orb_calls < max_calc_orb_calls - 5:
        # 最適点周辺でより精密な探索
        search_range = (result.x - result.x * 0.1, result.x + result.x * 0.1)
        search_range = (
            max(search_range[0], bounds_magnification[0]),
            min(search_range[1], bounds_magnification[1])
        )
        
        result_refined = minimize_scalar(
            objective,
            bounds=search_range,
            method='bounded',
            options={'maxiter': 5}
        )
        
        if result_refined.fun < result.fun:
            best_magnification = result_refined.x
    
    # 最終的なcoeffの計算
    orb_data = get_orb_data(best_magnification)
    if orb_data is None:
        raise RuntimeError("Failed to compute orbital data for the best magnification")
    
    with open('orb_data.txt', 'w') as f:
        for i in range(len(orb_data)):
            f.write(f"{i} {orb_data[i]}\n")
    
    fitted_coeff, final_error = compute_optimal_coeff(orb_data)
    
    return best_magnification, fitted_coeff, orb_data


def fit_electron_density_adaptive(
    target_data: np.ndarray,
    r_min: float,
    r_max: float,
    settings: Settings,
    bounds_magnification: Tuple[float, float] = (0.1, 10.0),
    bounds_coeff: Tuple[float, float] = (0.1, 10.0),
    initial_grid_points: int = 7,
    refinement_factor: int = 3,
    tolerance: float = 1e-4
) -> Tuple[float, float, np.ndarray]:
    """
    適応的グリッド細分化による効率的なフィッティング
    
    Parameters
    ----------
    target_data : np.ndarray
        フィッティングするデータ
    r_min : float
        フィッティングする半径の最小値（単位：Å）
    r_max : float
        フィッティングする半径の最大値（単位：Å）
    bounds_magnification : Tuple[float, float]
        軌道半径の拡大/縮小係数の範囲
    bounds_coeff : Tuple[float, float]
        軌道の大きさの拡大/縮小係数の範囲
    initial_grid_points : int
        初期グリッド点数（少なめに設定）
    refinement_factor : int
        各ステップでの細分化係数
    tolerance : float
        収束判定の許容誤差
    
    Returns
    -------
    fitted_magnification : float
        得られた軌道半径の拡大/縮小の係数
    fitted_coeff : float
        得られた軌道の大きさの拡大/縮小の係数
    orb_data : np.ndarray
        フィッティングした軌道データ
    """
    
    mesh_grid = len(target_data)
    calc_orb_calls = 0
    
    # 評価済みの点を保存
    evaluated_points = {}
    
    def evaluate_point(magnification: float) -> Tuple[float, float]:
        """点を評価し、結果をキャッシュ"""
        nonlocal calc_orb_calls
        
        if magnification in evaluated_points:
            return evaluated_points[magnification]
        
        try:
            orb_data = calc_single_orb_density(magnification, r_min, r_max, mesh_grid, settings)
            calc_orb_calls += 1
            
            # 線形最小二乗法
            numerator = np.dot(target_data, orb_data)
            denominator = np.dot(orb_data, orb_data)
            
            if denominator > 0:
                coeff = numerator / denominator
                coeff = np.clip(coeff, *bounds_coeff)
                
                residual = target_data - coeff * orb_data
                error = np.sum(residual**2)
                
                evaluated_points[magnification] = (coeff, error)
                return coeff, error
            else:
                evaluated_points[magnification] = (0.0, np.inf)
                return 0.0, np.inf
                
        except Exception as e:
            warnings.warn(f"Error at magnification={magnification}: {e}")
            evaluated_points[magnification] = (0.0, np.inf)
            return 0.0, np.inf
    
    # 初期グリッド評価
    current_min, current_max = bounds_magnification
    best_magnification = None
    best_coeff = None
    best_error = np.inf
    
    while True:
        # 現在の範囲でグリッド生成
        grid_points = np.linspace(current_min, current_max, initial_grid_points)
        
        # 新しい点のみ評価
        for mag in grid_points:
            if mag not in evaluated_points:
                coeff, error = evaluate_point(mag)
                
                if error < best_error:
                    best_error = error
                    best_magnification = mag
                    best_coeff = coeff
        
        # 収束判定
        grid_width = (current_max - current_min) / (initial_grid_points - 1)
        if grid_width < tolerance:
            break
        
        # 最良点の周辺を細分化
        if best_magnification is not None:
            current_min = max(bounds_magnification[0], best_magnification - grid_width)
            current_max = min(bounds_magnification[1], best_magnification + grid_width)
            initial_grid_points = refinement_factor + 2  # 中心点 + 両側
        else:
            break
        
        # calc_orb呼び出し回数の制限
        if calc_orb_calls >= 20:  # ノートPC向けに控えめに設定
            break
    
    if best_magnification is None:
        raise RuntimeError("Fitting failed: could not find valid parameters")
    
    orb_data = calc_single_orb_density(best_magnification, r_min, r_max, mesh_grid, settings)
    
    return best_magnification, best_coeff, orb_data


def fit_electron_density_parabolic(
    target_data: np.ndarray,
    r_min: float,
    r_max: float,
    settings: Settings,
    bounds_magnification: Tuple[float, float] = (0.1, 10.0),
    bounds_coeff: Tuple[float, float] = (0.1, 10.0),
    initial_points: int = 5
) -> Tuple[float, float, np.ndarray]:
    """
    放物線補間を使用した高効率フィッティング
    
    少ない評価点数で高精度を実現
    
    Parameters
    ----------
    target_data : np.ndarray
        フィッティングするデータ
    r_min : float
        フィッティングする半径の最小値（単位：Å）
    r_max : float
        フィッティングする半径の最大値（単位：Å）
    bounds_magnification : Tuple[float, float]
        軌道半径の拡大/縮小係数の範囲
    bounds_coeff : Tuple[float, float]
        軌道の大きさの拡大/縮小係数の範囲
    initial_points : int
        初期評価点数
    
    Returns
    -------
    fitted_magnification : float
        得られた軌道半径の拡大/縮小の係数
    fitted_coeff : float
        得られた軌道の大きさの拡大/縮小の係数
    orb_data : np.ndarray
        フィッティングした軌道データ
    """
    
    mesh_grid = len(target_data)
    calc_orb_calls = 0
    
    # 評価関数
    def evaluate_and_fit(magnification: float) -> Tuple[float, float]:
        nonlocal calc_orb_calls
        
        try:
            orb_data = calc_single_orb_density(magnification, r_min, r_max, mesh_grid)
            calc_orb_calls += 1
            
            # 線形最小二乗法でcoeffを計算
            numerator = np.dot(target_data, orb_data)
            denominator = np.dot(orb_data, orb_data)
            
            if denominator > 0:
                coeff = numerator / denominator
                coeff = np.clip(coeff, *bounds_coeff)
                
                residual = target_data - coeff * orb_data
                error = np.sum(residual**2)
                
                return coeff, error
            else:
                return 0.0, np.inf
                
        except Exception as e:
            warnings.warn(f"Error at magnification={magnification}: {e}")
            return 0.0, np.inf
    
    # 初期点での評価
    magnifications = np.linspace(*bounds_magnification, initial_points)
    errors = []
    coeffs = []
    
    for mag in magnifications:
        coeff, error = evaluate_and_fit(mag)
        errors.append(error)
        coeffs.append(coeff)
    
    errors = np.array(errors)
    
    # 最小誤差の点を見つける
    min_idx = np.argmin(errors)
    
    # 放物線補間による精密化
    if 0 < min_idx < len(magnifications) - 1:
        # 3点を使って放物線フィッティング
        x = magnifications[min_idx-1:min_idx+2]
        y = errors[min_idx-1:min_idx+2]
        
        # 放物線の係数を計算
        denom = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
        if abs(denom) > 1e-10:
            a = ((x[1] - x[2]) * (y[0] - y[2]) - (x[0] - x[2]) * (y[1] - y[2])) / denom
            b = ((x[0] - x[2])**2 * (y[1] - y[2]) - (x[1] - x[2])**2 * (y[0] - y[2])) / denom
            
            # 放物線の最小値
            if a > 0:  # 下に凸
                x_min = -b / (2 * a)
                
                # 境界内に収める
                x_min = np.clip(x_min, bounds_magnification[0], bounds_magnification[1])
                
                # 新しい点で評価
                if abs(x_min - magnifications[min_idx]) > 1e-6:
                    coeff_new, error_new = evaluate_and_fit(x_min)
                    
                    if error_new < errors[min_idx]:
                        best_magnification = x_min
                        best_coeff = coeff_new
                        best_error = error_new
                    else:
                        best_magnification = magnifications[min_idx]
                        best_coeff = coeffs[min_idx]
                        best_error = errors[min_idx]
                else:
                    best_magnification = magnifications[min_idx]
                    best_coeff = coeffs[min_idx]
                    best_error = errors[min_idx]
            else:
                best_magnification = magnifications[min_idx]
                best_coeff = coeffs[min_idx]
                best_error = errors[min_idx]
        else:
            best_magnification = magnifications[min_idx]
            best_coeff = coeffs[min_idx]
            best_error = errors[min_idx]
    else:
        best_magnification = magnifications[min_idx]
        best_coeff = coeffs[min_idx]
        best_error = errors[min_idx]

    orb_data = calc_single_orb_density(best_magnification, r_min, r_max, mesh_grid, settings)
    
    return best_magnification, best_coeff, orb_data


# シンプルなインターフェース（後方互換性のため）
def fit_electron_density(
    target_data: np.ndarray,
    r_min: float,
    r_max: float,
    settings: Settings,
    method: str = 'smart',
    **kwargs
) -> Tuple[float, float, np.ndarray]:
    """
    電子密度フィッティングのメインインターフェース
    
    Parameters
    ----------
    target_data : np.ndarray
        フィッティングするデータ
    r_min : float
        フィッティングする半径の最小値（単位：Å）
    r_max : float
        フィッティングする半径の最大値（単位：Å）
    method : str
        使用する手法 ('smart', 'adaptive', 'parabolic')
    **kwargs
        各手法固有のパラメータ
    
    Returns
    -------
    fitted_magnification : float
        得られた軌道半径の拡大/縮小の係数
    fitted_coeff : float
        得られた軌道の大きさの拡大/縮小の係数
    """
    
    if method == 'smart':
        mag, coeff, orb_data = fit_electron_density_smart(
            target_data, r_min, r_max, settings, **kwargs
        )
    elif method == 'adaptive':
        mag, coeff, orb_data = fit_electron_density_adaptive(
            target_data, r_min, r_max, settings, **kwargs
        )
    elif method == 'parabolic':
        mag, coeff, orb_data = fit_electron_density_parabolic(
            target_data, r_min, r_max, settings, **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return mag, coeff, orb_data

def fit_orbital(
    data: np.ndarray,
    settings: Settings
) -> tuple[float, float, np.ndarray]:
    """
    軌道フィッティングのメインインターフェース
    
    Parameters
    ----------
    data : np.ndarray
        フィッティングするデータ
    settings : Settings
        設定
    
    Returns
    -------
    mag : float
        得られた軌道半径の拡大/縮小の係数
    coeff : float
        得られた軌道の大きさの拡大/縮小の係数
    """
    r_min = settings.r_max * 0.7
    r_max = settings.r_max * 0.95
    mesh_grid = 100
    target_data = np.linspace(r_min, r_max, mesh_grid)
    center_idx_float = [settings.center[i] * settings.v[i] for i in range(3)]
    lattice_params = settings.lattice_params
    
    for i in range(mesh_grid):
        r = r_min + (r_max - r_min) * i / (mesh_grid - 1)
        res = 0.0
        for j in range(3):
            x = (center_idx_float[0] + (r * settings.basis_set[j][0] / lattice_params[0]) * settings.v[0]) % settings.v[0]
            y = (center_idx_float[1] + (r * settings.basis_set[j][1] / lattice_params[1]) * settings.v[1]) % settings.v[1]
            z = (center_idx_float[2] + (r * settings.basis_set[j][2] / lattice_params[2]) * settings.v[2]) % settings.v[2]
            tmp = 0.0
            # weight = 1.0
            for k in range(2):
                # weight *= np.abs(np.ceil(x) - x - k) 
                for l in range(2):
                    # weight *= np.abs(np.ceil(y) - y - l)
                    for m in range(2):
                        weight = np.abs(np.ceil(x) - x - k) * np.abs(np.ceil(y) - y - l) * np.abs(np.ceil(z) - z - m)
                        tmp += weight * data[int(x + k)][int(y + l)][int(z + m)]
            res += tmp


            x = (center_idx_float[0] - (r * settings.basis_set[j][0] / lattice_params[0]) * settings.v[0]) % settings.v[0]
            y = (center_idx_float[1] - (r * settings.basis_set[j][1] / lattice_params[1]) * settings.v[1]) % settings.v[1]
            z = (center_idx_float[2] - (r * settings.basis_set[j][2] / lattice_params[2]) * settings.v[2]) % settings.v[2]
            tmp = 0.0
            weight = 0.0
            for k in range(2):
                weight *= np.abs(np.ceil(x) - x - k) 
                for l in range(2):
                    weight *= np.abs(np.ceil(y) - y - l)
                    for m in range(2):
                        weight *= np.abs(np.ceil(z) - z - m)
                        tmp += weight * data[int(x + k)][int(y + l)][int(z + m)]
            res += tmp
        target_data[i] = res / 6
    mag, coeff, orb_data = fit_electron_density(target_data, r_min, r_max, settings, method='smart')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(target_data)), [i / target_data.max() for i in target_data], c='r', label='target')
    ax.plot(np.arange(len(orb_data)), [i / orb_data.max() for i in orb_data], c='b', label='orb')
    ax.set_xlabel('r')
    ax.set_ylabel('density')
    ax.legend()
    plt.savefig('orb_data.png')