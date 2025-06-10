from pymanopt.manifolds import Grassmann, Product, Euclidean
from src.tasks.pre_processing.settings import Settings
import numpy as np
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient
import autograd.numpy as anp
from src.tasks.pre_processing.create_orbitals import create_orbitals
from src.helpers.fourier_truncation import fourier_truncation
from scipy.ndimage import zoom
from prefect import task
from pymanopt.function import autograd as pymanopt_autograd # Alias to avoid confusion
from src.helpers import make_xplor

def anp_aware_zoom(input_array, zoom_factors, order=1):
    """
    Performs zoom operation compatible with autograd.
    
    Args:
        input_array: Input array to zoom, can be numpy array or autograd array
        zoom_factors: Zoom factors for each dimension
        order: Interpolation order
        
    Returns:
        Zoomed array as autograd.numpy array
    """
    # Ensure we're working with a numpy array for scipy.ndimage.zoom
    if hasattr(input_array, '_value'):
        input_array_np = np.array(input_array._value)
    elif isinstance(input_array, anp.ndarray):
        input_array_np = np.array(input_array)
    else:
        input_array_np = np.array(input_array)
    
    # Apply zoom function
    output_array_np = zoom(input_array_np, zoom_factors, order=order, prefilter=False)
    
    # Convert back to autograd array and ensure it's trackable
    return anp.array(output_array_np)

@task(name="main_loop")
def main_loop(data: np.ndarray[tuple[int, int, int], float], settings: Settings) -> None:
    print("Stage 1: magnification = 4")
    z_list_mag4_types, orbitals_magnification_4 = create_orbitals(np.array([]), 4, settings)

    coarse_mesh_shape_mag4 = orbitals_magnification_4[0][1].shape
    num_total_orbitals = len(orbitals_magnification_4) 
    
    # Product manifold: Grassmann多様体 × Euclidean空間（betaパラメータ用）
    manifold_stage1 = Product([
        Grassmann(num_total_orbitals, 7),
        Euclidean(1)  # betaパラメータ用の1次元空間
    ])

    zoom_factors_mag4 = tuple(c_dim / f_dim for c_dim, f_dim in zip(coarse_mesh_shape_mag4, data.shape))
    
    data_anp = anp.array(data)
    data_coarse_mag4 = anp_aware_zoom(data_anp, zoom_factors_mag4, order=1)
    
    # 同時最適化のコスト関数
    @pymanopt_autograd(manifold_stage1) 
    def cost_for_mag4(U, beta_vec):
        beta = beta_vec[0]  # スカラー値として取得
        
        gamma = U @ U.T
        
        rho_fit_before = anp.zeros(coarse_mesh_shape_mag4, dtype=float)
        for i in range(num_total_orbitals):
            for j in range(num_total_orbitals):
                gamma_ij = gamma[i, j]
                orb_i_data = orbitals_magnification_4[i][1]
                orb_j_data = orbitals_magnification_4[j][1]
                if not isinstance(orb_i_data, (anp.ndarray, anp.numpy_boxes.ArrayBox)):
                    orb_i_data = anp.array(orb_i_data)
                if not isinstance(orb_j_data, (anp.ndarray, anp.numpy_boxes.ArrayBox)):
                    orb_j_data = anp.array(orb_j_data)
                term = gamma_ij * orb_i_data * orb_j_data
                rho_fit_before += term

        if not isinstance(rho_fit_before, (anp.ndarray, anp.numpy_boxes.ArrayBox)):
            rho_fit_before = anp.array(rho_fit_before)
            
        rho_filtered = fourier_truncation(rho_fit_before, settings)

        target_data_for_test = data_coarse_mag4
        if not isinstance(target_data_for_test, (anp.ndarray, anp.numpy_boxes.ArrayBox)):
            target_data_for_test = anp.array(target_data_for_test)
            
        diff_sq = (rho_filtered * beta - target_data_for_test) ** 2
        
        cost_val = anp.sum(diff_sq) / anp.sum(target_data_for_test ** 2)
        return cost_val
    
    # 最適化の実行
    problem_mag4 = Problem(manifold=manifold_stage1, cost=cost_for_mag4)
    solver = ConjugateGradient(log_verbosity=2, max_iterations=100)
    
    # 初期点を取得（Productマニフォールドの正しい初期化方法）
    initial_point = manifold_stage1.random_point()
    # betaの初期値を設定
    initial_point[1][0] = 30.0
    
    print("Stage 1: Starting simultaneous optimization of U and beta")
    result_mag4 = solver.run(problem_mag4, initial_point=initial_point)
    print("Stage 1 complete.")

    U_opt_final, beta_opt_final = result_mag4.point
    beta_final = beta_opt_final[0]
    print(f"Final beta (Stage 1): {beta_final}")

    gamma = U_opt_final @ U_opt_final.T
        
    rho_fit_before = anp.zeros(coarse_mesh_shape_mag4, dtype=float)
    for i in range(num_total_orbitals):
        for j in range(num_total_orbitals):
            gamma_ij = gamma[i, j]
            orb_i_data = orbitals_magnification_4[i][1]
            orb_j_data = orbitals_magnification_4[j][1]
            if not isinstance(orb_i_data, (anp.ndarray, anp.numpy_boxes.ArrayBox)):
                orb_i_data = anp.array(orb_i_data)
            if not isinstance(orb_j_data, (anp.ndarray, anp.numpy_boxes.ArrayBox)):
                orb_j_data = anp.array(orb_j_data)
            term = gamma_ij * orb_i_data * orb_j_data
            rho_fit_before += term

    rho_filtered = fourier_truncation(rho_fit_before, settings)

    make_xplor(rho_fit_before, "output/rho_fit_before.xplor", "rho_fit_before", settings)
    make_xplor(rho_filtered, "output/rho_filtered.xplor", "rho_filtered", settings)
    make_xplor(data_coarse_mag4, "output/data_coarse_mag4.xplor", "data_coarse_mag4", settings)

    print("Stage 2: magnification = 2")
    z_list_mag2_types, orbitals_magnification_2 = create_orbitals(z_list_mag4_types, 2, settings)

    coarse_mesh_shape_mag2 = orbitals_magnification_2[0][1].shape
    
    # Stage 2でも同じProduct manifoldアプローチ
    manifold_stage2 = Product([
        Grassmann(num_total_orbitals, 4),
        Euclidean(1)  # betaパラメータ用の1次元空間
    ])
    
    zoom_factors_mag2 = tuple(c_dim / f_dim for c_dim, f_dim in zip(coarse_mesh_shape_mag2, data.shape))
    data_anp = anp.array(data)
    data_coarse_mag2 = anp_aware_zoom(data_anp, zoom_factors_mag2, order=1)

    # Stage 2の同時最適化コスト関数
    @pymanopt_autograd(manifold_stage2) 
    def cost_for_mag2(U, beta_vec):
        beta = beta_vec[0]  # スカラー値として取得
        
        gamma = U @ U.T
        rho_fit_before = anp.zeros(coarse_mesh_shape_mag2, dtype=float)
        for i in range(num_total_orbitals): 
            for j in range(num_total_orbitals):
                gamma_ij = gamma[i, j]
                orb_i_data = orbitals_magnification_2[i][1]
                orb_j_data = orbitals_magnification_2[j][1]
                if not isinstance(orb_i_data, (anp.ndarray, anp.numpy_boxes.ArrayBox)):
                    orb_i_data = anp.array(orb_i_data)
                if not isinstance(orb_j_data, (anp.ndarray, anp.numpy_boxes.ArrayBox)):
                    orb_j_data = anp.array(orb_j_data)
                term = gamma_ij * orb_i_data * orb_j_data
                rho_fit_before += term
                
        if not isinstance(rho_fit_before, (anp.ndarray, anp.numpy_boxes.ArrayBox)):
            rho_fit_before = anp.array(rho_fit_before)
            
        rho_filtered = fourier_truncation(rho_fit_before, settings) 

        target_data_for_test_s2 = data_coarse_mag2
        if not isinstance(target_data_for_test_s2, (anp.ndarray, anp.numpy_boxes.ArrayBox)):
            target_data_for_test_s2 = anp.array(target_data_for_test_s2)
            
        diff_sq = (rho_filtered * beta - target_data_for_test_s2) ** 2
        cost_val = anp.sum(diff_sq) / anp.sum(target_data_for_test_s2 ** 2)
        return cost_val

    # Stage 2の最適化
    problem_mag2 = Problem(manifold=manifold_stage2, cost=cost_for_mag2)
    
    # Stage 1の結果を初期値として使用（Productマニフォールドの正しい初期化方法）
    initial_point_stage2 = manifold_stage2.random_point()
    # Stage 1の結果から初期値を設定
    initial_point_stage2[0][:, :4] = result_mag4.point[0][:, :4]  # Uの4次元部分
    initial_point_stage2[1] = result_mag4.point[1]  # betaは継続使用
    
    print("Stage 2: Starting simultaneous optimization of U and beta")
    result_mag2 = solver.run(problem_mag2, initial_point=initial_point_stage2)
    print("Stage 2 complete.")

    U_opt_final, beta_opt_final = result_mag2.point
    beta_final = beta_opt_final[0]
    print(f"Final beta (Stage 2): {beta_final}")

    gamma = U_opt_final @ U_opt_final.T
        
    rho_fit_before = anp.zeros(coarse_mesh_shape_mag2, dtype=float)
    for i in range(num_total_orbitals):
        for j in range(num_total_orbitals):
            gamma_ij = gamma[i, j]
            orb_i_data = orbitals_magnification_2[i][1]
            orb_j_data = orbitals_magnification_2[j][1]
            if not isinstance(orb_i_data, (anp.ndarray, anp.numpy_boxes.ArrayBox)):
                orb_i_data = anp.array(orb_i_data)
            if not isinstance(orb_j_data, (anp.ndarray, anp.numpy_boxes.ArrayBox)):
                orb_j_data = anp.array(orb_j_data)
            term = gamma_ij * orb_i_data * orb_j_data
            rho_fit_before += term

    rho_filtered = fourier_truncation(rho_fit_before, settings)

    make_xplor(rho_filtered, "output/rho_filtered_mag2.xplor", "rho_filtered_mag2", settings)

    print(f"Final U:\n{U_opt_final}")
    print(f"Final U shape: {U_opt_final.shape}")
    print(f"Final beta: {beta_final}") 