from pymanopt.manifolds import Grassmann, Product, Euclidean
from src.tasks.pre_processing.settings import Settings
import numpy as np
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient
import autograd.numpy as anp
from autograd import grad as ag_grad # For testing gradients
from src.tasks.pre_processing.create_orbitals import create_orbitals
from src.helpers.fourier_truncation import fourier_truncation
from scipy.ndimage import zoom
from prefect import task
from pymanopt.function import autograd as pymanopt_autograd # Alias to avoid confusion
from pymanopt.tools import diagnostics
import matplotlib.pyplot as plt # For histogram

def anp_aware_zoom(input_array, zoom_factors, order=1):
    if hasattr(input_array, '_value'): 
        input_array_np = input_array._value
    else:
        input_array_np = input_array
    output_array_np = zoom(input_array_np, zoom_factors, order=order, prefilter=False)
    return anp.array(output_array_np)

@task(name="main_loop")
def main_loop(data: np.ndarray[tuple[int, int, int], float], settings: Settings) -> None:
    print("Stage 1: magnification = 4")
    z_list_mag4_types, orbitals_magnification_4 = create_orbitals(np.array([]), 4, settings)

    coarse_mesh_shape_mag4 = orbitals_magnification_4[0][1].shape
    num_total_orbitals = len(orbitals_magnification_4) 
    manifold_stage1 = Grassmann(num_total_orbitals, 4)

    zoom_factors_mag4 = tuple(c_dim / f_dim for c_dim, f_dim in zip(coarse_mesh_shape_mag4, data.shape))
    data_coarse_mag4 = anp_aware_zoom(data, zoom_factors_mag4, order=1)

    @pymanopt_autograd(manifold_stage1) 
    def cost_for_mag4(U):
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

        rho_filtered = fourier_truncation(rho_fit_before, settings) # Simplified fourier_truncation

        target_data_for_test = data_coarse_mag4 # Simplified target
        diff_sq = (rho_filtered - target_data_for_test) ** 2
        
        cost_val = anp.sum(diff_sq)
        return cost_val

    print("Performing manual gradient check for cost_for_mag4...")
    U_test_point_mag4 = manifold_stage1.random_point()
    try:
        grad_cost_for_mag4_at_U_test = ag_grad(cost_for_mag4)(U_test_point_mag4)
        grad_val_to_print = grad_cost_for_mag4_at_U_test._value if hasattr(grad_cost_for_mag4_at_U_test, '_value') else grad_cost_for_mag4_at_U_test
        print(f"MAIN_LOOP GRAD TEST (cost_for_mag4): Grad of cost_for_mag4 w.r.t. U non-zero: {anp.any(grad_cost_for_mag4_at_U_test != 0)}")
        print(f"MAIN_LOOP GRAD TEST (cost_for_mag4): Max abs of grad_cost_for_mag4: {anp.max(anp.abs(grad_val_to_print))}")
    except Exception as e_main_grad_test:
        print(f"MAIN_LOOP GRAD TEST (cost_for_mag4): Error during test: {e_main_grad_test}")

    problem_mag4 = Problem(manifold=manifold_stage1, cost=cost_for_mag4)
    print("Checking gradient for Stage 1 (magnification 4) using Pymanopt diagnostics...")
    diagnostics.check_gradient(problem_mag4) 
    print("Pymanopt diagnostics check for Stage 1 complete.")

    solver = ConjugateGradient(log_verbosity=2)
    result_mag4 = solver.run(problem_mag4) 
    print("Stage 1 complete (solver.run might be skipped).")

    print("Stage 2: magnification = 2")
    z_list_mag2_types, orbitals_magnification_2 = create_orbitals(z_list_mag4_types, 2, settings)

    coarse_mesh_shape_mag2 = orbitals_magnification_2[0][1].shape
    manifold_stage2 = Grassmann(num_total_orbitals, 4) 
    
    zoom_factors_mag2 = tuple(c_dim / f_dim for c_dim, f_dim in zip(coarse_mesh_shape_mag2, data.shape))
    data_coarse_mag2 = anp_aware_zoom(data, zoom_factors_mag2, order=1)


    @pymanopt_autograd(manifold_stage2) 
    def cost_for_mag2(U):
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
        rho_filtered = fourier_truncation(rho_fit_before, settings) 

        target_data_for_test_s2 = data_coarse_mag2 # Simplified target (all ones)
        diff_sq = (rho_filtered - target_data_for_test_s2) ** 2

        cost_val = anp.sum(diff_sq)
        return cost_val

    print("Performing manual gradient check for cost_for_mag2...")
    U_test_point_mag2 = manifold_stage2.random_point()
    try:
        grad_cost_for_mag2_at_U_test = ag_grad(cost_for_mag2)(U_test_point_mag2)
        grad_val_to_print_s2 = grad_cost_for_mag2_at_U_test._value if hasattr(grad_cost_for_mag2_at_U_test, '_value') else grad_cost_for_mag2_at_U_test
        print(f"MAIN_LOOP GRAD TEST (cost_for_mag2): Grad of cost_for_mag2 w.r.t. U non-zero: {anp.any(grad_cost_for_mag2_at_U_test != 0)}")
        print(f"MAIN_LOOP GRAD TEST (cost_for_mag2): Max abs of grad_cost_for_mag2: {anp.max(anp.abs(grad_val_to_print_s2))}")
    except Exception as e_main_grad_test_s2:
        print(f"MAIN_LOOP GRAD TEST (cost_for_mag2): Error during test: {e_main_grad_test_s2}")

    problem_mag2 = Problem(manifold=manifold_stage2, cost=cost_for_mag2)
    print("Checking gradient for Stage 2 (magnification 2) using Pymanopt diagnostics...")
    diagnostics.check_gradient(problem_mag2) 
    print("Pymanopt diagnostics check for Stage 2 complete.")

    result_mag2 = solver.run(problem_mag2, initial_point=result_mag4.point) 
    print("Stage 2 complete (solver.run might be skipped).")

    U_opt_final = result_mag2.point
    print(f"Final U:\n{U_opt_final}")
    print(f"Final U shape: {U_opt_final.shape}") 