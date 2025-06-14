import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks.pre_processing.settings import import_settings
from src.tasks.pre_processing.load_data import load_data
from src.helpers.xplor_loader import load_xplor

def plot_data(t, target_data, optimized_data, residual_data, filename):
    plt.rcParams["font.sans-serif"] = "Arial"
    _fontsize = 16
    fig, ax1 = plt.subplots(figsize=(6,6))
    
    # 左軸（target と optimized）
    ax1.plot(t, target_data, "o", label="target", color="k", markersize=6)
    ax1.plot(t, optimized_data, "-", label="optimized", color="blue", linewidth=2)
    ax1.set_xlabel("r", fontsize=_fontsize)
    ax1.set_ylabel("electron density (e/A$^3$)", fontsize=_fontsize)
    ax1.tick_params(axis='x', labelsize=_fontsize)
    ax1.tick_params(axis='y', labelsize=_fontsize)
    ax1.set_ylim(-max(target_data) * 0.3, max(target_data) * 1.2)
    ax1.axhline(0, color="k", linewidth=1)
    ax1.axvline(0, color="k", linewidth=1)
    
    # 右軸（residual）
    ax2 = ax1.twinx()
    ax2.plot(t, residual_data, "o", label="residual", color="red", markersize=6)
    ax2.set_ylabel("relative residual", fontsize=_fontsize)
    ax2.tick_params(axis='y', labelsize=_fontsize)
    ax2.set_ylim(-0.25, 1.0)

    ax1.tick_params(direction='in', top=True, right=False, labelsize=_fontsize)
    ax2.tick_params(direction='in', top=True, right=True, labelsize=_fontsize)

    plt.xlim(-0.5, 0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)

if __name__ == "__main__":
    settings = import_settings("data/input/settings.yaml")
    _ = load_data("data/input/data.xplor", settings)
    target_data = load_xplor("data/input/data.xplor")
    optimized_data = load_xplor("output/rho_output.xplor")
    residual_data = load_xplor("output/normalized_residual_output.xplor")
    
    center = settings.center

    t = np.linspace(-settings.r_max, settings.r_max, 100)
    target_x = np.zeros(len(t))
    optimized_x = np.zeros(len(t))
    residual_x = np.zeros(len(t))

    for i in range(len(t)):
        pos = [center[0] + t[i] * settings.basis_set[0][0] / settings.lattice_params[0], center[1] + t[i] * settings.basis_set[0][1] / settings.lattice_params[1], center[2] + t[i] * settings.basis_set[0][2] / settings.lattice_params[2]]
        idx = [(pos[j] * settings.v[j]) % settings.v[j] for j in range(3)]
        target_x[i] = target_data.get_value(idx[0], idx[1], idx[2])
        optimized_x[i] = optimized_data.get_value(idx[0], idx[1], idx[2])
        residual_x[i] = residual_data.get_value(idx[0], idx[1], idx[2])
    
    plot_data(t, target_x, optimized_x, residual_x, "output/after_processing_x.png")

    target_y = np.zeros(len(t))
    optimized_y = np.zeros(len(t))
    residual_y = np.zeros(len(t))

    for i in range(len(t)):
        pos = [center[0] + t[i] * settings.basis_set[1][0] / settings.lattice_params[0], center[1] + t[i] * settings.basis_set[1][1] / settings.lattice_params[1], center[2] + t[i] * settings.basis_set[1][2] / settings.lattice_params[2]]
        idx = [(pos[j] * settings.v[j]) % settings.v[j] for j in range(3)]
        target_y[i] = target_data.get_value(idx[0], idx[1], idx[2])
        optimized_y[i] = optimized_data.get_value(idx[0], idx[1], idx[2])
        residual_y[i] = residual_data.get_value(idx[0], idx[1], idx[2])
    
    plot_data(t, target_y, optimized_y, residual_y, "output/after_processing_y.png")

    target_z = np.zeros(len(t))
    optimized_z = np.zeros(len(t))
    residual_z = np.zeros(len(t))

    for i in range(len(t)):
        pos = [center[0] + t[i] * settings.basis_set[2][0] / settings.lattice_params[0], center[1] + t[i] * settings.basis_set[2][1] / settings.lattice_params[1], center[2] + t[i] * settings.basis_set[2][2] / settings.lattice_params[2]]
        idx = [(pos[j] * settings.v[j]) % settings.v[j] for j in range(3)]
        target_z[i] = target_data.get_value(idx[0], idx[1], idx[2])
        optimized_z[i] = optimized_data.get_value(idx[0], idx[1], idx[2])
        residual_z[i] = residual_data.get_value(idx[0], idx[1], idx[2])
    
    plot_data(t, target_z, optimized_z, residual_z, "output/after_processing_z.png")