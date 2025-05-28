from pymanopt.manifolds import Grassmann, Euclidean, Product
from src.tasks.pre_processing.settings import Settings
import autograd.numpy as anp


def define_manifold(settings: Settings) -> Product:
    return Product(
        Grassmann(9, 7),
        Euclidean(3)
    )


def cost(point):
    # point = (U, log_alpha)  … U:(9,r)  log_alpha:(3,)
    U, log_alpha = point
    alpha = anp.exp(log_alpha)           # 正数化
    # 4s,4p,3d ３群とも同じ α にするなら alpha = anp.exp(log_alpha[0])
    chi = scale_basis(chi_raw, alpha[0]) # ここでは3つを同一に
    phi = anp.dot(chi, U)                # (Ngrid,r)
    rho_fit = anp.sum(phi**2, axis=1)    # ∑|φ_a|²
    return anp.sum(w * (rho_fit - rho_exp) ** 2)
