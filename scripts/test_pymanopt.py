# pip install pymanopt autograd
import autograd.numpy as np
from pymanopt.manifolds import Grassmann
from pymanopt.function import autograd          # ✔ 小文字
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient

# Grassmann manifold Gr(k, n)
n, k = 5, 2
manifold = Grassmann(n, k)

# 対称行列 A（例として乱数）
A = np.random.randn(n, n)
A = 0.5 * (A + A.T)

# -------- コスト関数 --------
@autograd(manifold)                            # ← manifold を渡す
def cost(U):
    return -np.trace(U.T @ A @ U)              # Rayleigh quotient のマイナス
# --------------------------------

problem   = Problem(manifold, cost)            # 勾配は自動生成
optimizer = ConjugateGradient()

result = optimizer.run(problem)
U_opt = result.point

print("U_opt shape:", U_opt.shape)             # (n, k)
print("objective :", cost(U_opt))
