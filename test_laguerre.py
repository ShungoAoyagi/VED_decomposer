import numpy as np
from scipy.special import genlaguerre
import math

def calc_laguerre_manual(n: int, alpha: int, x: float) -> float:
    """
    手動実装のラゲール陪多項式
    L_n^α(x) = Σ_{i=0}^n (-1)^i * C(n+α, n-i) * x^i / i!
    """
    L = 0.0
    x_power = 1.0  # x^i
    factorial_i = 1.0  # i!
    
    for i in range(n + 1):
        # 二項係数 C(n+α, n-i) を計算
        # 分子: (n+α) * (n+α-1) * ... * (α+i+1)
        numerator = 1.0
        for j in range(n - i):
            numerator *= (n + alpha - j)
        
        # 分母: (n-i)!
        denominator = 1.0
        for j in range(1, n - i + 1):
            denominator *= j
        
        binom_coeff = numerator / denominator
        
        # 項を追加: (-1)^i * C(n+α, n-i) * x^i / i!
        term = ((-1.0)**i) * binom_coeff * x_power / factorial_i
        L += term
        
        # 次の反復のための更新
        if i < n:
            x_power *= x
            factorial_i *= (i + 1)
    
    return L

def test_cases():
    """テストケース"""
    test_data = [
        # (n, alpha, x, expected_name)
        (0, 1, 1.0, "L_0^(1)(1)"),
        (0, 3, 1.0, "L_0^(3)(1)"),
        (0, 5, 1.0, "L_0^(5)(1)"),
        (1, 1, 1.0, "L_1^(1)(1)"),
        (2, 3, 1.0, "L_2^(3)(1)"),
        (1, 1, 2.0, "L_1^(1)(2)"),
        (2, 3, 2.0, "L_2^(3)(2)"),
    ]
    
    print("ラゲール陪多項式のテスト:")
    print("=" * 50)
    
    for n, alpha, x, name in test_data:
        # SciPy版
        scipy_result = genlaguerre(n, alpha)(x)
        
        # 手動実装版
        manual_result = calc_laguerre_manual(n, alpha, x)
        
        # 差
        diff = abs(scipy_result - manual_result)
        
        print(f"{name:12} | SciPy: {scipy_result:10.6f} | Manual: {manual_result:10.6f} | Diff: {diff:.2e}")
        
        if diff > 1e-10:
            print(f"  ⚠️  大きな差が検出されました!")

def test_hydrogen_cases():
    """水素様原子軌道の具体例"""
    print("\n水素様原子軌道での具体例:")
    print("=" * 50)
    
    cases = [
        # (n, l, description)
        (1, 0, "1s"),
        (2, 0, "2s"),
        (2, 1, "2p"),
        (3, 0, "3s"),
        (3, 1, "3p"),
        (3, 2, "3d"),
        (4, 0, "4s"),
        (4, 1, "4p"),
        (4, 2, "4d"),
        (4, 3, "4f"),
    ]
    
    x = 1.0  # テスト用の値
    
    for n, l, desc in cases:
        k = n - l - 1  # ラゲール多項式の次数
        alpha = 2 * l + 1  # 上付き指数
        
        if k < 0:
            print(f"{desc:4} | n={n}, l={l} | k={k} < 0 → 無効な軌道")
            continue
            
        # SciPy版
        scipy_result = genlaguerre(k, alpha)(x)
        
        # 手動実装版
        manual_result = calc_laguerre_manual(k, alpha, x)
        
        # 差
        diff = abs(scipy_result - manual_result)
        
        print(f"{desc:4} | n={n}, l={l} | L_{k}^({alpha})(1) | SciPy: {scipy_result:8.4f} | Manual: {manual_result:8.4f} | Diff: {diff:.2e}")

if __name__ == "__main__":
    test_cases()
    test_hydrogen_cases() 