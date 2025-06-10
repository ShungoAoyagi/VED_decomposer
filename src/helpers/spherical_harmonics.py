import numpy as np
import autograd.numpy as anp
from scipy.special import sph_harm
from src.utils import ErrorHandler, ErrorCode, ErrorLevel

def spherical_harmonics(l: int, m: int, theta: float, phi: float) -> np.complex128:
    """
    Calculate the complex spherical harmonics Y_l^m(θ,φ) for given l and m values.
    
    Uses the standard physics convention:
    Y_l^m(θ,φ) = sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!) * P_l^|m|(cos θ) * e^(imφ)
    
    Args:
        l: Angular momentum quantum number (0=s, 1=p, 2=d, 3=f)
        m: Magnetic quantum number (-l ≤ m ≤ l)
        theta: Polar angle (0 to π)
        phi: Azimuthal angle (0 to 2π)
    
    Returns:
        Complex value of the spherical harmonic Y_l^m(θ,φ)
    """

    error_handler = ErrorHandler()
    if l < 0 or l > 3 or m < -l or m > l:
        error_handler.handle(
            f"Invalid l or m values. l must be between 0 and 3, and m must be between -l and l. l={l}, m={m}",
            ErrorCode.INVALID_INPUT,
            ErrorLevel.ERROR
        )
    
    # scipy.special.sph_harmを使用（引数の順序に注意: sph_harm(m, l, phi, theta)）
    # scipyの実装は標準的な物理学の慣例に従っている
    return sph_harm(m, l, phi, theta)

def associated_legendre(l: int, m: int, x: float) -> float:
    """
    Calculate the associated Legendre polynomial P_l^m(x).
    
    Args:
        l: Degree of the polynomial
        m: Order of the polynomial (0 ≤ m ≤ l)
        x: Argument (-1 ≤ x ≤ 1)
    
    Returns:
        Value of P_l^m(x)
    """
    from scipy.special import lpmv
    return lpmv(m, l, x)

def manual_spherical_harmonics(l: int, m: int, theta: float, phi: float) -> np.complex128:
    """
    Manual implementation of complex spherical harmonics for reference.
    
    Y_l^m(θ,φ) = sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!) * P_l^|m|(cos θ) * e^(imφ)
    
    Args:
        l: Angular momentum quantum number
        m: Magnetic quantum number
        theta: Polar angle
        phi: Azimuthal angle
    
    Returns:
        Complex spherical harmonic value
    """
    import math
    from scipy.special import factorial
    
    # 正規化定数の計算
    abs_m = abs(m)
    normalization = anp.sqrt((2*l + 1) / (4 * anp.pi) * 
                            factorial(l - abs_m) / factorial(l + abs_m))
    
    # ルジャンドル陪関数 P_l^|m|(cos θ)
    cos_theta = anp.cos(theta)
    legendre_val = associated_legendre(l, abs_m, cos_theta)
    
    # 複素指数部分 e^(imφ)
    complex_phase = anp.exp(1j * m * phi)
    
    # Condon-Shortley位相因子の考慮
    if m < 0:
        phase_factor = (-1)**abs_m
        return phase_factor * normalization * legendre_val * complex_phase
    else:
        return normalization * legendre_val * complex_phase