import numpy as np
from src.utils import ErrorHandler, ErrorCode, ErrorLevel

def spherical_harmonics(l: int, m: int, theta: float, phi: float) -> float:
    """
    Calculate the real spherical harmonics for given l and m values.
    
    Args:
        l: Angular momentum quantum number (0=s, 1=p, 2=d, 3=f)
        m: Magnetic quantum number, follows real-space convention:
           - l=0: m=0 for s orbital
           - l=1: m=0 for pz, m=1 for px, m=-1 for py
           - l=2: m=0 for dz², m=1 for dzx, m=-1 for dyz, m=2 for dx²-y², m=-2 for dxy
        theta: Polar angle (0 to π)
        phi: Azimuthal angle (0 to 2π)
    
    Returns:
        Value of the real spherical harmonic
    """

    error_handler = ErrorHandler()
    if l < 0 or l > 3 or m < -l or m > l:
        error_handler.handle(
            f"Invalid l or m values. l must be between 0 and 3, and m must be between -l and l. l={l}, m={m}",
            ErrorCode.INVALID_INPUT,
            ErrorLevel.ERROR
        )
        
    # l=0: s orbital
    if l == 0:
        # m=0: s
        return np.sqrt(1 / (4 * np.pi))
    
    # l=1: p orbitals
    elif l == 1:
        if m == 0:
            # pz orbital: proportional to cos(θ)
            return np.sqrt(3 / (4 * np.pi)) * np.cos(theta)
        elif m == 1:
            # px orbital: proportional to sin(θ)cos(φ)
            return np.sqrt(3 / (4 * np.pi)) * np.sin(theta) * np.cos(phi)
        elif m == -1:
            # py orbital: proportional to sin(θ)sin(φ)
            return np.sqrt(3 / (4 * np.pi)) * np.sin(theta) * np.sin(phi)
    
    # l=2: d orbitals
    elif l == 2:
        if m == 0:
            # dz² orbital: proportional to (3cos²(θ) - 1)
            return np.sqrt(5 / (16 * np.pi)) * (3 * np.cos(theta)**2 - 1)
        elif m == 1:
            # dzx orbital: proportional to sin(θ)cos(θ)cos(φ)
            return np.sqrt(15 / (4 * np.pi)) * np.sin(theta) * np.cos(theta) * np.cos(phi)
        elif m == -1:
            # dyz orbital: proportional to sin(θ)cos(θ)sin(φ)
            return np.sqrt(15 / (4 * np.pi)) * np.sin(theta) * np.cos(theta) * np.sin(phi)
        elif m == 2:
            # dx²-y² orbital: proportional to sin²(θ)cos(2φ)
            return np.sqrt(15 / (16 * np.pi)) * np.sin(theta)**2 * np.cos(2 * phi)
        elif m == -2:
            # dxy orbital: proportional to sin²(θ)sin(2φ)
            return np.sqrt(15 / (16 * np.pi)) * np.sin(theta)**2 * np.sin(2 * phi)
    
    # l=3: f orbitals (if needed)
    elif l == 3:
        if m == 0:
            # fz³ orbital: proportional to (5cos³(θ) - 3cos(θ))
            return np.sqrt(7 / (16 * np.pi)) * (5 * np.cos(theta)**3 - 3 * np.cos(theta))
        elif m == 1:
            # fz²x orbital: proportional to sin(θ)(5cos²(θ) - 1)cos(φ)
            return np.sqrt(21 / (32 * np.pi)) * np.sin(theta) * (5 * np.cos(theta)**2 - 1) * np.cos(phi)
        elif m == -1:
            # fz²y orbital: proportional to sin(θ)(5cos²(θ) - 1)sin(φ)
            return np.sqrt(21 / (32 * np.pi)) * np.sin(theta) * (5 * np.cos(theta)**2 - 1) * np.sin(phi)
        elif m == 2:
            # fzx²-zy² orbital: proportional to sin²(θ)cos(θ)cos(2φ)
            return np.sqrt(105 / (16 * np.pi)) * np.sin(theta)**2 * np.cos(theta) * np.cos(2 * phi)
        elif m == -2:
            # fzxy orbital: proportional to sin²(θ)cos(θ)sin(2φ)
            return np.sqrt(105 / (16 * np.pi)) * np.sin(theta)**2 * np.cos(theta) * np.sin(2 * phi)
        elif m == 3:
            # fx³-3xy² orbital: proportional to sin³(θ)cos(3φ)
            return np.sqrt(35 / (32 * np.pi)) * np.sin(theta)**3 * np.cos(3 * phi)
        elif m == -3:
            # f3yx²-y³ orbital: proportional to sin³(θ)sin(3φ)
            return np.sqrt(35 / (32 * np.pi)) * np.sin(theta)**3 * np.sin(3 * phi)
    
    # Return 0 for invalid l, m combinations
    return 0.0