import numpy as np
# It's good practice to import autograd.numpy if it might be involved with inputs,
# though this function aims to work with plain numpy arrays internally after unboxing.
import autograd.numpy as anp 
from src.tasks.pre_processing.settings import Settings

def fourier_truncation(f_in, settings: Settings) -> anp.ndarray:
    """
    Calculate the wavefunction f_filtered by the Fourier truncation method using FFT.
    The truncation is performed based on the k-space representation of the input f_in,
    using its own grid dimensions. Autograd-aware operations are used for FFT
    as this function is part of a Pymanopt cost function.

    Args:
        f_in: The 3D wavefunction to be truncated. Can be a real-valued numpy array
              or an autograd ArrayBox wrapping such an array.
        settings: The settings of the simulation, containing d_min for k_cutoff,
                  and lattice_params for physical dimensions.

    Returns:
        The filtered wavefunction f_filtered as an autograd-aware array (real part), 
        with the same shape as the original data within f_in.
    """
    f_ndarray = f_in
    if hasattr(f_in, '_value'):
        f_ndarray = f_in._value
    if not isinstance(f_ndarray, (anp.ndarray, anp.numpy_boxes.ArrayBox, np.ndarray)):
         # If f_ndarray is not already an anp array after unboxing (e.g. plain np.ndarray from f_in)
         # or if f_in was an unboxed anp array that got its _value extracted as np.ndarray
        f_ndarray = anp.array(f_ndarray, dtype=getattr(f_ndarray, 'dtype', float)) 
    elif isinstance(f_ndarray, np.ndarray) and not isinstance(f_ndarray, (anp.ndarray, anp.numpy_boxes.ArrayBox)):
        # Explicitly convert plain numpy ndarray to autograd numpy ndarray
        f_ndarray = anp.array(f_ndarray, dtype=f_ndarray.dtype)


    if not (hasattr(f_ndarray, 'ndim') and f_ndarray.ndim == 3 and hasattr(f_ndarray, 'shape')):
        raise ValueError(
            f"Input wavefunction (after potential unboxing and anp conversion) must be a 3D array-like object. Got type {type(f_ndarray)} with ndim {getattr(f_ndarray, 'ndim', 'N/A')}"
        )

    f_ndarray_val_for_np_checks = f_ndarray._value if hasattr(f_ndarray, '_value') else f_ndarray
    # Ensure f_ndarray_val_for_np_checks is a concrete numpy array for np functions like np.isnan, np.isinf, np.min, np.max
    if not isinstance(f_ndarray_val_for_np_checks, np.ndarray):
        # This might happen if f_ndarray was an anp.ArrayBox and ._value still returned something else, 
        # or if f_ndarray was an anp.ndarray not directly usable by np.isnan (though usually it is)
        # Try to convert to a plain np.ndarray for these checks if it's not already one.
        try:
            f_ndarray_val_for_np_checks = np.array(f_ndarray_val_for_np_checks)
        except Exception as e_conv:
            print(f"Could not convert f_ndarray_val_for_np_checks to np.ndarray for checks: {e_conv}")
            # Skip checks if conversion fails
            pass 


    nx, ny, nz = f_ndarray.shape

    if len(settings.lattice_params) >= 3:
        Lx = settings.lattice_params[0]
        Ly = settings.lattice_params[1]
        Lz = settings.lattice_params[2]
    else:
        raise ValueError("lattice_params must contain at least a, b, c values for physical dimensions.")

    F_k = anp.fft.fftn(f_ndarray)

    kx_vals = np.fft.fftfreq(nx, d=Lx/nx) * 2 * np.pi
    ky_vals = np.fft.fftfreq(ny, d=Ly/ny) * 2 * np.pi
    kz_vals = np.fft.fftfreq(nz, d=Lz/nz) * 2 * np.pi

    Kx, Ky, Kz = np.meshgrid(kx_vals, ky_vals, kz_vals, indexing='ij')
    K_squared = Kx**2 + Ky**2 + Kz**2
    filter_mask = K_squared <= (2 * np.pi / settings.d_min)**2
    F_k_filtered = F_k * filter_mask 

    f_filtered_complex = anp.fft.ifftn(F_k_filtered)
    return f_filtered_complex.real
