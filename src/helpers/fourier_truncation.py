import numpy as np
from src.tasks.pre_processing.settings import Settings

def fourier_truncation(f: np.ndarray, settings: Settings) -> np.ndarray:
    """
    Calculate the wavefunction f_filtered by the Fourier truncation method using FFT.

    Args:
        f: The 3D wavefunction to be truncated. Assumed to be a real-valued numpy array
           representing the function on a regular grid.
        settings: The settings of the simulation, containing r_max, d_min,
                  and mesh sizes (r_mesh, theta_mesh, phi_mesh are interpreted as Nx, Ny, Nz).

    Returns:
        The filtered wavefunction f_filtered as a real-valued numpy array.
    """
    if not isinstance(f, np.ndarray) or f.ndim != 3:
        raise ValueError("Input wavefunction f must be a 3D numpy array.")
    if not np.isrealobj(f):
        # If f is complex, proceed, but the output will be its real part.
        # Consider if a warning or different handling is needed for complex input.
        pass

    # Grid dimensions from input array shape
    nx, ny, nz = f.shape

    # For simplicity, we assume the input f is defined on a Cartesian grid.
    # The original code used spherical coordinates (r, theta, phi meshes).
    # For a direct FFT approach on a Cartesian grid, we interpret settings
    # related to r_mesh, theta_mesh, phi_mesh as nx, ny, nz if they match f.shape.
    # If they don't match, it implies a discrepancy or a need for interpolation
    # from spherical to Cartesian, which is not handled here.
    # We will use f.shape directly.

    # Physical lengths of the box, derived from lattice parameters.
    # Assuming an orthorhombic cell for simplicity, where a, b, c are Lx, Ly, Lz.
    if len(settings.lattice_params) >= 3:
        Lx = settings.lattice_params[0]
        Ly = settings.lattice_params[1]
        Lz = settings.lattice_params[2]
        # Optional: Add a check for orthorhombic cell if necessary
        # if len(settings.lattice_params) == 6 and \
        #    not (np.isclose(settings.lattice_params[3], 90.0) and \
        #         np.isclose(settings.lattice_params[4], 90.0) and \
        #         np.isclose(settings.lattice_params[5], 90.0)):
        #    # Handle non-orthorhombic cases or raise a warning/error
        #    # For now, we proceed with a,b,c as Lx,Ly,Lz
        #    pass
    else:
        # Fallback or error handling if lattice_params are not sufficiently defined
        raise ValueError("lattice_params must contain at least a, b, c values.")

    # Perform FFT
    F_k = np.fft.fftn(f)

    # Create k-space grid
    kx = np.fft.fftfreq(nx, d=Lx/nx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=Ly/ny) * 2 * np.pi
    kz = np.fft.fftfreq(nz, d=Lz/nz) * 2 * np.pi

    # Shift k-vectors for correct ordering if performing operations on shifted spectrum
    # kx = np.fft.fftshift(kx)
    # ky = np.fft.fftshift(ky)
    # kz = np.fft.fftshift(kz)
    # F_k_shifted = np.fft.fftshift(F_k) # And operate on F_k_shifted

    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')

    # Calculate k_squared for each point in k-space
    # Note: if using fftshift, Kx, Ky, Kz should also be generated from shifted kx,ky,kz
    # or the comparison needs to be careful about the order.
    # Here, we work with the unshifted k-vectors and unshifted F_k.
    K_squared = Kx**2 + Ky**2 + Kz**2

    # Cutoff in k-space
    k_cutoff = 2 * np.pi / settings.d_min
    k_cutoff_squared = k_cutoff**2

    # Apply truncation: set k-space components beyond k_cutoff to zero
    F_k_filtered = F_k * (K_squared <= k_cutoff_squared)

    # Perform inverse FFT
    # If F_k_shifted was used, it should be unshifted before ifftn or use F_k directly
    f_filtered_complex = np.fft.ifftn(F_k_filtered)

    # Return the real part, as the original wavefunction f was real.
    # Due to numerical precision, ifftn might return small imaginary parts.
    return f_filtered_complex.real
