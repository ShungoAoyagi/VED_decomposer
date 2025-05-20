from prefect import task
import numpy as np
from pyscf import gto, scf, tools
import os
from collections import defaultdict

lmap = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6}
a0 = 0.529177 # Å

@task(name="calculate Zeff")
def calculate_Zeff(atom: str = "Fe", basis_set: list[str] = ["3d", "4s", "4p"], spin: int = 4, output_path: str = "cache/Zeff_list/Fe_Zeff.npy") -> dict[tuple[int, int], float]:
    """
    Calculate orbital properties for an isolated atom
    """
    # Check if calculation has already been done
    if os.path.exists(output_path):
        return np.load(output_path, allow_pickle=True)

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs('cache/tmp', exist_ok=True)

    # Use a smaller basis set
    mol = gto.M(atom=f"{atom} 0 0 0", basis="cc-pV5Z", spin=spin)
    
    # Adjust SCF convergence parameters
    mf = scf.RHF(mol)
    mf.max_cycle = 150         # Increase maximum iterations
    mf.conv_tol = 1e-8         # Relax convergence criteria
    mf.level_shift = 0.2       # Add level shift
    mf.diis_space = 12         # Expand DIIS space
    mf.damp = 0.5              # Strong damping
    
    # Provide initial guess
    mf.init_guess = 'atom'     # Initial guess from atomic density
    mf = mf.run()

    if not mf.converged:
        print(f"Warning: SCF calculation did not converge. Continuing, but results may be inaccurate.")

    ao_labels = mol.ao_labels()
    groups = defaultdict(list)

    for i, ao in enumerate(ao_labels):
        n_l = ao.split()[-1]
        n = int(n_l[0])
        ell = lmap[n_l[1]]
        groups[(n, ell)].append(i)

    target_nl = {}
    for basis in basis_set:
        _b = basis.split()[0]
        n = int(_b[0])
        ell = lmap[_b[1]]
        target_nl[(n, ell)] = basis

    orbital_radii = {}

    for (n, ell), tag in target_nl.items():
        cubefile = f'cache/tmp/{tag}.cube'
        # Identify the MO with the largest contribution using AO → MO projection strength
        proj = mf.mo_coeff[groups[(n,ell)],:]         # shape: (#AO, #MO)
        weights = np.linalg.norm(proj, axis=0)**2
        mo_idx = weights.argmax()                     # MO with maximum contribution
        # n_pick = {0:1, 1:3, 2:5}.get(ell, 1)
        # top = weights.argsort()[-n_pick:]           # Top n_pick by weight
        # p orbitals are equivalent, so combine all and normalize
        # coeff_comb = mf.mo_coeff[:, top].mean(axis=1)          # (nao,)
        # coeff_comb /= np.linalg.norm(coeff_comb) 
        # tools.cubegen.orbital(
        #     mol, cubefile, coeff_comb, nx=120, ny=120, nz=120, margin=10.0)
        # ----- 4) Grid the MO into a cube -----
        
        tools.cubegen.orbital(mol, cubefile, mf.mo_coeff[:,mo_idx], nx=120, ny=120, nz=120, margin=10.0)
        # 4-1) Read and calculate <r>
        cube   = tools.cubegen.Cube(mol)          # Initialize with mol
        field  = cube.read(cubefile)           # (nx,ny,nz) array
        coords = cube.get_coords()
        r_bohr = np.linalg.norm(coords, axis=1)  # Bohr
        rho    = np.abs(field.ravel())**2                  # |ψ|²
        mean_r_bohr = (r_bohr*rho).sum() / rho.sum()         # Bohr
        mean_r_ang = mean_r_bohr * a0
        orbital_radii[tag] = mean_r_ang
        print(f"  mean radius: {mean_r_ang:.4f} Å")

    # ----- 5) Calculate Z_eff -----
    Zeffs = {}

    for (n, ell), tag in target_nl.items():
        if tag not in orbital_radii:
            continue
        print(f"Calculating Z_eff: {n}, {ell}, {tag}")
        # 正しい水素様原子モデルの有効核電荷計算式
        # Z_eff = n^2 / (a0 * <r>)
        # Zeffs[(n, ell)] = n**2 / (a0 * orbital_radii[tag])
        coeff = (3*n**2 - ell*(ell+1)) * a0 / 2
        Zeffs[(n, ell)] = coeff / orbital_radii[tag]
        print(f"  Z_eff = {Zeffs[(n, ell)]:.4f}")

    np.save(output_path, Zeffs, allow_pickle=True)
        
    return Zeffs
