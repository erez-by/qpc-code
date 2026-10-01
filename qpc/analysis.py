"""Analysis of converged SCF states (docs/SPEC.md M5.2 / M6). Atomic units inside."""
import numpy as np

from .occupation import fermi
from .solver import XAveragedPreconditioner

_PAD = 1e5          # XAveragedPreconditioner pads its blocks with 1e6 on the diagonal


def transverse_levels(V, ham, n_levels=1, basis="grid"):
    """Lowest eigenvalues of the transverse operator h(x) = -(1/2) d^2/dy^2 + V(x, y) at every x.

    Plane waves in y: <G_y|h(x)|G_y'> = G_y^2/2 delta + V_y(x)[(iy - iy') mod Ny],
    V_y(x) = fft_y(V(x,.))/Ny.
      basis="grid" (default): all Ny plane waves of the y grid -- exact for the grid-sampled V
                     (bare QPC barrier top (V0/2 + (w_y + V0)/2 = 4.000 meV) reproduced to 1e-5).
      basis="cutoff": only the G_x = 0 plane waves with |G_y|^2/2 <= e_cut (the KS basis); 17
                     waves at E_cut = 15 meV, which overestimates the stiff x = 0 level (+0.013 meV
                     for the bare barrier).
    V: (Nx, Ny) Ha*. Returns (Nx, n_levels) Ha*.
    """
    g = ham.grid
    if basis == "grid":
        iy = np.arange(g.Ny)
        kin = 0.5 * g.G_axes()[1] ** 2
    elif basis == "cutoff":
        sel = ham.ix == 0
        iy = ham.iy[sel]
        kin = ham.kinetic[sel]
    else:
        raise ValueError(basis)
    Vy = np.fft.fft(np.asarray(V, dtype=float), axis=1) / g.Ny            # (Nx, Ny)
    H = Vy[:, (iy[:, None] - iy[None, :]) % g.Ny] + np.diag(kin)[None]    # (Nx, m, m)
    return np.linalg.eigvalsh(H)[:, :n_levels]


def barrier_profile(res, ham, basis="grid"):
    """Effective 1D barrier e_0,s(x) (PRL Fig. 1(a)-(c)): lowest eigenvalue of
    -(1/2) d_y^2 + V_eff,s(x, y) at each x, per spin (transverse_levels).
    Returns (Nx, 2) Ha*, in grid order (x_j = j dx; physical x = min_image)."""
    if res.V is None:
        raise ValueError("SCFResult has no stored potentials (V)")
    return np.stack([transverse_levels(res.V[s], ham, basis=basis)[:, 0] for s in range(2)], axis=1)


def n1d(n, grid):
    """Line density n_1D(x) = int n(x, y) dy  [a*^-1], shape (Nx,)."""
    return np.asarray(n).sum(axis=1) * grid.dy


def wire_subbands(ham, V, mu, kT):
    """Transverse subbands of an x-independent potential V (clean wire), one spin channel.

    For x-independent V the KS states are e^{i G_x x} phi_{n,G_x}(y); the G_x blocks of
    XAveragedPreconditioner give them exactly (column n = transverse index n, ascending).
    Returns (bottoms, N_sub): bottoms[n] = e_n(k_x = 0) [Ha*]; N_sub[n] = sum_{G_x} f(e_n(G_x))
    = electrons in subband n for this spin (multiply by 2 for an unpolarised wire).
    """
    ham.set_potential(V)
    P = XAveragedPreconditioner(ham)
    d = np.where(P.d > _PAD, np.inf, P.d)
    g0 = int(np.nonzero(np.unique(ham.ix) == 0)[0][0])
    return np.sort(d[g0]), fermi(d, mu, kT).sum(axis=0)


def net_spin(res, grid):
    """M = int (n_up - n_dn) dx dy  (electrons)."""
    return float((res.n_up - res.n_dn).sum()) * grid.dx * grid.dy


def net_spin_local(res, wire_res, grid):
    """Local moment of the QPC: M_loc = int (n_up - n_dn) dx dy - (N_up - N_dn of the clean wire at
    the same B, same cell). Removes the Zeeman polarisation of the leads; equals M at B = 0.
    This is the number to compare with HMW (0.85, 0.93, 0.90). wire_res: SCFResult or CleanWire."""
    w = getattr(wire_res, "res", wire_res)
    return net_spin(res, grid) - net_spin(w, grid)
