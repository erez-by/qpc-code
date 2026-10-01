"""Analysis of converged SCF states (docs/SPEC.md M5.2 / M6). Atomic units inside."""
import numpy as np


def transverse_levels(V, ham, n_levels=1):
    """Lowest eigenvalues of the transverse operator h(x) = -(1/2) d^2/dy^2 + V(x, y) at every x.

    Basis: the G_x = 0 plane waves of ham (|G_y|^2/2 <= e_cut), i.e. the same y basis as the KS
    problem. <G_y|h(x)|G_y'> = G_y^2/2 delta + V_y(x)[(iy - iy') mod Ny], V_y(x) = fft_y(V(x,.))/Ny.
    V: (Nx, Ny) Ha*. Returns (Nx, n_levels) Ha*.
    """
    g = ham.grid
    sel = ham.ix == 0
    iy = ham.iy[sel]
    kin = ham.kinetic[sel]
    Vy = np.fft.fft(np.asarray(V, dtype=float), axis=1) / g.Ny            # (Nx, Ny)
    H = Vy[:, (iy[:, None] - iy[None, :]) % g.Ny] + np.diag(kin)[None]    # (Nx, m, m)
    return np.linalg.eigvalsh(H)[:, :n_levels]


def barrier_profile(res, ham):
    """Effective 1D barrier e_0,s(x): lowest transverse level of the KS potential V_s at each x,
    per spin. Returns (Nx, 2) Ha*, in grid order (x_j = j dx; physical x = min_image)."""
    if res.V is None:
        raise ValueError("SCFResult has no stored potentials (V)")
    return np.stack([transverse_levels(res.V[s], ham)[:, 0] for s in range(2)], axis=1)


def n1d(n, grid):
    """Line density n_1D(x) = int n(x, y) dy  [a*^-1], shape (Nx,)."""
    return np.asarray(n).sum(axis=1) * grid.dy
