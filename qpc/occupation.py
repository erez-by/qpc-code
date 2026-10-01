"""Fermi occupations, chemical potential and Kohn-Sham density (atomic units).

All energies in Ha*, densities in a*^-2. Conventions: see CLAUDE.md (psi = sum_p c_p e^{iG.r},
sum|c|^2 = 1, physical orbital psi / sqrt(Lx Ly)).
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


def fermi(e, mu, kT):
    """Fermi-Dirac occupation f(e) = 1 / (1 + exp((e - mu)/kT)).

    Evaluated as expit(-(e - mu)/kT), which never overflows (|e - mu|/kT = 1e4 gives exactly 0 or 1).
    e, mu, kT in the same energy unit (Ha* in the code); kT > 0.
    """
    if kT <= 0:
        raise ValueError("kT must be > 0")
    return expit(-(np.asarray(e, dtype=float) - mu) / kT)


def total_number(eigs_list, mu, kT):
    """N(mu) = sum_s sum_i f(e_is, mu, kT); eigs_list = [eigs_up, eigs_dn] (or one array)."""
    return float(sum(fermi(e, mu, kT).sum() for e in eigs_list))


def find_mu(eigs_list, N, kT, xtol=1e-15):
    """Chemical potential mu with sum_s sum_i f(e_is, mu, kT) = N (Ha*).

    N(mu) is strictly increasing for kT > 0, so the root is unique even when a level is exactly
    degenerate at the Fermi energy. Bracket [min(e) - 40 kT, max(e) + 40 kT]; brentq.
    eigs_list: sequence of eigenvalue arrays (one per spin; a single array for one spin channel).
    Requires 0 < N < total number of states.
    """
    eigs_list = [np.asarray(e, dtype=float) for e in eigs_list]
    n_states = sum(e.size for e in eigs_list)
    if not 0 < N < n_states:
        raise ValueError(f"N = {N} must lie in (0, {n_states})")
    lo = min(e.min() for e in eigs_list) - 40 * kT
    hi = max(e.max() for e in eigs_list) + 40 * kT
    return brentq(lambda m: total_number(eigs_list, m, kT) - N, lo, hi,
                  xtol=xtol, rtol=4 * np.finfo(float).eps, maxiter=500)


def density(ham, X, f, chunk=32):
    """Density of one spin channel n(r) = sum_i f_i |psi_i(r)|^2 / (Lx Ly)   [a*^-2].

    ham : Hamiltonian (provides to_real_space and grid); X : (n_pw, nb) coefficient columns with
    sum|c|^2 = 1; f : (nb,) occupations (0..1, no spin factor). Columns are transformed in
    chunks of `chunk` to bound memory ((Nx, Ny, chunk) complex). Returns (Nx, Ny) real.
    Parseval: sum(n) dx dy = sum(f).
    """
    X = np.asarray(X)
    f = np.asarray(f, dtype=float)
    g = ham.grid
    n = np.zeros((g.Nx, g.Ny))
    idx = np.nonzero(f > 0)[0]
    for start in range(0, idx.size, chunk):
        sel = idx[start:start + chunk]
        psi = ham.to_real_space(X[:, sel])
        n += np.einsum("xyk,k->xy", psi.real ** 2 + psi.imag ** 2, f[sel])
    return n / (g.Lx * g.Ly)


def check_band_margin(eigs, mu, kT, margin=15):
    """True if max(eigs) - mu > margin * kT, i.e. the highest computed band has f < e^-15 ~ 3e-7."""
    return float(np.max(eigs)) - mu > margin * kT
