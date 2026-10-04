"""Kohn-Sham total energy and grand potential (atomic units, Ha*).

For KS orbitals psi_is (eigenvectors of H_s = T + V_s with V_s the potential that produced them)
and occupations f_is, with n_s built from the same orbitals:

  eigenvalue form:  E = sum_s sum_i f_is e_is - sum_s int n_s (V_s - V_ext,s - Z_s)
                        + E_H[n] + E_xc[n] + sum_s int n_s Z_s ... (Z_s inside V_s, see below)
  direct form:      E = T_s + sum_s int n_s (V_ext,s + Z_s) + E_H[n] + E_xc[n]

with Z_s = -sigma_s E_Z / 2 (Zeeman), E_H = (s_H/2) int n V_H[n] (hartree_scale s_H),
E_xc = int n eps_xc(n_up, n_dn). At self-consistency V_s - V_ext,s - Z_s = s_H V_H[n] + v_xc,s[n],
and the eigenvalue form is the usual E = sum f e - E_H - sum int n_s v_xc,s + E_xc. The two forms
agree identically (sum f e = T_s + sum int n_s V_s for the same orbitals) -- this is the test.

Grand potential at fixed mu, Fermi-Dirac smearing:
  Omega = E - kT S - mu N,   S = -sum [f ln f + (1 - f) ln(1 - f)]   (k_B = 1).
"""
import numpy as np

from .occupation import density, fermi
from .scf import _per_spin
from .xc import exc_vxc


def _entropy(f):
    f = np.clip(np.asarray(f, dtype=float), 1e-300, 1 - 1e-16)
    g = 1.0 - f
    return float(-np.sum(f * np.log(f) + np.where(g > 0, g * np.log(np.maximum(g, 1e-300)), 0.0)))


def total_energy(res, ham, hartree, V_ext, p, units=None):
    """Energy terms of a stored state (needs res.eigs, res.X, res.V per spin, res.mu).

    Returns dict (Ha*): E_eig (eigenvalue form), E_direct, T_s, E_H, E_xc, E_ext, E_Z, TS, N,
    Omega = E_eig - TS - mu N. Orbitals beyond the stored bands are assumed empty.
    """
    g = ham.grid
    dA = g.dx * g.dy
    kT, EZ = p.kT_au(), p.EZ_au()
    Vext = _per_spin(V_ext)
    Z = [-0.5 * EZ, 0.5 * EZ]
    n, sum_fe, Ts, TS, Eext, EZt, Vterm = [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    spins = 2 if p.spin_polarized else 1
    weight = 1.0 if p.spin_polarized else 2.0          # unpolarised: one channel counted twice
    for s in range(spins):
        e = np.asarray(res.eigs[s])
        X = np.asarray(res.X[s])
        f = fermi(e, res.mu, kT)
        ns = density(ham, X, f)
        sum_fe += weight * float(np.sum(f * e))
        Ts += weight * float(np.sum(f * (np.abs(X) ** 2 * ham.kinetic[:, None]).sum(axis=0)))
        TS += weight * kT * _entropy(f)
        Eext += weight * float(np.sum(ns * Vext[s])) * dA
        EZt += weight * Z[s] * float(ns.sum()) * dA
        Vterm += weight * float(np.sum(ns * res.V[s])) * dA
        n.append(ns)
    if spins == 1:
        n = [n[0], n[0]]
    ntot = n[0] + n[1]
    E_H = 0.5 * p.hartree_scale * float(np.sum(ntot * hartree.potential(ntot))) * dA
    eps, _, _ = exc_vxc(n[0], n[1], p.interp)
    E_xc = float(np.sum(ntot * eps)) * dA
    N = float(ntot.sum()) * dA
    # potentials V_s include V_ext,s and Z_s: subtract them to get the mean-field part
    E_eig = sum_fe - (Vterm - Eext - EZt) + E_H + E_xc
    E_direct = Ts + Eext + EZt + E_H + E_xc
    return dict(E_eig=E_eig, E_direct=E_direct, T_s=Ts, E_H=E_H, E_xc=E_xc, E_ext=Eext, E_Z=EZt,
                TS=TS, N=N, Omega=E_eig - TS - res.mu * N)
