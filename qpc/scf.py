"""Kohn-Sham self-consistency for the 2D wire / QPC (spin-DFT, LSDA). docs/SPEC.md M5.

Effective potential for spin s (s = 0 up, 1 down; sigma_0 = +1, sigma_1 = -1), Ha*:

    V_s = V_ext + V_H[n_up + n_dn] + v_xc,s[n_up, n_dn] - sigma_s E_Z / 2,   E_Z = g* mu_B B
    V_ext = (1/2) w_y^2 y^2 + V_QPC(x, y)          (HMW Eq. (1); V_QPC = 0 for the clean wire)

Loop: V_s -> lowest KS states per spin -> occupations (fixed mu: f = fermi(e, mu); fixed N:
mu = find_mu) -> more bands if check_band_margin fails -> n_out,s -> Pulay -> repeat, until
sum_s int |n_out,s - n_in,s| < tol AND |M_out - M_in| < tol (M = int (n_up - n_dn)).

Model (docs/MODEL.md): the clean wire rho0 is the INTERACTING wire of Hirose & Wingreen
(cond-mat/0106581, Eq. 1: bare parabola + gated Hartree + TC xc, no positive background), so
HMW's V_H[rho0] + delta V_H = V_H[rho] and the full potential above is the PRL Eq. (2) model.

Inputs in physical units (meV, T, nm) are converted here; everything inside is atomic units.
"""
import dataclasses
import json
import time
from dataclasses import dataclass, field

import numpy as np

from .fourier import min_image
from .mixing import Pulay
from .occupation import check_band_margin, density, fermi, find_mu
from .potential import V_qpc
from .solver import XAveragedPreconditioner, lowest_states
from .units import Units
from .xc import exc_vxc

# Day-2 timing (notes/day02_output.txt): dense lowest-only (evr) 3.4 s vs warm LOBPCG 9.8 s.
DEFAULT_METHOD = "dense"
MU_B_MEV_PER_T = 5.788e-2      # Bohr magneton, meV/T
_PAD = 1e5                     # XAveragedPreconditioner pads blocks with 1e6 on the diagonal


@dataclass
class SCFParams:
    """SCF settings. kT in meV, B_T in tesla, a_image_nm in nm (input units); converted internally.

    a_image_nm: distance charge <-> image charge (HW Eq. (1), HMW delta V_H kernel
    1/rho - 1/sqrt(rho^2 + a_image^2), a_image = 100 nm). The Hartree class takes the
    metal-plane distance a_m = a_image / 2 (its kernel is 1/rho - 1/sqrt(rho^2 + 4 a_m^2)).

    PHYSICS-CHOICE defaults (docs/OPEN_QUESTIONS.md): kT = 0.05 meV, a = 100 nm, interp.
    """
    kT: float = 0.05
    B_T: float = 0.0
    g: float = 0.44
    a_image_nm: float = 100.0
    interp: str = "quadratic"
    alpha: float = 0.2
    history: int = 8
    tol: float = 1e-4
    maxiter: int = 300
    method: str = DEFAULT_METHOD
    nb_init: int = 160               # production: avoids 3 re-diagonalisations in iteration 1
    spin_polarized: bool = True

    def a_metal_au(self, units=Units()):
        """Metal-plane distance a_m = a_image / 2 in a* (the argument of Hartree)."""
        return units.nm_to_au(0.5 * self.a_image_nm)

    def kT_au(self, units=Units()):
        return units.meV_to_au(self.kT)

    def EZ_au(self, units=Units()):
        """Zeeman splitting E_Z = g* mu_B B in Ha*."""
        return units.meV_to_au(self.g * MU_B_MEV_PER_T * self.B_T)


@dataclass
class SCFResult:
    n_up: np.ndarray
    n_dn: np.ndarray
    eigs: list                   # [eigs_up, eigs_dn] (Ha*); for unpolarised both identical
    X: list                      # [X_up, X_dn] coefficient blocks
    mu: float                    # Ha*
    N: float
    M: float
    iterations: int
    converged: bool
    params: SCFParams
    residual: float = np.nan
    history: list = field(default_factory=list)   # per iteration (it, residual, mu, N, M, time)
    V: list = None               # [V_up, V_dn] KS potentials (Ha*) of the last iteration

    def save_npz(self, path):
        """Save everything (params as JSON) to an .npz file."""
        np.savez_compressed(
            path, n_up=self.n_up, n_dn=self.n_dn,
            eigs_up=self.eigs[0], eigs_dn=self.eigs[1], X_up=self.X[0], X_dn=self.X[1],
            mu=self.mu, N=self.N, M=self.M, iterations=self.iterations,
            converged=self.converged, residual=self.residual,
            history=np.array(self.history, dtype=float).reshape(-1, 6),
            params=json.dumps(dataclasses.asdict(self.params)),
            **({} if self.V is None else {"V_up": self.V[0], "V_dn": self.V[1]}))

    @classmethod
    def load_npz(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(n_up=z["n_up"], n_dn=z["n_dn"], eigs=[z["eigs_up"], z["eigs_dn"]],
                   X=[z["X_up"], z["X_dn"]], mu=float(z["mu"]), N=float(z["N"]),
                   M=float(z["M"]), iterations=int(z["iterations"]),
                   converged=bool(z["converged"]), residual=float(z["residual"]),
                   history=[tuple(r) for r in z["history"]],
                   params=SCFParams(**json.loads(str(z["params"]))),
                   V=[z["V_up"], z["V_dn"]] if "V_up" in z else None)


# ----------------------------------------------------------------------------- potentials
def physical_xy(grid):
    """Physical coordinate arrays (Nx, Ny): x = min_image(j dx, Lx), y = min_image(j dy, Ly)."""
    x, y = grid.real_axes()
    return np.meshgrid(min_image(x, grid.Lx), min_image(y, grid.Ly), indexing="ij")


def external_potential(grid, qpc_params, include_qpc=True):
    """V_ext = (1/2) w_y^2 y^2 + V_QPC(x, y)  [Ha*], HMW Eq. (1). include_qpc=False: clean wire."""
    X, Y = physical_xy(grid)
    V = 0.5 * qpc_params.wy ** 2 * Y ** 2
    if include_qpc:
        V = V + V_qpc(X, Y, qpc_params)
    return V


def ks_potentials(V_ext, V_H, n_up, n_dn, EZ, interp):
    """[V_up, V_dn] = V_ext + V_H + v_xc,s - sigma_s E_Z/2  (Ha*)."""
    _, v_up, v_dn = exc_vxc(n_up, n_dn, interp)
    base = V_ext + V_H
    return [base + v_up - 0.5 * EZ, base + v_dn + 0.5 * EZ]


# ----------------------------------------------------------------------------- eigen-solvers
def wire_states(ham, n_bands):
    """Exact lowest eigenpairs when V is x-independent (clean wire).

    Then H = T + <V>_x is block diagonal in G_x and the blocks of XAveragedPreconditioner
    (qpc/solver.py) are exact: psi = e^{i G_x x} phi_n(y). Returns (eigvals, X) with X columns
    in the basis order of ham, sum|c|^2 = 1.
    """
    P = XAveragedPreconditioner(ham)
    d = np.where(P.d > _PAD, np.inf, P.d)                     # drop padding levels
    flat = np.argsort(d, axis=None)[:n_bands]
    g_sel, s_sel = np.unravel_index(flat, d.shape)
    X = np.zeros((ham.n_pw, n_bands), dtype=complex)
    for j, (g, s) in enumerate(zip(g_sel, s_sel)):
        members = P.gid == g
        X[members, j] = P.U[g, P.pos[members], s]
    return d[g_sel, s_sel], X


def solve_states(ham, nb, method, X0=None):
    """Lowest nb KS states: method 'wire' (exact block route), 'dense' or 'lobpcg'."""
    if method == "wire":
        return wire_states(ham, nb)
    res = lowest_states(ham, nb, method=method, X0=X0)
    X_next = getattr(res, "X_full", res.X)
    return res.eigvals, res.X, X_next


def _solve(ham, nb, method, X0):
    out = solve_states(ham, nb, method, X0)
    if len(out) == 2:
        return out[0], out[1], out[1]
    return out


# ----------------------------------------------------------------------------- SCF
def run_scf(ham, hartree, V_ext, p: SCFParams, mu=None, N=None, n_init=None, X_init=None,
            checkpoint=None, checkpoint_every=10, verbose=True, units=Units()):
    """Kohn-Sham SCF at fixed mu (Ha*) or fixed N (electrons). Returns SCFResult.

    n_init: (n_up, n_dn) start densities [a*^-2] (default: non-interacting V_ext states);
    X_init: [X_up, X_dn] start blocks for LOBPCG warm starts. checkpoint: path; every
    checkpoint_every iterations the current input density is saved (resume via n_init).
    """
    if (mu is None) == (N is None):
        raise ValueError("give exactly one of mu, N")
    if abs(hartree.a - p.a_metal_au(units)) > 1e-9 * hartree.a:
        raise ValueError("hartree.a must equal SCFParams.a_image_nm / 2 (metal-plane distance)")
    g = ham.grid
    dA = g.dx * g.dy
    kT = p.kT_au(units)
    EZ = p.EZ_au(units)
    n_spin = 2 if p.spin_polarized else 1
    nb = int(p.nb_init)
    X_warm = list(X_init) if X_init is not None else [None, None]
    meV = units.au_to_meV

    def diagonalise(V_list, nb):
        eigs, Xs, Xn = [], [], []
        for s in range(n_spin):
            ham.set_potential(V_list[s])
            e, X, X_next = _solve(ham, nb, p.method, X_warm[s])
            eigs.append(e); Xs.append(X); Xn.append(X_next)
        return eigs, Xs, Xn

    def occupy(eigs):
        """Occupations per spin channel and mu. Unpolarised: the single channel counts twice."""
        e_list = eigs if n_spin == 2 else [eigs[0], eigs[0]]
        m = find_mu(e_list, N, kT) if N is not None else mu
        return [fermi(e, m, kT) for e in eigs], m

    if n_init is None:
        eigs, Xs, _ = diagonalise([V_ext, V_ext], nb)
        f, _ = occupy(eigs)
        n0 = [density(ham, Xs[s], f[s]) for s in range(n_spin)]
        n_in = n0 if n_spin == 2 else [n0[0], n0[0].copy()]
    else:
        n_in = [np.array(n_init[0], dtype=float), np.array(n_init[1], dtype=float)]

    mixer = Pulay(alpha=p.alpha, history=p.history)
    history = []
    converged = False
    t_start = time.perf_counter()
    if verbose:
        print(f"SCF: {'spin-polarised' if n_spin == 2 else 'unpolarised'}, "
              f"{'N = %.4f' % N if N is not None else 'mu = %.4f meV' % meV(mu)}, "
              f"B = {p.B_T} T, kT = {p.kT} meV, method = {p.method}, interp = {p.interp}",
              flush=True)
        print(f"{'it':>4} {'residual':>10} {'mu(meV)':>10} {'N':>10} {'M':>10} "
              f"{'nb':>5} {'t(s)':>7}", flush=True)
    for it in range(1, p.maxiter + 1):
        t0 = time.perf_counter()
        V_H = hartree.potential(n_in[0] + n_in[1])
        V_list = ks_potentials(V_ext, V_H, n_in[0], n_in[1], EZ, p.interp)
        while True:
            eigs, Xs, Xn = diagonalise(V_list, nb)
            f, mu_it = occupy(eigs)
            if all(check_band_margin(e, mu_it, kT) for e in eigs):
                break
            nb += 20
            if verbose:
                print(f"     band margin too small: nb -> {nb}", flush=True)
        X_warm = Xn
        if n_spin == 2:
            n_out = [density(ham, Xs[s], f[s]) for s in range(2)]
        else:
            half = density(ham, Xs[0], f[0])
            n_out = [half, half.copy()]

        resid = sum(np.abs(n_out[s] - n_in[s]).sum() for s in range(2)) * dA
        M_in = (n_in[0] - n_in[1]).sum() * dA
        M_out = (n_out[0] - n_out[1]).sum() * dA
        N_out = (n_out[0] + n_out[1]).sum() * dA
        dt = time.perf_counter() - t0
        history.append((it, resid, mu_it, N_out, M_out, dt))
        if verbose:
            print(f"{it:4d} {resid:10.3e} {meV(mu_it):10.5f} {N_out:10.4f} {M_out:10.5f} "
                  f"{nb:5d} {dt:7.2f}", flush=True)
        if resid < p.tol and abs(M_out - M_in) < p.tol:
            converged = True
            n_in = n_out
            break
        rho = mixer.step(np.concatenate([n_in[0].ravel(), n_in[1].ravel()]),
                         np.concatenate([n_out[0].ravel(), n_out[1].ravel()]))
        n_in = [rho[:g.Nx * g.Ny].reshape(g.Nx, g.Ny), rho[g.Nx * g.Ny:].reshape(g.Nx, g.Ny)]
        if n_spin == 1:
            avg = 0.5 * (n_in[0] + n_in[1])
            n_in = [avg, avg.copy()]
        if checkpoint is not None and it % checkpoint_every == 0:
            np.savez_compressed(checkpoint, n_up=n_in[0], n_dn=n_in[1], it=it)

    if verbose:
        print(f"SCF {'converged' if converged else 'NOT converged'} after {it} iterations, "
              f"{time.perf_counter() - t_start:.1f} s", flush=True)
    if n_spin == 1:
        eigs = [eigs[0], eigs[0]]
        Xs = [Xs[0], Xs[0]]
    return SCFResult(n_up=n_in[0], n_dn=n_in[1], eigs=eigs, X=Xs, mu=mu_it, N=N_out, M=M_out,
                     iterations=it, converged=converged, params=p, residual=resid,
                     history=history, V=[V_list[0], V_list[1]])
