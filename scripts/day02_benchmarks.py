"""Day 2 benchmarks for the non-interacting 2D plane-wave solver.

    python scripts/day02_benchmarks.py            # everything (~2-4 min)
    python scripts/day02_benchmarks.py --no-timing

1. Clean wire  V = (1/2) w_y^2 y^2 (folded):   E_{n,m} = hbar w_y (n + 1/2) + (hbar k_m)^2 / 2m*,
   k_m = 2 pi m / Lx.  Must hold for Ly = 320 nm and 480 nm alike.
2. Attractive Poschl-Teller  V = -V0 sech^2(x/d) + (1/2) w_y^2 y^2  (separable):
   E = E_n^PT + hbar w_y (n_y + 1/2),  E_n^PT = -(hbar^2 / 2 m* d^2) (lam - 1 - n)^2,
   lam (lam - 1) = 2 m* V0 d^2 / hbar^2   (Landau & Lifshitz QM, Sec. 23, problems).
3. Non-interacting QPC, paper Eq. (1) + wire confinement: spectrum, adiabatic subbands, states.
4. Timing at production size: dense vs LOBPCG (cold and warm start).
Atomic units inside (hbar = m* = 1), meV / nm only for printing.
"""
import argparse
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qpc.units import Units
from qpc.grid import grid_from_cutoff
from qpc.hamiltonian import Hamiltonian
from qpc.potential import QPCParams, V_qpc, V_barrier, sech2
from qpc.fourier import min_image
from qpc.solver import lowest_states, dense_lowest

U = Units()
E_CUT = U.meV_to_au(15.0)
LX = U.nm_to_au(5000.0)
HBAR_WY = 2.0                       # meV
WY = U.meV_to_au(HBAR_WY)


def setup(Ly_nm):
    grid = grid_from_cutoff(LX, U.nm_to_au(Ly_nm), E_CUT)
    x, y = grid.real_axes()
    X, Y = np.meshgrid(min_image(x, grid.Lx), min_image(y, grid.Ly), indexing="ij")
    return grid, Hamiltonian(grid, E_CUT), X, Y


def meV(e):
    return U.au_to_meV(np.asarray(e))


# ----------------------------------------------------------------------------- 1. clean wire
def clean_wire(n_bands=100):
    print("=" * 78 + "\n1. Clean wire, hbar w_y = 2 meV, Lx = 5 um, E_cut = 15 meV")
    m = np.arange(-400, 401)
    exact = np.sort((WY * (np.arange(4)[:, None] + 0.5) + 0.5 * (2 * np.pi * m[None, :] / LX) ** 2).ravel())[:n_bands]
    out = {}
    for Ly in (320.0, 480.0):
        grid, H, X, Y = setup(Ly)
        H.set_potential(0.5 * WY ** 2 * Y ** 2)
        r = lowest_states(H, n_bands, tol=1e-7)
        err = meV(np.abs(r.eigvals - exact)).max()
        out[Ly] = r.eigvals
        print(f"  Ly = {Ly:.0f} nm: Nx x Ny = {grid.Nx} x {grid.Ny}, n_pw = {H.n_pw}, "
              f"LOBPCG {r.n_iter} it, max |E - E_exact| = {err:.2e} meV")
    i1 = np.searchsorted(exact, WY * 1.5 - 1e-9)
    print(f"  first n=1 level is state #{i1}: E = {meV(out[320.0][i1]):.6f} meV (exact 3.000000)")
    print(f"  max |E(320) - E(480)| = {meV(np.abs(out[320.0] - out[480.0])).max():.2e} meV")
    print(f"  edge of cell: V(Ly/2) = {meV(0.5 * WY**2 * (U.nm_to_au(160.0))**2):.1f} meV, "
          f"Ly/(2 l_y) = {U.nm_to_au(160.0) * np.sqrt(WY):.1f}")


# ----------------------------------------------------------------------------- 2. Poschl-Teller
def poschl_teller():
    print("=" * 78 + "\n2. Attractive Poschl-Teller  -V0 sech^2(x/d), V0 = 3 meV, d = 82.6 nm")
    V0, d = U.meV_to_au(3.0), U.nm_to_au(82.6)
    lam = 0.5 * (1 + np.sqrt(1 + 8 * V0 * d ** 2))              # root of lam(lam-1) = 2 V0 d^2
    nb = int(np.ceil(lam - 1))                                   # n = 0 .. ceil(lam-1)-1
    E_pt = -(lam - 1 - np.arange(nb)) ** 2 / (2 * d ** 2)
    print(f"  lam = {lam:.4f},  2 m V0 d^2 / hbar^2 = {2 * V0 * d**2:.3f},  bound states: {nb}")
    exact = np.sort(np.concatenate([E_pt + WY * (ny + 0.5) for ny in range(3)]))
    exact = exact[exact < WY / 2]                                # below the continuum edge
    grid, H, X, Y = setup(320.0)
    H.set_potential(-V0 * sech2(X / d) + 0.5 * WY ** 2 * Y ** 2)
    r = lowest_states(H, exact.size + 3, tol=1e-8)
    print(f"  {'state':>5} {'exact (meV)':>14} {'numeric (meV)':>15} {'diff (meV)':>11}")
    for i, (a, b) in enumerate(zip(exact, r.eigvals)):
        print(f"  {i:5d} {meV(a):14.8f} {meV(b):15.8f} {meV(b - a):11.2e}")
    print(f"  next (continuum, box-quantised): {meV(r.eigvals[exact.size]):.6f} meV  (edge {HBAR_WY/2:.1f} meV)")


# ----------------------------------------------------------------------------- 3. non-interacting QPC
def qpc_states(hbar_wx=1.5, n_bands=260):
    print("=" * 78 + f"\n3. Non-interacting QPC, hbar w_x = {hbar_wx} meV, V0 = 3 meV")
    p = QPCParams(hbar_wx_meV=hbar_wx)
    grid, H, X, Y = setup(320.0)
    V = V_qpc(X, Y, p) + 0.5 * p.wy ** 2 * Y ** 2
    H.set_potential(V)
    w, v = dense_lowest(H, n_bands)
    top = p.V0 + p.wy / 2                                        # adiabatic n=0 barrier top
    print(f"  d = {U.au_to_nm(p.d):.1f} nm, n=0 barrier top = {meV(top):.3f} meV, "
          f"lowest state {meV(w[0]):.5f} meV")
    # pairs below the barrier: symmetric / antisymmetric combinations of the two sides
    below = w[w < top - U.meV_to_au(0.5)]
    print(f"  {below.size} states lie > 0.5 meV below the top; nearest-neighbour spacing there "
          f"min {meV(np.diff(below).min()):.2e} meV, max {meV(np.diff(below).max()):.2e} meV")

    x_nm = U.au_to_nm(np.fft.fftshift(X[:, 0]))
    y_nm = U.au_to_nm(np.fft.fftshift(Y[0, :]))
    sel = np.abs(x_nm) < 800
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))

    a = ax[0, 0]
    im = a.pcolormesh(x_nm[sel], y_nm, meV(np.fft.fftshift(V))[sel].T, shading="auto",
                      vmax=15, cmap="viridis")
    fig.colorbar(im, ax=a, label="V (meV)")
    a.set(xlabel="x (nm)", ylabel="y (nm)", title="(a) bare potential, Eq. (1) + wire")

    a = ax[0, 1]
    xs = np.linspace(-800, 800, 400)
    Vb = meV(V_barrier(U.nm_to_au(xs), p))
    for n in range(3):
        a.plot(xs, Vb / 2 + (HBAR_WY + Vb) * (n + 0.5), label=f"n={n}: V/2 + hbar(w_y+V)(n+1/2)")
    for e in meV(w[::6]):
        a.axhline(e, color="0.85", lw=0.5, zorder=0)
    a.set(xlabel="x (nm)", ylabel="E (meV)", ylim=(0.8, 6), title="(b) adiabatic subbands; grey: every 6th level")
    a.legend(fontsize=7, loc='center right')

    # two example states, one just below and one just above the n=0 barrier top. Above the
    # top, many levels are n=1 states (fully reflected by the n=1 barrier at ~9 meV), so pick
    # in each window the state with the largest weight at the QPC centre x = 0.
    w0 = np.array([np.sum(np.abs(H.to_real_space(v[:, i])[0, :]) ** 2) for i in range(w.size)])
    def pick(lo, hi):
        idx = np.nonzero((w > lo) & (w < hi))[0]
        return idx[np.argmax(w0[idx])]
    i_lo = pick(top - U.meV_to_au(0.3), top)
    i_hi = pick(top, top + U.meV_to_au(0.5))
    for a, i in ((ax[1, 0], i_lo), (ax[1, 1], i_hi)):
        dens = np.abs(H.to_real_space(v[:, i])) ** 2 / (grid.Lx * grid.Ly)     # |psi|^2 in a*^-2
        a.plot(x_nm, np.fft.fftshift(dens.sum(axis=1) * grid.dy) * U.nm_to_au(1.0) * 1e3,
               lw=0.8)
        a.set(xlabel="x (nm)", ylabel=r"$\int|\psi|^2 dy$  ($10^{-3}$ nm$^{-1}$)",
              title=f"({'c' if i == i_lo else 'd'}) state {i}, E = {meV(w[i]):.3f} meV "
                    f"({'below' if w[i] < top else 'above'} top)")
        a.axvspan(-U.au_to_nm(p.d), U.au_to_nm(p.d), color="orange", alpha=0.2)
    fig.tight_layout()
    fig.savefig("figures/day02_qpc_states.png", dpi=150)
    print("  saved figures/day02_qpc_states.png")


# ----------------------------------------------------------------------------- 4. timing
def timing(n_bands=90):
    print("=" * 78 + f"\n4. Timing at production size, {n_bands} bands")
    p = QPCParams()
    grid, H, X, Y = setup(320.0)
    V = V_qpc(X, Y, p) + 0.5 * p.wy ** 2 * Y ** 2
    H.set_potential(V)
    rows = []

    t = time.perf_counter(); w_full, _ = H.eigh_dense(); rows.append(("dense, full eigh", time.perf_counter() - t, "-"))
    t = time.perf_counter(); w_ref, _ = dense_lowest(H, n_bands); rows.append(("dense, lowest only (evr)", time.perf_counter() - t, "-"))
    t = time.perf_counter(); cold = lowest_states(H, n_bands); dt = time.perf_counter() - t
    rows.append(("LOBPCG cold, block precond", dt, f"{cold.n_iter} it"))
    t = time.perf_counter(); tpa = lowest_states(H, n_bands, precond="tpa"); dt = time.perf_counter() - t
    rows.append(("LOBPCG cold, TPA precond", dt, f"{tpa.n_iter} it"))
    # SCF-like update: a smooth 0.05 meV bump at the QPC (the size of a late SCF correction)
    H.set_potential(V + U.meV_to_au(0.05) * np.exp(-(X / U.nm_to_au(200.0)) ** 2))
    w_new, _ = dense_lowest(H, n_bands)
    t = time.perf_counter(); warm = lowest_states(H, n_bands, X0=cold.X_full); dt = time.perf_counter() - t
    rows.append(("LOBPCG warm (after dV = 0.05 meV)", dt, f"{warm.n_iter} it"))

    for name, dt, extra in rows:
        print(f"  {name:36s} {dt:7.2f} s  {extra}")
    print(f"  max |E_lobpcg - E_dense|: cold {meV(np.abs(cold.eigvals - w_ref)).max():.1e} meV, "
          f"warm {meV(np.abs(warm.eigvals - w_new)).max():.1e} meV")
    best_dense = min(rows[0][1], rows[1][1])
    choice = "LOBPCG (warm start)" if rows[-1][1] < best_dense else "dense (lowest only)"
    print(f"  -> per SCF diagonalisation: {choice}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-timing", action="store_true")
    args = ap.parse_args()
    clean_wire()
    poschl_teller()
    qpc_states()
    if not args.no_timing:
        timing()
