"""Day 1: Fourier content of the QPC barrier V(x) vs the plane-wave cutoff."""
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qpc.units import Units
from qpc.potential import QPCParams, V_barrier
from qpc.fourier import min_image, ft_numpy, ft_sech2_exact

u = Units()
E_CUT_MEV = 15.0
G_cut = np.sqrt(2 * u.meV_to_au(E_CUT_MEV))          # G^2/2 = E_cut  (atomic units)

L = u.nm_to_au(5000.0)
N = 512
dx = L / N
x = min_image(np.arange(N) * dx, L)
G = 2 * np.pi * np.fft.fftfreq(N, d=dx)
pos = G > 0

fig, ax = plt.subplots(figsize=(7, 4.5))
for hwx, color in [(1.0, "C0"), (1.5, "C1"), (2.0, "C2")]:
    p = QPCParams(hbar_wx_meV=hwx)
    Vt = ft_numpy(V_barrier(x, p), dx)
    rel_fft = np.abs(Vt[pos]) / (2 * p.V0 * p.d)
    Gp = G[pos]
    order = np.argsort(Gp)
    ax.semilogy(u.au_to_nm(1) ** -1 * Gp[order], rel_fft[order], color=color,
                label=f"FFT, d={u.au_to_nm(p.d):.1f} nm")
    exact = [float(ft_sech2_exact(g, p.V0, p.d) / (2 * p.V0 * p.d)) for g in Gp[order][::4]]
    ax.semilogy(u.au_to_nm(1) ** -1 * Gp[order][::4], exact, ".", color=color, ms=3)

ax.axvline(G_cut / u.length_nm, color="k", ls="--", label=f"G_cut (E_cut={E_CUT_MEV:.0f} meV)")
ax.axhline(1e-16, color="gray", ls=":", label="double precision floor")
ax.set_xlabel("G (1/nm)")
ax.set_ylabel(r"$|\tilde V(G)|/|\tilde V(0)|$")
ax.set_ylim(1e-20, 2)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/day01_fourier.png", dpi=150)
print("saved figures/day01_fourier.png; G_cut = %.4f 1/nm" % (G_cut / u.length_nm))
