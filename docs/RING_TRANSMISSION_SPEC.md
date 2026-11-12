# Spec: transmission from the ring eigenstates (Meir's even/odd phase-shift method), for Claude Code

Put this file in docs/RING_TRANSMISSION_SPEC.md. Do not change the SCF physics. Report after Step 0 and after every phase.

## Physics goal
The periodic cell (the "ring", Lx = 5000 nm, one QPC at x = 0) has KS eigenstates that are exactly even or odd under
x -> -x, because V_sigma(x, y) is mirror-symmetric. In the lead region (far from the QPC) a single-channel state is
psi(x) = A e^{ikx} + B e^{-ikx}, with the local chiral components
    a(x) = (psi + psi'/(ik))/2 = A e^{ikx},   b(x) = (psi - psi'/(ik))/2 = B e^{-ikx}.
A mirror-symmetric scatterer has t = (e^{2 i d_e} + e^{2 i d_o})/2, r = (e^{2 i d_e} - e^{2 i d_o})/2, hence
    T = |t|^2 = cos^2(d_e - d_o),
where d_e(E), d_o(E) are the phase shifts of the even and odd states:
    psi_e ~ cos(k|x| + d_e),  psi_o ~ sin(k|x| + d_o)  (lead region),
and the ring quantisation (Neumann/Dirichlet at x = +-Lx/2 for even/odd) gives  k_n Lx/2 + d(k_n) = n pi.
One state gives only one phase shift. T needs d_e and d_o at the SAME energy, so interpolate d_e(E) and d_o(E)
separately (smooth, monotone mod pi) and evaluate at E = mu or on an energy grid. cos^2 has period pi, so the pi
ambiguity of d does not matter, but d_e and d_o must be unwrapped consistently along E.
Checked by me on a 1D Poschl-Teller ring (L = 200, d = 2, V0 = 0.6, hbar = m = 1): agrees with the exact T to ~1e-3
from T ~ 0 to T ~ 1 (see tests below). The 2D/LSDA application is NOT yet checked.

Goal of the work: (A) T_sigma(E) in a frozen converged potential, (B) T_sigma at fixed mu as a function of the barrier
height V0 (full SCF per V0). Case: model C (hartree_scale 0.8, interacting leads), interp = "exchange",
hbar*omega_x = 1.5 meV, hbar*omega_y = 2.0 meV, kT = 0.05 meV, Lx = 5000 nm, E_cut = 15 meV, the usual Janak ramp.
Conductance (mean-field, static, no Kondo physics): G = (e^2/h) (T_up + T_dn).

## Step 0: inspect and report before coding
1. How do I get real-space KS orbitals per spin from SCFResult (X blocks -> psi(x, y) on the grid)? How many bands per
   spin are stored (nb_init = 160) and what is the highest eigenvalue relative to mu? Is res.V = [V_up, V_dn] saved in the
   checkpoints (.npz)?
2. Are the self-consistent V_sigma(x, y) exactly mirror-symmetric in x (and y) to numerical precision? Report
   max|V(x,y) - V(-x,y)| for a converged B = 0 polarised state. If not, the even/odd classification must be tolerant
   (use the parity expectation value <P> and a threshold 0.99), and say so.
3. Which clean-wire quantities are available: the lead transverse mode phi_1(y) (CleanWire.subbands / res), the lead
   dispersion E_1(k), the bottom of subband 2 (E_2). In model C, E_2 should be only ~0.1 meV above mu: report the
   exact number, because the single-channel method is valid only for E < E_2.
4. Where is the non-interacting 2D solver (Poschl-Teller / HMW Eq. 1 benchmarks from day 2) and can it take the same
   ring cell? HMW Eq. (1) gives omega(x) = omega_y + V(x)/hbar, so the lowest transverse mode sees exactly
   eps_0(x) = hbar*omega_y/2 + V(x) (a Poschl-Teller barrier in x); mode mixing is non-adiabatic and small for large d.

## New module: qpc/ring_transmission.py (keep qpc/ otherwise untouched)
Functions (docstrings with the formulas above):
1. `ring_states(V_sigma, grid, E_window, nb)`: dense or LOBPCG eigenstates of the single-spin KS Hamiltonian with the
   frozen potential, all states with E in the window. Return energies, psi(x, y).
2. `classify(psi)`: x-parity (+1 even / -1 odd) from <psi|P_x|psi>, y-parity, and the channel index = overlap with the
   lead transverse modes. Keep only x-even/odd states of channel 1 (y-even, overlap with phi_1 > 0.95 in the lead
   window |x| in [x1, x2]). Report the dropped states (energy, parity, overlap) so I can see them.
3. `lead_phase(psi_1d, k, x1, x2)`: method M2. psi_1(x) = integral dy phi_1(y) psi(x, y). Get psi_1' by spectral
   derivative (FFT) on the periodic grid. Compute a(x), b(x), check the conserved flux
   |a|^2 - |b|^2 = 0 (standing wave) and fit psi_1(x) = C cos(k|x| + d) (even) or C sin(k|x| + d) (odd) in
   |x| in [x1, x2] (default 600-1500 nm; scan x1, x2 as a robustness test). k fit as a free parameter and compared with
   the lead dispersion k(E); report both.
4. `level_phase(E_n, n_index, ...)`: method M1. d = n pi - k(E_n) Lx/2 with k(E_n) from the clean-wire dispersion. The
   integer n comes from counting the even/odd levels of channel 1 from the subband bottom. Needed only mod pi.
5. `transmission(E_grid, E_e, d_e, E_o, d_o)`: unwrap d_e, d_o along E, interpolate (cubic or monotone PCHIP),
   return T(E) = cos^2(d_e(E) - d_o(E)) on the common range of even and odd levels, plus T averaged over
   E in mu +- 2 kT with weight -df/dE (Fermi derivative) as an option (Meir: "take the Fermi level state or a few
   states and average").
6. Everything per spin sigma = up, dn. Output arrays + JSON.
Local k(x): in model C the effective lead potential has Friedel oscillations (period ~71 nm) near the QPC, so use the
fit window far away and report the scatter of d versus (x1, x2) as the systematic error.

## Tests (tests/test_ring_transmission.py)
1. 1D Poschl-Teller ring, hbar = m = 1, L = 200, N = 1024 spectral kinetic matrix, d = 2, V0 = 0.6, x grid
   symmetric around x = 0. Exact T(k) = sinh^2(pi k d) / (sinh^2(pi k d) + cosh^2(pi sqrt(2 V0 d^2 - 1/4))). The
   method must agree to better than 5e-3 for k in [0.45, 1.5].
2. Same 1D problem with L = 300: T unchanged within 5e-3 (cell-size independence).
3. Symmetry: a potential without mirror symmetry must raise an error or warning, not silently return T.
4. Unitarity-type sanity: 0 <= T <= 1; T -> 1 for V0 -> 0 (then d_e = d_o up to pi).

## Phase 0: non-interacting validation (2D, same cell as production)
- Potential: bare V_ext(x, y) = (1/2) omega_y^2 y^2 + V_QPC(x, y) (HMW Eq. 1), no Hartree, no xc, no spin.
- hbar*omega_x in {1.0, 1.5, 2.0} meV, V0 = 3.0 meV; also a longer case hbar*omega_x = 0.5 meV (d is larger, so the
  adiabatic reference is better).
- Compare T(E) with the 1D adiabatic Poschl-Teller formula on E_1D = E - hbar*omega_y/2 (exact for an adiabatic
  channel). Report the maximum deviation, and where it occurs. Expected: small and decreasing with d; if the deviation
  is a few percent for 1.0 meV, say so rather than tuning.
- Compare M1 and M2 phases; run the Lx = 5000 vs 7500 nm check.
- Gate: do not start Phase A until Phase 0 agrees. Report the table and one figure (T vs E: ring method points, interpolated
  curve, adiabatic exact curve).

## Phase A: frozen-potential T(E) in the LSDA state
- Load the converged B = 0 polarised state for model C, exchange-like, hbar*omega_x = 1.5 (from the existing results;
  if not on disk, rerun the Janak ramp from 5 T and report the time first).
- For each spin: diagonalise with V_sigma frozen, extract T_sigma(E) for E from the subband-1 bottom up to
  E_2 - 0.02 meV (single-channel limit, see Step 0.3). State clearly in the figure that this range ends below the
  barrier top, so T_dn stays small there.
- Report T_up(mu), T_dn(mu), G(mu)/G0 with the window-averaged value and the scatter from (x1, x2).
- Figure `figures/transmission/fig_T_vs_E.png`: T_up(E), T_dn(E) vs E - mu, mu marked, the unpolarised-state T for
  reference, and the non-interacting T from Phase 0 at the same hbar*omega_x.
- Caveat to print on the figure/notes: this is T(E) in the potential self-consistent at mu, not a self-consistent
  function of mu.

## Phase B: T at fixed mu versus V0
- hbar*omega_x = 1.5 meV fixed, so d = sqrt(2 V0 hbar^2 / m*) / (hbar*omega_x) changes with V0 (say so in the notes).
- V0 grid (meV): 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0; add points where T crosses 0.5 after seeing the first pass.
- For each V0: full Janak ramp 5, 4, 2, 1, 0.5, 0.25, 0 T, each field seeded from the previous one and the 5 T start
  seeded from the converged 5 T state of the neighbouring V0; mu(B) from the cached model C clean wire. Resumable with
  done-markers (overnight.py style); push a summary after each V0. Time the first V0, print the total estimate, wait
  for my go before the full sweep.
- At each V0: T_up(mu), T_dn(mu) from the B = 0 state (Phase A procedure), net spin, zeta0, convergence flag.
- Unconverged states are listed but never used in conclusions.
- Figure `figures/transmission/fig_T_vs_V0.png`: T_up, T_dn, and G/G0 = T_up + T_dn vs V0, with the non-interacting
  T(mu) (Phase 0 potential, same mu) as dashed line; second panel: net spin and zeta0 vs V0 on the same axis.
  Optional: mark where the unpolarised branch is unstable (lambda > 1).

## Deliverables and rules
- Code: qpc/ring_transmission.py, scripts/run_ring_T.py (resumable), tests/test_ring_transmission.py.
- Results: results/transmission/<tag>/*.json and summary.csv; notes/RING_T_SUMMARY.md (tables, method, caveats,
  list of dropped states); figures in figures/transmission/.
- Never state a number that is not in a JSON/CSV file. Say explicitly what was not run.
- Do not use Kwant; do not write a recursive Green's function solver for this task.
- Python 3.12, numpy/scipy only; keep the same units module and conventions.
- Scientific caveats to put in the notes: KS-Landauer on LSDA is a static mean-field conductance, not the Kondo picture
  of HMW; the method needs a mirror-symmetric potential and a single propagating channel; Friedel oscillations make k
  position dependent near the QPC; the ring levels are discrete (spacing ~0.06 meV), so T(E) is interpolated, not sampled.
