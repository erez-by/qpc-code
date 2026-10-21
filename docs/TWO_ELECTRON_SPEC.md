# Spec: two localized electrons in a longer QPC (for Claude Code)

## Physics goal
A longer QPC (smaller hbar*omega_x, so d = sqrt(2 V_0 hbar^2/m*) / (hbar*omega_x) grows) has a longer low-density
region, which may host TWO localized electrons instead of one. Find out:
(a) at which length the spin density s(x) = ∫dy (n_up - n_dn) splits into two lobes;
(b) whether the two localized spins prefer parallel (P, triplet-like) or antiparallel (AP, singlet-like) alignment,
    by comparing the grand potentials of the two self-consistent solutions.
Model C (hartree_scale=0.8, interacting leads), interp="exchange", the same cell / E_cut / grid as before.
Do not change SCF physics except the optional staggered Zeeman term below.

## Step 0: inspect and report before coding
1. Is a total-energy / grand-potential function implemented? If yes, which formula, and does it include the
   Hartree scale 0.8 consistently?
2. Where the Zeeman term enters (ks_potentials), and whether it can take an array E_Z(x) instead of a scalar.
3. Confirm d stays << Lx/2 for hbar*omega_x down to 0.25 meV at V_0 up to 3.5 meV (d is about 290 nm there).

## Phase 1: uniform-field scan (existing code)
- Grid: hbar*omega_x in {1.0 (reference, compare with the existing run), 0.7, 0.5, 0.35, 0.25} meV
  x V_0 in {2.5, 3.0, 3.5} meV.
- Ramp B = 5, 4, 2, 1, 0.5, 0.25, 0 T (uniform Zeeman, as usual); mu(B) from the cached model C clean wire.
- Seed each run's 5 T start from the converged 5 T density of the neighbouring V_0 at the same hbar*omega_x.
- kT = 0.05 meV. Afterwards, rerun at kT = 0.02 meV the cases that show two lobes or a broad moment.
- Resumable with done-markers (overnight.py style). Time the first run, print the total estimate, wait for go.

## Analysis module: qpc/analysis_moments.py
For each converged B = 0 state:
1. Profiles: n(x) = ∫dy (n_up + n_dn), s(x) = ∫dy (n_up - n_dn).
2. Depleted region: D = {x : n(x) < 0.9 n_lead}, where n_lead is the clean-wire 1D density. Report its length L_D.
3. Lobe detection inside D only: peaks of |s(x)| with prominence > 0.3e-2 nm^-1 (scipy.signal.find_peaks on s and
   on -s). For each lobe: position, sign, and integrated spin over the lobe (between the neighbouring zeros of s).
   Classification: 0 lobes = no moment; 1 lobe = single moment; 2 lobes of the same sign = P; opposite signs = AP.
   Also report the lobes outside D and their spacing, which should be about pi/k_F = 2/n_lead (Friedel
   oscillations). Do not count those as localized electrons.
4. Net spin ∫ s dx, and the number of electrons in D: N_D = ∫_D n dx.
5. Effective spin-resolved barriers eps_{0,sigma}(x): the lowest transverse eigenvalue of V_eff,sigma(x, y) for each x
   (finite differences in y, eigh_tridiagonal), with mu marked.
6. LDOS at the lobe positions (reuse the existing LDOS code): count the quasi-bound peaks.
Save everything to results/two_e/<tag>/analysis.json plus a summary.csv row
(hbar_wx, V0, L_D, N_D, net_spin, n_lobes, lobe_signs, lobe_spins, converged).

## Phase 2: staggered ramp (only for cases with 2 lobes or a broad single lobe)
- Add an optional position-dependent Zeeman energy to ks_potentials:
  E_Z,sigma(x) = sigma * (1/2) g* mu_B B * tanh(x / w), with w = 20 nm, centred at x = 0.
  The default stays the uniform field (all existing results unchanged; add a regression test for this).
- Run the same field list 5 -> 0 T with the staggered field. At B = 0 the Hamiltonian is identical to Phase 1;
  only the starting branch differs.
- Expected: AP solution with s(x) roughly antisymmetric and net spin about 0. If it instead converges back to P (or
  to no moment), report that; it is a result too (AP is not even metastable there).

## Phase 3: compare the energies of P and AP
- The leads are at fixed mu and finite kT, so compare the GRAND POTENTIAL, not E:
  Omega = -kT * sum_{i,sigma} ln(1 + exp(-(eps_{i,sigma} - mu)/kT))
          - E_H[n]  - sum_sigma ∫ v_xc,sigma n_sigma  + E_xc[n_up, n_dn]
  (the eps_i already contain V_ext, V_H and v_xc; the last three terms remove double counting).
  E_H = (1/2) * hartree_scale * ∫∫ n K n, with the same gate-screened kernel and the same 0.8 scale used in the SCF.
- Delta = Omega_AP - Omega_P. Delta < 0: antiparallel ground state; Delta > 0: parallel.
- Precision: Delta may be micro-eV. Tighten the SCF tolerance for these runs until Omega changes < 1e-4 meV between
  the last iterations, and report that convergence. Both states use the same cell, grid and mu, so the supercell
  errors cancel in the difference.
- Test: for a single-lobe case, Omega from the formula agrees with a finite-difference check d Omega / d mu = -N
  (vary mu by +-1e-3 meV), to within 1e-3.

## Figures (scripts/plot_two_e.py -> figures/two_e/)
1. Grid of panels (rows hbar*omega_x, columns V_0): s(x) solid, n(x)/2 dashed, the depleted region D shaded,
   in the same style as the existing Fig. 1 density plot.
2. Number of lobes and net spin vs L_D (or vs hbar*omega_x), with the estimate L_D ≈ 2/n_center marked.
3. P vs AP for the selected cases: s(x) overlaid; eps_{0,up}(x), eps_{0,dn}(x) for both.
4. Delta = Omega_AP - Omega_P vs hbar*omega_x.

## Do not
- Count Friedel lobes outside D as electrons.
- Decide the ground state from which run converged; only Delta decides.
- Change the default (uniform) Zeeman behaviour.
