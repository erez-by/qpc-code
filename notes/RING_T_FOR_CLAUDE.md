# Transmission of a spin-DFT quantum point contact: ring (even/odd phase-shift) method — results summary

Context for the reader: this is a self-contained summary of a numerical study, written for analysis of the results.
All numbers below are copied from result files in the repository (paths at the end). Nothing here was estimated.

## 1. The system

- **Model.** 2D plane-wave Kohn–Sham spin-DFT (LSDA) of a quantum point contact (QPC) in a GaAs wire, following
  Hirose, Meir & Wingreen, PRL 90, 026804 (2003) (HMW).
  - Effective GaAs units: m* = 0.067, ε = 12.9; 1 Ha* = 10.956 meV, 1 a* = 10.189 nm.
  - Periodic cell Lx × Ly = 5000 nm × 320 nm, plane-wave cutoff 15 meV (525 × 35 grid, 3379 plane waves).
  - Fermi smearing kT = 0.05 meV. The cell has one QPC at x = 0.
- **External potential (HMW Eq. 1).**
  - V(x) = V0 / cosh²(x/d), with d = √(2V0/m*)/ω_x, so the curvature ħω_x is fixed while V0 varies.
  - The transverse confinement frequency is ω_y + V(x)/ħ, with ħω_y = 2.0 meV.
  - The lowest transverse level along the axis is ε0(x) = ħω_y/2 + V(x), a Pöschl–Teller barrier.
- **Interactions ("model C").** Interacting clean-wire leads at fixed electron density 2.8e-2 nm⁻¹, Hartree scaled by
  0.8, and the "exchange" spin interpolation of the 2D correlation energy (Tanatar–Ceperley).
  - The Hartree kernel is gate-screened: image charge at 100 nm.
  - The 0.8 Hartree scale is a sensitivity knob chosen so that μ − e0(lead) = 1.07 meV, close to HMW's ~1.1 meV. Its
    clean wire has 1.9 electrons in subband 2 at kT = 0.05 meV.
- **Clean lead (model C).**
  - Subband bottoms: E1 = 6.1364 meV, E2 = 7.3378 meV (first y-odd mode), E3 = 9.06 meV (second y-even mode).
  - μ = 7.2087 meV, so μ − E1 = 1.072 meV and E2 − μ = +0.129 meV.
- **SCF states.**
  - Fixed μ (grand canonical); Pulay mixing; convergence on the density residual, tol 1e-4.
  - The B = 0 polarised state is reached by a "Janak ramp" in uniform in-plane Zeeman field: 5, 4, 2, 1, 0.5, 0.25, 0 T,
    each field started from the previous converged density.
  - "Net spin" M_loc = N_up − N_dn of the QPC (leads unpolarised at B = 0); ζ0 = spin polarisation of the line density
    at x = 0.
  - The unpolarised reference is the spin-restricted SCF at B = 0.

## 2. The transmission method (Meir's even/odd phase shifts on the ring)

- **Principle.** The KS potential V_σ(x, y) of each spin is mirror symmetric in x (checked: max|V(x) − V(−x)| ≈ 1e-12 meV),
  so the eigenstates of the periodic cell are x-even or x-odd standing waves. In a clean lead:
  - ψ_e ∝ cos(k|x| + d_e) and ψ_o ∝ sign(x) sin(k|x| + d_o);
  - t = (e^{2i d_e} + e^{2i d_o})/2, so **T(E) = cos²(d_e(E) − d_o(E))**;
  - ring quantisation: k_n Lx/2 + d(E_n) = nπ for both parities.
- **Steps.**
  - Dense diagonalisation of the single-spin KS Hamiltonian in the frozen self-consistent potential.
  - Classification by x-parity, y-parity and overlap with the lead's lowest transverse mode (> 0.95).
  - Phases from either:
    - **M1:** d = nπ − k(E_n) Lx/2, with k(E) taken from the exact lead dispersion of the plane-wave Hamiltonian,
      including the cutoff circle;
    - **M2:** fit of the channel-projected wavefunction in a lead window 600 ≤ |x| ≤ 1500 nm.
  - d_e(E) and d_o(E) are interpolated separately (PCHIP) and T = cos²(d_e − d_o) is formed on an energy grid.
  - **M1 is the headline method.** M2 is biased by the Friedel oscillations of the self-consistent potential near the QPC.
- **Range.** V is exactly mirror symmetric in y, so the y-odd channel 2 (opening at E2) never mixes with the y-even
  channel 1. The single-channel formula therefore holds up to E3, not only E2. T is not evaluated for E < E1 + 0.05 meV:
  the band-bottom phases are ill-defined there.
- **Lead padding.** For the converged SCF states the frozen potential is re-embedded in a 10 µm cell, the 5 µm cell
  with 2.5 µm of its own flat lead added on each side. This halves the ring level spacing. The SCF itself is always
  the 5 µm cell.
- **Conductance.** Mean-field, static: G = (e²/h)(T_up(μ) + T_dn(μ)); 2e²/h corresponds to G = 2 in these units.
  A Fermi-window average over μ ± 2kT is also listed.

## 3. Validation (no interactions)

- **1D Pöschl–Teller ring** (ħ = m = 1, L = 200, N = 1024, d = 2, V0 = 0.6):
  - |T − T_exact| ≤ 1.1e-4 for k ∈ [0.45, 1.5], with M1 and M2 identical (spec tolerance 5e-3);
  - L = 300 gives 3.6e-5;
  - an asymmetric potential raises an error;
  - V0 → 0 gives T = 1 to 1e-6.
- **Phase 0: bare HMW potential in the production 2D cell** (V0 = 3 meV, no Hartree/xc/spin). Maximum |T difference|
  for E > E1 + 0.05 meV:

| ħω_x (meV) | d (nm) | ring vs 1D Numerov, same Hamiltonian | M1 vs M2 | ring vs analytic Pöschl–Teller |
|---|---|---|---|---|
| 2.0 | 41 | 1.6e-3 | 1e-5 | 8.0e-3 |
| 1.5 | 55 | 1.8e-3 (also 1.8e-3 at Lx = 7500) | 1e-5 | 1.1e-2 |
| 1.0 | 83 | 3.3e-3 | 1e-5 | 1.7e-2 |
| 0.5 | 165 | 4.1e-3 | 4e-4 | 4.5e-2 |

- **Reading of Phase 0.**
  - The ring method agrees with a 1D Numerov solution on the same transverse operator to < 4e-3.
  - The larger gap to the analytic formula comes entirely from the plane-wave cutoff, which raises the channel-1
    barrier top from 4.0000 to 4.0135 meV. The predicted gap, (dT/dE) × 0.0135 meV, matches the measured gap
    (≈ 0.042 predicted vs 0.045 measured for ħω_x = 0.5).

## 4. Phase A: frozen B = 0 potential, model C, ħω_x = 1.5 meV, V0 = 3.0 meV

- **Interpolation check (padding).** P state, spin up, T(μ) = 0.7957 / 0.7806 / 0.7811 for the ring at Lx = 5000 / 10000 /
  20000 nm. The result is converged at 10 µm; the 5 µm value carries ≈ +0.015 interpolation error because T_up(E) has a
  dip at μ narrower than the level spacing (0.06 meV per parity).

| state | spin | T(μ), ring at Lx 10000 | T(μ), ring at Lx 5000 | ⟨T⟩ over μ ± 2kT | 1D Numerov, same Hamiltonian | 1D Numerov, continuum barrier | barrier top − μ (meV) |
|---|---|---|---|---|---|---|---|
| polarised P (M = 0.834) | up | 0.7805 | 0.7956 | 0.8198 | 0.7700 | 0.7762 | −0.601 |
| polarised P | down | 0.0752 | 0.0734 | 0.0858 | 0.0765 | 0.0785 | +0.346 |
| unpolarised | either | 0.8791 | 0.8791 | 0.8721 | 0.8706 | 0.8823 | −0.159 |

- **Conductance.**
  - Polarised: G = 0.856 e²/h (0.43 × 2e²/h); window average 0.906 e²/h. The old 1D Numerov method gives 0.855 e²/h.
  - Unpolarised: G = 1.758 e²/h (0.88 × 2e²/h).
- **Non-interacting barrier at the same energy above the band bottom** (μ − E1 = 1.072 meV): T ≈ 4e-5.
- **Converged ring vs 1D Numerov (same Hamiltonian).** The ring T_up(μ) is 0.011 higher, and this gap does not shrink
  with Lx. It is interpreted as transverse mode mixing (2D, non-adiabatic), which the ring method includes and the
  1D Numerov leaves out.
- **Shape of T(E) (figure fig_T_vs_E.png).**
  - T_up(E) reaches ≈ 1 near μ − 0.6 meV and has a dip to ≈ 0.78 right at μ.
  - T_dn(E) has a shoulder near μ and rises to 1 by μ + 1 meV.

## 5. Phase B: T at fixed μ versus barrier height V0 (ħω_x = 1.5 meV fixed, so d grows with V0)

- **Runs.** A full SCF (Janak ramp 5 → 0 T) per V0. The 5 T start of each V0 is seeded from the neighbouring V0's 5 T
  state; V0 = 3.0 starts from the clean wire. T is computed from the B = 0 polarised state (headline: ring M1, leads
  padded to 10 µm) and from the unpolarised reference.
- **Convergence.** All 7 polarised B = 0 states and all 7 unpolarised references converged.
- **V0 = 5.0 was not completed.** Its SCF converged (net spin 1.12, ζ0 0.95) but its unpolarised reference did not
  (300 iterations). The transmission step stopped on an over-strict symmetry tolerance (5e-9 meV asymmetry, which is
  rounding noise). The user chose to skip it; no transmission exists for V0 = 5.0.

| V0 (meV) | d (nm) | net spin M_loc | ζ0 | n_1D(0) (nm⁻¹) | T_up(μ) | T_dn(μ) | G (e²/h) | ⟨T_up⟩, ⟨T_dn⟩ over μ ± 2kT | T_up, T_dn at Lx 5000 | unpolarised: T per spin, G (e²/h) | non-interacting T, same E − band bottom |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 31.8 | 0.000 | 0.00 | 0.01907 | 0.9658 | 0.9658 | 1.9315 | 0.9647, 0.9647 | 0.9657, 0.9657 | 0.9658, 1.9315 | 0.711 |
| 1.5 | 38.9 | 0.000 | 0.00 | 0.01670 | 0.9544 | 0.9544 | 1.9087 | 0.9524, 0.9524 | 0.9543, 0.9543 | 0.9544, 1.9087 | 0.173 |
| 2.0 | 45.0 | 0.000 | 0.00 | 0.01471 | 0.9391 | 0.9389 | 1.8779 | 0.9360, 0.9358 | 0.9390, 0.9388 | 0.9390, 1.8779 | 0.0144 |
| 2.5 | 50.3 | 0.357 | 0.50 | 0.01310 | 0.9997 | 0.4267 | 1.4265 | 0.9938, 0.4442 | 0.9996, 0.4245 | 0.9156, 1.8313 | 8.7e-4 |
| 3.0 | 55.1 | 0.834 | 0.81 | 0.01101 | 0.7806 | 0.0752 | 0.8558 | 0.8198, 0.0858 | 0.7957, 0.0734 | 0.8791, 1.7582 | 4e-5 |
| 3.5 | 59.5 | 1.040 | 0.91 | 0.00895 | 0.5425 | 0.0141 | 0.5566 | 0.5941, 0.0168 | 0.5542, 0.0127 | 0.8208, 1.6416 | < 1e-5 |
| 4.0 | 63.6 | 1.116 | 0.94 | 0.00692 | 0.3897 | 0.0040 | 0.3937 | 0.4372, 0.0048 | 0.3957, 0.0038 | 0.7268, 1.4536 | < 1e-5 |

- **Observations.**
  1. **No moment for V0 ≤ 2.0.** The polarised and unpolarised solutions coincide and G ≈ 1.88–1.93 e²/h, i.e. both
     spins nearly fully transmitted.
  2. **The moment appears between V0 = 2.0 and 2.5** and grows to ≈ 1.1 at V0 = 4.0.
  3. **Spin down closes first.** T_dn drops from 0.94 to 0.43 to 0.075 to 0.014 between V0 = 2.0 and 3.5, while T_up
     stays ≥ 0.78 up to V0 = 3.0. At V0 = 2.5 spin up is even fully open (T_up = 0.9997).
  4. **G passes through ≈ 0.5 × 2e²/h.** It goes 1.43 (V0 2.5) → 0.86 (3.0) → 0.56 (3.5) → 0.39 (4.0) e²/h, and is
     carried almost entirely by spin up for V0 ≥ 3.
  5. **The unpolarised reference stays much more open** at the same V0 (G = 1.83 → 1.45 e²/h). The drop in G is tied to
     the spin polarisation, not to the barrier alone.
  6. **The non-interacting barrier is closed by V0 ≈ 2** (T = 0.014). Electron screening (Hartree) keeps the
     interacting channel open to much higher V0.

## 6. Earlier method on other states (for comparison)

Earlier, a direct scattering calculation was run on the frozen potentials of longer QPCs (two-electron study, ħω_x = 0.5
and 0.35 meV): 1D adiabatic Numerov plus a 2D Kwant tight-binding check, which agreed to ≤ 2e-3 at equal spacing. These
results are kept (git tag `transport-v1`, notes/TRANSPORT_RESULTS.md).

- **P states (parallel spins, net spin 2–3):** T_dn(μ) ≈ 1e-4 to 1e-6, T_up(μ) = 0.05–0.20.
- **Antiparallel (AP) states:** T_up = T_dn = 0.11–0.25.
- **Unpolarised states:** T = 1.

These states are not the same as Phase A/B, so the numbers are not directly comparable.

## 7. Caveats (please keep in mind when interpreting)

- **Static mean field.** KS-Landauer on an LSDA ground state is a static mean-field conductance. It is not the Kondo /
  many-body picture of HMW, and the polarised solution is a broken-symmetry mean-field state.
- **T(E) is evaluated in one frozen potential.** It is T(E) in the potential self-consistent at μ, not a
  self-consistent function of μ. Only T(μ) is fully self-consistent.
- **Temperature.** kT = 0.05 meV is a numerical smearing needed by the discrete levels. The physical HMW temperature is
  0.1 K (0.0086 meV). An earlier test showed that lowering kT to 0.02 meV increases the moment at short QPCs.
- **Interpolation.** The ring levels are discrete (0.06 meV per parity at Lx = 5 µm), so T(E) is interpolated, not
  sampled. Sharp features near μ need the padded leads.
- **Single channel.** The method needs mirror symmetry and one channel per symmetry sector (valid up to E3). Channel 2
  (y-odd) opens 0.129 meV above μ and is not included in G.
- **Plane-wave cutoff.** The cutoff raises the channel-1 barrier by ≈ 0.0135 meV compared with the continuum.
- **Model C.** The Hartree scale 0.8 is a sensitivity choice, not physical screening.

## 8. Questions an analyst might address

- Does G(V0) show a structure resembling the 0.7 (or 0.5) anomaly, and how does it depend on the moment onset between
  V0 = 2.0 and 2.5?
- Is the dip of T_up(E) exactly at μ (Phase A) a generic consequence of 2k_F Friedel oscillations in a self-consistent
  potential?
- How would kT closer to the physical 0.1 K, or full Hartree screening (scale 1.0), change the onset and the G(V0) curve?

## 9. Files

| what | path |
|---|---|
| method | qpc/ring_transmission.py |
| tests | tests/test_ring_transmission.py |
| driver | scripts/run_ring_T.py |
| Phase 0 / A results | results/transmission/phase0/*.json, results/transmission/phaseA/*.json |
| Phase B table | results/transmission/phaseB/summary.csv |
| Phase B per V0 | results/transmission/phaseB/C_exch_wx1.5_V<V0>/ringT.json |
| figures | figures/transmission/fig_phase0.png, fig_T_vs_E.png, fig_T_vs_V0.png |
| notes | notes/RING_T_SUMMARY.md |
| specification | docs/RING_TRANSMISSION_SPEC.md |
