# Ring (even/odd phase-shift) transmission (2026-11-12 19:52)

Method (Meir): T(E) = cos^2(d_e(E) - d_o(E)) from the x-even / x-odd channel-1 eigenstates of the periodic cell; M1: d = n pi - k(E_n) Lx/2 with the exact lead dispersion of the plane-wave Hamiltonian; M2: fit of psi_1(x) = int phi_1 psi dy to cos(k|x| + d) / sign(x) sin(k|x| + d) in 600 <= |x| <= 1500 nm (free k). Phases interpolated with PCHIP; T is not evaluated for E < E1 + 0.05 meV (band bottom, k -> 0: ill-defined phases of the first levels gave errors up to 4e-2 there). Code: qpc/ring_transmission.py, scripts/run_ring_T.py. Old Kwant/Numerov results: git tag transport-v1, results/archive/transport_kwant_numerov_v1/.

Range: channel 1 is the only y-EVEN channel up to E3 (V is exactly mirror symmetric in y, so the y-odd channel 2 above E2 does not mix); the spec's limit was E2. Both edges are listed; the curves go to E3 - 0.02 meV.

## Phase 0: bare HMW potential (no Hartree, no xc, no spin), V0 = 3.0 meV

| case | d [nm] | E1 / E2 / E3 [meV] | barrier top: cutoff / grid basis [meV] | ch-1 states | dropped | max\|ring - PT\| (at E) | max\|ring - Numerov cutoff\| | max\|ring - Numerov grid\| | max\|T_M1 - T_M2\| | M2 window scatter | M1-M2 phase (E > E1 + 0.02) [rad] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wx 1, Lx 5000 | 83 | 1.0000 / 3.0000 / 5.0001 | 4.0135 / 4.0000 | 131 | 0 | 1.70e-02 (3.994) | 3.28e-03 | 1.70e-02 | 8.94e-06 | 2.52e-05 | 1.77e-03 |
| wx 1.5, Lx 5000 | 55 | 1.0000 / 3.0000 / 5.0001 | 4.0135 / 4.0000 | 131 | 0 | 1.10e-02 (4.036) | 1.77e-03 | 1.10e-02 | 1.42e-05 | 1.74e-05 | 1.61e-03 |
| wx 2, Lx 5000 | 41 | 1.0000 / 3.0000 / 5.0001 | 4.0135 / 4.0000 | 133 | 0 | 7.98e-03 (3.916) | 1.57e-03 | 7.98e-03 | 1.12e-05 | 2.36e-05 | 1.19e-03 |
| wx 0.5, Lx 5000 | 165 | 1.0000 / 3.0000 / 5.0001 | 4.0135 / 4.0000 | 129 | 0 | 4.54e-02 (4.014) | 4.12e-03 | 4.54e-02 | 4.42e-04 | 5.56e-03 | 3.75e-02 |
| wx 1.5, Lx 7500 | 55 | 1.0000 / 3.0000 / 5.0001 | 4.0135 / 4.0000 | 199 | 0 | 1.09e-02 (4.028) | 1.76e-03 | 1.09e-02 | 7.53e-06 | 1.48e-05 | 7.22e-04 |

## Phase A: frozen B = 0 potential, model C, exchange, hbar w_x 1.5 meV, V0 3.0 meV, kT 0.05 meV

Headline: ring method M1 with the frozen leads padded to Lx = 10000 nm (the SCF cell is 5000 nm). Interpolation test on the P state, spin up: T(mu) = 0.7957 / 0.7806 / 0.7811 for Lx = 5000 / 10000 / 20000 nm, i.e. converged at 10000 nm; the 5000 nm value carries ~+0.015 interpolation error because T(E) has structure on the scale of the level spacing (0.06 meV per parity).

| state | spin | T(mu) ring Lx 10000 | T(mu) ring Lx 5000 | M2 (600-1500) | M2 window spread | <T> mu +- 2kT | 1D Numerov cutoff basis | 1D Numerov old (continuum) | barrier top - mu: cutoff / continuum [meV] | ch-1 states | dropped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P (conv y, M = 0.834) | up | 0.7805 | 0.7956 | 0.8004 | 4.1e-02 | 0.8198 | 0.7700 | 0.7762 | -0.6008 / -0.6220 | 227 | 0 |
| P (conv y, M = 0.834) | dn | 0.0752 | 0.0734 | 0.0785 | 7.6e-03 | 0.0858 | 0.0765 | 0.0785 | +0.3462 / +0.3369 | 227 | 0 |
| P | G = T_up + T_dn | **0.8558 e^2/h** (= 0.4279 x 2e^2/h); <G>_2kT 0.9057; old Numerov 0.8547 | | | | | | | | | |
| unpol (conv y, M = 0.000) | up | 0.8791 | 0.8791 | 0.8788 | 1.0e-03 | 0.8721 | 0.8706 | 0.8823 | -0.1588 / -0.1740 | 227 | 0 |
| unpol (conv y, M = 0.000) | dn | 0.8791 | 0.8791 | 0.8788 | 1.0e-03 | 0.8721 | 0.8706 | 0.8823 | -0.1588 / -0.1740 | 227 | 0 |
| unpol | G = T_up + T_dn | **1.7582 e^2/h** (= 0.8791 x 2e^2/h); <G>_2kT 1.7441; old Numerov 1.7647 | | | | | | | | | |

Ring vs 1D Numerov: the converged ring value is 2D-exact in the frozen potential (it includes transverse mode mixing); for the P spin-up channel it lies 0.011 above 1D Numerov on the same Hamiltonian, a difference that does not shrink with Lx (non-adiabatic, not numerical). M2 fitted close to the QPC (400-1500 nm) is biased by the Friedel oscillations of the self-consistent potential (window spread listed); M1 uses only level energies and the clean lead at the cell edge.


Reading: 'Numerov cutoff' isolates the ring method (same transverse operator as the KS Hamiltonian; the difference is non-adiabatic mixing + level interpolation). The PT formula and 'Numerov grid' use the exact continuum barrier; their difference to the ring result is dominated by the plane-wave cutoff, which raises the channel-1 barrier top (cutoff vs grid column).

Caveats: KS-Landauer on LSDA is a static mean-field conductance (not the Kondo picture of HMW); the method needs a mirror-symmetric potential and a single channel per symmetry sector; the ring levels are discrete, so T(E) is interpolated between levels, not sampled; Friedel oscillations make k position dependent near the QPC (M2 window scan). Phase B: not run yet.
