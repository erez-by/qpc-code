# Two-electron scan, Phase 1 (2026-10-21 14:17)

Model C, exchange, E_cut 15 meV, Ly 320 nm, uniform field, ramp 5 -> 0 T. B = 0 states. L_D: length of D = {n < 0.9 n_lead} [nm]; N_D electrons in D; lobes in D as x(spin) [nm (e)]; Friedel: mean same-sign spacing of the lobes outside D vs pi/k_F = 2/n_lead [nm]; eps0max-mu: top of eps_0,s(x) minus mu (up/dn) [meV]; dOmega = Omega_pol - Omega_unpol [meV]; LDOS peaks below mu at the lobes (up/dn).

class: moments in D after merging same-sign lobes with a shallow dip (s > 0.5 peak) and dropping lobes with |spin| < 0.3 (satellites); raw: the spec's literal peak count (OPEN_QUESTIONS #13).

| case | d | conv/it | M_loc | net spin | L_D | N_D | class (raw) | moments x(spin) | satellites | broad | Friedel / 2/n | eps0max-mu | dOmega | LDOS pk<mu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_exch_wx0.7_V2.5 | 108 | NO/300 | 0.883 | 0.883 | 295 | 5.953 | single (multi5) | +10(+1.09) | -133(+0.09) -86(-0.16) +86(-0.16) +133(+0.09) | - | nan / 71 | -0.432/+0.087 | -0.00034 | 1/0 |
| C_exch_wx0.7_V3.0 | 118 | NO/300 | 2.136 | 2.136 | 352 | 6.844 | single (multi5) | +0(+2.36) | -162(+0.11) -114(-0.18) +114(-0.18) +162(+0.11) | y | 76 / 71 | -0.448/+0.586 | -0.02940 | 2/0 |
| C_exch_wx0.7_V3.5 | 127 | y/18 | 2.011 | 2.011 | 390 | 7.290 | single (multi5) | +0(+2.30) | -171(+0.12) -124(-0.22) +124(-0.22) +171(+0.12) | y | 76 / 71 | -0.370/+0.736 | -0.10169 | 2/0 |
| C_exch_wx1.0_V2.5 | 75 | y/24 | 0.232 | 0.232 | 200 | 3.956 | single (multi3) | +0(+0.46) | -57(-0.15) +57(-0.15) | - | nan / 71 | -0.458/+0.089 | -0.00657 | 1/0 |
| C_exch_wx1.0_V3.0 | 83 | y/15 | 1.094 | 1.094 | 238 | 4.535 | single (multi4) | -0(+1.30) | -86(-0.15) +86(-0.15) | - | 67 / 71 | -0.542/+0.349 | -0.03215 | 1/0 |
| C_exch_wx1.0_V3.5 | 89 | y/19 | 1.214 | 1.214 | 276 | 5.116 | single (multi3) | -38(+1.48) | -95(-0.18) +95(-0.18) | - | 67 / 71 | -0.478/+0.476 | -0.06839 | 2/0 |

## Ramp: M_loc at each field (NO = not converged)

| case | 5 T | 4 T | 2 T | 1 T | 0.5 T | 0.25 T | 0 T |
|---|---|---|---|---|---|---|---|
| C_exch_wx0.7_V2.5 | 1.223 | 1.107 | 0.796 | 0.831 | 0.829 | 0.786 NO | 0.883 NO |
| C_exch_wx0.7_V3.0 | 1.713 | 1.631 | 1.664 | 1.690 | 1.621 NO | 1.784 NO | 2.136 NO |
| C_exch_wx0.7_V3.5 | 2.134 | 2.170 | 1.848 | 1.897 | 1.951 | 1.981 | 2.011 |
| C_exch_wx1.0_V2.5 | 0.850 | 0.815 | 0.769 | 0.640 | 0.303 | 0.255 | 0.232 |
| C_exch_wx1.0_V3.0 | 1.252 | 1.226 | 1.064 | 1.057 | 1.072 | 1.083 | 1.094 |
| C_exch_wx1.0_V3.5 | 1.484 | 1.500 | 1.481 | 1.265 | 1.214 | 1.210 | 1.214 |

## Ramp start: 5 T (this scan) vs 6 T (existing C_exch_wx1.0), wx 1.0, V0 3.0, B = 0

M_loc 6 T start 1.0935 (15 it), 5 T start 1.0935 (15 it): difference +0.0000; zeta0 0.921 vs 0.921; M_win300 1.1180 vs 1.1180

## d Omega / d mu = -N check (wx 1.0, V0 3.0, B = 0, mu +- 1e-3 meV, tol 1e-6)

class multi4: dOmega/dmu = -137.72449, -N = -137.72449, relative deviation 3.06e-08 -> PASS (1e-3); converged [True, True]

## Cell size: wx 0.25, V0 3.0, Lx 7500 vs 5000 nm (B = 0)

(not yet)

## Timing estimate
```
TE01 (wx 1.0, V0 3.0, ramp 5 -> 0 T + unpol + energies + analysis + dOmega/dmu check): 14.1 min
remaining Lx 5000 cases: 14 x ~14 min = 3.3 h (longer QPCs may need more iterations per field: treat as a lower bound)
Lx 7500 check: ~48 min (dense diagonalisation ~ n_pw^3: x 3.4)
kT 0.02 reruns: ~14 min each, number decided by the scan (0 ... 15)
Phase 1 total without the reruns: ~4.1 h
```
