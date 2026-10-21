# Two-electron scan, Phase 1 (2026-10-21 23:28)

Model C, exchange, E_cut 15 meV, Ly 320 nm, uniform field, ramp 5 -> 0 T. B = 0 states. L_D: length of D = {n < 0.9 n_lead} [nm]; N_D electrons in D; lobes in D as x(spin) [nm (e)]; Friedel: mean same-sign spacing of the lobes outside D vs pi/k_F = 2/n_lead [nm]; eps0max-mu: top of eps_0,s(x) minus mu (up/dn) [meV]; dOmega = Omega_pol - Omega_unpol [meV]; LDOS peaks below mu at the lobes (up/dn).

class: moments in D after merging same-sign lobes with a shallow dip (s > 0.5 peak) and dropping lobes with |spin| < 0.3 (satellites); raw: the spec's literal peak count (OPEN_QUESTIONS #13).

| case | d | conv/it | M_loc | net spin | L_D | N_D | class (raw) | moments x(spin) | satellites | broad | Friedel / 2/n | eps0max-mu | dOmega | LDOS pk<mu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_exch_wx0.35_V2.5 | 215 | NO/300 | 2.657 | 2.657 | 600 | 12.258 | single (multi9) | +29(+2.84) | -267(+0.08) -229(-0.11) -181(+0.19) -133(-0.21) +133(-0.21) +181(+0.19) +229(-0.11) +267(+0.08) | y | nan / 71 | -0.261/+0.519 | +0.07664 | 3/0 |
| C_exch_wx0.35_V3.0 | 236 | y/208 | 2.952 | 2.952 | 733 | 14.632 | multi3 (multi12) | -152(-0.35) -0(+3.32) +152(-0.35) | -333(-0.09) -295(+0.12) -248(-0.17) -200(+0.27) +200(+0.27) +248(-0.17) +295(+0.12) +333(-0.09) | - | nan / 71 | -0.104/+0.607 | -0.07574 | 2/1; 2/0; 2/1 |
| C_exch_wx0.35_V3.5 | 255 | y/20 | 3.934 | 3.934 | 829 | 15.959 | multi3 (multi11) | -190(-0.36) -67(+4.32) +190(-0.36) | -371(-0.09) -333(+0.13) -286(-0.18) -238(+0.27) +238(+0.27) +286(-0.18) +333(+0.13) +371(-0.09) | - | 76 / 71 | -0.081/+0.765 | -0.20517 | 2/1; 4/0; 2/1 |
| C_exch_wx0.5_V2.5 | 151 | NO/300 | 1.999 | 1.999 | 429 | 8.750 | single (multi7) | +0(+2.24) | -200(-0.08) -162(+0.14) -114(-0.20) +114(-0.20) +162(+0.14) +200(-0.08) | y | nan / 71 | -0.397/+0.350 | +0.03206 | 2/0 |
| C_exch_wx0.5_V3.0 | 165 | y/17 | 1.980 | 1.980 | 505 | 9.985 | single (multi7) | +0(+2.33) | -210(-0.11) -171(+0.19) -124(-0.30) +124(-0.30) +171(+0.19) +210(-0.11) | y | 76 / 71 | -0.221/+0.615 | -0.05829 | 2/0 |
| C_exch_wx0.5_V3.5 | 178 | y/20 | 2.920 | 2.920 | 581 | 11.147 | single (multi9) | -38(+3.23) | -286(+0.07) -248(-0.10) -200(+0.17) -152(-0.26) +152(-0.26) +200(+0.17) +248(-0.10) +286(+0.07) | y | nan / 71 | -0.256/+0.706 | -0.14008 | 3/0 |
| C_exch_wx0.7_V2.5 | 108 | NO/300 | 0.883 | 0.883 | 295 | 5.953 | single (multi5) | +10(+1.09) | -133(+0.09) -86(-0.16) +86(-0.16) +133(+0.09) | - | nan / 71 | -0.432/+0.087 | -0.00034 | 1/0 |
| C_exch_wx0.7_V3.0 | 118 | NO/300 | 2.136 | 2.136 | 352 | 6.844 | single (multi5) | +0(+2.36) | -162(+0.11) -114(-0.18) +114(-0.18) +162(+0.11) | y | 76 / 71 | -0.448/+0.586 | -0.02940 | 2/0 |
| C_exch_wx0.7_V3.5 | 127 | y/18 | 2.011 | 2.011 | 390 | 7.290 | single (multi5) | +0(+2.30) | -171(+0.12) -124(-0.22) +124(-0.22) +171(+0.12) | y | 76 / 71 | -0.370/+0.736 | -0.10169 | 2/0 |
| C_exch_wx1.0_V2.5 | 75 | y/24 | 0.232 | 0.232 | 200 | 3.956 | single (multi3) | +0(+0.46) | -57(-0.15) +57(-0.15) | - | nan / 71 | -0.458/+0.089 | -0.00657 | 1/0 |
| C_exch_wx1.0_V3.0 | 83 | y/15 | 1.094 | 1.094 | 238 | 4.535 | single (multi4) | -0(+1.30) | -86(-0.15) +86(-0.15) | - | 67 / 71 | -0.542/+0.349 | -0.03215 | 1/0 |
| C_exch_wx1.0_V3.5 | 89 | y/19 | 1.214 | 1.214 | 276 | 5.116 | single (multi3) | -38(+1.48) | -95(-0.18) +95(-0.18) | - | 67 / 71 | -0.478/+0.476 | -0.06839 | 2/0 |

## Ramp: M_loc at each field (NO = not converged)

| case | 5 T | 4 T | 2 T | 1 T | 0.5 T | 0.25 T | 0 T |
|---|---|---|---|---|---|---|---|
| C_exch_wx0.35_V2.5 | 1.901 | 1.944 | 2.305 NO | 2.712 NO | 2.575 NO | 2.701 NO | 2.657 NO |
| C_exch_wx0.35_V3.0 | 3.320 | 2.902 | 3.148 NO | 3.487 NO | 3.811 NO | 3.599 NO | 2.952 |
| C_exch_wx0.35_V3.5 | 4.183 | 3.825 | 4.056 | 4.017 NO | 3.767 | 3.851 | 3.934 |
| C_exch_wx0.5_V2.5 | 1.424 | 1.352 | 1.288 NO | 1.609 NO | 2.001 NO | 1.979 NO | 1.999 NO |
| C_exch_wx0.5_V3.0 | 2.285 | 2.236 | 2.414 NO | 2.821 NO | 1.880 | 1.930 | 1.980 |
| C_exch_wx0.5_V3.5 | 2.915 | 2.960 | 2.554 | 2.728 | 2.825 | 2.874 | 2.920 |
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
