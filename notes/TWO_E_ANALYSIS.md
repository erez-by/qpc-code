# Two-electron scan: Phase 1 analysis (2026-10-22 08:39)

Existing B = 0 states (Lx 5000 nm, kT 0.05 meV); no new SCF. NOT CONV. = maxiter (300) reached: shown, never fitted, and its dOmega is not used. n-peaks: maxima of n(x) inside D with prominence >= 0.1e-2 nm^-1 (weak: >= 0.05e-2). Figures: figures/two_e/fig1_profiles.png, fig2_netspin.png, fig3_ldos_x0.png.

## 1. Peaks of n(x) inside D

| wx | V0 | conv | net spin | L_D | N_D | 1/n_center | n-peaks (x, nm) | spacings | weak peaks | dOmega pol-unpol |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.5 | y | 0.23 | 200 | 3.96 | 64 | - | - | 0 | -0.00657 |
| 1 | 3 | y | 1.09 | 238 | 4.54 | 82 | - | - | 0 | -0.03215 |
| 1 | 3.5 | y | 1.21 | 276 | 5.12 | 94 | - | - | 0 | -0.06839 |
| 0.7 | 2.5 | NO | 0.88 | 295 | 5.95 | 68 | - | - | 0 | excl. (not conv.) |
| 0.7 | 3 | NO | 2.14 | 352 | 6.84 | 65 | +0 | - | 1 | excl. (not conv.) |
| 0.7 | 3.5 | y | 2.01 | 390 | 7.29 | 72 | +0 | - | 1 | -0.10169 |
| 0.5 | 2.5 | NO | 2.00 | 429 | 8.75 | 60 | +0 | - | 1 | excl. (not conv.) |
| 0.5 | 3 | y | 1.98 | 505 | 9.99 | 63 | +0 | - | 1 | -0.05829 |
| 0.5 | 3.5 | y | 2.92 | 581 | 11.15 | 79 | -38 +38 | 76 | 2 | -0.14008 |
| 0.35 | 2.5 | NO | 2.66 | 600 | 12.26 | 63 | - | - | 4 | excl. (not conv.) |
| 0.35 | 3 | y | 2.95 | 733 | 14.63 | 73 | -29 +29 | 57 | 4 | -0.07574 |
| 0.35 | 3.5 | y | 3.93 | 829 | 15.96 | 68 | -67 +0 +67 | 67 67 | 5 | -0.20517 |
| 0.25 | 2.5 | NO | 3.85 | 867 | 17.86 | 59 | -171 -57 +0 +57 +171 | 114 57 57 114 | 5 | excl. (not conv.) |
| 0.25 | 3 | NO | 4.99 | 1019 | 20.31 | 71 | -210 -162 -95 -29 +29 +95 +162 +210 | 48 67 67 57 67 67 48 | 10 | excl. (not conv.) |
| 0.25 | 3.5 | NO | 5.42 | 1171 | 22.64 | 73 | -238 -124 +124 +238 | 114 248 114 | 8 | excl. (not conv.) |

Mean n-peak spacing (converged): 67 nm; mean 1/n_center (converged): 74 nm.

## 2. Net spin vs L_D and N_D (linear fit on the converged states)

- net spin vs L_D: slope 0.00493 per nm, intercept -0.27, r = 0.962; i.e. one spin per 203 nm (compare 1/n_center = 74 nm)
- net spin vs N_D: slope 0.24653 per electron in D, intercept -0.20, r = 0.950
- converged states used: 8 of 15

## 3. LDOS at x = 0 (eta 0.05 meV), V0 = 3.0

| wx | conv | spin-up peaks below mu (e - mu, meV) | spin-down peaks above mu (e - mu, meV) |
|---|---|---|---|
| 1 | y | 1: -0.802 | 1: +0.563 |
| 0.5 | y | 2: -1.067 -0.227 | 1: +0.613 |
| 0.35 | y | 2: -1.187 -0.612 | 1: +0.658 |
