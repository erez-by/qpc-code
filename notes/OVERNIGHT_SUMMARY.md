# Overnight summary (2026-10-04 23:31)

Columns: M_loc (local moment), M_win (|x|<300 nm), zeta0 = spin polarisation of n_1D at x = 0, n1d0 [1e-2 nm^-1], barrier heights in meV above e_0(far) as height@x(nm), e0up(0) = spin-up value at x = 0, D_dn = top_dn - top_unpol, D_up0 = e0up(0) - top_unpol, LDOS (eta 0.05/0.1 mean): resonance e - mu, Gamma = FWHM_fit - 2 eta, U (HMW definition), all meV.

## A_exch_wx1.0

unpolarised B = 0: top 0.628 meV, n1d0 1.382, mu - e0(far) 0.930, converged True (30 it)

| B | conv | it | M_loc | M_win | zeta0 | n1d0 | up peaks | e0up(0) | dn peaks | D_dn | D_up0 | mu-e0far | res-mu | Gamma | U |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | y | 70 | 1.074 | 1.096 | 0.90 | 1.298 | 0.42@-86 0.42@+86 | -0.029 | 0.12@-124 1.22@+0 0.12@+124 | +0.592 | -0.658 | 0.930 | -0.781 | 0.031 | 1.014 |

## A_exch_wx2.0

unpolarised B = 0: top 0.837 meV, n1d0 1.049, mu - e0(far) 0.930, converged True (30 it)

| B | conv | it | M_loc | M_win | zeta0 | n1d0 | up peaks | e0up(0) | dn peaks | D_dn | D_up0 | mu-e0far | res-mu | Gamma | U |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | y | 57 | 0.525 | 0.521 | 0.57 | 1.062 | 0.56@+0 | 0.556 | 1.16@+0 | +0.319 | -0.281 | 0.930 | -0.356 | 0.897 | 0.386 |
| 0.25 | y | 42 | 0.516 | 0.487 | 0.61 | 1.060 | 0.55@+0 | 0.550 | 1.16@+0 | +0.323 | -0.287 | 0.947 | -0.348 | 0.324 | 0.398 |
| 0.5 | y | 45 | 0.430 | 0.406 | 0.64 | 1.057 | 0.54@+0 | 0.542 | 1.16@+0 | +0.328 | -0.294 | 0.957 | -0.380 | 0.301 | 0.418 |
| 1 | y | 40 | 0.328 | 0.314 | 0.68 | 1.052 | 0.53@+0 | 0.526 | 1.17@+0 | +0.338 | -0.310 | 0.970 | -0.431 | 0.270 | 0.496 |
| 2 | y | 36 | 0.244 | 0.237 | 0.74 | 1.044 | 0.50@+0 | 0.501 | 1.19@+0 | +0.355 | -0.336 | 0.987 | -0.505 | 0.234 | 0.641 |
| 4 | y | 37 | 0.181 | 0.168 | 0.82 | 1.036 | 0.47@+0 | 0.471 | 1.22@+0 | +0.379 | -0.365 | 1.014 | -0.639 | 0.917 | 0.850 |
| 6 | y | 41 | 0.144 | 0.115 | 0.86 | 1.033 | 0.46@+0 | 0.458 | 1.23@+0 | +0.397 | -0.379 | 1.039 | -0.732 | 1.637 | 1.055 |

## B = 0 of every folder vs PRL targets

| folder | conv | M_loc | zeta0 | n1d0 | up peaks | e0up(0) | dn peaks | D_dn | D_up0 | mu-e0far | res-mu | Gamma | U |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_exch_wx1.0 | y | 1.074 | 0.90 | 1.298 | 0.42@-86 0.42@+86 | -0.029 | 0.12@-124 1.22@+0 0.12@+124 | +0.592 | -0.658 | 0.930 | -0.781 | 0.031 | 1.014 |
| A_exch_wx2.0 | y | 0.525 | 0.57 | 1.062 | 0.56@+0 | 0.556 | 1.16@+0 | +0.319 | -0.281 | 0.930 | -0.356 | 0.897 | 0.386 |

PRL targets (user's readings, polarised, B = 0):
- wx 1.0: net spin 0.85; up peaks ~0.6 at +-75 nm, centre ~0.4; dn ~1.18; zeta(0) ~0.68; Gamma ~0.1; U ~0.6
- wx 1.5: net spin 0.93; up centre ~0.58, sides ~0.33 at +-60 nm; dn ~1.3; zeta(0) ~0.74; U ~0.6, Gamma ~0.3
- wx 2.0: net spin 0.90; up peak ~0.88, sides ~0.18; dn ~1.5; zeta(0) ~0.70; Gamma ~0.6; U ~0.6
- all: mu - e_0(far) ~ 1.1 meV; Gamma ~0.1 (1.0) -> ~0.6 (2.0); U ~0.6 nearly constant

## Clean wire (B = 0) per folder

| folder | mu | e1-e0 | mu-e0 | N(subband 1) | hartree_scale |
|---|---|---|---|---|---|
| A_exch_wx1.0 | 8.9565 | 0.954 | 0.930 | 11.46 | 1.0 |
| A_exch_wx2.0 | 8.9565 | 0.954 | 0.930 | 11.46 | 1.0 |

## Spin gain (B = 0, linear mixing, 1 % seed)
```
(none yet)
```

## Grand potential Omega_pol - Omega_unpol at B = 0

(none yet)

## Unconverged states (maxiter reached)

(none)
