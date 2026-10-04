# Overnight summary (2026-10-04 23:08)

Columns: M_loc (local moment), M_win (|x|<300 nm), zeta0 = spin polarisation of n_1D at x = 0, n1d0 [1e-2 nm^-1], barrier heights in meV above e_0(far) as height@x(nm), e0up(0) = spin-up value at x = 0, D_dn = top_dn - top_unpol, D_up0 = e0up(0) - top_unpol, LDOS (eta 0.05/0.1 mean): resonance e - mu, Gamma = FWHM_fit - 2 eta, U (HMW definition), all meV.

## A_exch_wx1.0

unpolarised B = 0: top 0.628 meV, n1d0 1.382, mu - e0(far) 0.930, converged True (30 it)

| B | conv | it | M_loc | M_win | zeta0 | n1d0 | up peaks | e0up(0) | dn peaks | D_dn | D_up0 | mu-e0far | res-mu | Gamma | U |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | y | 70 | 1.074 | 1.096 | 0.90 | 1.298 | 0.42@-86 0.42@+86 | -0.029 | 0.12@-124 1.22@+0 0.12@+124 | +0.592 | -0.658 | 0.930 | -0.781 | 0.031 | 1.014 |

## B = 0 of every folder vs PRL targets

| folder | conv | M_loc | zeta0 | n1d0 | up peaks | e0up(0) | dn peaks | D_dn | D_up0 | mu-e0far | res-mu | Gamma | U |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_exch_wx1.0 | y | 1.074 | 0.90 | 1.298 | 0.42@-86 0.42@+86 | -0.029 | 0.12@-124 1.22@+0 0.12@+124 | +0.592 | -0.658 | 0.930 | -0.781 | 0.031 | 1.014 |

PRL targets (user's readings, polarised, B = 0):
- wx 1.0: net spin 0.85; up peaks ~0.6 at +-75 nm, centre ~0.4; dn ~1.18; zeta(0) ~0.68; Gamma ~0.1; U ~0.6
- wx 1.5: net spin 0.93; up centre ~0.58, sides ~0.33 at +-60 nm; dn ~1.3; zeta(0) ~0.74; U ~0.6, Gamma ~0.3
- wx 2.0: net spin 0.90; up peak ~0.88, sides ~0.18; dn ~1.5; zeta(0) ~0.70; Gamma ~0.6; U ~0.6
- all: mu - e_0(far) ~ 1.1 meV; Gamma ~0.1 (1.0) -> ~0.6 (2.0); U ~0.6 nearly constant

## Clean wire (B = 0) per folder

| folder | mu | e1-e0 | mu-e0 | N(subband 1) | hartree_scale |
|---|---|---|---|---|---|
| A_exch_wx1.0 | 8.9565 | 0.954 | 0.930 | 11.46 | 1.0 |

## Spin gain (B = 0, linear mixing, 1 % seed)
```
(none yet)
```

## Grand potential Omega_pol - Omega_unpol at B = 0

(none yet)

## Unconverged states (maxiter reached)

(none)
