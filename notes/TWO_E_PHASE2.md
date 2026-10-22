# Two-electron spec: Phase 2 (alternating seed) and Phase 3 (2026-10-22 16:46)

Model C, exchange, Lx 5000 nm, kT 0.05 meV. AP runs: ramp 5 -> 0 T with the alternating field (sign (-1)^j per n(x)-peak segment in D, zero outside D; leads at zero field), started from the unpolarised B = 0 state. B = 0 Hamiltonian = Phase 1. NOT CONV. states are marked and never used for Delta.

| wx | V0 | field pattern | AP conv/it | net spin AP | net spin P | corr(s_AP, s_P) | class AP | outcome | Delta = Omega_AP - Omega_P (meV) |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 3 | fallback: 2 equal segments of the spin moment [-95, 95] nm (only 1 n(x) peak(s) in D; net spin 1.98): -48 +48 | y/76 | 0.000 | 1.980 | +0.000 | AP | AP-like (distinct from P) | +0.047158 |
| 0.35 | 3 | 2 n(x) peaks in D: -29 +29 | y/62 | 0.000 | 2.952 | +0.000 | multi4 | AP-like (distinct from P) | +0.046378 |
| 0.5 | 3.5 | 2 n(x) peaks in D: -38 +38 | y/38 | 0.000 | 2.920 | +0.000 | multi4 | AP-like (distinct from P) | +0.119019 |

## Ramp (net spin M of the cell at each field; leads at zero field, so M = local moment)

| case | 5 T | 4 T | 2 T | 1 T | 0.5 T | 0.25 T | 0 T |
|---|---|---|---|---|---|---|---|
| wx 0.5 V0 3 | -0.000 | 0.000 | -0.000 | 0.000 | -0.000 | 0.000 | 0.000 |
| wx 0.35 V0 3 | 0.000 | -0.000 | -0.000 | 0.000 | 0.000 | -0.000 | 0.000 |
| wx 0.5 V0 3.5 | -0.000 | -0.000 | 0.033 NO | -0.000 | -0.000 | 0.000 | 0.000 |

## Phase 3: Omega convergence with the SCF tolerance (meV)

- wx 0.5 V0 3: P (parallel) ground state; Delta = +0.047158 meV; M_P 1.9803, M_AP -0.0000
  - P: tol 1e-05: -534.994366 (5 it); tol 1e-06: -534.994366 (6 it)
  - AP: tol 1e-05: -534.947209 (25 it); tol 1e-06: -534.947209 (17 it)
- wx 0.35 V0 3: P (parallel) ground state; Delta = +0.046378 meV; M_P 2.9524, M_AP 0.0000
  - P: tol 1e-05: -523.565963 (9 it); tol 1e-06: -523.565963 (9 it)
  - AP: tol 1e-05: -523.519584 (35 it); tol 1e-06: -523.519584 (58 it)
- wx 0.5 V0 3.5: P (parallel) ground state; Delta = +0.119019 meV; M_P 2.9201, M_AP 0.0000
  - P: tol 1e-05: -529.352111 (7 it); tol 1e-06: -529.352111 (10 it)
  - AP: tol 1e-05: -529.233092 (36 it); tol 1e-06: -529.233092 (30 it)
