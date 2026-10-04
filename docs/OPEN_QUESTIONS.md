# Open questions and physics choices

Choices not fixed by the PRL (HMW 2003) or by docs/SPEC.md. Each one is marked in the code with
`# PHYSICS-CHOICE:` where it enters. The complete model with sources is in docs/MODEL.md.

## From docs/SPEC.md
1. ~~Tanatar-Ceperley coefficients and energy unit~~ **Resolved:** both rows verified against
   TC, PRB 39, 5005 (1989): Eq. (14), Table IV, energies in Ry* (Sec. I). Fit ranges rs = 1-50
   (zeta = 0), 5-75 (zeta = 1); wire peak rs ~ 2 (zeta = 1 row extrapolated). Tests: Pade vs
   printed Table I/II E_c, rel. 1e-2 (zeta = 0) and 1.5e-2 (zeta = 1: the Pade deviates from its
   own data by ~1.1 % at rs = 10, ~4 % at rs = 5); E(rs,1) - E(rs,0) changes sign in (30, 40).
2. **PHYSICS-CHOICE: spin interpolation.** HMW do not state it. Default `quadratic` (TC's own
   prescription: E_c(rs,z) = E_c(rs,0) + z^2 [E_c(rs,1) - E_c(rs,0)], exact exchange), in
   `xc.py` and `SCFParams`. The `exchange` interpolation (Koskinen, Manninen & Reimann, PRL 79,
   1389 (1997); alias `vbh`) is run as a convergence/sensitivity check in M8.
3. ~~Gate model and distance~~ **Resolved:** HW Eq. (1) and HMW's delta V_H use the kernel
   1/rho - 1/sqrt(rho^2 + a_image^2) with a_image = 100 nm the charge <-> image-charge distance.
   `SCFParams.a_image_nm = 100`; `Hartree` takes the metal-plane distance a_m = a_image/2 = 50 nm.
   (The earlier run with a = 100 nm as the metal distance is kept as a diagnostic:
   notes/m5_wire_diag_a_metal100.txt.)
4. **Temperature / smearing (numerical).** The PRL uses T = 0.1 K in an OPEN system. Our
   kT = 0.05 meV is a numerical smearing required by the discrete levels of the 5 um periodic cell
   (spacing ~0.06 meV near mu). In the clean wire kT = 0.0086 vs 0.05 meV changes mu - e_0 by
   ~0.02 meV (0.954 vs 0.930) and the subband-1 electrons by ~2 (9.8 vs 11.5). To be checked in
   M8 (Lx = 7.5 um, kT = 0.025 meV).
5. ~~Which HMW Fig. 1 panel corresponds to which hbar w_x~~ **Resolved:** Fig. 1 (a),(d) =
   1.0 meV, (b),(e) = 1.5 meV, (c),(f) = 2.0 meV; PRL net spin 0.85, 0.93, 0.90 respectively.

## Added during implementation
6. **Hartree y-quadrature (numerical, M2).** SPEC M2 prescribes the plain cell-averaged kernel
   (n piecewise constant per y cell). Its error is O(dy^2): 1.7e-3 relative for the 25 nm line
   charge of test (a) at production dy = 9.14 nm, so it fails the spec's 1e-3 tolerance (and is
   -4.6e-4 / -1.2e-4 at dy/2, dy/4, confirming pure discretisation error). Default is now
   `Hartree(..., y_rule="quadratic")`: same exact kernel w(k, y), but n quadratic (central
   differences) inside each cell, error O(dy^4): 8.9e-5 at production dy. `y_rule="cell"` keeps
   the spec's rule (QPC barrier changes by < 0.001 meV, mu_wire by 3 ueV). Accepted by the user.
7. ~~Clean wire / M5.1~~ **Resolved (user decision).** The model is the full one (bare parabola +
   V_QPC + V_H[n] + v_xc,s[n] + Zeeman): HMW Eq. (2) has delta V_H = V_H[rho] - V_H[rho0] with
   rho0 the INTERACTING clean wire of HW (cond-mat/0106581, Eq. 1), so V_H[rho0] + delta V_H =
   V_H[rho]. No reference-subtracted mode. The earlier failure (spacing 0.889 meV) came from
   the Hartree convention bug (#3). With a_image = 100 nm: spacing 0.954 meV, mu - e_0 =
   0.930 meV, 11.5 electrons in subband 1 at kT = 0.05 meV; acceptance rewritten (subband-0
   density), PASS. The subband-1 electrons are reflected by the QPC (n = 1 adiabatic barrier
   ~9 meV >> mu) and do not carry current.
8. **Dielectric constant.** epsilon = 12.9 (HMW PRL) is kept; the HW wire paper uses kappa = 13.1.
9. **Lead (clean-wire) spin polarisation at B > 0.** Because the subband-1 edge is pinned at mu
   (1D band-edge DOS), the clean wire polarises strongly: N_up - N_dn = 8.0 (1 T), 25.4 (6 T),
   35.4 (10 T) of 140 electrons; at 6 T subband 0 gives 10.4 (vs ~5.3 from E_Z x DOS without
   exchange: enhancement ~2, spin splitting 0.29 meV vs E_Z = 0.153 meV) and subband 1 gives 15.0
   (15.15 up, 0.11 down). The QPC moment is therefore reported as M_loc = M - M_wire(B)
   (`analysis.net_spin_local`); M_loc = M at B = 0.
10. **Model C (sensitivity, not physics):** `SCFParams.hartree_scale = 0.8` scales V_H everywhere (leads,
    QPC, E_H). At 0.8 the clean wire has a single subband (mu - e_0 = 1.100 meV, spacing 1.197 meV at
    kT = 0.0086), i.e. it mimics HMW's mu - e_0(far) ~ 1.1 meV. Used only to test how the moment depends on
    the lead screening; default 1.0 unchanged.
11. **Grand potential (qpc/energy.py):** Omega = E - kT S - mu N with E in eigenvalue form; the eigenvalue
    and direct forms agree to 1e-8 (tests/test_energy.py). Used for Omega_pol - Omega_unpol at B = 0 (J19).
