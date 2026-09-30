"""Effective GaAs atomic units: hbar = m* = e^2/(4 pi eps0 eps_r) = 1."""
from dataclasses import dataclass

HARTREE_EV = 27.211386245988   # CODATA: vacuum Hartree in eV
BOHR_NM = 0.0529177210903      # CODATA: vacuum Bohr radius in nm


@dataclass(frozen=True)
class Units:
    m_eff: float = 0.067   # m*/m_e
    eps_r: float = 12.9    # relative permittivity (kappa) -- verify against the article

    @property
    def energy_meV(self) -> float:
        """1 Ha* in meV: Ha * (m*/eps_r^2)."""
        return HARTREE_EV * 1e3 * self.m_eff / self.eps_r**2

    @property
    def length_nm(self) -> float:
        """1 a* in nm: a0 * (eps_r/m*)."""
        return BOHR_NM * self.eps_r / self.m_eff

    def nm_to_au(self, x_nm):
        return x_nm / self.length_nm

    def au_to_nm(self, x_au):
        return x_au * self.length_nm

    def meV_to_au(self, e_meV):
        return e_meV / self.energy_meV

    def au_to_meV(self, e_au):
        return e_au * self.energy_meV
