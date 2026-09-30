"""QPC potential of Hirose, Meir, Wingreen, PRL 90, 026804 (2003), Eq. (1).

Effective atomic units: hbar = m* = 1, energies in Ha*, lengths in a*.

    V_QPC(x, y) = V(x)/2 + (m*/2) [omega_y + V(x)/hbar]^2 y^2 - (m*/2) omega_y^2 y^2
    V(x)        = V0 / cosh^2(x/d),     d = sqrt(2 V0 / m*) / omega_x

V_QPC is the *difference* from the clean-wire confinement, so it vanishes for
|x| -> infinity. The full bare potential is V_QPC + (1/2) omega_y^2 y^2.
"""
from dataclasses import dataclass, field
import numpy as np
from .units import Units


@dataclass(frozen=True)
class QPCParams:
    """Physical inputs in meV (paper defaults); derived quantities in atomic units."""
    hbar_wx_meV: float = 1.5     # 1.0, 1.5 or 2.0 in the paper
    hbar_wy_meV: float = 2.0
    V0_meV: float = 3.0
    units: Units = field(default_factory=Units)

    @property
    def V0(self) -> float:
        return self.units.meV_to_au(self.V0_meV)

    @property
    def wx(self) -> float:
        """omega_x in Ha* (hbar = 1)."""
        return self.units.meV_to_au(self.hbar_wx_meV)

    @property
    def wy(self) -> float:
        return self.units.meV_to_au(self.hbar_wy_meV)

    @property
    def d(self) -> float:
        """Decay length in a*: d = sqrt(2 V0 / m*) / omega_x."""
        return np.sqrt(2.0 * self.V0) / self.wx


def sech2(u):
    """sech^2(u) without overflow for large |u|."""
    e = np.exp(-2.0 * np.abs(u))
    return 4.0 * e / (1.0 + e) ** 2


def V_barrier(x, p: QPCParams):
    """Poschl-Teller barrier V(x) = V0 sech^2(x/d), in Ha*."""
    return p.V0 * sech2(np.asarray(x) / p.d)


def V_qpc(x, y, p: QPCParams):
    """Eq. (1). x, y broadcastable arrays in a*; returns Ha*."""
    V = V_barrier(x, p)
    y = np.asarray(y)
    return 0.5 * V + 0.5 * (p.wy + V) ** 2 * y ** 2 - 0.5 * p.wy ** 2 * y ** 2
