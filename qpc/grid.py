"""Real-space and plane-wave grids for the periodic supercell (atomic units)."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Grid:
    Lx: float
    Ly: float
    Nx: int
    Ny: int

    @property
    def dx(self) -> float:
        return self.Lx / self.Nx

    @property
    def dy(self) -> float:
        return self.Ly / self.Ny

    def real_axes(self):
        """x_j = j*dx with the endpoint excluded (last point is the periodic image of the first)."""
        return np.arange(self.Nx) * self.dx, np.arange(self.Ny) * self.dy

    def G_axes(self):
        """G_m = 2*pi*m/L in numpy FFT ordering (0, 1, ..., -N/2, ..., -1)."""
        return (2 * np.pi * np.fft.fftfreq(self.Nx, d=self.dx),
                2 * np.pi * np.fft.fftfreq(self.Ny, d=self.dy))

    def G2(self):
        Gx, Gy = self.G_axes()
        return Gx[:, None] ** 2 + Gy[None, :] ** 2

    def kinetic(self):
        """T(G) = G^2 / 2 in Ha*."""
        return 0.5 * self.G2()

    def cutoff_mask(self, e_cut):
        """Boolean mask of plane waves with G^2/2 <= e_cut (Ha*)."""
        return self.kinetic() <= e_cut

def grid_from_cutoff(Lx, Ly, e_cut):
    """FFT-friendly Grid that resolves the density: Nyquist pi/dx >= 2 G_cut.

    Lx, Ly, e_cut in atomic units; G_cut = sqrt(2 e_cut).
    N >= 2 G_cut L / pi, rounded up to a fast FFT length.
    """
    from scipy.fft import next_fast_len
    G_cut = np.sqrt(2.0 * e_cut)
    nx = next_fast_len(int(np.ceil(2 * G_cut * Lx / np.pi)))
    ny = next_fast_len(int(np.ceil(2 * G_cut * Ly / np.pi)))
    return Grid(Lx, Ly, nx, ny)
