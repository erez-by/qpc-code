"""Pulay / DIIS density mixing.

Pulay, Chem. Phys. Lett. 73, 393 (1980), in the form of Kresse & Furthmueller, PRB 54, 11169
(1996), Sec. IV B. State vector rho = concat(n_up.ravel(), n_dn.ravel()) [a*^-2], residual
F = rho_out - rho_in. With the stored history {rho_in,k, F_k}:

    minimise || sum_k c_k F_k ||^2  subject to  sum_k c_k = 1
    -> [B 1; 1^T 0][c; lambda] = [0; 1],   B_ij = <F_i, F_j>
    rho_in,next = sum_k c_k (rho_in,k + alpha F_k)

The inner product <.,.> = sum * dx dy only rescales B, which leaves c unchanged, so the plain
dot product is used.
"""
from collections import deque

import numpy as np

COND_MAX = 1e12


class Pulay:
    """DIIS mixer. alpha: linear mixing weight applied to the extrapolated residual;
    history: number of (rho_in, F) pairs kept (history = 1 is plain linear mixing)."""

    def __init__(self, alpha=0.2, history=8):
        if history < 1:
            raise ValueError("history >= 1")
        self.alpha = float(alpha)
        self.history = int(history)
        self.reset()

    def reset(self):
        """Forget the history (e.g. after a change of B or of the band count)."""
        self._rho = deque(maxlen=self.history)
        self._F = deque(maxlen=self.history)
        self.last_coeffs = None

    def _coefficients(self):
        """Solve the constrained least-squares problem; drop the oldest entries while
        cond(B) > COND_MAX (B scaled by its largest diagonal element)."""
        while True:
            F = np.array(self._F)
            m = len(F)
            B = F @ F.T
            if m == 1:
                return np.ones(1)
            scale = np.max(np.diag(B))
            Bs = B / scale if scale > 0 else B
            if np.linalg.cond(Bs) > COND_MAX:
                self._rho.popleft()
                self._F.popleft()
                continue
            M = np.zeros((m + 1, m + 1))
            M[:m, :m] = Bs
            M[:m, m] = M[m, :m] = 1.0
            rhs = np.zeros(m + 1)
            rhs[m] = 1.0
            sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
            return sol[:m]

    def step(self, rho_in, rho_out):
        """Next input density from the current (rho_in, rho_out) pair; negatives clipped to 0."""
        rho_in = np.asarray(rho_in, dtype=float).ravel().copy()
        F = np.asarray(rho_out, dtype=float).ravel() - rho_in
        self._rho.append(rho_in)
        self._F.append(F)
        c = self._coefficients()
        self.last_coeffs = c
        rho = np.array(self._rho)
        Fs = np.array(self._F)
        rho_next = c @ (rho + self.alpha * Fs)
        return np.maximum(rho_next, 0.0)
