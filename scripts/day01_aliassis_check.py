import mpmath as mp
from qpc.fourier import ft_mpmath, ft_sech2_exact

mp.mp.dps = 40
d, V0, k = mp.mpf("4.05"), mp.mpf("0.27"), mp.mpf("0.5")
f = lambda x: V0 / mp.cosh(x / d) ** 2

print("   h      T - true     sum of copies   nearest copy")
for h in [4, 2, 1, 0.5]:
    h = mp.mpf(h)
    T = ft_mpmath(f, k, h, 40 * d)
    true = ft_sech2_exact(k, V0, d)
    copies = sum(ft_sech2_exact(k + 2 * mp.pi * m / h, V0, d)
                 for m in range(-5, 6) if m != 0)
    near = ft_sech2_exact(2 * mp.pi / h - k, V0, d)
    print(mp.nstr(h, 3), mp.nstr(T - true, 4), mp.nstr(copies, 4), mp.nstr(near, 4))