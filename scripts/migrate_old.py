"""One-time: copy the pre-overnight result files into the folder layout of scripts/qpc_driver.py.
JSON / LDOS are created later by `run_qpc.py --model .. --postprocess` (job J03). Originals are kept."""
import os
import shutil

FIELDS = ["6", "4", "2", "1", "0.5", "0.25", "0"]
MAP = {  # folder: (old B-file pattern, old unpol file)
    "A_exch_wx1.0": ("qpc_wx1.0_A_B{B}.npz", "qpc_wx1.0_unpol.npz"),
    "A_exch_wx1.5": ("qpc_wx1.5_A_B{B}.npz", "qpc_wx1.5_unpol.npz"),
    "A_quad_wx1.0": ("qpc_wx1.0_A_quad_B{B}.npz", "qpc_wx1.0_unpol.npz"),
    "B_exch_wx1.0": ("qpc_wx1.0_B_B{B}.npz", "qpc_wx1.0_unpol_bare.npz"),
}
for folder, (pat, unpol) in MAP.items():
    d = os.path.join("results", folder)
    os.makedirs(d, exist_ok=True)
    pairs = [(pat.format(B=B), f"B{B}.npz") for B in FIELDS] + [(unpol, "unpol.npz")]
    for src, dst in pairs:
        s, t = os.path.join("results", src), os.path.join(d, dst)
        if os.path.exists(s) and not os.path.exists(t):
            shutil.copy2(s, t)
            print(f"{s} -> {t}")
