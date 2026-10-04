"""Rebuild a ramp table (incl. the window moment M_win, |x| < 300 nm) from saved npz files; no SCF.

Usage: python scripts/ramp_table.py --wx 1.5 --dir results/ramp_10to0 --B 10 9 8 7 6 5 4 3 2 1 0.5 0.25 0
Clean-wire states are read from <dir>/wire_B{B}.npz (model A). Time = sum of the per-iteration
times stored in the SCF history.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from run_qpc import HEADER, bstr, ramp_end_lines, row, setup, spin_summary   # noqa: E402

from qpc.scf import SCFResult                                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wx", type=float, default=1.5)
    ap.add_argument("--dir", default="results")
    ap.add_argument("--B", type=float, nargs="+", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="extra qpc npz names in --dir (B = 0 rows)")
    ap.add_argument("--qpc", default="qpc_wx{wx}_B{B}.npz", help="qpc file pattern in --dir")
    ap.add_argument("--wire", default="wire_B{B}.npz", help="clean-wire file pattern in --dir")
    ap.add_argument("--lead", default="interacting")
    args = ap.parse_args()
    U, grid, ham, hart, q = setup(args.wx)
    print(HEADER)
    feats = []
    for B in args.B:
        res = SCFResult.load_npz(os.path.join(args.dir, args.qpc.format(wx=args.wx, B=bstr(B))))
        wire = SCFResult.load_npz(os.path.join(args.dir, args.wire.format(wx=args.wx, B=bstr(B))))
        d = spin_summary(res, wire, ham, U, args.wx, args.lead)
        feats.append((B, d, d["feat"]))
        print(row(B, d, res.iterations, sum(h[5] for h in res.history), res.converged))
    for name in args.extra:
        res = SCFResult.load_npz(os.path.join(args.dir, name))
        wire = SCFResult.load_npz(os.path.join(args.dir, args.wire.format(wx=args.wx, B="0")))
        d = spin_summary(res, wire, ham, U, args.wx, args.lead)
        print(row(0.0, d, res.iterations, sum(h[5] for h in res.history), res.converged) + f"   ({name})")
    print(ramp_end_lines(feats))


if __name__ == "__main__":
    main()
