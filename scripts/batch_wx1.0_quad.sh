#!/bin/bash
# Janak ramp, model A, wx = 1.0 meV, interp = QUADRATIC (6 4 2 1 0.5 0.25 0 T); if M_loc(B=0) < 0.3,
# also the same ramp with interp = mixed:0.5. Summary: notes/wx1.0_quad_summary.txt
#     bash scripts/batch_wx1.0_quad.sh          progress: tail -f logs/ramp_wx1.0_A_quad.log   (~40-60 min, x2 if mixed)
cd "$(dirname "$0")/.." && source .venv/bin/activate
BS="6 4 2 1 0.5 0.25 0"
python -u scripts/run_qpc.py --wx 1.0 --lead interacting --interp quadratic --B $BS > logs/ramp_wx1.0_A_quad.log 2>&1
M0=$(python -c "
import sys; sys.path.insert(0, 'scripts')
from run_qpc import setup; from qpc.scf import SCFResult, SCFParams, clean_wire; from qpc.analysis import net_spin_local
U, g, ham, hart, q = setup(1.0); r = SCFResult.load_npz('results/qpc_wx1.0_A_quad_B0.npz')
w = clean_wire(ham, hart, q, SCFParams(B_T=0.0, interp='quadratic'), 140.0)
print(f'{net_spin_local(r, w, g):.4f}')" 2>/dev/null | tail -1)
echo "M_loc(B=0, quadratic) = $M0"
RUNS="quad"
if python -c "import sys; sys.exit(0 if float('$M0') < 0.3 else 1)"; then
  echo "M_loc(B=0) < 0.3 -> running mixed:0.5 ramp"
  python -u scripts/run_qpc.py --wx 1.0 --lead interacting --interp mixed:0.5 --B $BS > logs/ramp_wx1.0_A_mixed0.5.log 2>&1
  RUNS="quad mixed0.5"
fi
S=notes/wx1.0_quad_summary.txt
{
  echo "=== wx = 1.0 meV, model A, a_image = 100 nm, kT = 0.05 meV ($(date)); M_loc(B=0, quadratic) = $M0 ==="
  for r in $RUNS; do
    L=logs/ramp_wx1.0_A_$r.log
    echo; echo "##### Janak ramp model A, interp = $r (6 -> 0 T)"
    grep -E "^ B\[T\]" $L | tail -1
    grep -E "^ +[0-9.]+ +-?[0-9]+\.[0-9]{4} +-?[0-9]+\.[0-9]{4} " $L | tail -7
    grep -A3 "^largest B with two" $L
    echo "-- per-B features (M_loc > 0.3 with LDOS):"
    grep -E "^##### B =|M_loc =|\[up\]|\[dn\]|n_1D\(0\)|D_dn =|LDOS eta" $L
    cp $L notes/
  done
} > $S
echo "DONE -> $S"
