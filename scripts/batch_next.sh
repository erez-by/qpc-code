#!/bin/bash
# Queue: (1) B = 0 rerun wx 1.0 A quadratic; (2) if not converged: same with kT = 0.08;
# (3) Janak ramp wx 1.5 A exchange; (4) if M_loc(B=0) < 0.3: same ramp with power:1.3333;
# (5) Fig.-1 panels for wx 1.0 (quad, exch); (6) notes/RESULTS_for_claude.md (all results).
#     bash scripts/batch_next.sh          progress: tail -f logs/batch_next.out      (~1.5-3 h)
cd "$(dirname "$0")/.." && source .venv/bin/activate
exec > >(tee logs/batch_next.out) 2>&1
conv() { python -c "from qpc.scf import SCFResult; print(SCFResult.load_npz('$1').converged)"; }
mloc0() { python -c "
import sys; sys.path.insert(0, 'scripts')
from run_qpc import setup; from qpc.scf import SCFResult, SCFParams, clean_wire; from qpc.analysis import net_spin_local
U, g, ham, hart, q = setup($1); r = SCFResult.load_npz('$2')
w = clean_wire(ham, hart, q, SCFParams(B_T=0.0, interp='$3'), 140.0)
print(f'{net_spin_local(r, w, g):.4f}')" 2>/dev/null | tail -1; }

echo "== (1) B = 0 rerun, wx 1.0 A quadratic  $(date)"
python -u scripts/b0_rerun.py --wx 1.0 --interp quadratic --start results/qpc_wx1.0_A_quad_B0.25.npz \
    --alpha 0.1 --history 12 --maxiter 400 > logs/b0_rerun_quad.log 2>&1
C1=$(conv results/qpc_wx1.0_A_quad_B0_rerun.npz); echo "converged: $C1"
if [ "$C1" != "True" ]; then
  echo "== (2) B = 0 rerun with kT = 0.08  $(date)"
  python -u scripts/b0_rerun.py --wx 1.0 --interp quadratic --start results/qpc_wx1.0_A_quad_B0.25.npz \
      --alpha 0.1 --history 12 --maxiter 300 --kT 0.08 > logs/b0_rerun_quad_kT0.08.log 2>&1
fi
echo "== (3) Janak ramp wx 1.5 A exchange  $(date)"
python -u scripts/run_qpc.py --wx 1.5 --lead interacting --interp exchange --B 6 4 2 1 0.5 0.25 0 \
    > logs/ramp_wx1.5_A_exch.log 2>&1
M15=$(mloc0 1.5 results/qpc_wx1.5_A_B0.npz exchange); echo "M_loc(B=0, wx 1.5 exch) = $M15"
if python -c "import sys; sys.exit(0 if float('$M15') < 0.3 else 1)"; then
  echo "== (4) Janak ramp wx 1.5 A power:1.3333  $(date)"
  python -u scripts/run_qpc.py --wx 1.5 --lead interacting --interp power:1.3333 --B 6 4 2 1 0.5 0.25 0 \
      > logs/ramp_wx1.5_A_power1.3333.log 2>&1
fi
echo "== (5) figures  $(date)"
QF=results/qpc_wx1.0_A_quad_B0.npz
[ "$C1" == "True" ] && QF=results/qpc_wx1.0_A_quad_B0_rerun.npz
python scripts/make_fig1.py --wx 1.0 --state quad=$QF --state exch=results/qpc_wx1.0_A_B0.npz > logs/fig1.log 2>&1

echo "== (6) results file  $(date)"
R=notes/RESULTS_for_claude.md
sec() { echo; echo "### $1"; echo '```'; }
end() { echo '```'; }
{
  cat notes/results_head.md
  echo; echo "## New results ($(date +%F))"
  sec "1) B = 0 rerun, wx 1.0, A, quadratic, from the converged 0.25 T state (Pulay alpha 0.1, history 12, maxiter 400)"
  grep -vE "Hartree: kernel" logs/b0_rerun_quad.log; end
  echo "(full per-iteration table: notes/wx1.0_A_quad_B0_rerun_table.txt)"
  if [ -f logs/b0_rerun_quad_kT0.08.log ] && [ "$C1" != "True" ]; then
    sec "2) same B = 0 rerun with kT = 0.08 meV (smearing sensitivity)"
    grep -vE "Hartree: kernel" logs/b0_rerun_quad_kT0.08.log | grep -vE "^ +[0-9]+ +-?[0-9]" ; end
    echo "(per-iteration table: notes/wx1.0_A_quad_B0_rerun_kT0.08_table.txt)"
  fi
  for L in logs/ramp_wx1.5_A_exch.log logs/ramp_wx1.5_A_power1.3333.log; do
    [ -f "$L" ] || continue
    sec "3/4) Janak ramp $(basename $L .log) (6 -> 0 T); M_loc(B=0) exch = $M15"
    grep -E "^ B\[T\]" $L | tail -1
    grep -E "^ +[0-9.]+ +-?[0-9]+\.[0-9]{4} +-?[0-9]+\.[0-9]{4} " $L | tail -7
    grep -A3 "^largest B with two" $L
    echo "-- per-B features:"
    grep -E "^##### B =|M_loc =|\[up\]|\[dn\]|n_1D\(0\)|D_dn =|LDOS eta" $L; end
  done
  sec "5) figures"; cat logs/fig1.log | grep -v kernel; end
  echo "Figures: figures/fig1_wx1.0_quad.png, figures/fig1_wx1.0_exch.png (PRL Fig. 1 axes; n_up - n_dn"
  echo "below 0 in the Friedel oscillations is clipped by the paper's 0 ... 1.5 axis)."
} > $R
cp logs/b0_rerun_quad*.log logs/ramp_wx1.5_A_*.log notes/ 2>/dev/null
echo "DONE -> $R"
