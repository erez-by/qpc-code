#!/bin/bash
# Remaining wx = 1.0 meV work (exchange interp): large-seed B (A is done), LDOS of all states,
# then a compact summary in notes/wx1.0_summary.txt. Run from the project root:
#     bash scripts/batch_wx1.0_rest.sh
# Progress: tail -f logs/batch_wx1.0_rest.out      (about 30-40 min)
cd "$(dirname "$0")/.." && source .venv/bin/activate
exec > >(tee logs/batch_wx1.0_rest.out) 2>&1
set -x
python -u scripts/run_qpc.py --wx 1.0 --lead bare --interp exchange --large-seed > logs/largeseed_wx1.0_B.log 2>&1
python -u scripts/ldos.py --wx 1.0 --files \
    results/qpc_wx1.0_A_B0.npz results/qpc_wx1.0_A_B0.25.npz results/qpc_wx1.0_A_B0.5.npz \
    results/qpc_wx1.0_A_B0_largeseed.npz \
    results/qpc_wx1.0_B_B0.npz results/qpc_wx1.0_B_B6.npz results/qpc_wx1.0_B_B0_largeseed.npz \
    > logs/ldos_wx1.0.log 2>&1
set +x
S=notes/wx1.0_summary.txt
{
  echo "=== wx = 1.0 meV, interp = exchange, a_image = 100 nm, kT = 0.05 meV ($(date)) ==="
  for m in A B; do
    echo; echo "##### Janak ramp model $m (6 -> 0 T)"
    sed -n '/^ B\[T\].*conv$/,$p' logs/ramp_wx1.0_$m.log | grep -E "^ B\[T\]|^ +[0-9.]+ +-?[0-9]+\.[0-9]{4} " | tail -8
    grep "largest B" logs/ramp_wx1.0_$m.log
    echo "-- features at B = 0:"; sed -n '/##### B = 0 T/,$p' logs/ramp_wx1.0_$m.log | grep -E "M_loc =|\[up\]|\[dn\]|n_1D\(0\)"
  done
  for m in A B; do
    echo; echo "##### large seed (M_init = 1, Pulay, maxiter 200) model $m"
    grep -E "large-seed|converged after|NOT converged" logs/largeseed_wx1.0_$m.log
    grep -E "M_loc =|\[up\]|\[dn\]|n_1D\(0\)" logs/largeseed_wx1.0_$m.log
  done
  echo; echo "##### LDOS at x = 0 (Lorentzian)"; grep -E "^===|eta =" logs/ldos_wx1.0.log
} > $S
cp logs/ramp_wx1.0_B.log notes/ramp_wx1.0_B.txt
cp logs/largeseed_wx1.0_A.log notes/largeseed_wx1.0_A.txt
cp logs/largeseed_wx1.0_B.log notes/largeseed_wx1.0_B.txt
cp logs/ldos_wx1.0.log notes/ldos_wx1.0.txt
echo "DONE -> $S"
