#!/bin/bash
#SBATCH --job-name=sdg6-baselines
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=124G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Urban/rural baseline and stratified AUROC.
#
#   sbatch scripts/slurm/eval_baselines.sbatch

set -uo pipefail

ROOT="${SDG6_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$ROOT" || exit 1
PY="${PY:-$ROOT/.venv/bin/python}"
OUT="$ROOT/outputs/tables/baseline_stratified_auroc.csv"

log() { echo "[$(date +'%F %T')] $*"; }
rm -f "$OUT"

for task in pw sw; do
  for scheme in random spatial; do
    log "=== scheme=$scheme task=$task"
    "$PY" scripts/eval_baselines.py --scheme "$scheme" --task "$task" --out "$OUT" || exit 1
  done
done

log "=== summary"
"$PY" - <<'PY'
import pandas as pd
df = pd.read_csv("outputs/tables/baseline_stratified_auroc.csv")
g = df.groupby(["task","scheme","features","stratum"]).agg(
    folds=("auroc","size"), mean=("auroc","mean"), sd=("auroc","std"),
    n=("n","mean"), pos=("pos_rate","mean")).reset_index()
print(f"\n{'task':<4} {'scheme':<8} {'features':<8} {'stratum':<7} {'folds':>5} {'AUROC':>8} {'sd':>6} {'mean n':>8} {'pos':>6}")
for r in g.itertuples():
    print(f"{r.task:<4} {r.scheme:<8} {r.features:<8} {r.stratum:<7} {r.folds:>5} "
          f"{r.mean*100:>7.2f}% {0 if pd.isna(r.sd) else r.sd*100:>5.2f} {r.n:>8.0f} {r.pos:>5.2f}")

print("\n=== imagery advantage over the urban/rural baseline (same folds) ===")
p = df.pivot_table(index=["task","scheme","stratum","fold"], columns="features", values="auroc")
p = p.dropna().reset_index()
d = p.groupby(["task","scheme","stratum"]).apply(
    lambda x: pd.Series({"delta_pts": (x["dinov2"]-x["urbrur"]).mean()*100,
                         "folds": len(x)}), include_groups=False).reset_index()
for r in d.itertuples():
    print(f"{r.task:<4} {r.scheme:<8} {r.stratum:<7} delta={r.delta_pts:+6.2f} pts  (folds={int(r.folds)})")
PY
log "DONE"
