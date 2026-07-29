#!/bin/bash
#SBATCH --job-name=sdg6-splits
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=124G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Score the same embeddings under every split scheme.
#
# The `original` scheme runs first and doubles as an end-to-end sanity check:
# it must reproduce the published AUROC (~91.54% PW, ~93.24% SW at k=200). If it
# does not, the embeddings or the manifest join are wrong and the rest of the
# numbers are meaningless, so the job stops there.
#
#   sbatch scripts/slurm/eval_splits.sbatch

set -uo pipefail

ROOT="${SDG6_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$ROOT" || exit 1
PY="${PY:-$ROOT/.venv/bin/python}"
OUT="$ROOT/outputs/tables/split_scheme_auroc.csv"

log() { echo "[$(date +'%F %T')] $*"; }

# Start clean so re-runs do not append duplicates.
rm -f "$OUT"

for task in pw sw; do
  log "=== original split (reproduction check) task=$task"
  "$PY" scripts/eval_splits.py --scheme original --task "$task" --out "$OUT" || exit 1
done

for task in pw sw; do
  for scheme in random spatial country; do
    log "=== scheme=$scheme task=$task"
    "$PY" scripts/eval_splits.py --scheme "$scheme" --task "$task" --out "$OUT" || exit 1
  done
done

log "=== summary"
"$PY" - <<'PY'
import pandas as pd
df = pd.read_csv("outputs/tables/split_scheme_auroc.csv")
k = df["k"].max()
d = df[df["k"] == k]
print(f"\nAUROC at k={k}, by scheme (fold-level mean +/- sd):\n")
print(f"{'task':<5} {'scheme':<9} {'folds':>6} {'mean':>8} {'median':>8} {'sd':>7} {'min':>8} {'max':>8}")
for (t, s), g in d.groupby(["task", "scheme"]):
    print(f"{t:<5} {s:<9} {len(g):>6} {g.auroc.mean()*100:>7.2f}% {g.auroc.median()*100:>7.2f}% "
          f"{g.auroc.std()*100 if len(g)>1 else 0:>6.2f} {g.auroc.min()*100:>7.2f}% {g.auroc.max()*100:>7.2f}%")
PY
log "DONE"
