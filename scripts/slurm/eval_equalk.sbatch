#!/bin/bash
#SBATCH --job-name=sdg6-equalk
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=124G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Equal-k comparison across encoders.
#
# The published ranking gave DINOv2 a k grid up to 200 while dino, dinov3 and
# galileo were capped at 100. AUROC rises with k, and DINOv2's margin over dino
# is 0.25 points on piped water and 0.15 on sewage, so the comparison was not
# like for like. This re-scores every available encoder on the identical grid.

set -uo pipefail
ROOT="${SDG6_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$ROOT" || exit 1
PY="${PY:-$ROOT/.venv/bin/python}"
OUT="$ROOT/outputs/tables/equal_k_auroc.csv"
rm -f "$OUT"

for model in dinov2 dino galileo; do
  d="$ROOT/runs/embeddings/$model"
  [ -d "$d" ] || { echo "skip $model (no embeddings)"; continue; }
  for task in pw sw; do
    echo "[$(date +'%F %T')] === $model / $task"
    "$PY" scripts/eval_splits.py --scheme original --task "$task" \
        --emb-dir "$d" --model "$model" --out "$OUT" || exit 1
  done
done

echo "[$(date +'%F %T')] === equal-k summary"
"$PY" - <<'PY'
import pandas as pd
d = pd.read_csv("outputs/tables/equal_k_auroc.csv")
if "model" not in d.columns:
    raise SystemExit("eval_splits did not record the model column")
for task in ("pw","sw"):
    g = d[d.task==task]
    p = g.pivot_table(index="model", columns="k", values="auroc")*100
    print(f"\n=== {task.upper()} AUROC by k ===")
    print(p.round(2).to_string())
    if 200 in p.columns:
        s = p[200].sort_values(ascending=False)
        print(f"  ranking at k=200: " + ", ".join(f"{m}={v:.2f}%" for m,v in s.items()))
        print(f"  spread top-to-bottom: {s.iloc[0]-s.iloc[-1]:.2f} pts")
PY
echo "[$(date +'%F %T')] DONE"
