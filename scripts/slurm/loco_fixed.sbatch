#!/bin/bash
#SBATCH --job-name=sdg6-loco
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=124G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
ROOT="${SDG6_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$ROOT"||exit 1
PY="${PY:-$ROOT/.venv/bin/python}"
OUT="$ROOT/outputs/tables/loco_by_country_auroc.csv"; rm -f "$OUT"
for t in pw sw; do
  "$PY" scripts/eval_splits.py --scheme country --task "$t" --emb-dir runs/embeddings/dinov2 --model dinov2 --out "$OUT" || exit 1
done
echo "=== LOCO summary (real countries, k=200) ==="
"$PY" - <<'PY'
import pandas as pd
d=pd.read_csv("outputs/tables/loco_by_country_auroc.csv"); d=d[d.k==200]
for t in ("pw","sw"):
    g=d[d.task==t]
    print(f"{t}: n={len(g)} mean={g.auroc.mean()*100:.2f} sd={g.auroc.std()*100:.2f} "
          f"median={g.auroc.median()*100:.2f} range {g.auroc.min()*100:.1f}-{g.auroc.max()*100:.1f} "
          f"<80%: {(g.auroc<0.8).sum()}/{len(g)}")
PY
echo DONE
