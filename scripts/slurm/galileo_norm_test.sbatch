#!/bin/bash
#SBATCH --job-name=sdg6-galnorm
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=124G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Diagnostic: is Galileo's poor result an input-normalization artifact?
# The tiles are uint8 (0-255) but Galileo normalizes against Sentinel-2
# reflectance stats (~0-10000). Re-extract under corrected input scales and
# compare held-out AUROC against the broken baseline (value_scale=1, normalize).

set -uo pipefail
ROOT="${SDG6_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$ROOT" || exit 1
PY="${PY:-$ROOT/.venv/bin/python}"
W="$ROOT/runs/pretrained/galileo/base"
OUT="$ROOT/outputs/tables/galileo_norm_test_auroc.csv"
rm -f "$OUT"

# condition name -> extract args
run () {  # $1=tag  $2...=extra args
  local tag="$1"; shift
  local d="$ROOT/runs/embeddings/galileo_$tag"
  echo "[$(date +%T)] === extract galileo/$tag"
  "$PY" scripts/extract_embeddings.py --model galileo --weights "$W" \
      --out "$ROOT/runs/embeddings" "$@" || return 1
  # extract_embeddings writes to <out>/galileo; move to a tagged dir
  rm -rf "$d"; mv "$ROOT/runs/embeddings/galileo" "$d"
  for t in pw sw; do
    "$PY" scripts/eval_splits.py --scheme original --task "$t" \
        --emb-dir "$d" --model "galileo_$tag" --out "$OUT" || return 1
  done
}

# scale 39.2 maps 255 -> ~10000 (into the reflectance range Galileo expects)
run "scale39_norm"   --value-scale 39.2 --galileo-normalize
# no reflectance normalization: let the raw 0-255 dynamic range through
run "raw_nonorm"     --value-scale 1.0  --no-galileo-normalize
# baseline reproduction of the broken setting, for a clean side-by-side
run "baseline"       --value-scale 1.0  --galileo-normalize

echo "[$(date +%T)] === AUROC at k=200 by condition"
"$PY" - <<'PY'
import pandas as pd
d = pd.read_csv("outputs/tables/galileo_norm_test_auroc.csv")
d = d[d.k==200]
for (m,t),g in d.groupby(["model","task"]):
    print(f"  {m:<22} {t}  AUROC={g.auroc.iloc[0]*100:.2f}%")
PY
echo "[$(date +%T)] DONE"
