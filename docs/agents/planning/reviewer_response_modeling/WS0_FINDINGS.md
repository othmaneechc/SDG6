# WS-0 findings — k-NN scoring probe (investigation only, no code changed)

Evidence: `src/sdg6/knn.py`, `scripts/analysis/compute_dino_family_auroc.py`,
`outputs/tables/dino_family_auroc_pws_sws.csv`, and the controlled reproduction in
`ws0_probe_knn_scoring.py`.

## Headline: the paper's two headline AUROC values are VALID

DINOv2 uses `k_values = "5,10,20,50,100,200"`, and both reported headline numbers
sit at **k = 200 = max_k**:

| dataset | model | k | AUROC |
|---|---|---|---|
| PW-s | dinov2 | **200 (= max_k)** | **91.54%** |
| SW-s | dinov2 | **200 (= max_k)** | **93.24%** |

At `k == max_k` the softmax weights are summed over all the neighbours they were
normalized over, so probabilities sum to exactly 1, the signed-confidence score is
a strictly monotone transform of P(y=1), and the AUROC is exactly correct. The
probe confirms agreement to **0.0000 points** at that k.

**No headline number needs to change on account of this defect.**

## But every AUROC at k < max_k is understated

The probe reproduces the exact code path. Probability mass actually summed:

| k | Σp (mean) | Σp (min) | AUROC current | AUROC correct | understated by |
|---|---|---|---|---|---|
| 5 | 0.161 | 0.077 | 62.33% | 63.96% | **1.62 pts** |
| 10 | 0.241 | 0.135 | 64.33% | 66.26% | **1.93 pts** |
| 20 | 0.351 | 0.231 | 67.27% | 69.11% | 1.83 pts |
| 50 | 0.554 | 0.454 | 71.00% | 72.17% | 1.16 pts |
| 100 | 0.757 | 0.701 | 73.69% | 74.20% | 0.51 pts |
| 200 | 1.000 | 1.000 | 75.54% | 75.54% | 0.00 pts |

At k = 5 the "probabilities" sum to **0.16**, not 1.

## Three consequences that do matter

### 1. The AUROC-vs-k curve overstates the benefit of larger k
In the probe, the current path rises +13.20 pts across k while the correct path
rises +11.58 pts. **~1.6 pts (≈12%) of the apparent gain from increasing k is a
normalization artifact, not signal.** `outputs/figures/dino_family_auroc_vs_k_pws_sws.png`
and the corresponding table inherit this.

### 2. The model ranking rests on an unequal hyperparameter grid ⚠

| model | k grid | max_k |
|---|---|---|
| dino | 5,10,20,50,100 | 100 |
| **dinov2** | 5,10,20,50,100,**200** | **200** |
| dinov3 | 5,10,20,50,100 | 100 |
| galileo | 5,10,20,50,100 | 100 |

Only DINOv2 was allowed k = 200. Since AUROC genuinely increases with k (+11.58 pts
even under correct scoring), and DINOv2's winning margin is:

- PW-s: 91.54 (dinov2, k=200) vs **91.29** (dino, k=100) → **+0.25 pts**
- SW-s: 93.24 (dinov2, k=200) vs **93.09** (dino, k=100) → **+0.15 pts**

…the margin is far smaller than the gain DINOv2 obtains from its extra k value.
**The "DINOv2 is the best encoder" claim is not established by this evidence.**
Each model's own max_k row is individually undistorted, so this is a *grid*
problem, not a normalization problem — and it is cheap to settle: re-evaluate
dino/dinov3/galileo at k = 200 reusing the existing embeddings. No re-encoding.

### 3. DINOv3 is specifically penalized
DINOv3's best PW-s value is at **k = 50**, i.e. *not* at its max_k, so that row is
one of the distorted ones and is understated (the probe suggests ~1 pt at that k).
DINOv3 is being compared at a handicap.

## Recommendation (no action taken pending your decision)

1. **Do not restate the headline AUROCs** — they are correct.
2. **Re-run all encoders on an identical k grid including k = 200** before any
   claim about which encoder wins. Cheap: embeddings are split- and k-independent.
3. **Fix the normalization** so every k is a proper distribution, then regenerate
   the AUROC-vs-k table/figure. This changes non-headline numbers upward.
4. Keep the signed-confidence AUROC only as a legacy cross-check; emit true
   P(y=1) once the fix lands.

Items 2–4 are code changes and are **not** started, per the investigate-only
instruction.
