# DINO-family AUROC (PW-s and SW-s)

Models: dino, dinov2, dinov3

Note: dinov3 values are computed from per-sample prediction confidences.

## PW-s

| model | k | AUROC |
|---|---:|---:|
| dino | 5 | 0.854984 |
| dino | 10 | 0.865134 |
| dino | 20 | 0.878825 |
| dino | 50 | 0.899134 |
| dino | 100 | 0.912907 |
| dinov2 | 5 | 0.842821 |
| dinov2 | 10 | 0.854347 |
| dinov2 | 20 | 0.865184 |
| dinov2 | 50 | 0.881460 |
| dinov2 | 100 | 0.898158 |
| dinov2 | 200 | 0.915367 |
| dinov3 | 5 | 0.870960 |
| dinov3 | 10 | 0.882518 |
| dinov3 | 20 | 0.893129 |
| dinov3 | 50 | 0.902239 |
| dinov3 | 100 | 0.901391 |

| model | best k | best AUROC |
|---|---:|---:|
| dino | 100 | 0.912907 |
| dinov2 | 200 | 0.915367 |
| dinov3 | 50 | 0.902239 |

## SW-s

| model | k | AUROC |
|---|---:|---:|
| dino | 5 | 0.857822 |
| dino | 10 | 0.865351 |
| dino | 20 | 0.879323 |
| dino | 50 | 0.906347 |
| dino | 100 | 0.930927 |
| dinov2 | 5 | 0.857377 |
| dinov2 | 10 | 0.865957 |
| dinov2 | 20 | 0.874921 |
| dinov2 | 50 | 0.893276 |
| dinov2 | 100 | 0.912159 |
| dinov2 | 200 | 0.932391 |
| dinov3 | 5 | 0.880926 |
| dinov3 | 10 | 0.891678 |
| dinov3 | 20 | 0.902385 |
| dinov3 | 50 | 0.917650 |
| dinov3 | 100 | 0.918950 |

| model | best k | best AUROC |
|---|---:|---:|
| dino | 100 | 0.930927 |
| dinov2 | 200 | 0.932391 |
| dinov3 | 100 | 0.918950 |

