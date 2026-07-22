"""Test configuration — keep BLAS/OpenMP single-threaded.

Rorqual login nodes cap the per-user process/thread count, and torch's OpenMP
runtime aborts with "libgomp: Thread creation failed: Resource temporarily
unavailable" when it tries to spawn a thread per core. pytest then dies with no
output at all, which is confusing to debug. These variables must be set before
torch is imported, so this module is deliberately import-light.
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import torch  # noqa: E402  (must follow the env vars above)

torch.set_num_threads(1)
