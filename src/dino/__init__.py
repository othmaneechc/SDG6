from __future__ import annotations

import sys
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
if str(_pkg_dir) not in sys.path:
    # Preserve legacy intra-repo imports like `import utils`.
    sys.path.insert(0, str(_pkg_dir))
