# Lightweight package shim to help editors and runtime find the real package
# implementation located under `paper_ready/src/paper_ready`.
# This file is intentionally minimal to avoid import-time side effects.
from pathlib import Path

# Insert the `paper_ready/src/paper_ready` directory to the front of __path__
# so `import paper_ready...` will resolve to the implementation under src/.
_this_file = Path(__file__).resolve()
_repo_root = _this_file.parents[1]
_src_pkg = _repo_root / 'src' / 'paper_ready'
if _src_pkg.exists():
    __path__.insert(0, str(_src_pkg))
