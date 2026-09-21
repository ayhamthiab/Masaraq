"""
Utility package shim.

The repository imports `utils` in several modules. The real implementations live in
`ml_models.utils`. This shim re-exports the commonly used names so imports like
`from utils import ensure_dirs, save_json` continue to work.
"""
# Use an absolute import so the shim works both when `src` is on sys.path
# and when the package is imported as `src.*`.
try:
    from ml_models.utils import *  # re-export everything for backward compatibility
except Exception:
    # Fallback to the src-prefixed import (covers different import layouts)
    from src.ml_models.utils import *  # type: ignore
