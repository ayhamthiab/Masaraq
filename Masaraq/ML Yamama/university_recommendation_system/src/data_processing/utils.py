# Utils functions (see canvas full code)
from __future__ import annotations
import random
import os
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Any, Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
RECS_DIR = OUTPUTS_DIR / "recommendations"




def ensure_dirs() -> None:
    for d in [DATA_DIR, RAW_DIR, PROC_DIR, INTERIM_DIR, MODELS_DIR, OUTPUTS_DIR, RECS_DIR]:
        d.mkdir(parents=True, exist_ok=True)




def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)




def load_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def save_json(obj: Any, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)




def save_model(obj: Any, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)




def load_model(path: Path | str) -> Any:
    return joblib.load(path)




def find_column(df, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    if required:
        raise ValueError(f"None of the expected columns found: {candidates}")
    return None