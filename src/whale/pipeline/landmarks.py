from __future__ import annotations

from typing import List

import numpy as np

from whale.methodology import selection

try:
    from whale.sampling.hybrid_sampler import hybrid_maxmin_kde  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hybrid_maxmin_kde = None


def parse_methods(arg: str) -> List[str]:
    parts = [token.strip() for token in arg.split(",") if token.strip()]
    if not parts:
        raise ValueError("No landmark methods specified.")
    return parts


def select_landmarks(
    method: str,
    X: np.ndarray,
    m: int,
    *,
    seed: int,
    selection_c: int,
    hybrid_alpha: float,
) -> np.ndarray:
    if m >= X.shape[0]:
        return np.arange(X.shape[0], dtype=np.int64)

    if method == "random":
        rng = np.random.default_rng(seed)
        return rng.choice(X.shape[0], size=m, replace=False).astype(np.int64)

    if method == "density":
        idx, _meta = selection.select_landmarks(X, m=m, c=selection_c, rng_seed=seed)
        return np.asarray(idx, dtype=np.int64)

    if method == "hybrid":
        if hybrid_maxmin_kde is None:
            raise RuntimeError("Hybrid sampler unavailable (scikit-learn may be missing); install requirements and retry.")
        idx = np.asarray(hybrid_maxmin_kde(X, m=m, alpha=hybrid_alpha, seed=seed), dtype=np.int64)
        if idx.size < m:
            candidates = np.setdiff1d(np.arange(X.shape[0]), idx, assume_unique=False)
            rng = np.random.default_rng(seed + 911)
            needed = m - idx.size
            if needed <= 0 or candidates.size == 0:
                return idx[:m]
            extras = rng.choice(candidates, size=min(needed, candidates.size), replace=False)
            idx = np.concatenate([idx, extras])
        return idx[:m]

    raise ValueError(f"Unsupported method '{method}'")
