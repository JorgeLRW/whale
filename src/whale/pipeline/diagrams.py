from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from numpy.typing import NDArray

from whale.methodology.witness_ph import compute_witness_persistence


def normalize_diagram(diagram: Iterable) -> NDArray[np.float64]:
    arr = np.asarray(diagram, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    arr = arr.reshape(-1, 2)
    return np.ascontiguousarray(arr, dtype=np.float64)


def compute_witness_diagrams(
    X: NDArray[np.float64],
    indices: Sequence[int],
    *,
    max_dim: int,
    k_witness: int,
) -> List[NDArray[np.float64]]:
    raw = compute_witness_persistence(X, list(indices), max_dim=max_dim, k_witness=k_witness)
    diagrams: Dict[int, Iterable] = {}
    if isinstance(raw, dict):
        diagrams = raw  # type: ignore[assignment]
    else:
        for dim, diag in enumerate(raw):
            diagrams[dim] = diag
    return [normalize_diagram(diagrams.get(dim, [])) for dim in range(max_dim + 1)]
