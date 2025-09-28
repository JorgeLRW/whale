from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist

from whale.pipeline.diagrams import normalize_diagram
from whale.pointcloud import PointCloud

try:  # optional dependency for Vietoris--Rips reference
    import gudhi  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    gudhi = None


@dataclass
class RipsReference:
    diagrams: list[NDArray[np.float64]]
    sample_size: int
    max_edge: Optional[float]
    elapsed: float


def prepare_rips_reference(
    pc: PointCloud,
    *,
    max_dim: int,
    sample_size: int,
    percentile: float,
    seed: int,
) -> Optional[RipsReference]:
    if gudhi is None:
        return None
    if sample_size <= 0 or pc.points.shape[0] < sample_size:
        sample_size = pc.points.shape[0]
    if sample_size < max_dim + 2:
        return None

    rng = np.random.default_rng(seed)
    idx = rng.choice(pc.points.shape[0], size=sample_size, replace=False)
    subset = pc.points[idx]

    pairwise = pdist(subset)
    if pairwise.size == 0:
        max_edge = None
    else:
        max_edge = float(np.percentile(pairwise, percentile))

    start = time.perf_counter()
    rips_cls = getattr(gudhi, "RipsComplex", None)
    if rips_cls is None:
        raise RuntimeError("Installed Gudhi package does not expose RipsComplex().")
    rips_complex = rips_cls(points=subset, max_edge_length=max_edge)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
    simplex_tree.persistence(homology_coeff_field=2)
    elapsed = time.perf_counter() - start

    diagrams = [normalize_diagram(simplex_tree.persistence_intervals_in_dimension(dim)) for dim in range(max_dim + 1)]
    return RipsReference(diagrams=diagrams, sample_size=sample_size, max_edge=max_edge, elapsed=elapsed)


def bottleneck_distance(
    diag_a: Any,
    diag_b: Any,
    *,
    epsilon: float = 0.0,
) -> float:
    if gudhi is None:
        raise RuntimeError("Gudhi is not available; install gudhi to compute bottleneck distances.")
    func = getattr(gudhi, "bottleneck_distance", None)
    if func is None:
        raise RuntimeError("Installed Gudhi package does not expose bottleneck_distance().")
    return float(func(diag_a, diag_b, e=epsilon))
