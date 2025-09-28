from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors


def ensure_weights(intensities: NDArray[np.float64]) -> NDArray[np.float64]:
    weights = intensities.astype(np.float64, copy=True)
    weights -= weights.min()
    if np.allclose(weights, 0.0):
        weights = np.ones_like(weights) / weights.size
    else:
        weights = weights / weights.sum()
    return weights


def coverage_metrics(
    X: NDArray[np.float64],
    intensities: NDArray[np.float64],
    indices: Sequence[int],
    *,
    radius: float,
) -> Dict[str, float]:
    if len(indices) == 0:
        return {
            "coverage_mean": math.nan,
            "coverage_p95": math.nan,
            "coverage_weighted_mean": math.nan,
            "coverage_weighted_p95": math.nan,
            "coverage_ratio": 0.0,
            "coverage_weighted_ratio": 0.0,
        }

    nn = NearestNeighbors(n_neighbors=1).fit(X[indices])
    distances, _ = nn.kneighbors(X, return_distance=True)
    distances = distances.reshape(-1)
    weights = ensure_weights(intensities)

    coverage_mean = float(distances.mean())
    coverage_p95 = float(np.percentile(distances, 95))
    weighted_mean = float(np.dot(distances, weights))
    weighted_sorted = np.sort(distances)
    cdf = np.cumsum(np.sort(weights))
    weighted_p95 = float(weighted_sorted[np.searchsorted(cdf, 0.95, side="right")])

    mask = distances <= radius
    coverage_ratio = float(np.mean(mask))
    coverage_weighted_ratio = float(np.dot(mask.astype(float), weights))

    return {
        "coverage_mean": coverage_mean,
        "coverage_p95": coverage_p95,
        "coverage_weighted_mean": weighted_mean,
        "coverage_weighted_p95": weighted_p95,
        "coverage_ratio": coverage_ratio,
        "coverage_weighted_ratio": coverage_weighted_ratio,
    }


def landmark_intensity_stats(intensities: NDArray[np.float64], indices: Sequence[int]) -> Dict[str, float]:
    if len(indices) == 0:
        return {
            "landmark_intensity_mean": math.nan,
            "landmark_intensity_median": math.nan,
            "landmark_intensity_p90": math.nan,
        }
    vals = intensities[list(indices)]
    return {
        "landmark_intensity_mean": float(np.mean(vals)),
        "landmark_intensity_median": float(np.median(vals)),
        "landmark_intensity_p90": float(np.percentile(vals, 90)),
    }
