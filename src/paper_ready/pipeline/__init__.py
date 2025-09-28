"""Compatibility layer that forwards to :mod:`whale.pipeline`."""

from whale.pipeline import (
    RipsReference,
    bottleneck_distance,
    compute_witness_diagrams,
    coverage_metrics,
    ensure_weights,
    gudhi,
    landmark_intensity_stats,
    normalize_diagram,
    parse_methods,
    prepare_rips_reference,
    select_landmarks,
)

__all__ = [
    "compute_witness_diagrams",
    "normalize_diagram",
    "parse_methods",
    "select_landmarks",
    "coverage_metrics",
    "ensure_weights",
    "landmark_intensity_stats",
    "RipsReference",
    "bottleneck_distance",
    "gudhi",
    "prepare_rips_reference",
]
