"""Compatibility shim forwarding to :mod:`whale.pipeline.metrics`."""

from whale.pipeline.metrics import (
    coverage_metrics,
    ensure_weights,
    landmark_intensity_stats,
)

__all__ = ["ensure_weights", "coverage_metrics", "landmark_intensity_stats"]
