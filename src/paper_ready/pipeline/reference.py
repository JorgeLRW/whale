"""Compatibility shim forwarding to :mod:`whale.pipeline.reference`."""

from whale.pipeline.reference import (
    RipsReference,
    bottleneck_distance,
    gudhi,
    prepare_rips_reference,
)

__all__ = ["RipsReference", "prepare_rips_reference", "bottleneck_distance", "gudhi"]
