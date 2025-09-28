"""Legacy namespace that forwards to :mod:`whale.io`."""

from whale.io import (
    ensure_nibabel_for_real_data,
    generate_synthetic_volume,
    load_mri_volume,
    volume_to_point_cloud,
)

__all__ = [
    "ensure_nibabel_for_real_data",
    "generate_synthetic_volume",
    "load_mri_volume",
    "volume_to_point_cloud",
]
