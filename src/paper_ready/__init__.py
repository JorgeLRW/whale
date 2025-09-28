"""Public package for the MRI landmark-based persistence pipeline.

The code is organised into small, composable submodules so the CLI wrappers
can be reused or adapted in isolation.  The main entry points remain::

    python -m paper_ready.mri_deep_dive
    python -m paper_ready.mri_deep_dive_fast
"""

from . import mri_deep_dive, mri_deep_dive_fast
from .io import volume
from .pipeline import diagrams, landmarks, metrics, reference
from .pointcloud import PointCloud

__all__ = [
    "PointCloud",
    "diagrams",
    "landmarks",
    "metrics",
    "reference",
    "volume",
    "mri_deep_dive",
    "mri_deep_dive_fast",
]
