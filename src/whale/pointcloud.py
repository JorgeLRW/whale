from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PointCloud:
    """Container for a normalised point cloud with optional intensity weights."""

    points: NDArray[np.float64]
    intensities: NDArray[np.float64]
    bounds_min: NDArray[np.float64]
    bounds_max: NDArray[np.float64]

    @property
    def diameter(self) -> float:
        """Return the Euclidean diameter of the original coordinate bounding box."""

        return float(np.linalg.norm(self.bounds_max - self.bounds_min))
