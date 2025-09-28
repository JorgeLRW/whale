from __future__ import annotations

import pathlib
from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from whale.pointcloud import PointCloud

try:  # optional dependency for real MRI volumes
    import nibabel as nib
    from nibabel.filebasedimages import FileBasedImage
    from nibabel.loadsave import load as nib_load
except Exception:  # pragma: no cover - optional dependency
    nib = None  # type: ignore[assignment]
    nib_load = None  # type: ignore[assignment]
    FileBasedImage = Any  # type: ignore[misc,assignment]


def ensure_nibabel_for_real_data(use_synthetic: bool) -> None:
    """Validate that nibabel is available when the user requests real MRI volumes."""

    if use_synthetic:
        return
    if nib is None or nib_load is None:
        raise RuntimeError(
            "Nibabel is required for loading MRI volumes. Install it via `pip install nibabel` "
            "or enable the synthetic mode with --synthetic."
        )


def load_mri_volume(path: pathlib.Path) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load an MRI volume (and its affine) from disk using nibabel."""

    if nib is None or nib_load is None:
        raise RuntimeError(
            "Nibabel is required for loading MRI volumes. Install it via `pip install nibabel`."
        )
    img: FileBasedImage = nib_load(str(path))  # type: ignore[assignment]
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float64)
    affine = np.asarray(img.affine, dtype=np.float64)
    return data, affine


def generate_synthetic_volume(size: int = 120, seed: int = 0) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate a smooth MRI-like phantom volume for benchmarking."""

    rng = np.random.default_rng(seed)
    grid = np.indices((size, size, size)).astype(np.float64)
    coords = np.stack([grid[i] for i in range(3)], axis=-1)

    center1 = np.array([size * 0.45, size * 0.5, size * 0.5])
    center2 = np.array([size * 0.62, size * 0.55, size * 0.45])
    radius1 = size * 0.22
    radius2 = size * 0.18

    d1 = np.linalg.norm(coords - center1, axis=-1)
    d2 = np.linalg.norm(coords - center2, axis=-1)
    volume = np.exp(-(d1 / radius1) ** 2) + 0.8 * np.exp(-(d2 / radius2) ** 2)

    lesion_center = np.array([size * 0.55, size * 0.35, size * 0.58])
    lesion_radius = size * 0.08
    d3 = np.linalg.norm(coords - lesion_center, axis=-1)
    volume += 1.4 * np.exp(-(d3 / lesion_radius) ** 2)

    noise = rng.normal(scale=0.05, size=volume.shape)
    taper = np.prod([
        np.clip(1.0 - np.abs((grid[i] - size / 2) / (size / 2)), 0.0, 1.0) for i in range(3)
    ], axis=0)
    volume = (volume + noise) * taper
    volume = np.clip(volume, 0.0, None)

    affine = np.eye(4, dtype=np.float64)
    return volume.astype(np.float64), affine


def volume_to_point_cloud(
    volume: NDArray[np.float64],
    affine: NDArray[np.float64],
    *,
    percentile: float,
    intensity_threshold: Optional[float],
    max_points: int,
    seed: int,
) -> PointCloud:
    """Convert a volumetric MRI into a normalised point cloud with intensities."""

    coords = np.argwhere(np.ones_like(volume, dtype=bool))
    intensities = volume.reshape(-1)

    if intensity_threshold is None:
        intensity_threshold = float(np.percentile(intensities, percentile))
    mask = intensities >= intensity_threshold
    if not np.any(mask):
        raise ValueError(
            "No voxels survived the intensity threshold. Consider lowering --mask-percentile or --intensity-threshold."
        )
    coords = coords[mask]
    intensities = intensities[mask]

    homogeneous = np.concatenate([coords, np.ones((coords.shape[0], 1), dtype=np.float32)], axis=1)
    physical = (affine @ homogeneous.T).T[:, :3]

    bounds_min = physical.min(axis=0)
    bounds_max = physical.max(axis=0)
    span = bounds_max - bounds_min
    span[span == 0.0] = 1.0
    normalised = (physical - bounds_min) / span

    rng = np.random.default_rng(seed)
    if normalised.shape[0] > max_points:
        weights = intensities.copy()
        weights -= weights.min()
        weights = np.clip(weights, 0.0, None)
        if weights.sum() <= 0:
            indices = rng.choice(normalised.shape[0], size=max_points, replace=False)
        else:
            weights = weights / weights.sum()
            indices = rng.choice(normalised.shape[0], size=max_points, replace=False, p=weights)
        normalised = normalised[indices]
        intensities = intensities[indices]

    intensities = intensities.astype(np.float64, copy=False)
    intensities -= intensities.min()
    intensities += 1e-9

    return PointCloud(
        points=normalised.astype(np.float64, copy=False),
        intensities=intensities,
        bounds_min=bounds_min.astype(np.float64, copy=False),
        bounds_max=bounds_max.astype(np.float64, copy=False),
    )
