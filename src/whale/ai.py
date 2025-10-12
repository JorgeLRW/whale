"""Utilities for integrating Whale with deep-learning pipelines.

This module provides helpers that turn high-dimensional embeddings (NumPy arrays or
PyTorch tensors) into :class:`~whale.pointcloud.PointCloud` objects, compute witness
persistence summaries, and expose the results as torch layers for feature extraction.

**Important**: The witness persistence computation is performed in NumPy and is not
differentiable. :class:`WitnessFeatureLayer` can be used for:

- Feature extraction: embeddings → TDA features → classifier
- Regularization terms (using ``.detach()``)

You cannot backpropagate through the TDA computation to optimize embeddings for specific
topological properties. This is a fundamental limitation of most TDA operations.

**Usage Example**::

    import torch.nn as nn
    from whale.ai import WitnessFeatureLayer

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(784, 128)
            # TDA layer computes ~20-30 features depending on max_dim
            self.tda_layer = WitnessFeatureLayer(
                method="hybrid", m=100, tda_mode="fast", summary_keys=None
            )
            # Concatenate encoder output (128) with TDA features
            self.classifier = nn.Linear(128 + 30, 10)  # adjust based on actual TDA feature count
        
        def forward(self, x):
            embeddings = self.encoder(x)  # (batch, 128)
            # WitnessFeatureLayer expects (batch, n_points, dim)
            tda_features = self.tda_layer(embeddings.unsqueeze(1))  # (batch, num_tda_features)
            combined = torch.cat([embeddings, tda_features], dim=-1)
            return self.classifier(combined)

**Performance Considerations**:

Computing witness persistence on every forward pass can be slow during training. Consider:

- **Precomputing features offline**: Use :func:`batch_witness_summaries` to generate
  TDA features once and cache them.
- **Periodic computation**: Only compute TDA features every N batches.
- **Fast mode during training**: Use ``tda_mode="fast"`` (dim-1 only) for training and
  ``tda_mode="regular"`` (dim-2 aware) for validation/testing.

**Summary Key Handling**:

:class:`WitnessFeatureLayer` infers feature names from the first forward pass if
``summary_keys`` is not provided. Once inferred, the schema is locked. To avoid schema
mismatches across batches, either:

- Provide ``summary_keys`` explicitly in the constructor.
- Ensure the first forward pass is representative of all subsequent inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .pointcloud import PointCloud
from .pipeline import coverage_metrics, landmark_intensity_stats, select_landmarks
from .pipeline.diagrams import compute_witness_diagrams

try:  # pragma: no cover - optional dependency
    import torch
    from torch import Tensor, nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment]
    nn = Any  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

ArrayInput = Any


def _to_numpy(data: ArrayInput) -> NDArray[np.float64]:
    if _TORCH_AVAILABLE and (torch is not None) and isinstance(data, torch.Tensor):
        array = data.detach().cpu().numpy()
    else:
        array = np.asarray(data)
    return array.astype(np.float64, copy=False)


def _normalise_points(points: NDArray[np.float64], *, center: bool, scale: bool) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if points.ndim != 2:
        raise ValueError("Expected a 2-D array shaped (n_points, embedding_dim).")
    if points.size == 0:
        raise ValueError("Cannot build a point cloud from an empty embedding array.")

    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    centred = points
    if center:
        centred = centred - (bounds_min + bounds_max) / 2.0

    if scale:
        diameter = float(np.linalg.norm(bounds_max - bounds_min))
        if diameter > 0.0:
            centred = centred / diameter
    return centred.astype(np.float64, copy=False), bounds_min.astype(np.float64, copy=False), bounds_max.astype(np.float64, copy=False)


def embedding_to_point_cloud(
    embeddings: ArrayInput,
    *,
    weights: Optional[ArrayInput] = None,
    center: bool = True,
    scale: bool = True,
) -> PointCloud:
    """Convert a single embedding table into a :class:`PointCloud`.

    Parameters
    ----------
    embeddings:
        Array shaped ``(n_points, embedding_dim)``. Supports NumPy arrays and torch tensors.
    weights:
        Optional per-point weights; defaults to a vector of ones.
    center, scale:
        Control normalisation of the coordinates. By default the embedding is centred and
        scaled by the diameter of its bounding box so downstream landmark heuristics have
        sensible defaults.
    """

    array = _to_numpy(embeddings)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("`embeddings` must be a 2-D array once flattened.")

    if weights is None:
        intensities = np.ones(array.shape[0], dtype=np.float64)
    else:
        intensities = _to_numpy(weights).reshape(-1)
        if intensities.shape[0] != array.shape[0]:
            raise ValueError("`weights` must match the first dimension of `embeddings`.")

    normalised, bounds_min, bounds_max = _normalise_points(array, center=center, scale=scale)
    return PointCloud(points=normalised, intensities=intensities, bounds_min=bounds_min, bounds_max=bounds_max)


def batch_embeddings_to_point_clouds(
    embeddings: ArrayInput,
    *,
    weights: Optional[ArrayInput] = None,
    center: bool = True,
    scale: bool = True,
) -> List[PointCloud]:
    """Convert a batch of embeddings into point clouds.

    Accepts inputs shaped ``(batch, n_points, embedding_dim)`` or ``(n_points, embedding_dim)``.
    When ``weights`` is provided, it must either be 1-D (broadcast to every batch item) or
    match the leading dimension of ``embeddings``.
    """

    array = _to_numpy(embeddings)
    if array.ndim == 2:
        return [embedding_to_point_cloud(array, weights=weights, center=center, scale=scale)]
    if array.ndim != 3:
        raise ValueError("`embeddings` must be 2-D or 3-D (batch, n_points, embedding_dim).")

    weight_array: Optional[NDArray[np.float64]] = None
    if weights is not None:
        weight_array = _to_numpy(weights)
        if weight_array.ndim == 1:
            weight_array = np.broadcast_to(weight_array, (array.shape[1],))
        elif weight_array.ndim == 2 and weight_array.shape[0] == array.shape[0]:
            pass
        else:
            raise ValueError("`weights` must be 1-D or match the batch dimension of `embeddings`.")

    clouds: List[PointCloud] = []
    for i in range(array.shape[0]):
        batch_weights = None
        if weight_array is not None:
            batch_weights = weight_array if weight_array.ndim == 1 else weight_array[i]
        clouds.append(embedding_to_point_cloud(array[i], weights=batch_weights, center=center, scale=scale))
    return clouds


def _diagram_statistics(diagram: NDArray[np.float64]) -> Dict[str, float]:
    if diagram.size == 0:
        return {
            "count": 0.0,
            "max_persistence": 0.0,
            "mean_persistence": 0.0,
            "std_persistence": 0.0,
            "total_persistence": 0.0,
        }
    lifetimes = np.maximum(diagram[:, 1] - diagram[:, 0], 0.0)
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    return {
        "count": float(diagram.shape[0]),
        "max_persistence": float(lifetimes.max(initial=0.0)) if lifetimes.size else 0.0,
        "mean_persistence": float(lifetimes.mean()) if lifetimes.size else 0.0,
        "std_persistence": float(lifetimes.std(ddof=0)) if lifetimes.size else 0.0,
        "total_persistence": float(lifetimes.sum()) if lifetimes.size else 0.0,
    }


_TDA_MODES = {
    "fast": {"max_dim": 1, "k_witness": 64, "m": 128},
    "regular": {"max_dim": 2, "k_witness": 128, "m": 256},
}


def _resolve_tda_defaults(
    mode: str,
    *,
    max_dim: Optional[int],
    k_witness: Optional[int],
    m: Optional[int],
) -> Tuple[int, int, int]:
    resolved_mode = mode.lower()
    if resolved_mode == "auto":
        resolved_mode = "fast"
    if resolved_mode not in _TDA_MODES:
        raise ValueError(f"Unsupported tda_mode '{mode}'. Expected one of: auto, fast, regular.")
    defaults = _TDA_MODES[resolved_mode]
    resolved_max_dim = int(defaults["max_dim"] if max_dim is None else max_dim)
    resolved_k_witness = int(defaults["k_witness"] if k_witness is None else k_witness)
    resolved_m = int(defaults["m"] if m is None else m)
    return resolved_max_dim, resolved_k_witness, resolved_m


def embedding_witness_summary(
    embeddings: ArrayInput,
    *,
    weights: Optional[ArrayInput] = None,
    method: str = "hybrid",
    max_dim: Optional[int] = None,
    k_witness: Optional[int] = None,
    m: Optional[int] = None,
    seed: int = 42,
    selection_c: int = 8,
    hybrid_alpha: float = 0.35,
    coverage_radius: Optional[float] = None,
    center: bool = True,
    scale: bool = True,
    summary_keys: Optional[Sequence[str]] = None,
    tda_mode: str = "auto",
) -> Tuple[Dict[str, float], List[NDArray[np.float64]], Sequence[str]]:
    """Run witness persistence on a single embedding table and return summary features.

    The summary comprises coverage metrics, landmark intensity statistics, and basic
    persistence aggregates for each homology dimension up to ``max_dim``.
    """

    cloud = embedding_to_point_cloud(embeddings, weights=weights, center=center, scale=scale)
    X = cloud.points
    intensities = cloud.intensities
    n_points = X.shape[0]
    if n_points == 0:
        raise ValueError("Embeddings must contain at least one point.")

    resolved_max_dim, resolved_k_witness, resolved_m = _resolve_tda_defaults(
        tda_mode, max_dim=max_dim, k_witness=k_witness, m=m
    )

    budget = max(1, min(resolved_m, n_points))
    indices_array = select_landmarks(
        method,
        X,
        budget,
        seed=seed,
        selection_c=selection_c,
        hybrid_alpha=hybrid_alpha,
    )
    indices = np.asarray(indices_array, dtype=np.int64).tolist()

    diagrams = compute_witness_diagrams(X, indices, max_dim=resolved_max_dim, k_witness=resolved_k_witness)
    radius = coverage_radius
    if radius is None:
        radius = 0.05 if cloud.diameter == 0 else min(0.2, 0.05 * cloud.diameter)
    coverage = coverage_metrics(X, intensities, indices, radius=radius)
    intensity_stats = landmark_intensity_stats(intensities, indices)

    summary: Dict[str, float] = {**coverage, **intensity_stats}
    for dim, diagram in enumerate(diagrams):
        stats = _diagram_statistics(diagram)
        for key, value in stats.items():
            summary[f"h{dim}_{key}"] = value

    if summary_keys is None:
        ordered_keys = tuple(sorted(summary.keys()))
    else:
        ordered_keys = tuple(summary_keys)
        for key in ordered_keys:
            summary.setdefault(key, 0.0)

    return summary, diagrams, ordered_keys


def batch_witness_summaries(
    embeddings: ArrayInput,
    *,
    weights: Optional[ArrayInput] = None,
    method: str = "hybrid",
    max_dim: Optional[int] = None,
    k_witness: Optional[int] = None,
    m: Optional[int] = None,
    seed: int = 42,
    selection_c: int = 8,
    hybrid_alpha: float = 0.35,
    coverage_radius: Optional[float] = None,
    center: bool = True,
    scale: bool = True,
    summary_keys: Optional[Sequence[str]] = None,
    return_vectors: bool = False,
    tda_mode: str = "auto",
) -> Tuple[List[Dict[str, float]], Sequence[str], Optional[List[NDArray[np.float32]]]]:
    """Vectorised wrapper around :func:`embedding_witness_summary` for batched inputs."""

    clouds = batch_embeddings_to_point_clouds(embeddings, weights=weights, center=center, scale=scale)
    summaries: List[Dict[str, float]] = []
    derived_keys: Optional[Sequence[str]] = summary_keys

    for i, cloud in enumerate(clouds):
        item_weights = None
        if weights is not None:
            weight_array = _to_numpy(weights)
            if weight_array.ndim == 1:
                item_weights = weight_array
            elif weight_array.ndim == 2:
                item_weights = weight_array[i]
        summary, _diagrams, ordered_keys = embedding_witness_summary(
            cloud.points,
            weights=item_weights,
            method=method,
            max_dim=max_dim,
            k_witness=k_witness,
            m=m,
            seed=seed + i,
            selection_c=selection_c,
            hybrid_alpha=hybrid_alpha,
            coverage_radius=coverage_radius,
            center=False,
            scale=False,
            summary_keys=derived_keys,
            tda_mode=tda_mode,
        )
        summaries.append(summary)
        if derived_keys is None:
            derived_keys = ordered_keys

    vectors: Optional[List[NDArray[np.float32]]] = None
    if return_vectors:
        if derived_keys is None:
            derived_keys = tuple()
        vectors = []
        for summary in summaries:
            vec = np.array([summary.get(key, 0.0) for key in derived_keys], dtype=np.float32)
            vectors.append(vec)

    return summaries, tuple(derived_keys or ()), vectors


@dataclass
class WitnessSummary:
    summary: Dict[str, float]
    diagrams: List[NDArray[np.float64]]


if _TORCH_AVAILABLE:

    class WitnessFeatureLayer(nn.Module):  # type: ignore[misc]
        """Torch layer that emits witness persistence summaries for each batch element.
        
        **Non-Differentiability**: This layer computes witness persistence in NumPy and
        returns detached tensors. Gradients do not flow through the TDA computation.
        Use this layer for feature extraction or regularization (with ``.detach()``), not
        for optimizing embeddings toward specific topological properties.
        
        **Summary Key Locking**: If ``summary_keys`` is ``None``, the layer infers feature
        names from the first forward pass and locks the schema. Subsequent batches must
        produce the same set of keys. To avoid schema mismatches, provide ``summary_keys``
        explicitly or ensure the first forward pass is representative.
        
        **Performance**: Computing TDA on every forward pass can be slow. Consider:
        
        - Precomputing features offline with :func:`batch_witness_summaries`.
        - Using ``tda_mode="fast"`` during training, ``tda_mode="regular"`` for validation.
        - Computing features periodically (e.g., every N batches) instead of every step.
        
        Parameters
        ----------
        method : str, default="hybrid"
            Landmark selection method: "hybrid", "density", "random", etc.
        max_dim : int, optional
            Maximum homology dimension. If ``None``, derived from ``tda_mode``.
        k_witness : int, optional
            Number of nearest landmarks for witness complex. If ``None``, derived from ``tda_mode``.
        m : int, optional
            Landmark budget. If ``None``, derived from ``tda_mode``.
        selection_c : int, default=8
            Parameter for landmark selection heuristics.
        hybrid_alpha : float, default=0.35
            Blending parameter for hybrid sampling.
        coverage_radius : float, optional
            Radius for coverage metrics. Defaults to adaptive heuristic if ``None``.
        summary_keys : Sequence[str], optional
            Explicit list of feature names. If ``None``, inferred from first forward pass.
        dtype : torch.dtype, optional
            Output tensor dtype. Defaults to ``torch.float32``.
        tda_mode : str, default="auto"
            TDA configuration preset: "auto" (same as "fast"), "fast" (dim-1, lightweight),
            or "regular" (dim-2 aware).
        """

        def __init__(
            self,
            *,
            method: str = "hybrid",
            max_dim: Optional[int] = None,
            k_witness: Optional[int] = None,
            m: Optional[int] = None,
            selection_c: int = 8,
            hybrid_alpha: float = 0.35,
            coverage_radius: Optional[float] = None,
            summary_keys: Optional[Sequence[str]] = None,
            dtype: Any = None,
            tda_mode: str = "auto",
        ) -> None:
            super().__init__()
            self.method = method
            self.max_dim = max_dim
            self.k_witness = k_witness
            self.m = m
            self.selection_c = selection_c
            self.hybrid_alpha = hybrid_alpha
            self.coverage_radius = coverage_radius
            self._summary_keys: Optional[Tuple[str, ...]] = tuple(summary_keys) if summary_keys is not None else None
            default_dtype = torch.float32 if torch is not None else None
            if dtype is None:
                if default_dtype is None:  # pragma: no cover - defensive guard
                    raise RuntimeError("PyTorch default dtype unavailable; ensure torch is installed.")
                self.output_dtype = default_dtype
            else:
                self.output_dtype = dtype
            self.tda_mode = tda_mode

        def forward(self, embeddings: Tensor) -> Tensor:  # type: ignore[override]
            device = embeddings.device
            _summaries, keys, vectors = batch_witness_summaries(
                embeddings,
                method=self.method,
                max_dim=self.max_dim,
                k_witness=self.k_witness,
                m=self.m,
                selection_c=self.selection_c,
                hybrid_alpha=self.hybrid_alpha,
                coverage_radius=self.coverage_radius,
                summary_keys=self._summary_keys,
                return_vectors=True,
                tda_mode=self.tda_mode,
            )
            if vectors is None:
                raise RuntimeError("WitnessFeatureLayer expected vector outputs but none were produced.")
            if self._summary_keys is None:
                self._summary_keys = tuple(keys)
            array = np.stack(vectors, axis=0) if vectors else np.zeros((0, len(keys)), dtype=np.float32)
            if torch is None:  # pragma: no cover - defensive
                raise RuntimeError("PyTorch is required for WitnessFeatureLayer outputs.")
            return torch.as_tensor(array, device=device, dtype=self.output_dtype)

else:  # pragma: no cover - executed only when torch is missing

    class WitnessFeatureLayer:  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - simple guard
            raise ImportError("torch is required to use WitnessFeatureLayer; install torch>=1.13.")

        def forward(self, embeddings):  # type: ignore[no-untyped-def]
            raise ImportError("torch is required to use WitnessFeatureLayer; install torch>=1.13.")