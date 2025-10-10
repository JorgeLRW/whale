"""Whale: a reusable witness-complex pipeline for large point clouds.

The :mod:`whale` package exposes the core building blocks used throughout the
repository as a lightweight library so that downstream projects can reuse the
landmark selection, witness diagram construction, and diagnostic helpers beyond
the MRI-focused command line interfaces.

All public objects are re-exported at the package root for convenience::

    from whale import PointCloud, select_landmarks, compute_witness_diagrams

"""

from .ai import (
    WitnessFeatureLayer,
    WitnessSummary,
    batch_embeddings_to_point_clouds,
    batch_witness_summaries,
    embedding_to_point_cloud,
    embedding_witness_summary,
)
from .pointcloud import PointCloud
from .io import (
    ensure_nibabel_for_real_data,
    generate_synthetic_volume,
    load_mri_volume,
    volume_to_point_cloud,
)
from .pipeline import (
    RipsReference,
    bottleneck_distance,
    compute_witness_diagrams,
    coverage_metrics,
    gudhi,
    landmark_intensity_stats,
    normalize_diagram,
    parse_methods,
    prepare_rips_reference,
    select_landmarks,
)

__all__ = [
    "PointCloud",
    "WitnessFeatureLayer",
    "WitnessSummary",
    "batch_embeddings_to_point_clouds",
    "batch_witness_summaries",
    "ensure_nibabel_for_real_data",
    "generate_synthetic_volume",
    "load_mri_volume",
    "volume_to_point_cloud",
    "RipsReference",
    "bottleneck_distance",
    "compute_witness_diagrams",
    "coverage_metrics",
    "gudhi",
    "landmark_intensity_stats",
    "normalize_diagram",
    "parse_methods",
    "prepare_rips_reference",
    "select_landmarks",
    "embedding_to_point_cloud",
    "embedding_witness_summary",
]
