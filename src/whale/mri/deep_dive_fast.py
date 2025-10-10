"""Fast MRI deep-dive benchmarking presets built on the Whale pipeline."""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from whale.io import (
    ensure_nibabel_for_real_data,
    generate_synthetic_volume,
    load_mri_volume,
    volume_to_point_cloud,
)
from whale.pipeline import (
    bottleneck_distance,
    coverage_metrics,
    gudhi,
    landmark_intensity_stats,
    parse_methods,
    prepare_rips_reference,
    select_landmarks,
    compute_witness_diagrams,
)
from whale.pointcloud import PointCloud

from .deep_dive import add_result_row, compute_scaled_landmarks


def _replace_point_cloud(
    pc: PointCloud,
    *,
    points: Optional[NDArray[np.float64]] = None,
    intensities: Optional[NDArray[np.float64]] = None,
) -> PointCloud:
    return PointCloud(
        points=points if points is not None else pc.points,
        intensities=intensities if intensities is not None else pc.intensities,
        bounds_min=pc.bounds_min,
        bounds_max=pc.bounds_max,
    )


def thin_point_cloud(pc: PointCloud, ratio: float, seed: int) -> PointCloud:
    if ratio >= 1.0 or ratio <= 0.0:
        return pc
    total = pc.points.shape[0]
    target = int(total * ratio)
    if target >= total or target < 32:
        return pc
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=target, replace=False)
    return _replace_point_cloud(pc, points=pc.points[idx], intensities=pc.intensities[idx])


def softclip_intensities(pc: PointCloud, percentile: float) -> PointCloud:
    if percentile is None or percentile <= 0.0 or percentile >= 100.0:
        return pc
    clip_value = float(np.percentile(pc.intensities, percentile))
    if clip_value <= 0:
        return pc
    clipped = np.clip(pc.intensities, 0.0, clip_value)
    return _replace_point_cloud(pc, intensities=clipped)


def ensure_min_points(pc: PointCloud, minimum: int, label: str) -> None:
    if pc.points.shape[0] < minimum:
        raise RuntimeError(
            f"Point cloud for '{label}' contains only {pc.points.shape[0]} points after thinning; "
            f"increase --thin-ratio or lower the intensity threshold."
        )


def run(args: argparse.Namespace) -> List[Dict[str, object]]:
    ensure_nibabel_for_real_data(args.synthetic)

    if args.synthetic and args.input:
        raise ValueError("Specify either --synthetic or --input, not both.")
    if not args.synthetic and not args.input:
        raise ValueError("Provide --input path for MRI volume or set --synthetic for phantom data.")

    if args.synthetic:
        volume, affine = generate_synthetic_volume(seed=args.seed)
        dataset_label = args.dataset_label or "synthetic_brain_phantom"
    else:
        input_path = pathlib.Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"MRI volume not found at {input_path}")
        volume, affine = load_mri_volume(input_path)
        dataset_label = args.dataset_label or input_path.stem

    point_cloud = volume_to_point_cloud(
        volume,
        affine,
        percentile=args.mask_percentile,
        intensity_threshold=args.intensity_threshold,
        max_points=args.max_points,
        seed=args.seed,
    )

    point_cloud = softclip_intensities(point_cloud, args.softclip_percentile)
    point_cloud = thin_point_cloud(point_cloud, args.thin_ratio, seed=args.seed + 101)
    ensure_min_points(point_cloud, args.min_points, dataset_label)

    if getattr(args, "auto_m", False):
        auto_m = compute_scaled_landmarks(
            point_cloud.points.shape[0],
            base=args.auto_m_base,
            exponent=args.auto_m_exponent,
            min_landmarks=args.auto_m_min,
            max_landmarks=args.auto_m_max if args.auto_m_max > 0 else None,
        )
        print(
            f"[auto-m] total_points={point_cloud.points.shape[0]:,} -> m={auto_m} "
            f"(base={args.auto_m_base}, exponent={args.auto_m_exponent})"
        )
        args.m = auto_m

    methods = parse_methods(args.methods)
    results: List[Dict[str, object]] = []

    rips_reference = None
    if args.rips_points > 0:
        rips_reference = prepare_rips_reference(
            point_cloud,
            max_dim=args.max_dim,
            sample_size=min(args.rips_points, point_cloud.points.shape[0]),
            percentile=args.rips_percentile,
            seed=args.seed,
        )

    for method_idx, method in enumerate(methods):
        method_seed = args.seed + 17 * (method_idx + 1)
        select_start = time.perf_counter()
        indices = select_landmarks(
            method,
            point_cloud.points,
            args.m,
            seed=method_seed,
            selection_c=args.selection_c,
            hybrid_alpha=args.hybrid_alpha,
        )
        select_time = time.perf_counter() - select_start

        indices_list = indices.tolist()

        witness_start = time.perf_counter()
        witness_diagrams = compute_witness_diagrams(
            point_cloud.points,
            indices_list,
            max_dim=args.max_dim,
            k_witness=args.k_witness,
        )
        witness_time = time.perf_counter() - witness_start

        bottleneck_metrics = {f"bottleneck_b{dim}": math.nan for dim in range(args.max_dim + 1)}
        if rips_reference is not None:
            for dim in range(args.max_dim + 1):
                rip_diag = rips_reference.diagrams[dim]
                wit_diag = witness_diagrams[dim]
                if rip_diag.size == 0 and wit_diag.size == 0:
                    continue
                distance = math.nan
                if gudhi is not None:
                    try:
                        distance = bottleneck_distance(rip_diag, wit_diag, epsilon=0.0)
                    except RuntimeError:
                        distance = math.nan
                bottleneck_metrics[f"bottleneck_b{dim}"] = distance

        coverage = coverage_metrics(
            point_cloud.points,
            point_cloud.intensities,
            indices_list,
            radius=args.coverage_radius,
        )
        intens_stats = landmark_intensity_stats(point_cloud.intensities, indices_list)

        base_row: Dict[str, object] = {
            "dataset": dataset_label,
            "method": method,
            "m": float(len(indices)),
            "total_points": float(point_cloud.points.shape[0]),
            "selection_time": select_time,
            "witness_time": witness_time,
            "seed": float(method_seed),
            "coverage_radius": args.coverage_radius,
            "diameter": point_cloud.diameter,
            "rips_sample_size": float(rips_reference.sample_size) if rips_reference else math.nan,
            "rips_max_edge": float(rips_reference.max_edge) if (rips_reference and rips_reference.max_edge is not None) else math.nan,
            "rips_time": float(rips_reference.elapsed) if rips_reference else math.nan,
        }
        base_row.update(bottleneck_metrics)
        base_row.update(coverage)
        base_row.update(intens_stats)
        add_result_row(results, base_row)

        print(
            f"[{method}] m={len(indices)} selection={select_time:.2f}s witness={witness_time:.2f}s "
            f"coverage_mean={coverage['coverage_mean']:.4f} coverage_p95={coverage['coverage_p95']:.4f}"
        )

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in results for key in row.keys()})
    with open(out_path, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"[whale-deep-fast] Saved {len(results)} rows to {out_path}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Accelerated MRI deep-dive evaluation with fast presets."
    )
    parser.add_argument("--input", type=str, default=None, help="Path to MRI volume (.nii/.nii.gz). Required unless --synthetic.")
    parser.add_argument("--dataset-label", type=str, default=None, help="Optional label for outputs.")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic phantom instead of reading a file.")
    parser.add_argument("--mask-percentile", type=float, default=99.0, help="Intensity percentile for voxel masking (higher -> fewer voxels).")
    parser.add_argument("--intensity-threshold", type=float, default=None, help="Absolute intensity threshold (overrides percentile).")
    parser.add_argument("--max-points", type=int, default=120_000, help="Maximum number of voxels converted to points before thinning.")
    parser.add_argument("--thin-ratio", type=float, default=0.85, help="Additional random thinning ratio applied after masking (0-1].")
    parser.add_argument("--softclip-percentile", type=float, default=99.5, help="Clip intensities at this percentile to stabilise weights.")
    parser.add_argument("--min-points", type=int, default=40_000, help="Minimum point count after thinning; abort if fewer.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--methods", type=str, default="hybrid", help="Comma-separated landmark methods (default hybrid only).")
    parser.add_argument("--m", type=int, default=800, help="Number of landmarks per method (overridden by --auto-m).")
    parser.add_argument("--auto-m", action="store_true", help="Scale landmark count based on the number of retained points.")
    parser.add_argument("--auto-m-base", type=float, default=43.0, help="Coefficient in m = base * n^exponent when --auto-m is enabled.")
    parser.add_argument("--auto-m-exponent", type=float, default=0.26, help="Exponent in m = base * n^exponent when --auto-m is enabled.")
    parser.add_argument("--auto-m-min", type=int, default=500, help="Minimum landmarks when --auto-m is enabled.")
    parser.add_argument("--auto-m-max", type=int, default=2200, help="Maximum landmarks when --auto-m is enabled (<=0 disables the cap).")
    parser.add_argument("--selection-c", type=int, default=3, help="Hybrid density oversampling factor.")
    parser.add_argument("--hybrid-alpha", type=float, default=0.4, help="Alpha parameter for hybrid sampler.")
    parser.add_argument("--k-witness", type=int, default=4, help="Witness count per simplex (smaller for speed).")
    parser.add_argument("--max-dim", type=int, default=1, help="Maximum homology dimension (1 -> H0/H1 only).")
    parser.add_argument("--coverage-radius", type=float, default=0.028, help="Radius for coverage metrics in normalised space.")
    parser.add_argument("--rips-points", type=int, default=0, help="Optional Rips reference sample size (0 disables).")
    parser.add_argument("--rips-percentile", type=float, default=70.0, help="Percentile for Rips max edge length if enabled.")
    parser.add_argument("--out", type=str, default="artifacts/mri_deep_dive_fast.csv", help="Output CSV path.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


def cli(argv: Optional[Sequence[str]] = None) -> None:
    """Alias for :func:`main` to support console scripts."""

    main(argv)
