"""MRI deep-dive benchmarking built on top of the Whale pipeline."""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import time
from typing import Dict, List, Optional, Sequence

from whale.io import (
    ensure_nibabel_for_real_data,
    generate_synthetic_volume,
    load_mri_volume,
    volume_to_point_cloud,
)
from whale.pipeline import (
    RipsReference,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MRI deep-dive evaluation for landmark-based persistence."
    )
    parser.add_argument("--input", type=str, default=None, help="Path to MRI volume (.nii or .nii.gz). Required unless --synthetic is set.")
    parser.add_argument("--dataset-label", type=str, default=None, help="Optional label for the dataset in outputs.")
    parser.add_argument("--synthetic", action="store_true", help="Generate a synthetic MRI-like phantom instead of loading a file.")
    parser.add_argument("--mask-percentile", type=float, default=97.5, help="Percentile threshold for voxel intensities.")
    parser.add_argument("--intensity-threshold", type=float, default=None, help="Absolute intensity threshold (overrides percentile).")
    parser.add_argument("--max-points", type=int, default=80_000, help="Maximum number of points sampled from the volume.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--methods", type=str, default="random,hybrid", help="Comma-separated landmark methods to evaluate.")
    parser.add_argument("--m", type=int, default=600, help="Number of landmarks per method.")
    parser.add_argument("--selection-c", type=int, default=4, help="Density sampler parameter (if used).")
    parser.add_argument("--hybrid-alpha", type=float, default=0.5, help="Alpha parameter for the hybrid sampler.")
    parser.add_argument("--k-witness", type=int, default=8, help="Number of witnesses per simplex for witness complex.")
    parser.add_argument("--max-dim", type=int, default=2, help="Maximum homology dimension to compute.")
    parser.add_argument("--coverage-radius", type=float, default=0.03, help="Radius (in normalised units) for coverage ratios.")
    parser.add_argument("--rips-points", type=int, default=6000, help="Number of points for Vietoris--Rips reference (0 disables).")
    parser.add_argument("--rips-percentile", type=float, default=85.0, help="Percentile for max edge length in Rips reference.")
    parser.add_argument("--out", type=str, default="artifacts/mri_deep_dive.csv", help="CSV file for detailed results.")
    return parser


def add_result_row(rows: List[Dict[str, object]], base: Dict[str, object]) -> None:
    rows.append(base)


def _prepare_volume(args: argparse.Namespace) -> tuple[PointCloud, str]:
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
    return point_cloud, dataset_label


def run(args: argparse.Namespace) -> List[Dict[str, object]]:
    point_cloud, dataset_label = _prepare_volume(args)

    methods = parse_methods(args.methods)
    results: List[Dict[str, object]] = []

    rips_reference: Optional[RipsReference] = None
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
        indices_list = indices.tolist()
        select_time = time.perf_counter() - select_start

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
                    bottleneck_metrics[f"bottleneck_b{dim}"] = math.nan
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
            "m": float(len(indices_list)),
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

    print(f"[whale-deep] Saved {len(results)} rows to {out_path}")
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


def cli(argv: Optional[Sequence[str]] = None) -> None:
    """Alias for :func:`main` to support console scripts."""

    main(argv)