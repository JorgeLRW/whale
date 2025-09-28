"""Cross-domain benchmark harness for the witness pipeline on generic point clouds.

This module mirrors the high-level structure of ``mri_deep_dive_fast`` but removes
MRI-specific assumptions so we can stress-test the sampler on synthetic and
real-world datasets drawn from outside of medical imaging.

Example usage
-------------

Run the default Swiss roll benchmark and write results to ``artifacts``::

    python -m paper_ready.pointcloud_benchmark --builtin swiss_roll \
        --m 256 --max-points 4000 --out artifacts/swiss_roll_fast.csv

Load a custom CSV (rows = points) and disable the Rips reference::

    python -m paper_ready.pointcloud_benchmark --input data/airfoil.csv \
        --format csv --methods hybrid,density --rips-points 0

The CLI exposes a handful of built-in generators (Swiss roll, torus, sphere,
S-curve, two moons, and Gaussian mixtures) so we can produce reproducible point
clouds without external assets. When Gudhi is available, the script will also
compute Vietoris–Rips references for bottleneck distance comparisons.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn import datasets

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


@dataclass
class LoadedPointCloud:
    point_cloud: PointCloud
    label: str


def _ensure_positive(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.asarray(arr, dtype=np.float64)
    out -= out.min()
    out += 1e-9
    return out


def _normalize_points(points: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    data = np.asarray(points, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("Point cloud must be 2-D (samples x features).")
    bounds_min = data.min(axis=0)
    bounds_max = data.max(axis=0)
    span = bounds_max - bounds_min
    span[span == 0.0] = 1.0
    normalized = (data - bounds_min) / span
    return normalized, bounds_min, bounds_max


def _subset_points(
    points: NDArray[np.float64],
    intensities: NDArray[np.float64],
    *,
    max_points: int,
    seed: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    total = points.shape[0]
    if total <= max_points:
        return points, intensities
    rng = np.random.default_rng(seed)
    weights = intensities.copy()
    weights -= weights.min()
    weights = np.clip(weights, 0.0, None)
    if weights.sum() <= 0.0:
        idx = rng.choice(total, size=max_points, replace=False)
    else:
        probs = weights / weights.sum()
        idx = rng.choice(total, size=max_points, replace=False, p=probs)
    return points[idx], intensities[idx]


def _make_point_cloud(points: NDArray[np.float64], intensities: Optional[NDArray[np.float64]], *, max_points: int, seed: int) -> PointCloud:
    normalized, bounds_min, bounds_max = _normalize_points(points)
    if intensities is None:
        intensities_arr = np.ones(normalized.shape[0], dtype=np.float64)
    else:
        intensities_arr = np.asarray(intensities, dtype=np.float64)
    intensities_arr = _ensure_positive(intensities_arr)
    normalized, intensities_arr = _subset_points(normalized, intensities_arr, max_points=max_points, seed=seed)
    return PointCloud(points=normalized, intensities=intensities_arr, bounds_min=bounds_min, bounds_max=bounds_max)


def _load_from_file(path: pathlib.Path, *, fmt: str) -> NDArray[np.float64]:
    if fmt == "npy":
        return np.load(path)
    if fmt == "csv":
        return np.loadtxt(path, delimiter=",", dtype=np.float64)
    raise ValueError(f"Unsupported input format '{fmt}'.")


def _infer_format(path: pathlib.Path) -> str:
    ext = path.suffix.lower()
    if ext == ".npy":
        return "npy"
    if ext in {".csv", ".txt"}:
        return "csv"
    raise ValueError("Could not infer file format. Use --format {npy,csv} explicitly.")


def _make_torus(n: int, seed: int, *, major: float = 1.0, minor: float = 0.35) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * math.pi, size=(n, 2))
    theta = angles[:, 0]
    phi = angles[:, 1]
    x = (major + minor * np.cos(phi)) * np.cos(theta)
    y = (major + minor * np.cos(phi)) * np.sin(theta)
    z = minor * np.sin(phi)
    return np.stack([x, y, z], axis=1)


def _make_sphere(n: int, seed: int, *, radius: float = 1.0) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * math.pi, size=n)
    theta = np.arccos(u)
    r = np.full(n, radius)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=1)


def _make_cylinder(n: int, seed: int, *, radius: float = 1.0, height: float = 2.0) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
    r = np.sqrt(rng.uniform(0.0, radius ** 2, size=n))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = rng.uniform(-height / 2.0, height / 2.0, size=n)
    return np.stack([x, y, z], axis=1)


def _builtin_dataset(name: str, n_samples: int, seed: int) -> Tuple[NDArray[np.float64], NDArray[np.float64], str]:
    key = name.lower()
    if key == "swiss_roll":
        data, color = datasets.make_swiss_roll(n_samples=n_samples, noise=0.05, random_state=seed)
        return data, _ensure_positive(color), "swiss_roll"
    if key == "s_curve":
        data, color = datasets.make_s_curve(n_samples=n_samples, noise=0.03, random_state=seed)
        return data, _ensure_positive(color), "s_curve"
    if key == "moons":
        data, labels = datasets.make_moons(n_samples=n_samples, noise=0.06, random_state=seed)
        return np.column_stack([data, np.zeros(n_samples)]), _ensure_positive(labels.astype(float)), "two_moons"
    if key == "blobs":
        blob_result = datasets.make_blobs(
            n_samples=n_samples,
            centers=4,
            cluster_std=0.5,
            random_state=seed,
            return_centers=False,
        )
        data, labels = blob_result  # type: ignore[misc]
        return data, _ensure_positive(labels.astype(float)), "gaussian_blobs"
    if key == "torus":
        data = _make_torus(n_samples, seed)
        return data, np.ones(n_samples, dtype=np.float64), "torus"
    if key == "sphere":
        data = _make_sphere(n_samples, seed)
        return data, np.ones(n_samples, dtype=np.float64), "sphere"
    if key == "cylinder":
        data = _make_cylinder(n_samples, seed)
        return data, np.ones(n_samples, dtype=np.float64), "cylinder"
    raise ValueError(
        f"Unknown built-in dataset '{name}'. Available options: swiss_roll, s_curve, moons, blobs, torus, sphere, cylinder."
    )


def load_point_cloud(
    *,
    builtin: Optional[str],
    input_path: Optional[pathlib.Path],
    fmt: str,
    n_samples: int,
    max_points: int,
    seed: int,
) -> LoadedPointCloud:
    builtin_name = builtin
    if builtin_name is None and input_path is None:
        builtin_name = "swiss_roll"

    if builtin_name is not None and input_path is not None:
        raise ValueError("Pass either --builtin or --input, not both.")

    if input_path is not None:
        points = _load_from_file(input_path, fmt=fmt)
        label = input_path.stem
        intensities = np.ones(points.shape[0], dtype=np.float64)
    else:
        if builtin_name is None:
            raise ValueError("Internal error: builtin_name resolved to None.")
        points, intensities, label = _builtin_dataset(builtin_name, n_samples, seed)

    pc = _make_point_cloud(points, intensities, max_points=max_points, seed=seed + 17)
    return LoadedPointCloud(point_cloud=pc, label=label)


def _auto_radius(pc: PointCloud) -> float:
    diag = float(np.linalg.norm(pc.bounds_max - pc.bounds_min))
    if not math.isfinite(diag) or diag == 0.0:
        return 0.08
    return 0.08 * diag


def run(args: argparse.Namespace) -> List[Dict[str, object]]:
    if args.format == "auto" and args.input is not None:
        args.format = _infer_format(pathlib.Path(args.input))

    loaded = load_point_cloud(
        builtin=args.builtin,
        input_path=pathlib.Path(args.input) if args.input else None,
        fmt=args.format,
        n_samples=args.samples,
        max_points=args.max_points,
        seed=args.seed,
    )
    point_cloud = loaded.point_cloud
    dataset_label = args.dataset_label or loaded.label

    coverage_radius = args.coverage_radius
    if coverage_radius is None:
        coverage_radius = _auto_radius(point_cloud)

    methods = parse_methods(args.methods)
    results: List[Dict[str, object]] = []

    rips_reference = None
    if args.rips_points > 0:
        rips_reference = prepare_rips_reference(
            point_cloud,
            max_dim=args.max_dim,
            sample_size=min(args.rips_points, point_cloud.points.shape[0]),
            percentile=args.rips_percentile,
            seed=args.seed + 31,
        )

    for method_idx, method in enumerate(methods):
        method_seed = args.seed + 101 * (method_idx + 1)
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

        witness_start = time.perf_counter()
        witness_diagrams = compute_witness_diagrams(
            point_cloud.points,
            indices.tolist(),
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
            indices.tolist(),
            radius=coverage_radius,
        )
        intensity_stats = landmark_intensity_stats(point_cloud.intensities, indices.tolist())

        row: Dict[str, object] = {
            "dataset": dataset_label,
            "method": method,
            "m": float(len(indices)),
            "total_points": float(point_cloud.points.shape[0]),
            "selection_time": select_time,
            "witness_time": witness_time,
            "seed": float(method_seed),
            "coverage_radius": coverage_radius,
            "diameter": point_cloud.diameter,
            "rips_sample_size": float(rips_reference.sample_size) if rips_reference else math.nan,
            "rips_max_edge": float(rips_reference.max_edge) if (rips_reference and rips_reference.max_edge is not None) else math.nan,
            "rips_time": float(rips_reference.elapsed) if rips_reference else math.nan,
        }
        row.update(bottleneck_metrics)
        row.update(coverage)
        row.update(intensity_stats)
        results.append(row)

        print(
            f"[{dataset_label}:{method}] m={len(indices)} selection={select_time:.2f}s witness={witness_time:.2f}s "
            f"coverage_mean={coverage['coverage_mean']:.4f} coverage_p95={coverage['coverage_p95']:.4f}"
        )

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in results for key in row.keys()})
    with open(out_path, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"[pointcloud_benchmark] Saved {len(results)} rows to {out_path}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the witness pipeline on generic point clouds.")
    parser.add_argument("--builtin", type=str, default=None, help="Name of a built-in dataset (swiss_roll, s_curve, moons, blobs, torus, sphere, cylinder).")
    parser.add_argument("--input", type=str, default=None, help="Optional path to a custom point cloud (.npy or .csv).")
    parser.add_argument("--format", choices=["auto", "npy", "csv"], default="auto", help="Input format when --input is supplied (default: infer).")
    parser.add_argument("--samples", type=int, default=6000, help="Number of samples to generate for built-in datasets (default: 6000).")
    parser.add_argument("--max-points", type=int, default=6000, help="Maximum number of points to keep after sampling (default: 6000).")
    parser.add_argument("--m", type=int, default=512, help="Number of landmarks per method (default: 512).")
    parser.add_argument("--methods", type=str, default="hybrid", help="Comma-separated landmark methods (e.g., hybrid,density,random).")
    parser.add_argument("--selection-c", type=int, default=3, help="Oversampling factor for density-based selection (default: 3).")
    parser.add_argument("--hybrid-alpha", type=float, default=0.4, help="Alpha parameter for the hybrid sampler (default: 0.4).")
    parser.add_argument("--k-witness", type=int, default=4, help="Witness count per simplex (default: 4).")
    parser.add_argument("--max-dim", type=int, default=1, help="Maximum homology dimension (default: 1).")
    parser.add_argument("--rips-points", type=int, default=400, help="Sample size for optional Rips reference (0 disables).")
    parser.add_argument("--rips-percentile", type=float, default=80.0, help="Percentile for Rips max edge length (default: 80).")
    parser.add_argument("--coverage-radius", type=float, default=None, help="Coverage radius in normalised space (default: auto).")
    parser.add_argument("--dataset-label", type=str, default=None, help="Override label used in the CSV output.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed (default: 0).")
    parser.add_argument("--out", type=str, default="artifacts/pointcloud_benchmark.csv", help="Output CSV path (default: artifacts/pointcloud_benchmark.csv).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
