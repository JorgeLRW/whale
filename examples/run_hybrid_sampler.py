"""Command-line driver for the hybrid MaxMin/KDE sampler.

Usage examples
--------------
Generate a synthetic point cloud and select landmarks (no input file):
    python run_hybrid_sampler.py --landmarks 48 --alpha 0.6 --bandwidth 0.4 --seed 13

Use a saved NumPy array as input and export indices:
    python run_hybrid_sampler.py --input data/cloud.npy --landmarks 120 --alpha 0.4 --save results/indices.npy

CSV input works as well (comma-delimited, rows are points):
    python run_hybrid_sampler.py --input data/cloud.csv --format csv --landmarks 80
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
from typing import Optional

import numpy as np

from paper_ready.sampling.hybrid_sampler import hybrid_maxmin_kde


def _load_point_cloud(path: Path, kind: str) -> np.ndarray:
    if kind == "npy":
        return np.load(path)
    if kind == "csv":
        return np.loadtxt(path, delimiter=",")
    raise ValueError(f"Unsupported input format '{kind}'.")


def _infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return "npy"
    if suffix in {".csv", ".txt"}:
        return "csv"
    raise ValueError(
        "Could not infer input format; please pass --format {npy,csv}."
    )


def _make_synthetic_cloud(n: int, seed: Optional[int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    clusters = []
    centers = [(-2.0, 0.5), (0.0, 2.5), (2.0, -1.0)]
    scales = [0.35, 0.55, 0.25]
    per_cluster = max(1, n // len(centers))
    for (cx, cy), scale in zip(centers, scales):
        pts = rng.normal(loc=(cx, cy), scale=scale, size=(per_cluster, 2))
        clusters.append(pts)
    cloud = np.vstack(clusters)
    if cloud.shape[0] > n:
        cloud = cloud[:n]
    return cloud


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hybrid MaxMin/KDE sampler on a point cloud.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional path to a point cloud stored as .npy or .csv (rows = points).",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "npy", "csv"],
        default="auto",
        help="Input format hint when --input is supplied (default: infer from extension).",
    )
    parser.add_argument(
        "--landmarks",
        "-m",
        type=int,
        default=64,
        help="Number of landmarks to select (default: 64).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Trade-off between density (1.0) and max-min coverage (0.0); default 0.5.",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        help="Optional KDE bandwidth; leave unset to use k-NN density proxy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional output path (.npy) to save the selected landmark indices.",
    )
    parser.add_argument(
        "--synthetic-size",
        type=int,
        default=600,
        help="Number of synthetic points to generate when no input file is given (default: 600).",
    )

    args = parser.parse_args()

    if args.input is not None:
        fmt = args.format
        if fmt == "auto":
            fmt = _infer_format(args.input)
        X = _load_point_cloud(args.input, fmt)
    else:
        X = _make_synthetic_cloud(args.synthetic_size, args.seed)
        print(
            f"[info] No --input provided; generated synthetic cloud with shape {X.shape}."
        )

    m = args.landmarks
    if m <= 0:
        raise ValueError("--landmarks must be a positive integer.")
    if m > len(X):
        raise ValueError(
            f"Requested {m} landmarks but point cloud only has {len(X)} points."
        )

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must lie in the interval [0, 1].")

    indices = hybrid_maxmin_kde(
        X,
        m=m,
        alpha=args.alpha,
        bandwidth=args.bandwidth,
        seed=args.seed,
    )

    print("Selected landmark indices (first 20 shown):", indices[:20])
    print("Total landmarks:", len(indices))

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save, np.asarray(indices, dtype=int))
        print(f"Saved landmark indices to {args.save.resolve()}")


if __name__ == "__main__":
    main()
