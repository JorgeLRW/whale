"""Utility script to download and subsample real point clouds for the baseline sweep.

This script retrieves the Stanford bunny and dragon meshes from a public mirror,
parses their ASCII PLY representation, and saves 2,000-point subsamples as
NumPy arrays. The resulting files match the expectations of
`paper_ready_tests/baseline_sweep.py` and enable real-data validation within the
existing benchmarking pipeline.
"""

from __future__ import annotations

import argparse
import io
import pathlib
import tarfile
import urllib.request
from typing import Iterable

import numpy as np


DATA_DIR = pathlib.Path(__file__).resolve().parent


PLY_SOURCES = {
    "bunny": {
        "url": "https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz",
        "format": "tar",
        "member": "bunny/reconstruction/bun_zipper.ply",
        "output": DATA_DIR / "bunny_2000.npy",
        "sample_size": 2000,
        "seed": 42,
    },
    "dragon": {
        "url": "https://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz",
        "format": "tar",
        "member": "dragon_recon/dragon_vrip.ply",
        "output": DATA_DIR / "dragon_2000.npy",
        "sample_size": 2000,
        "seed": 1337,
    },
}


class PlyFormatError(RuntimeError):
    """Raised when the downloaded PLY file cannot be interpreted as ASCII vertices."""


def download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:  # nosec: B310 - trusted source
        return response.read()


def parse_ascii_ply_vertices(data: bytes) -> np.ndarray:
    stream = io.StringIO(data.decode("utf8"))
    header_line = stream.readline().strip()
    if header_line != "ply":
        raise PlyFormatError("PLY header missing")

    format_line = stream.readline().strip()
    if not format_line.startswith("format ascii"):
        raise PlyFormatError("Only ASCII PLY files are supported")

    vertex_count: int | None = None
    property_order: list[str] = []

    while True:
        line = stream.readline()
        if not line:
            raise PlyFormatError("PLY header terminated prematurely")
        stripped = line.strip()
        if stripped.startswith("comment"):
            continue
        if stripped.startswith("element vertex"):
            parts = stripped.split()
            vertex_count = int(parts[2])
        elif stripped.startswith("property") and vertex_count is not None and len(property_order) < 3:
            property_order.append(stripped.split()[-1])
        elif stripped == "end_header":
            break

    if vertex_count is None:
        raise PlyFormatError("PLY file did not declare an element vertex count")
    if len(property_order) < 3:
        raise PlyFormatError("Expected at least three vertex properties (x, y, z)")

    vertices = np.zeros((vertex_count, 3), dtype=np.float64)
    for idx in range(vertex_count):
        line = stream.readline()
        if not line:
            raise PlyFormatError("Unexpected EOF while reading vertices")
        parts = line.strip().split()
        if len(parts) < 3:
            raise PlyFormatError(f"Vertex line {idx} missing coordinates")
        vertices[idx, 0] = float(parts[0])
        vertices[idx, 1] = float(parts[1])
        vertices[idx, 2] = float(parts[2])

    return vertices


def extract_ply_from_tar(data: bytes, member_name: str) -> bytes:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            if member.name == member_name:
                fileobj = archive.extractfile(member)
                if fileobj is None:
                    break
                return fileobj.read()
    raise PlyFormatError(f"Member '{member_name}' not found in tar archive")


def subsample_points(points: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    if points.shape[0] <= sample_size:
        return points.astype(np.float32, copy=True)
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=sample_size, replace=False)
    return points[indices].astype(np.float32, copy=True)


def ensure_dataset(key: str, *, force: bool = False) -> pathlib.Path:
    if key not in PLY_SOURCES:
        raise KeyError(f"Unknown dataset key '{key}'. Choices: {sorted(PLY_SOURCES)}")
    cfg = PLY_SOURCES[key]
    output_path: pathlib.Path = cfg["output"]
    if output_path.exists() and not force:
        return output_path

    print(f"[fetch] Downloading {key} point cloud…")
    blob = download_bytes(cfg["url"])

    fmt = cfg.get("format", "ply")
    if fmt == "tar":
        member = cfg.get("member")
        if not member:
            raise PlyFormatError(f"Tar source for {key} missing 'member' entry")
        ply_bytes = extract_ply_from_tar(blob, member)
    elif fmt == "ply":
        ply_bytes = blob
    else:
        raise PlyFormatError(f"Unsupported source format '{fmt}' for {key}")

    points = parse_ascii_ply_vertices(ply_bytes)
    print(f"[fetch] Loaded {points.shape[0]} vertices from {key} mesh")

    sampled = subsample_points(points, cfg["sample_size"], cfg["seed"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, sampled)
    print(f"[fetch] Saved {sampled.shape[0]} points to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and subsample real point clouds for the sweep.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="bunny,dragon",
        help="Comma-separated dataset keys to download (choices: bunny, dragon)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite existing .npy files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested: Iterable[str] = [name.strip() for name in args.datasets.split(",") if name.strip()]
    if not requested:
        raise SystemExit("No datasets specified")

    for key in requested:
        path = ensure_dataset(key, force=args.force)
        print(f"[fetch] Ready: {path}")


if __name__ == "__main__":
    main()
