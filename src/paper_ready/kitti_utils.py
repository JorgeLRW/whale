"""Small KITTI helpers: load velodyne .bin, prune, save PLY.
This module avoids heavy deps so it can run in minimal envs.
"""
from pathlib import Path
import numpy as np


def load_velodyne_bin(path: str) -> np.ndarray:
    """Load KITTI velodyne .bin and return Nx3 float64 array (x,y,z).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Velodyne file not found: {p}")
    arr = np.fromfile(str(p), dtype=np.float32)
    if arr.size % 4 != 0:
        raise ValueError(f"Unexpected velodyne file size: {arr.size} floats")
    pts = arr.reshape(-1, 4)[:, :3].astype(np.float64)
    return pts


def prune_points_random(pts: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    """Randomly downsample to at most max_points.
    If pts <= max_points, returns pts unchanged.
    """
    n = pts.shape[0]
    if n <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return pts[idx]


def write_ply_ascii(path: str, pts: np.ndarray) -> None:
    """Write Nx3 points to an ASCII PLY file (no colors).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = pts.shape[0]
    header = [
        'ply',
        'format ascii 1.0',
        f'element vertex {n}',
        'property float x',
        'property float y',
        'property float z',
        'end_header'
    ]
    with p.open('w', encoding='utf8') as f:
        f.write('\n'.join(header) + '\n')
        for x, y, z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
