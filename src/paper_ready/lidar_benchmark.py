"""LiDAR benchmark harness that runs the existing pointcloud_benchmark on KITTI frames.

This module prepares a pruned point cloud from a KITTI velodyne .bin (using
`paper_ready.kitti_utils`), writes a temporary CSV or NPY, and calls
`paper_ready.pointcloud_benchmark.run` programmatically to gather timings for
specified methods and TDA modes.
"""
from __future__ import annotations

import argparse
import pathlib
import tempfile
import numpy as np
from typing import List, Optional

from paper_ready.kitti_utils import load_velodyne_bin, prune_points_random
from paper_ready.pointcloud_benchmark import build_parser, run


def prepare_pointcloud_file(seq: str, frame: str, *, max_points: int, out_path: pathlib.Path, seed: int = 0) -> str:
    # project root is two levels above this file (src/paper_ready/<this>)
    repo = pathlib.Path(__file__).resolve().parents[2]
    vel_path = repo / 'data' / 'lidar' / 'sequences' / seq / 'velodyne' / f'{frame}.bin'
    pts = load_velodyne_bin(str(vel_path))
    pts2 = prune_points_random(pts, max_points, seed=seed)
    # write as CSV (x,y,z)
    np.savetxt(str(out_path), pts2, delimiter=',', fmt='%.6f')
    return str(out_path)


def run_benchmark_on_frame(seq: str, frame: str, *, modes: List[str], m: int = 1700, max_points: int = 200000, rips_points: int = 0, out_dir: Optional[pathlib.Path] = None) -> List[pathlib.Path]:
    # project root (paper_ready) is two levels above this file
    repo = pathlib.Path(__file__).resolve().parents[2]
    if out_dir is None:
        out_dir = repo / 'artifacts'
    out_dir.mkdir(parents=True, exist_ok=True)

    results_files: List[pathlib.Path] = []
    with tempfile.TemporaryDirectory() as td:
        tmp_in = pathlib.Path(td) / 'input.csv'
        prepare_pointcloud_file(seq, frame, max_points=max_points, out_path=tmp_in)
        for mode in modes:
            out_file = (out_dir / f'kitti_seq{seq}_{frame}_m{m}_mode{mode}.csv').resolve()
            parser = build_parser()
            # build argparse.Namespace equivalent
            args = parser.parse_args([
                '--input', str(tmp_in),
                '--format', 'csv',
                '--max-points', str(max_points),
                '--m', str(m),
                '--methods', 'hybrid',
                '--rips-points', str(rips_points),
                '--max-dim', '2' if mode == 'regular' else '1',
                '--out', str(out_file),
            ])
            # run with tda_mode set via environment: pointcloud_benchmark doesn't accept tda_mode directly
            # but pipeline defaults may be read from whale.ai; for now we call run(args)
            run(args)
            results_files.append(out_file)
    return results_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', default='00')
    parser.add_argument('--frame', default='000000')
    parser.add_argument('--m', type=int, default=1700)
    parser.add_argument('--max-points', type=int, default=200000)
    parser.add_argument('--rips-points', type=int, default=0)
    parser.add_argument('--modes', type=str, default='fast,regular')
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    out = run_benchmark_on_frame(args.seq, args.frame, modes=modes, m=args.m, max_points=args.max_points, rips_points=args.rips_points)
    print('Generated:', out)


if __name__ == '__main__':
    main()
