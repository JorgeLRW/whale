"""Convenience driver for running the generic point cloud benchmark suite.

This script executes a small battery of non-medical datasets (Swiss roll, torus,
Gaussian blobs) to showcase that the witness pipeline generalises beyond MRI
volumes. Each case reuses the main CLI from ``paper_ready.pointcloud_benchmark``
and stores the resulting CSVs under ``examples/sample_outputs``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from paper_ready.pointcloud_benchmark import run


def _default_cases() -> List[Dict[str, object]]:
    base = {
        "methods": "hybrid,density,random",
        "samples": 5000,
        "max_points": 5000,
        "m": 400,
        "rips_points": 300,
        "max_dim": 1,
        "selection_c": 3,
        "hybrid_alpha": 0.4,
        "k_witness": 4,
        "seed": 42,
    }
    cases: List[Dict[str, object]] = []
    cases.append({**base, "builtin": "swiss_roll", "dataset_label": "swiss_roll", "output": "pointcloud_swiss_roll.csv"})
    cases.append({**base, "builtin": "torus", "dataset_label": "torus_surface", "output": "pointcloud_torus.csv"})
    cases.append({**base, "builtin": "blobs", "dataset_label": "gaussian_blobs", "output": "pointcloud_gaussian_blobs.csv"})
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the generic point cloud benchmark suite.")
    parser.add_argument("--cases", type=Path, default=None, help="Optional path to a JSON file describing custom benchmark cases.")
    parser.add_argument("--output-dir", type=Path, default=Path("examples/sample_outputs"), help="Directory to write per-case CSV outputs.")
    args = parser.parse_args()

    if args.cases is not None:
        with open(args.cases, "r", encoding="utf8") as fh:
            case_overrides: List[Dict[str, object]] = json.load(fh)
    else:
        case_overrides = _default_cases()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows: List[Dict[str, object]] = []
    for idx, case in enumerate(case_overrides, start=1):
        namespace = argparse.Namespace(
            builtin=case.get("builtin"),
            input=case.get("input"),
            format=case.get("format", "auto"),
            samples=case.get("samples", 5000),
            max_points=case.get("max_points", 5000),
            m=case.get("m", 400),
            methods=case.get("methods", "hybrid"),
            selection_c=case.get("selection_c", 3),
            hybrid_alpha=case.get("hybrid_alpha", 0.4),
            k_witness=case.get("k_witness", 4),
            max_dim=case.get("max_dim", 1),
            rips_points=case.get("rips_points", 0),
            rips_percentile=case.get("rips_percentile", 80.0),
            coverage_radius=case.get("coverage_radius"),
            dataset_label=case.get("dataset_label"),
            seed=case.get("seed", 0),
            out=str(output_dir / case.get("output", f"pointcloud_case_{idx}.csv")),
        )
        print(f"[suite] Running case #{idx}: builtin={namespace.builtin} -> {namespace.out}")
        rows = run(namespace)
        aggregate_rows.extend(rows)

    if aggregate_rows:
        summary_path = output_dir / "pointcloud_suite_summary.csv"
        fieldnames = sorted({key for row in aggregate_rows for key in row.keys()})
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", newline="", encoding="utf8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregate_rows)
        print(f"[suite] Wrote aggregate summary to {summary_path}")


if __name__ == "__main__":
    main()
