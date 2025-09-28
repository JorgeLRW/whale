"""Batch runner for the fast MRI deep-dive pipeline.

This helper lets collaborators execute `paper_ready.mri_deep_dive_fast` over a
folder of MRI volumes (or synthetic phantoms) without remembering the long list
of flags. Parameter defaults mirror the values used in internal experiments but
can be overridden from the command line.

Example (PowerShell):

```
python examples/run_mri_deep_dive_fast_batch.py \
  --data-root C:/data/IXI \
  --limit 3 \
  --output-dir artifacts/ixi_fast_runs \
  --methods hybrid \
  --m 900 \
  --mask-percentile 98.5 \
  --thin-ratio 0.9 \
  --softclip-percentile 99.8 \
  --selection-c 3 \
  --k-witness 5 \
  --max-points 130000 \
  --coverage-radius 0.03
```

For a quick dry run you can use the `--synthetic` flag, which generates a single
phantom volume instead of scanning a directory.
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
from datetime import UTC, datetime
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from paper_ready.mri_deep_dive_fast import build_parser, run  # noqa: E402


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch runner for paper_ready.mri_deep_dive_fast."
    )
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        default=PROJECT_ROOT / "data" / "IXI",
        help="Directory containing MRI volumes (.nii/.nii.gz). Ignored if --synthetic is set.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.nii*",
        help="Glob pattern under the data root (default: *.nii*).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Maximum number of volumes to process (default: 1, use 0 for all).",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=PROJECT_ROOT / "artifacts",
        help="Directory where per-volume CSVs are written (default: artifacts/).",
    )
    parser.add_argument(
        "--aggregate",
        type=pathlib.Path,
        default=PROJECT_ROOT / "artifacts" / "mri_deep_dive_fast_batch.csv",
        help="Path for the combined CSV across all runs.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Process a synthetic phantom instead of scanning --data-root.",
    )
    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=1,
        help="Number of synthetic runs to generate when --synthetic is set (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed shared across runs (default: 42).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="hybrid",
        help="Comma separated list of landmark methods (default: hybrid).",
    )
    parser.add_argument("--m", type=int, default=900, help="Number of landmarks per method (default: 900).")
    parser.add_argument(
        "--selection-c",
        type=int,
        default=3,
        help="Hybrid density oversampling factor (default: 3).",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.4,
        help="Hybrid sampler alpha parameter (default: 0.4).",
    )
    parser.add_argument(
        "--mask-percentile",
        type=float,
        default=98.5,
        help="Intensity percentile for voxel masking (default: 98.5).",
    )
    parser.add_argument(
        "--intensity-threshold",
        type=float,
        default=None,
        help="Optional absolute intensity threshold overriding the percentile.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=130_000,
        help="Maximum number of voxels converted to points before thinning (default: 130000).",
    )
    parser.add_argument(
        "--thin-ratio",
        type=float,
        default=0.9,
        help="Random thinning ratio applied after masking (default: 0.9).",
    )
    parser.add_argument(
        "--softclip-percentile",
        type=float,
        default=99.8,
        help="Percentile for intensity soft clipping (default: 99.8).",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=40_000,
        help="Minimum point count required after thinning (default: 40000).",
    )
    parser.add_argument(
        "--k-witness",
        type=int,
        default=5,
        help="Witness count per simplex (default: 5).",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1,
        help="Maximum homology dimension (default: 1).",
    )
    parser.add_argument(
        "--coverage-radius",
        type=float,
        default=0.03,
        help="Radius for coverage metrics (default: 0.03).",
    )
    parser.add_argument(
        "--rips-points",
        type=int,
        default=0,
        help="Rips reference sample size (default: 0 = disabled).",
    )
    parser.add_argument(
        "--rips-percentile",
        type=float,
        default=70.0,
        help="Percentile for Rips max edge when enabled (default: 70.0).",
    )
    parser.add_argument(
        "--out-suffix",
        type=str,
        default="fast.csv",
        help="Filename suffix appended to each dataset label (default: fast.csv).",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print key metrics for each run to stdout.",
    )
    return parser.parse_args(argv)


def discover_volumes(data_root: pathlib.Path, pattern: str, limit: int | None) -> List[pathlib.Path]:
    files = sorted(data_root.glob(pattern))
    if limit is not None:
        files = files[:limit]
    return files


def ensure_output_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def aggregate_rows(rows: Iterable[Dict[str, object]], dest: pathlib.Path) -> None:
    rows = list(rows)
    if not rows:
        return
    headers = sorted({key for row in rows for key in row.keys()})
    with dest.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def build_namespace(base_args: argparse.Namespace, *, input_path: pathlib.Path | None, dataset_label: str) -> argparse.Namespace:
    parser = build_parser()
    defaults = vars(parser.parse_args([]))
    overrides = dict(defaults)
    overrides.update(
        input=str(input_path) if input_path is not None else None,
        dataset_label=dataset_label,
        synthetic=input_path is None,
        mask_percentile=base_args.mask_percentile,
        intensity_threshold=base_args.intensity_threshold,
        max_points=base_args.max_points,
        thin_ratio=base_args.thin_ratio,
        softclip_percentile=base_args.softclip_percentile,
        min_points=base_args.min_points,
        seed=base_args.seed,
        methods=base_args.methods,
        m=base_args.m,
        selection_c=base_args.selection_c,
        hybrid_alpha=base_args.hybrid_alpha,
        k_witness=base_args.k_witness,
        max_dim=base_args.max_dim,
        coverage_radius=base_args.coverage_radius,
        rips_points=base_args.rips_points,
        rips_percentile=base_args.rips_percentile,
    )
    overrides["out"] = str(
        base_args.output_dir / f"{dataset_label}_{base_args.out_suffix}"
    )
    return argparse.Namespace(**overrides)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    ensure_output_dir(args.output_dir)
    ensure_output_dir(args.aggregate.parent)

    results: List[Dict[str, Any]] = []

    if args.synthetic:
        run_count = max(args.synthetic_count, 1)
        for idx in range(run_count):
            timestamp = datetime.now(UTC).strftime("%Y%m%d")
            label = f"synthetic_{timestamp}_{args.seed + idx:04d}"
            namespace = build_namespace(args, input_path=None, dataset_label=label)
            namespace.seed = args.seed + idx
            run(namespace)
            out_path = pathlib.Path(namespace.out)
            with out_path.open("r", encoding="utf8", newline="") as f:
                reader = csv.DictReader(f)
                for raw_row in reader:
                    typed_row: Dict[str, Any] = dict(raw_row)
                    typed_row["dataset"] = typed_row.get("dataset", label)
                    typed_row["batch_label"] = label
                    results.append(typed_row)
    else:
        limit = None if args.limit == 0 else args.limit
        volumes = discover_volumes(args.data_root, args.pattern, limit)
        if not volumes:
            raise FileNotFoundError(f"No MRI volumes matched pattern '{args.pattern}' in {args.data_root}")

        for offset, volume in enumerate(volumes):
            label = volume.stem
            namespace = build_namespace(args, input_path=volume, dataset_label=label)
            namespace.seed = args.seed + offset
            run(namespace)
            out_path = pathlib.Path(namespace.out)
            with out_path.open("r", encoding="utf8", newline="") as f:
                reader = csv.DictReader(f)
                for raw_row in reader:
                    typed_row: Dict[str, Any] = dict(raw_row)
                    typed_row["dataset"] = typed_row.get("dataset", label)
                    typed_row["batch_label"] = label
                    results.append(typed_row)
                    if args.print_summary:
                        coverage_mean = typed_row.get("coverage_mean")
                        witness_time = typed_row.get("witness_time")
                        print(
                            f"[{label}] coverage_mean={coverage_mean} witness_time={witness_time}s -> {out_path}"
                        )

    aggregate_rows(results, args.aggregate)
    print(f"Saved aggregate CSV with {len(results)} rows to {args.aggregate}")


if __name__ == "__main__":
    main()
