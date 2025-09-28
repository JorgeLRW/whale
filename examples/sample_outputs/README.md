# Sample MRI Deep-Dive Run Recipes

This folder now contains small CSV snapshots produced directly by running the
main pipelines. Each section lists the exact command we executed, with the
resulting CSV saved beside this README for easy inspection.

All commands assume you run them from the repository root with
`$env:PYTHONPATH` pointing at `src`. No Vietoris–Rips references are enabled in
these examples (`--rips-points 0` in the standard pipeline), keeping runtime
lightweight.

See `run_recipes.txt` for a plaintext copy of the commands.

## Standard BrainWeb configuration

- **Command:**

  ```powershell
  python -m paper_ready.mri_deep_dive `
    --input data/t1_icbm_normal_1mm_pn3_rf20.nii.gz `
    --dataset-label brainweb_t1_standard `
    --methods hybrid `
    --m 600 `
    --mask-percentile 97.5 `
    --selection-c 4 `
    --k-witness 8 `
    --max-points 90000 `
    --coverage-radius 0.03 `
    --rips-points 0 `
    --out examples/sample_outputs/brainweb_t1_standard.csv
  ```

- **Output:** `brainweb_t1_standard.csv`

## Standard IXI configuration

- **Command:**

  ```powershell
  python -m paper_ready.mri_deep_dive `
    --input data/IXI/IXI050-Guys-0711-T1.nii.gz `
    --dataset-label ixi_t1_full `
    --methods random,hybrid `
    --m 650 `
    --mask-percentile 98.0 `
    --selection-c 5 `
    --k-witness 8 `
    --max-points 110000 `
    --coverage-radius 0.028 `
    --rips-points 0 `
    --out examples/sample_outputs/ixi_t1_full.csv
  ```

- **Output:** `ixi_t1_full.csv`

## Fast IXI configuration

- **Command:**

  ```powershell
  python -m paper_ready.mri_deep_dive_fast `
    --input data/IXI/IXI050-Guys-0711-T1.nii.gz `
    --dataset-label ixi_t1_fast_opt `
    --methods hybrid `
    --m 900 `
    --mask-percentile 98.5 `
    --thin-ratio 0.9 `
    --softclip-percentile 99.8 `
    --selection-c 3 `
    --k-witness 5 `
    --max-points 130000 `
    --coverage-radius 0.03 `
    --out examples/sample_outputs/ixi_t1_fast_opt.csv
  ```

- **Output:** `ixi_t1_fast_opt.csv`

## Fast BrainWeb configuration

- **Command:**

  ```powershell
  python -m paper_ready.mri_deep_dive_fast `
    --input data/t1_icbm_normal_1mm_pn3_rf20.nii.gz `
    --dataset-label brainweb_t1_fast `
    --methods hybrid `
    --m 500 `
    --mask-percentile 98.0 `
    --thin-ratio 0.85 `
    --softclip-percentile 99.6 `
    --selection-c 3 `
    --k-witness 5 `
    --max-points 95000 `
    --coverage-radius 0.029 `
    --out examples/sample_outputs/brainweb_t1_fast.csv
  ```

- **Output:** `brainweb_t1_fast.csv`

## Fast synthetic phantom

- **Command:**

  ```powershell
  python -m paper_ready.mri_deep_dive_fast `
    --synthetic `
    --dataset-label synthetic_fast_demo `
    --seed 123 `
    --methods hybrid `
    --m 400 `
    --mask-percentile 97.0 `
    --thin-ratio 0.9 `
    --softclip-percentile 99.5 `
    --selection-c 3 `
    --k-witness 4 `
    --max-points 80000 `
    --min-points 10000 `
    --coverage-radius 0.03 `
    --out examples/sample_outputs/synthetic_fast_demo.csv
  ```

- **Output:** `synthetic_fast_demo.csv`

Feel free to add more recipes (for example, alternative IXI subjects or new
fast presets) and keep their CSVs in this folder for quick demonstrations.

## Generic point cloud benchmarks

To showcase performance beyond medical imaging, we added a lightweight driver
around `paper_ready.pointcloud_benchmark` that operates on synthetic and
scikit-learn datasets.

- **Command (Swiss roll example):**

  ```powershell
  $env:PYTHONPATH = "${PWD}\src"
  python -m paper_ready.pointcloud_benchmark `
    --builtin swiss_roll `
    --samples 5000 `
    --max-points 5000 `
    --methods hybrid,density,random `
    --m 400 `
    --rips-points 300 `
    --seed 42 `
    --out examples/sample_outputs/pointcloud_swiss_roll.csv
  ```

- **Outputs:** `pointcloud_swiss_roll.csv`, `pointcloud_torus.csv`,
  `pointcloud_gaussian_blobs.csv`, and the aggregated
  `pointcloud_suite_summary.csv` (created via the helper script below).

- **Batch convenience:**

  ```powershell
  $env:PYTHONPATH = "${PWD}\src"
  python examples/run_pointcloud_benchmark.py --output-dir examples/sample_outputs
  ```

  The helper replays three non-medical datasets (Swiss roll, torus surface,
  Gaussian mixtures) so you can regenerate the CSVs with one command.

## Metric reference

Most CSV outputs share a common schema. Below is a quick glossary for the core
columns:

- `dataset` – name supplied via `--dataset-label` (or inferred from the source).
- `method` – landmark selector (`hybrid`, `density`, `random`, etc.).
- `m` – number of landmarks actually used (float in CSV for consistency).
- `total_points` – point-cloud size after masking/thinning.
- `selection_time` – seconds spent choosing landmarks.
- `witness_time` – seconds to compute witness complex persistence.
- `coverage_radius` – radius (in normalised units) used when evaluating
  coverage ratios.
- `coverage_mean` / `coverage_p95` – mean and 95th percentile of the distances
  from each point to its nearest landmark.
- `coverage_ratio` – fraction of points whose nearest landmark is within
  `coverage_radius`.
- `coverage_weighted_mean` / `coverage_weighted_p95` / `coverage_weighted_ratio`
  – same metrics as above but intensity-weighted (points with higher
  intensities count more).
- `landmark_intensity_mean` / `median` / `p90` – summary stats of the sampled
  landmark intensities.
- `diameter` – Euclidean diameter of the original (unnormalised) bounding box.
- `rips_sample_size` / `rips_max_edge` / `rips_time` – properties of the optional
  Vietoris–Rips reference built through Gudhi (NaN when disabled).
- `bottleneck_b{0,1,2,...}` – bottleneck distances between Rips and witness
  diagrams for each homology dimension (NaN/`inf` when diagrams are incompatible
  or Gudhi is unavailable).
- `seed` – RNG seed used by the current method.

Any additional columns come directly from the pipeline (for example, synthetic
experiments may add custom diagnostics). Feel free to extend this glossary when
new metrics appear.
