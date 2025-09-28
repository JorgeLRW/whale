# MRI Witness Pipeline

This folder contains the curated code, results, and manuscript assets that accompany the
upcoming arXiv submission on fast landmark-based witness persistence for MRI volumes.
It is self-contained so it can be open-sourced independently of the larger research
workspace.

## Directory layout

- `analysis/` – Human-readable summaries of the experiments (see `mri_deep_dive_summary.md`).
- `artifacts/` – Local workspace for generated CSV/JSON outputs (ignored by git).
- `data/` – Provenance notes (`real_dataset_manifest.md`) plus the mirrored datasets and helper
  scripts previously used in `paper_ready_tests/data/` (BrainWeb volume, IXI tree, Stanford point clouds).
- `paper/` – LaTeX sources for the manuscript.
- `scripts/` – Utility scripts, including an IXI dataset downloader for Windows PowerShell.
- `src/paper_ready/` – Minimal Python package implementing the deep-dive pipeline and
  supporting modules used in the experiments.
- `examples/sample_outputs/` – Documentation for representative run configurations and
  pointers to the corresponding CSVs that can be regenerated locally.

## Quick start

```powershell
# 1. Create a fresh virtual environment and activate it (example with conda)
conda create -n mri-witness python=3.10 -y
conda activate mri-witness

# 2. Install runtime dependencies
pip install -r paper_ready/requirements.txt

# 3. Make the package importable for local development
$env:PYTHONPATH = "${PWD}\paper_ready\src"

# 4. (Optional) Refresh IXI volumes into paper_ready/data/IXI
powershell -ExecutionPolicy Bypass -File paper_ready/scripts/download_ixi_dataset.ps1 -Archives T1

# 5. Reproduce the BrainWeb run
python -m paper_ready.mri_deep_dive `
    --input paper_ready/data/t1_icbm_normal_1mm_pn3_rf20.nii.gz `
    --dataset-label brainweb_t1_icbm `
    --methods hybrid `
    --m 600 `
    --mask-percentile 97.5 `
    --max-points 90000 `
    --rips-points 0 `
    --out paper_ready/artifacts/brainweb_t1_icbm_mri_deep_dive.csv

# 6. Reproduce the fast IXI configuration
python -m paper_ready.mri_deep_dive_fast `
    --input paper_ready/data/IXI/IXI050-Guys-0711-T1.nii.gz `
    --dataset-label ixi_t1_guys_0711_opt `
    --mask-percentile 98.5 `
    --thin-ratio 0.9 `
    --softclip-percentile 99.8 `
    --m 900 `
    --selection-c 3 `
    --k-witness 5 `
    --max-points 130000 `
    --coverage-radius 0.03 `
    --out paper_ready/artifacts/ixi_t1_guys_0711_opt.csv
```

> **Note**
>
> - Gudhi is optional. If installed, you can enable Vietoris–Rips references by
>   setting `--rips-points` and the pipeline will automatically compute bottleneck distances.
> - The hybrid sampler also benefits from `hnswlib` or `faiss-cpu`; both are optional
>   accelerators and can be installed when available.

## Cross-domain benchmarking

The new module `paper_ready.pointcloud_benchmark` exercises the witness pipeline on
generic point clouds (Swiss roll, torus surface, Gaussian mixtures, and any CSV/NPY you
provide). This demonstrates that our landmark heuristics generalise outside MRI.

```powershell
$env:PYTHONPATH = "${PWD}\paper_ready\src"
python -m paper_ready.pointcloud_benchmark `
  --builtin swiss_roll `
  --samples 5000 `
  --max-points 5000 `
  --methods hybrid,density,random `
  --m 400 `
  --rips-points 300 `
  --out paper_ready/artifacts/swiss_roll_benchmark.csv

# Batch three datasets (Swiss roll, torus, Gaussian blobs)
python examples/run_pointcloud_benchmark.py `
  --output-dir paper_ready/examples/sample_outputs
```

All generated CSVs mirror the schema used by the MRI pipelines, making it easy to
compare coverage, timing, and (when Gudhi is installed) bottleneck distances across
domains.

> **Tip**
> A concise glossary for the CSV columns (coverage metrics, intensity stats,
> Vietoris–Rips fields, etc.) lives in `examples/sample_outputs/README.md` under
> “Metric reference”. Extend it as you introduce new measurements.

## Reproducing the tables and plots

1. Run the commands above (or supply your own MRI data with matching parameters).
2. Inspect the resulting CSVs in `paper_ready/artifacts/`.
3. Consult `analysis/mri_deep_dive_summary.md` for a curated narrative of the
   performance/coverage trade-offs.

## Dataset considerations

- The IXI volumes are large and subject to their original license. The downloader script
  saves archives under `paper_ready/data/IXI` and leaves extraction markers so repeated runs
  are fast.
- Synthetic and BrainWeb volumes referenced in the paper can be fetched from their public
  sources. Place them under `paper_ready/data/` following the manifest.
- `data/real_dataset_manifest.md` is the canonical place to document acquisition details,
  voxel spacing, preprocessing, and commands for every subject used in the paper.

## Manuscript workflow

The LaTeX sources live in `paper_ready/paper/`. Use the provided `Makefile` (or the
instructions in `paper_ready/paper/README.md`) to compile a PDF:

```powershell
cd paper_ready/paper
latexmk -pdf main.tex
```

The LaTeX template already contains skeleton sections (introduction, methods, results,
conclusion) and injects the CSV artifacts into tables using `pgfplotstable` helpers.

## License and citation

This project is released under the [MIT License](LICENSE). Please cite the forthcoming
arXiv preprint once it is available; a `CITATION.cff` file will be published alongside
the preprint metadata.

## AI-assisted development

Portions of the codebase and documentation were produced with the help of AI coding
assistants. All generated content was reviewed, tested, and adapted by the author
prior to release.
