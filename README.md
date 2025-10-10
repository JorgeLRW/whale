# MRI Witness Pipeline

Fast landmark-based persistence tooling for MRI volumes and point clouds, powered by the
Whale library.

[![PyPI](https://img.shields.io/pypi/v/whale-tda.svg)](https://pypi.org/project/whale-tda/)
[![Tests](https://github.com/jorgeLRW/whale/actions/workflows/tests.yml/badge.svg)](https://github.com/jorgeLRW/whale/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgeLRW/whale/blob/main/examples/notebooks/synthetic_demo.ipynb)

This folder contains the curated code, results, and manuscript assets that accompany the
arXiv preprint on fast landmark-based witness persistence for MRI volumes. It is
self-contained so it can be open-sourced independently of the larger research workspace.

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

# 2. Install runtime dependencies (or grab the PyPI wheel)
pip install -r paper_ready/requirements.txt
# Alternatively, skip local requirements and install the published package
pip install whale-tda

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
  --auto-m `
    --selection-c 3 `
    --k-witness 5 `
    --max-points 130000 `
    --coverage-radius 0.03 `
    --out paper_ready/artifacts/ixi_t1_guys_0711_opt.csv

> **Auto-scaling landmarks**
> Both entry points accept `--auto-m` together with
> `--auto-m-base`, `--auto-m-exponent`, `--auto-m-min`, and
> `--auto-m-max`. The default formula uses
> $m = \text{base} \cdot n^{\text{exponent}}$ with $n$ equal to the
> final retained point count, clamped to the provided bounds. Swap any
> explicit `--m …` flag above for `--auto-m` when you want landmark budgets
> to adapt automatically to dataset size.
```

## Try it in 60 seconds

Run the smoke suite on the synthetic MRI phantom to validate your environment:

```powershell
cd paper_ready
python -m unittest tests.test_whale_smoke
```

Prefer the cloud? Launch the [Colab notebook](https://colab.research.google.com/github/jorgeLRW/whale/blob/main/examples/notebooks/synthetic_demo.ipynb) to clone the
repository, install dependencies, and execute the synthetic benchmark end-to-end.

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

## AI-ready embeddings

The `whale.ai` module streamlines integration with deep-learning workflows. Convert
vision or language embeddings (NumPy arrays *or* PyTorch tensors) into Whale
`PointCloud` objects, run witness persistence on each batch element, and obtain compact
feature summaries suitable for downstream models. Optional utilities expose a
`torch.nn.Module` that can be dropped into training loops so persistence statistics are
available alongside conventional features. Switch between fast (dim-1, lightweight) and
regular (dim-2 aware) modes by passing `tda_mode="fast"` or `tda_mode="regular"` to the
helpers. Install the PyTorch extras via `pip install "whale-tda[ai]"` to enable the neural
layer. See `tests/test_ai_integration.py` for a
minimal example and the inline docstrings in `whale.ai` for configuration details.

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

## Governance and roadmap

- Maintainer responsibilities and release cadence live in [`MAINTAINERS.md`](MAINTAINERS.md).
- Upcoming milestones and feature ideas are tracked in [`ROADMAP.md`](ROADMAP.md) and on the GitHub project board.
- Release history follows [`CHANGELOG.md`](CHANGELOG.md) using the Keep a Changelog format.

## Packaging and distribution

The project ships a modern `pyproject.toml`. Until wheels are published to PyPI you can install locally:

```powershell
pip install build
python -m build
pip install dist/whale_tda-0.1.0-py3-none-any.whl
```

After publishing to PyPI the workflow simplifies to `pip install whale-tda`.

The install exposes three console scripts:

- `whale-deep-dive`
- `whale-deep-dive-fast`
- `whale-pointcloud`

See [`pyproject.toml`](pyproject.toml) for optional extras (dev, docs) and run the smoke tests once the package is installed.

## Containers

Prefer containers over local Python? Build the images defined in [`docker/`](docker/):

```powershell
docker build -f docker/Dockerfile.whale -t whale .
docker build -f docker/Dockerfile.fast -t whale-fast .
```

The short tags (`whale`, `whale-fast`) work locally without a namespace. Retag before pushing to a registry (for example `ghcr.io/jorgeLRW/whale:0.1.0`). More guidance lives in [`docker/README.md`](docker/README.md).

> **Security note**
> Even though the images inherit from `python:3.11-slim-bookworm`, scan them with `docker scout cves whale` before shipping production artifacts.

## Community

- Open questions and showcase results in the GitHub **Discussions** tab (Announcements, Q&A, Show & Tell).
- Report bugs or feature requests via the issue templates.
- Join the upcoming community chat (Discord invite will be posted in Discussions).

## Compliance checklist

- Review dataset licences in [`COMPLIANCE.md`](COMPLIANCE.md) and keep raw MRI volumes out of Git.
- Track third-party package licences in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md); regenerate it with `pip-licenses` on every release.
- Security reports can be sent privately to the maintainer email listed in `MAINTAINERS.md`.

## License and citation

This project is released under the [MIT License](LICENSE). Please cite the arXiv
preprint *Fast Witness Persistence for MRI Volumes via Hybrid Landmarking*
(2025). A `CITATION.cff` file accompanies the repository with the full bibliographic
entry.

## AI-assisted development

Portions of the codebase and documentation were produced with the help of AI coding
assistants. All generated content was reviewed, tested, and adapted by the author
prior to release.
