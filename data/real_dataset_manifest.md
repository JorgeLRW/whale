# Real MRI Dataset Manifest

This document serves as a staging area for future clinical-grade evaluations. Populate it with MRI datasets, acquisition notes, and preprocessing steps before running `python -m paper_ready.mri_deep_dive` on real data.

## Candidate Public Datasets
| Dataset | Modality | Availability | Notes |
| --- | --- | --- | --- |
| BrainWeb | T1/T2/PD (phantom) | http://brainweb.bic.mni.mcgill.ca | Provides controlled noise levels and tissue labels; good for initial validation. |
| IXI | T1/T2/PD | https://brain-development.org/ixi-dataset/ | Multi-scanner images with demographic variety; direct tarballs downloadable via `paper_ready/scripts/download_ixi_dataset.ps1`. |
| fastMRI | Proton Density, T2 (k-space + images) | https://fastmri.org/ | Large-scale knee/brain data; includes reconstructions and raw k-space. |
| ADNI | T1/T2/PET | http://adni.loni.usc.edu | Requires application; Alzheimer’s cohorts with longitudinal scans. |
| OASIS-3 | T1/T2 | https://www.oasis-brains.org | Longitudinal aging study with varying pathology. |

## Recording Template
For each dataset or subject you ingest, append a section like this:

```
## Dataset ID: <unique identifier>
- Source: <BrainWeb / IXI / etc.>
- Modality: <T1 / T2 / PD / diffusion>
- Acquisition Size: <e.g., 256 x 256 x 180 voxels>
- Voxel Spacing: <e.g., 1.0 x 1.0 x 1.2 mm>
- Preprocessing:
  - Bias field correction: <yes/no>
  - Skull stripping: <tool used>
  - Intensity normalization: <method>
  - Registration: <atlas or transform>
- Point-Cloud Extraction:
  - Command:
    ```powershell
  python -m paper_ready.mri_deep_dive `
        --input <path/to/volume.nii.gz> `
        --dataset-label <short_name> `
        --methods hybrid `
        --m <landmark_count> `
        --mask-percentile <e.g., 97.5> `
        --max-points <e.g., 90000> `
        --rips-points <e.g., 8000> `
    --out artifacts/<label>_mri_deep_dive.csv
    ```
  - Seed(s): <list of seeds used>
- Outputs:
  - Result CSV: `<path>`
  - Visualizations: `<path>`
  - Notes: <observations, anomalies>
```

## Preprocessing Checklist
1. Convert DICOM to NIfTI (`dcm2niix` recommended).
2. Skull-stripping (e.g., FSL BET, HD-BET); inspect results manually.
3. Bias field correction (N4ITK) to stabilize intensity-based sampling.
4. Optional: register to a common atlas for cross-subject comparisons.
5. Decide on intensity thresholds or tissue masks.
6. Document all steps above in this manifest for reproducibility.

## Data Governance
- Ensure licensing terms allow research usage.
- Maintain any needed de-identification steps before storage.
- Track consent and IRB requirements where applicable.

Populate this manifest as real datasets are onboarded so downstream analyses and publications have clear provenance.

## Dataset ID: brainweb_t1_icbm_1mm_pn3_rf20
- Source: BrainWeb Simulated Brain Database (ICBM protocol, normal phantom)
- Modality: T1
- Acquisition Size: 181 × 217 × 181 voxels
- Voxel Spacing: 1.0 × 1.0 × 1.0 mm
- Preprocessing:
  - MINC → NIfTI conversion (nibabel)
  - Bias field correction: no (simulated RF already embedded)
  - Skull stripping: not required (phantom is brain-only)
  - Intensity normalization: none
  - Registration: none
- Point-Cloud Extraction:
  - Command:
    ```powershell
    python -m paper_ready.mri_deep_dive `
        --input paper_ready/data/t1_icbm_normal_1mm_pn3_rf20.nii.gz `
        --dataset-label brainweb_t1_icbm `
        --methods hybrid `
        --m 600 `
        --mask-percentile 97.5 `
        --max-points 90000 `
        --rips-points 8000 `
        --out artifacts/brainweb_t1_icbm_mri_deep_dive.csv
    ```
  - Seed(s): 17 (derived from base seed 0 inside script)
- Outputs:
  - Result CSV: `artifacts/brainweb_t1_icbm_mri_deep_dive.csv`
  - Visualizations: _pending_
  - Notes: 600 landmarks selected in 1.70 s; witness persistence 12.19 s; coverage_mean 0.0234, coverage_p95 0.0523; rips reference used 8 000 points.

## Dataset ID: ixi002_guys_0828_t1
- Source: IXI Dataset (Guy’s Hospital, Philips 1.5 T)
- Modality: T1-weighted MRI
- Acquisition Size: _pending (read from NIfTI header)_
- Voxel Spacing: _pending_
- Preprocessing:
  - Download: `paper_ready/scripts/download_ixi_dataset.ps1 -SkipExtract -Archives T1`
  - Extraction: `tar -xf paper_ready/data/IXI/IXI-T1.tar -C paper_ready/data/IXI`
  - Bias field correction: no
  - Skull stripping: none
  - Intensity normalization: none
  - Registration: none
- Point-Cloud Extraction:
  - Command:
    ```powershell
    python -m paper_ready.mri_deep_dive `
        --input paper_ready/data/IXI/IXI002-Guys-0828-T1.nii.gz `
        --dataset-label ixi_t1_guys_0828 `
        --methods hybrid `
        --m 1000 `
        --mask-percentile 97.5 `
        --max-points 150000 `
        --rips-points 0 `
        --out artifacts/ixi_t1_guys_0828_hybrid_m1000.csv
    ```
  - Seed(s): 17 (base seed offset)
- Outputs:
  - Result CSV: `artifacts/ixi_t1_guys_0828_hybrid_m1000.csv`
  - Visualizations: _pending_
  - Notes: coverage_mean 0.0231, coverage_p95 0.0506; coverage_ratio 0.7818, weighted_ratio 0.8321; selection 7.32 s, witness 22.04 s.
  - Fast presets: `python -m paper_ready.mri_deep_dive_fast --input paper_ready/data/IXI/IXI002-Guys-0828-T1.nii.gz --dataset-label ixi_fast --out artifacts/ixi_t1_guys_0828_fast.csv` → coverage_mean 0.0218, coverage_p95 0.0389; coverage_ratio 0.7652; selection 2.22 s, witness 3.23 s.

## Dataset ID: ixi050_guys_0711_t1
- Source: IXI Dataset (Guy’s Hospital, Philips 1.5 T)
- Modality: T1-weighted MRI
- Acquisition Size: _pending (read from NIfTI header)_
- Voxel Spacing: _pending_
- Preprocessing:
  - Download/extract via `paper_ready/scripts/download_ixi_dataset.ps1 -SkipExtract -Archives T1` and `tar -xf paper_ready/data/IXI/IXI-T1.tar -C paper_ready/data/IXI`
  - Bias field correction: no
  - Skull stripping: none
  - Intensity normalization: none
  - Registration: none
- Point-Cloud Extraction (optimized fast preset):
  - Command:
    ```powershell
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
        --out artifacts/ixi_t1_guys_0711_opt.csv
    ```
  - Seed(s): 17
- Outputs:
  - Result CSV: `artifacts/ixi_t1_guys_0711_opt.csv`
  - Visualizations: _pending_
  - Notes: coverage_mean 0.0246, coverage_p95 0.0448; coverage_ratio 0.7043, weighted_ratio 0.7499; selection 4.01 s, witness 4.82 s; total points 117 k after thinning.
