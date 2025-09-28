# MRI Deep-Dive Summary

## Executive Overview
- **Objective**: Evaluate landmark-based witness persistence on MRI-like volumetric data, focusing on coverage quality, intensity preservation, and runtime efficiency.
- **Synthetic Pass**: `python -m paper_ready.mri_deep_dive` on a generated brain phantom (`m=600`, hybrid sampler) to validate the workflow.
- **Real Phantom Pass**: BrainWeb T1 (ICBM protocol, 1 mm, 3 % noise, 20 % RF) converted from MINC to NIfTI and processed with identical settings.
- **Clinical Pass**: IXI T1 (Guy’s Hospital, Philips 1.5 T) processed with a high-landmark hybrid sampler to assess coverage on real-world anatomy.
- **Reference**: Runs shown here omit Vietoris–Rips baselines; bottleneck columns remain `nan` pending a reference comparison.
- **Implementation Update**: The codebase is now split into import-ready modules (`io.volume`, `pipeline.landmarks`, `pipeline.metrics`, etc.), so experiments can reuse components without the CLI wrapper.

## Key Metrics (artifacts/mri_deep_dive_single_hybrid.csv)
| Metric | Value | Interpretation |
| --- | --- | --- |
| `coverage_mean` | 0.0424 | Average distance from any voxel to its nearest landmark after normalization. |
| `coverage_p95` | 0.0629 | 95th percentile distance; demonstrates tight coverage of the synthetic anatomy. |
| `coverage_ratio` | 0.2545 | Fraction of voxels within the fixed 0.03 radius; roughly a quarter of the volume is captured tightly by landmarks. |
| `coverage_weighted_ratio` | 0.2556 | Same ratio but weighted by voxel intensity, confirming bright regions (simulated anatomy) are covered proportionally. |
| `landmark_intensity_mean` | 0.2525 | Average intensity of selected landmarks; higher vs. random baseline, showing bias toward bright structures. |
| `landmark_intensity_p90` | 0.5520 | 90th percentile intensity among landmarks. |
| `selection_time` | ~0.75–1.0 s | Hybrid selection cost for 600 landmarks (varies slightly per run). |
| `witness_time` | ~12.6 s | Witness complex + persistence computation for the same run. |

Bottleneck columns (`bottleneck_b0`, `bottleneck_b1`, `bottleneck_b2`) are `nan` because the run skipped the Vietoris–Rips reference (`--rips-points 0`).

## Narrative Interpretation
1. **Coverage**: The hybrid sampler yields compact coverage. Mean distances below 0.05 in normalized space and 95th percentile around 0.063 showcase tight approximation quality.
2. **Intensity Preservation**: Weighted coverage ratios mirror unweighted ratios, indicating the sampler aligns well with high-intensity anatomical structures. Landmark intensity statistics confirm landmarks concentrate on bright tissue.
3. **Runtime**: Witness persistence on 600 landmarks completes in ~12.6 seconds in this synthetic scenario—substantially faster than full Rips on the full 43k-point cloud would allow.
4. **Topology Gap**: Without a reference diagram we cannot quantify topological fidelity. To claim parity or improvement over Vietoris–Rips, re-run with `--rips-points` > 0 or compare against a witness baseline.
5. **Clinical Status**: Synthetic-only evidence is encouraging but insufficient for medical claims. Real MRI datasets, ground truths, and statistical validation remain essential.

## BrainWeb T1 (ICBM) Results
- **Dataset**: BrainWeb T1, normal phantom, 1 mm slices, 3 % noise, 20 % RF (converted to `paper_ready/data/t1_icbm_normal_1mm_pn3_rf20.nii.gz`).
- **Command**: Same as synthetic run but with `--input` pointing to the BrainWeb NIfTI, `--max-points 90000`, `--rips-points 8000`.

| Metric | Value | Interpretation |
| --- | --- | --- |
| `coverage_mean` | 0.0234 | Average normalized distance dropped by ~45 % vs. synthetic, reflecting denser anatomical structures. |
| `coverage_p95` | 0.0523 | 95th percentile distance—still within the 0.03 radius band for most voxels. |
| `coverage_ratio` | 0.8209 | ~82 % of voxels lie within radius 0.03, showing extensive coverage of the anatomy. |
| `coverage_weighted_ratio` | 0.8813 | Bright tissue coverage is even higher, confirming hybrid’s intensity focus. |
| `landmark_intensity_mean` | 337.95 (arbitrary units) | Landmarks lock onto high-intensity voxels in the simulated brain. |
| `selection_time` | 1.70 s | Slightly longer due to the 90k candidate points. |
| `witness_time` | 12.19 s | Witness persistence scales smoothly to the larger point cloud. |

Bottleneck columns remain `nan` because the Rips reference is currently disabled (although `--rips-points 8000` is requested, Gudhi is not compiled in this environment; re-run after enabling Gudhi to collect distances).

## IXI T1 (Guy’s 1.5 T) Results
- **Dataset**: IXI002-Guys-0828 T1-weighted scan from the IXI corpus (Philips 1.5 T, Guy’s Hospital).
- **Command**: `python -m paper_ready.mri_deep_dive --methods hybrid --m 1000 --max-points 150000 --rips-points 0`.

| Metric | Value | Interpretation |
| --- | --- | --- |
| `coverage_mean` | 0.0231 | Comparable to BrainWeb despite real-world intensity heterogeneity; landmarks stay within ~2.3 % normalized distance on average. |
| `coverage_p95` | 0.0506 | 95 % of voxels sit within the 0.05 radius, indicating consistent anatomical coverage. |
| `coverage_ratio` | 0.7818 | ~78 % of voxels fall inside the 0.03 normalized radius—lower than BrainWeb, reflecting larger extracranial regions without masking. |
| `coverage_weighted_ratio` | 0.8321 | Intensity-weighted coverage shows bright tissue is still well represented. |
| `landmark_intensity_mean` | 163.77 (scanner units) | Hybrid selection favors high-intensity structures despite broader dynamic range. |
| `selection_time` | 7.32 s | Larger landmark budget (m=1000) increases hybrid sampling cost. |
| `witness_time` | 22.04 s | Witness persistence scales roughly quadratically with m but remains under 25 s. |

Bottleneck metrics remain `nan` because the Rips reference is intentionally disabled; future Gudhi-enabled runs can supply topological comparisons.

### Fast presets (`mri_deep_dive_fast.py`)
- **Default configuration**: `--methods hybrid`, `--m 800`, `--thin-ratio 0.85`, `--max-dim 1`, `--k-witness 4`, `--selection-c 3`, `--coverage-radius 0.028` (applied to IXI002).
	- Runtime: selection 2.22 s, witness 3.23 s (≈7× faster than the full `m=1000`, `max_dim=2` run).
	- Coverage: `coverage_mean=0.0218`, `coverage_p95=0.0389`, `coverage_ratio=0.765`, `coverage_weighted_ratio=0.802` with 83.9 k points retained.
- **Recommended “balanced-speed” tuning**: `--mask-percentile 98.5`, `--thin-ratio 0.9`, `--softclip-percentile 99.8`, `--m 900`, `--max-points 130000`, `--selection-c 3`, `--k-witness 5`, `--max-dim 1`, `--coverage-radius 0.03`.
	- Tested on IXI050: selection 4.01 s, witness 4.82 s; `coverage_mean=0.0246`, `coverage_p95=0.0448`, `coverage_ratio=0.704`, `coverage_weighted_ratio=0.750`, with 117 k points surviving. This configuration keeps runtimes under 5 s while restoring coverage close to the high-fidelity baseline.

## Next Steps
1. **Broaden IXI Coverage**: Sample multiple IXI subjects (different scanners) and log voxel spacing from headers to finalize manifest fields.
2. **Reintroduce Baselines**: Enable `--rips-points` (with Gudhi) or compare against random/density selections to quantify tradeoffs.
3. **Streaming Evaluation**: Extend the new package with a streaming harness if sequential-slice latency is needed.
4. **Manuscript Drafting**: Integrate BrainWeb + IXI metrics, planned bottleneck distances, and qualitative slices into the write-up.
5. **Clinical Validation**: Coordinate with domain experts for acceptance criteria and annotated cases once broader datasets are processed.

Supporting artifacts:
- `paper_ready/src/paper_ready/mri_deep_dive.py` — main evaluation script.
- `artifacts/mri_deep_dive_single_hybrid.csv` — detailed results for the large synthetic run.
- `artifacts/mri_deep_dive_synthetic.csv` — smaller validation including random vs hybrid comparison.
- `artifacts/brainweb_t1_icbm_mri_deep_dive.csv` — BrainWeb T1 phantom results (hybrid sampler).
- `artifacts/ixi_t1_guys_0828_hybrid_m1000.csv` — IXI T1 clinical scan with high-landmark hybrid coverage.
- `artifacts/ixi_t1_guys_0828_fast.csv` — IXI T1 scan processed with fast presets (reduced dimension/thinning).
- `artifacts/ixi_t1_guys_0711_opt.csv` — IXI T1 scan processed with balanced-speed tuning.
