# Real point cloud datasets

This directory holds optional real-world point clouds used by `baseline_sweep.py` when the corresponding files are present.

You can populate them automatically by running `python fetch_real_point_clouds.py`, which downloads the Stanford meshes,
extracts their vertices, and saves subsampled point clouds in the expected format. Manual placement is still supported:

Place the following files here to include the real datasets in the sweep:

- `bunny_2000.npy`: NumPy array of shape `(N, 3)` containing the Stanford bunny (or similar) point cloud.
- `dragon_2000.npy`: NumPy array of shape `(N, 3)` with a second real scan (for example, the Stanford dragon).

Each array should be stored in floating point units (meters) and will be consumed directly without additional normalization.

If these files are absent, the sweep will skip the real datasets automatically and proceed with the synthetic benchmarks.
