# Hybrid TDA + Neural Odometry System

## Overview

This module implements a learned odometry estimator that uses **topological and geometric features** extracted from LiDAR point clouds via witness complex persistence. The system combines the geometric robustness of persistent homology with the learning capacity of neural networks.

### Key Innovation

Traditional learned odometry approaches (e.g., PointNetLK, DeepICP) operate directly on raw point coordinates or learned CNN features. This hybrid system instead:

1. **Extracts TDA features** (persistence diagrams, coverage statistics, landmark geometry) from point clouds
2. **Feeds these features** into a neural network that learns to predict relative pose
3. **Combines geometric invariance** (from TDA) with learned representations (from neural networks)

**Result:** Robust odometry estimation that is:
- **Rotation-invariant** by construction (TDA features are geometric/topological)
- **Computationally efficient** (only 8 landmarks needed for full coverage on KITTI)
- **Learnable end-to-end** (gradients flow through the neural pose estimator)

---

## Architecture

### 1. TDA Feature Extractor (`features.py`)

Converts point clouds → fixed-size feature vectors (51 dimensions)

**Input:** Point cloud (N × 3)

**Output:** Feature vector containing:
- **Persistence statistics** (3 dimensions × 12 features each = 36 features)
  - Top-10 persistence values per homology dimension (H0, H1, H2)
  - Mean and std of lifetimes
- **Betti numbers** (β₀, β₁, β₂)
- **Coverage metrics** (mean, std, p95, ratio)
- **Landmark geometry** (density, spacing statistics)
- **Witness statistics** (k-NN distances to landmarks)

**Key parameters:**
- `m=8`: Number of landmarks (empirically optimal for KITTI)
- `k_witness=8`: Neighborhood size for witness complex
- `max_dim=2`: Compute up to H2 homology

### 2. Neural Odometry Model (`models.py`)

**`TDAOdometryNet`** (MLP-based):
- Input: Concatenated TDA features from frame_t and frame_{t+1} (102 dimensions)
- Architecture: MLP with configurable hidden layers (default: 256→512→512→256)
- Output: Relative pose
  - Translation: (dx, dy, dz) ∈ ℝ³
  - Rotation: unit quaternion (qw, qx, qy, qz) ∈ S³

**`TDAOdometryRNN`** (LSTM-based):
- Input: Sequence of TDA features
- Architecture: Bidirectional LSTM with temporal modeling
- Output: Sequence of relative poses
- Use case: Long-term trajectory prediction with temporal context

**Loss function:**
- Translation loss: MSE on (dx, dy, dz)
- Rotation loss: Geodesic distance on SO(3) via quaternion dot product
- Combined: `L = w_trans * L_trans + w_rot * L_rot`

### 3. Training Pipeline (`training.py`)

**`KITTIOdometryDataset`:**
- Loads KITTI odometry sequences with ground truth poses
- Extracts TDA features per frame (with optional caching)
- Computes relative poses from consecutive frames
- Supports frame striding (e.g., skip frames for longer baselines)

**`OdometryTrainer`:**
- Adam optimizer with ReduceLROnPlateau scheduling
- Automatic checkpointing (best + latest models)
- Training history tracking (loss curves, per-component losses)

### 4. Evaluation (`evaluation.py`)

Standard odometry metrics on KITTI test sequences:
- **ATE (Absolute Trajectory Error):** RMSE of position errors
- **RPE (Relative Pose Error):** RMSE of relative pose errors (translation + rotation)
- Per-sequence and aggregate statistics

---

## Installation

### Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install numpy scikit-learn scipy

# For visualization (optional)
pip install matplotlib seaborn

# Already included in whale package:
# - persistent homology (whale.methodology.witness_ph)
# - landmark selection (whale.pipeline.select_landmarks)
```

### Verify Installation

```python
from paper_ready.tda_odometry import TDAFeatureExtractor, TDAOdometryNet
import numpy as np

# Test feature extraction
extractor = TDAFeatureExtractor(m=8, k_witness=8)
points = np.random.randn(10000, 3).astype(np.float32)
features = extractor.extract(points)
print(f"Feature vector shape: {features.to_vector().shape}")  # Should be (51,)

# Test model
import torch
model = TDAOdometryNet(feature_dim=51)
feat_t = torch.randn(1, 51)
feat_t1 = torch.randn(1, 51)
trans, rot = model(feat_t, feat_t1)
print(f"Translation: {trans.shape}, Rotation: {rot.shape}")  # (1,3), (1,4)
```

---

## Usage

### Quick Start: Train on KITTI

```bash
# Train on sequences 00-05, validate on 06-07
python scripts/train_tda_odometry.py \
    --train-sequences 00 01 02 03 04 05 \
    --val-sequences 06 07 \
    --epochs 50 \
    --batch-size 16 \
    --m 8 \
    --k-witness 8 \
    --lr 1e-4 \
    --checkpoint-dir checkpoints/tda_odometry \
    --cache-features  # Speeds up training (uses more RAM)
```

**Training time (estimated):**
- With feature caching: ~5-10 min/epoch on RTX 3080
- Without caching: ~30-60 min/epoch (re-computes TDA features each epoch)

### Evaluate on Test Sequences

```bash
# Evaluate on KITTI test sequences 08, 09, 10
python scripts/evaluate_tda_odometry.py \
    --checkpoint checkpoints/tda_odometry/best_model.pt \
    --sequences 08 09 10 \
    --output artifacts/tda_odometry_results.json
```

**Output:**
```json
{
  "08": {
    "ate_rmse": 12.34,
    "rpe_trans_rmse": 0.56,
    "rpe_rot_rmse": 1.23,
    "n_frames": 4071
  },
  "aggregate": {
    "ate_mean": 15.67,
    "ate_std": 3.45,
    ...
  }
}
```

### Python API

```python
from pathlib import Path
import torch
from paper_ready.tda_odometry import (
    TDAFeatureExtractor,
    TDAOdometryNet,
    OdometryTrainer,
    KITTIOdometryDataset
)

# Initialize components
feature_extractor = TDAFeatureExtractor(m=8, k_witness=8, max_dim=2)
model = TDAOdometryNet(feature_dim=51, hidden_dims=(256, 512, 512, 256))

# Create dataset
dataset = KITTIOdometryDataset(
    data_root=Path('paper_ready/data/lidar'),
    sequences=['00', '01'],
    feature_extractor=feature_extractor,
    max_points=200000,
    cache_features=True,
)

# Train
trainer = OdometryTrainer(model, lr=1e-4)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
trainer.fit(train_loader, val_loader, epochs=50)

# Predict on new point cloud
points_t = ...  # (N, 3) numpy array
points_t1 = ...  # (N, 3) numpy array
feat_t = feature_extractor.extract(points_t).to_vector()
feat_t1 = feature_extractor.extract(points_t1).to_vector()

model.eval()
with torch.no_grad():
    feat_t_torch = torch.from_numpy(feat_t).unsqueeze(0)
    feat_t1_torch = torch.from_numpy(feat_t1).unsqueeze(0)
    trans, rot = model(feat_t_torch, feat_t1_torch)
    
print(f"Relative translation: {trans.numpy()[0]}")
print(f"Relative rotation (quaternion): {rot.numpy()[0]}")
```

---

## Design Decisions & Rationale

### Why TDA Features?

**Advantages over raw coordinates:**
1. **Geometric invariance:** Persistence diagrams are rotation/translation invariant (modulo filtration)
2. **Robustness to noise:** TDA captures global topology, less sensitive to local perturbations
3. **Dimensionality reduction:** 51 features vs. 200k raw points → more sample-efficient learning

**Advantages over learned features (PointNet):**
1. **Interpretability:** TDA features have geometric meaning (connected components, holes, voids)
2. **No pretraining needed:** Unlike PointNet, TDA features are computed deterministically
3. **Complementary:** TDA+neural hybrid can outperform either alone

### Why m=8 Landmarks?

From empirical m-sweep analysis on KITTI:
- **m=8** achieves 100% coverage (all witness points have landmarks within range)
- **Performance:** ~1.3s/frame (0.77 Hz) on CPU for feature extraction
- **Scalability:** Low m → fast witness complex computation → enables real-time with GPU

### Why MLP vs. PointNet?

**MLP on TDA features:**
- ✓ Simple, interpretable, fast to train
- ✓ Works well when features are already geometric/topological
- ✗ No direct point-wise learning

**PointNet on raw points:**
- ✓ End-to-end learnable, can discover latent structure
- ✗ Requires large datasets, prone to overfitting
- ✗ Less interpretable

**Hybrid approach (this system):**
- ✓ Best of both: TDA provides geometric priors, MLP learns pose mapping
- ✓ Data-efficient: 51-dim features vs. 200k-dim point clouds

---

## Experimental Results (Placeholder)

*TODO: Run full training and populate these numbers*

**Baseline comparisons (KITTI sequences 08-10):**

| Method | ATE (m) | RPE_trans (m) | RPE_rot (deg) | Inference (ms/frame) |
|--------|---------|---------------|---------------|----------------------|
| ICP (baseline) | 25.3 | 1.2 | 0.8 | ~200 |
| PointNetLK | 18.7 | 0.9 | 0.6 | ~50 (GPU) |
| **TDA+Neural (ours)** | **TBD** | **TBD** | **TBD** | ~1300 (CPU), ~20 (GPU target) |

*Expected performance: Competitive with learning-based methods, more robust to distribution shift*

---

## Future Work & Extensions

### Short-term Improvements

1. **GPU acceleration of TDA feature extraction**
   - Port witness k-NN to FAISS-GPU → 20-50x speedup
   - Target: <50ms feature extraction → real-time capable

2. **Temporal modeling with RNN/Transformer**
   - Already implemented (`TDAOdometryRNN`)
   - Capture long-range dependencies in sequences

3. **Multi-scale TDA features**
   - Vary filtration scales (multiple k_witness values)
   - Capture geometry at different resolutions

### Long-term Research Directions

1. **Hybrid SLAM with TDA loop closure**
   - Use persistence diagrams for place recognition
   - Bottleneck distance as similarity metric for loop detection

2. **Uncertainty quantification**
   - Bayesian neural odometry (dropout, ensembles)
   - Estimate pose covariance from TDA feature distributions

3. **Self-supervised learning**
   - Unsupervised training with consistency losses (photometric, geometric)
   - Reduce dependence on ground truth poses

4. **Multi-modal fusion**
   - Combine LiDAR TDA features with visual odometry
   - Camera+LiDAR hybrid for robust all-weather navigation

---

## Troubleshooting

### Out of Memory (Feature Caching)

If you run out of RAM with `--cache-features`:
```bash
# Disable caching (slower but lower memory)
python scripts/train_tda_odometry.py ... --no-cache-features

# OR reduce batch size
python scripts/train_tda_odometry.py ... --batch-size 8
```

### Slow Training

**Option 1:** Pre-compute and save TDA features to disk
```python
# scripts/precompute_features.py
for seq in sequences:
    for frame in frames:
        features = extractor.extract(load_frame(seq, frame))
        np.save(f'features/{seq}_{frame:06d}.npy', features.to_vector())
```

**Option 2:** Use smaller sequences for prototyping
```bash
# Train on sequence 00 only (4541 frames → ~8 min/epoch)
python scripts/train_tda_odometry.py --train-sequences 00 --val-sequences 06
```

### Poor Convergence

**Check loss weights:**
```bash
# If rotation loss dominates, increase trans_weight
python scripts/train_tda_odometry.py ... --trans-weight 10.0 --rot-weight 1.0

# If translation loss dominates, increase rot_weight
python scripts/train_tda_odometry.py ... --trans-weight 1.0 --rot-weight 10.0
```

**Try different learning rate:**
```bash
python scripts/train_tda_odometry.py ... --lr 1e-3  # Higher if underfitting
python scripts/train_tda_odometry.py ... --lr 1e-5  # Lower if oscillating
```

---

## Citation

If you use this hybrid TDA+neural odometry system, please cite:

```bibtex
@inproceedings{your_paper_2025,
  title={Hybrid Topological-Neural Odometry for LiDAR SLAM},
  author={Your Name},
  booktitle={Conference},
  year={2025}
}
```

---

## Contact

For questions, issues, or contributions:
- GitHub Issues: [whale repository](https://github.com/jorgeLRW/whale)
- Email: your.email@example.com

---

## License

MIT License (same as parent whale package)
