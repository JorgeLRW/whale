# Installation Guide - whale-tda v0.3.0

## Quick Install

### Basic Installation
```bash
pip install whale-tda
```

### With LiDAR Odometry Support
```bash
pip install whale-tda[lidar]
```
This installs:
- PyTorch ≥2.0 (for GPU acceleration and neural odometry)
- tqdm (for progress bars during training)

### With General Mapping Support
```bash
pip install whale-tda[mapping]
```
Same dependencies as `lidar` extra.

### All Optional Dependencies
```bash
pip install whale-tda[dev,docs,ai,lidar]
```

---

## From Source (Development)

### Clone Repository
```bash
git clone https://github.com/JorgeLRW/whale.git
cd whale/paper_ready
```

### Install in Editable Mode
```bash
pip install -e .                  # Basic
pip install -e .[lidar]           # With LiDAR support
pip install -e .[dev,lidar]       # Development + LiDAR
```

---

## Feature-Specific Requirements

### 1. LiDAR Odometry (New in v0.3.0)
**Required:**
- `whale-tda[lidar]`
- KITTI odometry dataset (download separately)

**Includes:**
- Fast landmark selection (63.7x speedup)
- Hybrid TDA+Neural odometry
- GPU-accelerated witness complex
- Trained models for KITTI sequences

**Usage:**
```python
from paper_ready.tda_odometry import TDAFeatureExtractor, TDAOdometryNet
import torch

# Extract TDA features from LiDAR frame
extractor = TDAFeatureExtractor(m=8, k_witness=8, method="lidar")
features = extractor.extract(point_cloud)

# Load pre-trained odometry model
model = TDAOdometryNet.load("checkpoints/tda_odometry_ultrafast/best_model.pt")
trans, rot = model(features_t, features_t1)
```

### 2. MRI Brain Analysis (v0.1.x-0.2.x)
**Required:**
- `whale-tda` (basic install)

**Includes:**
- IXI dataset processing
- Brain MRI witness complex
- 3D volumetric TDA

### 3. Point Cloud Benchmarking
**Required:**
- `whale-tda` (basic install)

**Includes:**
- Synthetic point cloud generation
- Multiple landmark selection methods
- Scalability benchmarks

---

## GPU Support

### CUDA (NVIDIA GPUs)
PyTorch will automatically use CUDA if available:
```bash
pip install whale-tda[lidar]
```

To specify CUDA version:
```bash
pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install whale-tda[lidar] --no-deps
```

### CPU-Only
If you don't have a GPU, the package will fall back to CPU:
```bash
pip install whale-tda[lidar]
```
Performance: ~1.05s/frame on CPU (KITTI ~125k points)

### Check GPU Availability
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

---

## Verification

### Test Basic Install
```python
import whale
from whale.methodology.witness_ph import compute_witness_persistence
import numpy as np

points = np.random.randn(1000, 3)
landmarks = np.random.choice(1000, size=10, replace=False)
diagrams = compute_witness_persistence(points, landmarks, k_witness=8)
print(f"Computed diagrams: {list(diagrams.keys())}")  # Should print [0, 1, 2]
```

### Test LiDAR Odometry
```python
from paper_ready.tda_odometry import TDAFeatureExtractor
import numpy as np

extractor = TDAFeatureExtractor(m=8, method="lidar")
points = np.random.randn(10000, 3).astype(np.float32)
features = extractor.extract(points)
print(f"Feature vector: {features.to_vector().shape}")  # Should print (51,)
print(f"Computation time: {features.computation_time:.3f}s")
```

### Test GPU Acceleration
```python
import torch
from whale.methodology.witness_ph import compute_witness_persistence
import numpy as np

points = np.random.randn(100000, 3).astype(np.float32)
landmarks = np.random.choice(100000, size=8, replace=False)

# Should automatically use GPU if available
diagrams = compute_witness_persistence(points, landmarks, k_witness=8, max_dim=2)
print(f"Diagrams computed: {list(diagrams.keys())}")
```

---

## Troubleshooting

### Import Error: `No module named 'whale'`
```bash
# Ensure you're in the correct directory
cd paper_ready
pip install -e .
```

### Import Error: `No module named 'torch'`
```bash
pip install whale-tda[lidar]
```

### CUDA Out of Memory
Reduce `max_witnesses` parameter:
```python
from whale.methodology.witness_ph import compute_witness_persistence

diagrams = compute_witness_persistence(
    points, landmarks, 
    k_witness=8, 
    max_witnesses=5000  # Default: 10000
)
```

### Slow Performance on CPU
Expected timings on CPU (KITTI ~125k points):
- Landmark selection: ~5ms (LiDAR method)
- Witness complex: ~1,045ms (with subsampling)
- **Total: ~1,050ms/frame (0.95 Hz)**

For faster processing:
1. Use GPU: `pip install torch+cu118`
2. Reduce max_witnesses: `max_witnesses=5000`
3. Skip frames: process every 2nd or 3rd frame

### Missing KITTI Data
Download KITTI Odometry dataset:
1. Register at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
2. Download:
   - Velodyne laser data (80 GB)
   - Ground truth poses (4 MB)
3. Extract to: `paper_ready/data/lidar/sequences/`

Expected structure:
```
paper_ready/data/lidar/
├── sequences/
│   ├── 00/
│   │   └── velodyne/
│   │       ├── 000000.bin
│   │       ├── 000001.bin
│   │       └── ...
│   ├── 01/
│   └── ...
└── poses/
    ├── 00.txt
    ├── 01.txt
    └── ...
```

---

## Performance Expectations

### LiDAR Processing (KITTI ~125k points/frame)

| Component | Time (ms) | Speedup vs Baseline |
|-----------|-----------|---------------------|
| Landmark selection | 5.3 | 87.6x |
| Witness k-NN (GPU) | 3.0 | 5.3x |
| Simplex construction | 1,042 | 11.1x (via subsampling) |
| **Total** | **1,051** | **11.5x** |

**Throughput:** 0.95 Hz (vs 0.08 Hz baseline)

### Recommended Hardware

**Minimum:**
- CPU: 4 cores, 3+ GHz
- RAM: 16 GB
- Storage: 100 GB (for KITTI dataset)
- Throughput: ~0.5-1 Hz

**Recommended:**
- CPU: 8+ cores
- RAM: 32 GB
- GPU: NVIDIA RTX 3060 or better
- Storage: SSD for KITTI dataset
- Throughput: ~0.95 Hz (CPU-bound simplex construction)

**Optimal (Future):**
- CPU: 16+ cores
- RAM: 64 GB
- GPU: NVIDIA RTX 4090
- Storage: NVMe SSD
- Throughput: ~5-10 Hz (with incremental updates)

---

## Next Steps

### Learn More
- [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md): Complete optimization journey
- [LIDAR_PERFORMANCE_ANALYSIS.md](LIDAR_PERFORMANCE_ANALYSIS.md): LiDAR-specific benchmarks
- [FAST_LANDMARK_SELECTION.md](FAST_LANDMARK_SELECTION.md): Quantum-inspired methodology
- [src/paper_ready/tda_odometry/README.md](src/paper_ready/tda_odometry/README.md): Odometry system docs

### Example Scripts
```bash
# Train odometry model on KITTI
python scripts/train_tda_odometry.py --train-sequences 00 01 --val-sequences 06

# Evaluate on test sequences
python scripts/evaluate_tda_odometry.py --checkpoint checkpoints/best_model.pt --sequences 08 09 10

# Profile pipeline performance
python scripts/profile_tda_pipeline.py

# Benchmark landmark selection methods
python scripts/benchmark_landmark_selection.py
```

### Community
- GitHub: [JorgeLRW/whale](https://github.com/JorgeLRW/whale)
- Issues: [Report bugs or request features](https://github.com/JorgeLRW/whale/issues)
- PyPI: [whale-tda](https://pypi.org/project/whale-tda/)

---

**Version:** 0.3.0  
**Date:** October 14, 2025  
**License:** MIT
