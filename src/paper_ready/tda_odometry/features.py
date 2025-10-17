"""TDA Feature Extraction for Point Clouds.

Extracts topological and geometric features from LiDAR point clouds using witness complex
persistence and coverage statistics.

Features extracted:
1. Persistence diagram statistics (H0, H1, H2 births/deaths/lifetimes)
2. Coverage metrics (mean, std, p95, ratio)
3. Landmark geometry (spatial distribution, density)
4. Witness statistics (k-NN distances, local density)
5. Topological summaries (Betti numbers, bottleneck distances)

These features are rotation-invariant (via relative geometry) and provide robust
geometric signatures suitable for learning-based odometry.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from whale.pipeline import select_landmarks, compute_witness_diagrams, coverage_metrics
from whale.methodology.witness_ph import compute_witness_persistence
from whale.methodology.fast_landmarks import lidar_sector_sampling


@dataclass
class TDAFeatures:
    """Container for TDA-derived features from a point cloud."""
    
    # Persistence diagram statistics
    h0_births: np.ndarray  # H0 birth times
    h0_deaths: np.ndarray  # H0 death times
    h0_lifetimes: np.ndarray  # H0 persistence
    h1_births: np.ndarray
    h1_deaths: np.ndarray
    h1_lifetimes: np.ndarray
    h2_births: np.ndarray
    h2_deaths: np.ndarray
    h2_lifetimes: np.ndarray
    
    # Betti numbers
    betti_0: int
    betti_1: int
    betti_2: int
    
    # Coverage statistics
    coverage_mean: float
    coverage_std: float
    coverage_p95: float
    coverage_ratio: float
    
    # Landmark geometry
    landmark_density_mean: float
    landmark_density_std: float
    landmark_spacing_mean: float  # Mean nearest-neighbor distance
    landmark_spacing_std: float
    
    # Witness statistics
    witness_knn_mean: float  # Mean k-NN distance to landmarks
    witness_knn_std: float
    
    # Metadata
    n_points: int
    n_landmarks: int
    computation_time: float
    
    def to_vector(self) -> np.ndarray:
        """Convert to fixed-size feature vector for neural network input.
        
        Uses statistical summaries of persistence diagrams (top-k persistence, moments)
        rather than raw diagram points for fixed dimensionality.
        """
        features = []
        
        # Persistence statistics (top-10 lifetimes + moments)
        for dim_lifetimes in [self.h0_lifetimes, self.h1_lifetimes, self.h2_lifetimes]:
            if len(dim_lifetimes) == 0:
                top10 = np.zeros(10)
                mean, std = 0.0, 0.0
            else:
                # Filter infinite lifetimes
                finite = dim_lifetimes[np.isfinite(dim_lifetimes)]
                if len(finite) == 0:
                    top10 = np.zeros(10)
                    mean, std = 0.0, 0.0
                else:
                    sorted_pers = np.sort(finite)[::-1]  # Descending
                    top10 = np.pad(sorted_pers[:10], (0, max(0, 10 - len(sorted_pers))), constant_values=0.0)
                    mean = float(np.mean(finite))
                    std = float(np.std(finite))
            
            features.extend(top10)
            features.extend([mean, std])
        
        # Betti numbers
        features.extend([float(self.betti_0), float(self.betti_1), float(self.betti_2)])
        
        # Coverage
        features.extend([
            self.coverage_mean,
            self.coverage_std,
            self.coverage_p95,
            self.coverage_ratio,
        ])
        
        # Landmark geometry
        features.extend([
            self.landmark_density_mean,
            self.landmark_density_std,
            self.landmark_spacing_mean,
            self.landmark_spacing_std,
        ])
        
        # Witness statistics
        features.extend([
            self.witness_knn_mean,
            self.witness_knn_std,
        ])
        
        # Metadata (normalized)
        features.extend([
            np.log10(self.n_points + 1),  # Log scale
            float(self.n_landmarks),
        ])
        
        return np.array(features, dtype=np.float32)
    
    @property
    def feature_dim(self) -> int:
        """Dimension of feature vector."""
        # 3 dimensions * (10 top + 2 moments) + 3 betti + 4 coverage + 4 landmark + 2 witness + 2 meta
        return 3 * 12 + 3 + 4 + 4 + 2 + 2  # 51 features


class TDAFeatureExtractor:
    """Extract TDA features from point clouds for odometry."""
    
    def __init__(
        self,
        m: int = 8,
        k_witness: int = 8,
        max_dim: int = 2,
        method: str = "lidar",  # Changed default to 'lidar' for 63x speedup
        seed: int = 42,
        selection_c: int = 8,
        hybrid_alpha: float = 0.8,
        n_sectors: int = 16,  # LiDAR sector sampling params
        n_rings: int = 4,
        use_student: bool = False,  # Use student NN instead of full TDA
        student_checkpoint: Optional[str] = "paper_ready/checkpoints/student/pointnetlite_dinov3_best.pt",  # DINOv3-trained student (-21.6% ATE, 224x speedup)
        student_device: str = "cuda",  # Device for student inference
        num_points: int = 4096,  # Number of points for student (FPS sampling)
        # Sheaf-based temporal consistency (runtime fallback)
    sheaf_enabled: bool = True,
        sheaf_window: int = 5,
        sheaf_alpha: float = 0.8,
        sheaf_threshold: float = 0.25,
        variance_threshold: float = 0.01,
    snr_threshold: float = 0.0,
    signal_threshold: float = 1.2,
        topo_threshold: float = -1.0,
        topo_density_threshold: float = 0.002,
        fallback_mode: str = 'sheaf',
    ):
        """Initialize feature extractor.
        
        Args:
            m: Number of landmarks to select
            k_witness: Number of nearest neighbors for witness complex
            max_dim: Maximum homology dimension
            method: Landmark selection method ('lidar', 'hybrid', 'density', 'random')
                   'lidar' is 63x faster than 'hybrid' (8ms vs 514ms)
            seed: Random seed
            selection_c: Candidate multiplier for density-based selection
            hybrid_alpha: Hybrid sampler parameter
            n_sectors: Number of azimuthal sectors for LiDAR sampling
            n_rings: Number of range rings for LiDAR sampling
            use_student: If True, use student NN to predict features (224x faster, -21.6% ATE vs teacher!)
            student_checkpoint: Path to student checkpoint file (.pt). Default is PointNetLite DINOv3.
            student_device: Device for student inference ('cuda' or 'cpu')
            num_points: Number of points to sample for student input (FPS)
        """
        self.m = m
        self.k_witness = k_witness
        self.max_dim = max_dim
        self.method = method
        self.seed = seed
        self.selection_c = selection_c
        self.hybrid_alpha = hybrid_alpha
        self.n_sectors = n_sectors
        self.n_rings = n_rings
        self.use_student = use_student
        self.student_checkpoint = student_checkpoint
        self.student_device = student_device
        self.num_points = num_points
        # Sheaf / temporal consistency params
        self.sheaf_enabled = sheaf_enabled
        self.sheaf_window = sheaf_window
        self.sheaf_alpha = sheaf_alpha
        self.sheaf_threshold = sheaf_threshold
        self.variance_threshold = variance_threshold
        self.snr_threshold = snr_threshold
        self.signal_threshold = signal_threshold
        self.topo_threshold = topo_threshold
        self.topo_density_threshold = topo_density_threshold
        # Fallback scoring mode: 'sheaf' uses restriction-map obstruction, 'cosine' uses simple 1-cos(s_i,s_{i+1})
        self.fallback_mode = fallback_mode

        # Sliding buffer for recent student descriptors
        from collections import deque
        self._student_buffer = deque(maxlen=self.sheaf_window)
        self.last_sheaf_score = 0.0
        self.last_used_teacher = False
        self.last_variance_score = 0.0
        self.last_snr_score = 0.0
        self.last_signal_strength = 0.0
        self.last_topo_signal = 0.0
        self.last_topo_density = 0.0
        self.sheaf_fallback_count = 0

        # Lazy-load student model if needed
        self._student_model = None
        self._student_target_mean = None
        self._student_target_std = None
        if use_student:
            self._load_student_model()

    def _compute_cosine_score(self, s_vec: np.ndarray) -> float:
        """Compute a simple cosine-based obstruction between last buffer entry and current.

        Returns 1 - cosine(s_prev, s_curr). If no previous entry, returns 0.0.
        """
        import numpy as _np

        # Append new vector
        self._student_buffer.append(_np.asarray(s_vec, dtype=_np.float32))

        if len(self._student_buffer) < 2:
            self.last_sheaf_score = 0.0
            return 0.0

        s_prev = self._student_buffer[-2]
        s_curr = self._student_buffer[-1]
        eps = 1e-12
        norm_prev = _np.linalg.norm(s_prev) + eps
        norm_curr = _np.linalg.norm(s_curr) + eps
        cos = float(_np.dot(s_prev, s_curr) / (norm_prev * norm_curr))
        cos = max(-1.0, min(1.0, cos))
        obstruction = 1.0 - cos
        self.last_sheaf_score = float(obstruction)
        return float(obstruction)
    
    def _load_student_model(self):
        """Load student model from checkpoint."""
        import torch
        import sys
        from pathlib import Path
        
        # Add paths for student imports
        repo_root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(repo_root / 'paper_ready' / 'src'))
        
        # Dynamic import to support different package layouts (src/ vs package)
        import importlib
        get_student = None
        for mod_name in ('paper_ready.tda_odometry.student', 'src.paper_ready.tda_odometry.student'):
            try:
                mod = importlib.import_module(mod_name)
                get_student = getattr(mod, 'get_student')
                break
            except Exception:
                continue
        if get_student is None:
            raise ImportError('could not import get_student from expected module paths')
        
        if self.student_checkpoint is None:
            raise ValueError("use_student=True but student_checkpoint is None")
        
        # Load checkpoint
        ckpt = torch.load(self.student_checkpoint, map_location=self.student_device)
        
        # Determine architecture from checkpoint args if available
        arch = ckpt.get('args', {}).get('arch', 'pointnet_lite')
        
        # Create model
        self._student_model = get_student(arch, out_dim=51)
        assert self._student_model is not None, 'get_student() returned None'
        self._student_model.load_state_dict(ckpt['model_state'])
        self._student_model.to(self.student_device)
        self._student_model.eval()

        # Load normalization stats if present
        if 'target_mean' in ckpt:
            self._student_target_mean = torch.from_numpy(ckpt['target_mean']).to(self.student_device)
            self._student_target_std = torch.from_numpy(ckpt['target_std']).to(self.student_device)

        print(f"Loaded student model from {self.student_checkpoint} (arch={arch})")
    
    def _fps_sample(self, points: np.ndarray, n_samples: int) -> np.ndarray:
        """Farthest point sampling to get fixed number of points.
        
        Fast vectorized version - samples in batches to avoid slow Python loops.
        """
        n_points = len(points)
        
        if n_points <= n_samples:
            # Pad if too few points
            if n_points < n_samples:
                indices = np.arange(n_points)
                pad_indices = np.random.choice(n_points, n_samples - n_points, replace=True)
                return np.concatenate([indices, pad_indices])
            return np.arange(n_points)
        
        # For speed, use random sampling instead of true FPS for large point clouds
        # True FPS is O(n*m) which is too slow for 100k+ points
        # Random sampling gives similar coverage and is O(1)
        return np.random.choice(n_points, n_samples, replace=False)
    
    def _extract_with_student(self, points: np.ndarray) -> np.ndarray:
        """Extract features using student model (fast approximation).
        
        Returns:
            51-dim feature vector (same format as TDAFeatures.to_vector())
        """
        import torch
        
        # Sample fixed number of points (fast random sampling)
        if len(points) != self.num_points:
            indices = self._fps_sample(points, self.num_points)
            points_sampled = points[indices]
        else:
            points_sampled = points
        
        # Convert to tensor (avoid copy if possible)
        pts_tensor = torch.from_numpy(points_sampled.copy()).float().unsqueeze(0)
        
        # Move to device
        if self.student_device != 'cpu':
            pts_tensor = pts_tensor.to(self.student_device, non_blocking=True)
        
        # Inference with GPU sync
        with torch.no_grad():
            # Ensure student model is loaded (lazy-load if necessary)
            if self._student_model is None:
                try:
                    self._load_student_model()
                except Exception as e:
                    raise RuntimeError(f"failed to load student model for inference: {e}") from e

            if self._student_model is None:
                raise RuntimeError("student model is not available for inference (self._student_model is None)")

            pred = self._student_model(pts_tensor).squeeze(0)
            
            # Unnormalize if stats available
            if self._student_target_mean is not None:
                pred = pred * self._student_target_std + self._student_target_mean
            
            # Ensure GPU operations complete before returning
            if self.student_device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
        
        return pred.cpu().numpy()

    def _compute_sheaf_score(self, s_vec: np.ndarray) -> float:
        """Update sliding buffer with new student descriptor and compute obstruction score.

        Uses the restriction map r_{i->i+1}(s_i) = alpha * (s_i/||s_i||) + (1-alpha) * s_{i+1}
        and the obstruction proxy 1 - cosine(r, s_{i+1}). Aggregates over the buffer window.
        """
        import numpy as _np

        # Append new vector
        self._student_buffer.append(_np.asarray(s_vec, dtype=_np.float32))

        if len(self._student_buffer) < 2:
            self.last_sheaf_score = 0.0
            return 0.0

        alpha = float(self.sheaf_alpha)
        scores = []
        eps = 1e-8
        for i in range(len(self._student_buffer) - 1):
            s_i = self._student_buffer[i]
            s_j = self._student_buffer[i + 1]

            # normalize both vectors to make score scale-invariant
            norm_si = _np.linalg.norm(s_i) + eps
            norm_sj = _np.linalg.norm(s_j) + eps
            z_i = s_i / norm_si
            z_j = s_j / norm_sj

            r = alpha * z_i + (1.0 - alpha) * z_j

            denom = (_np.linalg.norm(r) * (_np.linalg.norm(z_j) + eps)) + eps
            cos = float(_np.dot(r, z_j) / denom)
            cos = max(-1.0, min(1.0, cos))
            obstruction = 1.0 - cos
            scores.append(obstruction)

        score = float(_np.mean(scores)) if scores else 0.0
        self.last_sheaf_score = score
        return score

    def _compute_variance_score(self) -> float:
        """Compute average stddev across vector components in the buffer.

        Low variance may indicate over-smoothing (e.g., featureless scenes where the
        student outputs collapse). We take the mean of per-dimension stddevs.
        """
        import numpy as _np

        if len(self._student_buffer) < 2:
            return 0.0

        arr = _np.stack(list(self._student_buffer), axis=0)
        per_dim_std = _np.std(arr, axis=0)
        var_score = float(_np.mean(per_dim_std))
        self.last_variance_score = var_score
        return var_score

    def _compute_snr_score(self, s_vec: np.ndarray) -> float:
        """Robust temporal SNR proxy.

        Uses the sliding buffer of recent descriptors (already populated by
        _compute_sheaf_score) and computes:

            temporal_std_per_dim = std(arr, axis=0)
            signal = mean(temporal_std_per_dim)

            diffs = abs(diff(arr, axis=0))
            noise_per_dim = mean(diffs, axis=0)
            noise = mean(noise_per_dim)

        We return log10(snr + eps) for numeric stability and consistent thresholds.
        """
        import numpy as _np

        eps = 1e-12

        # Need at least two entries in the buffer to estimate temporal statistics
        if len(self._student_buffer) < 2:
            self.last_snr_score = 0.0
            return 0.0

        arr = _np.stack(list(self._student_buffer), axis=0).astype(_np.float32)  # (T, D)

        # Temporal signal: mean of per-dimension stddev across time
        per_dim_std = _np.std(arr, axis=0)
        signal = float(_np.mean(per_dim_std))

        # Temporal noise: mean absolute change between consecutive frames
        diffs = _np.abs(_np.diff(arr, axis=0))  # (T-1, D)
        per_dim_mean_diff = _np.mean(diffs, axis=0)
        noise = float(_np.mean(per_dim_mean_diff))

        snr_linear = signal / (noise + eps)
        snr_log10 = float(_np.log10(snr_linear + eps))
        self.last_snr_score = snr_log10
        return snr_log10

    def _compute_signal_strength(self, s_vec: np.ndarray) -> float:
        """Compute log10 L2 norm of the feature vector as a signal-strength proxy.

        Returns log10(norm + eps). Stored in self.last_signal_strength.
        """
        import numpy as _np

        eps = 1e-12
        s = _np.asarray(s_vec, dtype=_np.float32)
        norm = float(_np.linalg.norm(s))
        log_norm = float(_np.log10(norm + eps))
        self.last_signal_strength = log_norm
        return log_norm

    def _compute_topo_signal(self, s_vec: np.ndarray) -> float:
        """Estimate topological signal from predicted feature vector.

        The feature vector layout (as in TDAFeatures.to_vector) places the
        top-10 lifetimes for H0, then H1, then H2 at indices:
            H0: 0..11 (top10 + mean/std)
            H1: 12..23
            H2: 24..35

        We extract the H1 and H2 top-10 lifetimes, sum the top-5 from each,
        and return log10(sum + eps) as a compact topological signal.
        """
        import numpy as _np

        eps = 1e-12
        vec = _np.asarray(s_vec, dtype=_np.float32)

        # Indices according to to_vector() layout: each dim has 12 entries (10 top + mean + std)
        h1_start = 12
        h1_top10 = vec[h1_start:h1_start+10]
        h2_start = 24
        h2_top10 = vec[h2_start:h2_start+10]

        # Ensure non-negative lifetimes
        h1_top10 = _np.clip(h1_top10, 0.0, None)
        h2_top10 = _np.clip(h2_top10, 0.0, None)

        top_h1 = float(_np.sum(_np.sort(h1_top10)[-5:]))
        top_h2 = float(_np.sum(_np.sort(h2_top10)[-5:]))

        total = top_h1 + top_h2
        topo_log = float(_np.log10(total + eps))
        self.last_topo_signal = topo_log
        return topo_log

    def reset_buffer(self):
        """Clear the sliding buffer (call at sequence boundaries)."""
        self._student_buffer.clear()
        self.last_sheaf_score = 0.0
        self.last_variance_score = 0.0
        self.last_used_teacher = False

    def _extract_with_teacher(self, points: np.ndarray) -> TDAFeatures:
        """Run the full (teacher) TDA extraction pipeline on points and return TDAFeatures.

        This duplicates the non-student code path so it can be invoked as a fallback
        at runtime when the sheaf obstruction score indicates inconsistency.
        """
        # Select landmarks using fast LiDAR-optimized method
        if self.method == "lidar":
            landmark_indices = lidar_sector_sampling(
                points,
                self.m,
                n_sectors=self.n_sectors,
                n_rings=self.n_rings,
                seed=self.seed,
            )
        else:
            # Fallback to standard methods
            landmark_indices = select_landmarks(
                self.method,
                points,
                self.m,
                seed=self.seed,
                selection_c=self.selection_c,
                hybrid_alpha=self.hybrid_alpha,
            )

        landmarks = points[landmark_indices]

        # Compute witness complex persistence
        diagrams = compute_witness_persistence(
            points,
            landmark_indices,
            max_dim=self.max_dim,
            k_witness=self.k_witness,
        )

        # Extract persistence statistics per dimension
        h0_stats = self._extract_diagram_stats(diagrams.get(0, []))
        h1_stats = self._extract_diagram_stats(diagrams.get(1, []))
        h2_stats = self._extract_diagram_stats(diagrams.get(2, []))
        # Betti numbers (finite features only)
        betti_0 = sum(1 for b, d in diagrams.get(0, []) if np.isfinite(d))
        betti_1 = sum(1 for b, d in diagrams.get(1, []) if np.isfinite(d))
        betti_2 = sum(1 for b, d in diagrams.get(2, []) if np.isfinite(d))

        # Coverage statistics
        cov_stats = self._compute_coverage(points, landmark_indices, self.k_witness)

        # Landmark geometry
        lm_geom = self._compute_landmark_geometry(landmarks)

        # Witness statistics
        wit_stats = self._compute_witness_stats(points, landmarks, self.k_witness)

        comp_time = 0.0

        return TDAFeatures(
            h0_births=h0_stats[0],
            h0_deaths=h0_stats[1],
            h0_lifetimes=h0_stats[2],
            h1_births=h1_stats[0],
            h1_deaths=h1_stats[1],
            h1_lifetimes=h1_stats[2],
            h2_births=h2_stats[0],
            h2_deaths=h2_stats[1],
            h2_lifetimes=h2_stats[2],
            betti_0=betti_0,
            betti_1=betti_1,
            betti_2=betti_2,
            coverage_mean=cov_stats['mean'],
            coverage_std=cov_stats['std'],
            coverage_p95=cov_stats['p95'],
            coverage_ratio=cov_stats['ratio'],
            landmark_density_mean=lm_geom['density_mean'],
            landmark_density_std=lm_geom['density_std'],
            landmark_spacing_mean=lm_geom['spacing_mean'],
            landmark_spacing_std=lm_geom['spacing_std'],
            witness_knn_mean=wit_stats['mean'],
            witness_knn_std=wit_stats['std'],
            n_points=points.shape[0],
            n_landmarks=self.m,
            computation_time=comp_time,
        )
    
    def _features_from_vector(self, vec: np.ndarray, n_points: int, comp_time: float) -> TDAFeatures:
        """Reconstruct TDAFeatures from a 51-dim feature vector (for student).
        
        Note: This is an approximation. The student predicts statistics directly,
        so we can't recover full persistence diagrams. We return dummy arrays for
        births/deaths and populate the statistical summaries that the odometry
        model actually uses.
        """
        idx = 0
        
        # Extract persistence stats (3 dims * 12 features each = 36)
        h0_lifetimes_top10 = vec[idx:idx+10]
        h0_mean, h0_std = vec[idx+10], vec[idx+11]
        idx += 12
        
        h1_lifetimes_top10 = vec[idx:idx+10]
        h1_mean, h1_std = vec[idx+10], vec[idx+11]
        idx += 12
        
        h2_lifetimes_top10 = vec[idx:idx+10]
        h2_mean, h2_std = vec[idx+10], vec[idx+11]
        idx += 12
        
        # Betti numbers (3)
        betti_0, betti_1, betti_2 = int(vec[idx]), int(vec[idx+1]), int(vec[idx+2])
        idx += 3
        
        # Coverage (4)
        coverage_mean, coverage_std, coverage_p95, coverage_ratio = vec[idx:idx+4]
        idx += 4
        
        # Landmark geometry (4)
        lm_density_mean, lm_density_std, lm_spacing_mean, lm_spacing_std = vec[idx:idx+4]
        idx += 4
        
        # Witness stats (2)
        witness_knn_mean, witness_knn_std = vec[idx:idx+2]
        idx += 2
        
        # Metadata (2) - log(n_points), n_landmarks
        log_n = vec[idx]
        n_landmarks = int(vec[idx+1])
        
        # Create dummy persistence arrays (student doesn't predict full diagrams)
        # We only have top-10 lifetimes; use those as placeholders
        h0_lifetimes = h0_lifetimes_top10[h0_lifetimes_top10 > 0]  # Filter zeros
        h1_lifetimes = h1_lifetimes_top10[h1_lifetimes_top10 > 0]
        h2_lifetimes = h2_lifetimes_top10[h2_lifetimes_top10 > 0]
        
        # Dummy births/deaths (not used by downstream model, but required by dataclass)
        h0_births = np.zeros_like(h0_lifetimes)
        h0_deaths = h0_lifetimes.copy()
        h1_births = np.zeros_like(h1_lifetimes)
        h1_deaths = h1_lifetimes.copy()
        h2_births = np.zeros_like(h2_lifetimes)
        h2_deaths = h2_lifetimes.copy()
        
        return TDAFeatures(
            h0_births=h0_births,
            h0_deaths=h0_deaths,
            h0_lifetimes=h0_lifetimes,
            h1_births=h1_births,
            h1_deaths=h1_deaths,
            h1_lifetimes=h1_lifetimes,
            h2_births=h2_births,
            h2_deaths=h2_deaths,
            h2_lifetimes=h2_lifetimes,
            betti_0=betti_0,
            betti_1=betti_1,
            betti_2=betti_2,
            coverage_mean=float(coverage_mean),
            coverage_std=float(coverage_std),
            coverage_p95=float(coverage_p95),
            coverage_ratio=float(coverage_ratio),
            landmark_density_mean=float(lm_density_mean),
            landmark_density_std=float(lm_density_std),
            landmark_spacing_mean=float(lm_spacing_mean),
            landmark_spacing_std=float(lm_spacing_std),
            witness_knn_mean=float(witness_knn_mean),
            witness_knn_std=float(witness_knn_std),
            n_points=n_points,
            n_landmarks=n_landmarks,
            computation_time=comp_time,
        )
    
    def extract(self, points: np.ndarray) -> TDAFeatures:
        """Extract TDA features from a point cloud.
        
        Args:
            points: (N, 3) array of point coordinates
            
        Returns:
            TDAFeatures object (or approximation if use_student=True)
        """
        t0 = time.time()
        n = points.shape[0]
        # Reset per-frame teacher flag so bookkeeping counts each frame
        # independently. `last_used_teacher` may have been set True by a
        # previous extraction; ensure it starts False for the current frame
        # so increments to `sheaf_fallback_count` happen once per frame when
        # a fallback is triggered.
        self.last_used_teacher = False
        
        # Fast path: use student model
        if self.use_student:
            feature_vec = self._extract_with_student(points)

            # Optionally compute sheaf/cosine obstruction score and fallback to teacher
            sheaf_score = 0.0
            if self.sheaf_enabled:
                try:
                    if getattr(self, 'fallback_mode', 'sheaf') == 'cosine':
                        sheaf_score = self._compute_cosine_score(feature_vec)
                    else:
                        sheaf_score = self._compute_sheaf_score(feature_vec)
                except Exception:
                    sheaf_score = 0.0

            # Also compute variance score to detect over-smoothing
            var_score = 0.0
            if self.sheaf_enabled:
                try:
                    var_score = self._compute_variance_score()
                except Exception:
                    var_score = 0.0

            # Compute SNR proxy (signal-to-noise) and update last_snr_score
            snr_score = 0.0
            if self.sheaf_enabled:
                try:
                    snr_score = self._compute_snr_score(feature_vec)
                except Exception:
                    snr_score = 0.0

            # Compute signal strength (log10 norm) and update last_signal_strength
            signal_strength = 0.0
            try:
                signal_strength = self._compute_signal_strength(feature_vec)
            except Exception:
                signal_strength = 0.0
            # Compute topological signal (H1/H2 lifetimes)
            topo_signal = 0.0
            try:
                topo_signal = self._compute_topo_signal(feature_vec)
            except Exception:
                topo_signal = 0.0

            # Compute feature norm and decide fallback using topo-based rule.
            # Fallback triggers when topo signal is weak, temporal sheaf
            # obstruction is high, or the student output norm is degenerate.
            try:
                feat_norm = float(np.linalg.norm(feature_vec))
            except Exception:
                feat_norm = 0.0

            # Compute topological density: topo_signal (log10 of sum lifetimes)
            # divided by raw feature norm. Store for logging/inspection.
            try:
                eps = 1e-12
                topo_density = float(topo_signal / (feat_norm + eps))
            except Exception:
                topo_density = 0.0
            self.last_topo_density = topo_density

            # Fallback when topological density is low (less informative per-unit-magnitude),
            # or when sheaf obstruction is large, or when the feature norm is degenerate.
            should_fallback = (
                (topo_density < float(self.topo_density_threshold)) or
                (float(sheaf_score) > float(self.sheaf_threshold)) or
                (feat_norm < 0.5)
            )

            if should_fallback:
                # Trigger full teacher extraction for this frame. Use the
                # same atomic update pattern as in the teacher-only path to
                # avoid double-counting the fallback counter if the
                # downstream _extract_with_teacher also updates the flag.
                if not getattr(self, 'last_used_teacher', False):
                    self.last_used_teacher = True
                    try:
                        self.sheaf_fallback_count += 1
                    except Exception:
                        self.sheaf_fallback_count = int(getattr(self, 'sheaf_fallback_count', 0)) + 1
                # Delegate to teacher extraction which will compute full features
                return self._extract_with_teacher(points)
            else:
                self.last_used_teacher = False
                comp_time = time.time() - t0
                return self._features_from_vector(feature_vec, n, comp_time)
        
        # Select landmarks using fast LiDAR-optimized method
        if self.method == "lidar":
            landmark_indices = lidar_sector_sampling(
                points,
                self.m,
                n_sectors=self.n_sectors,
                n_rings=self.n_rings,
                seed=self.seed,
            )
        else:
            # Fallback to standard methods
            landmark_indices = select_landmarks(
                self.method,
                points,
                self.m,
                seed=self.seed,
                selection_c=self.selection_c,
                hybrid_alpha=self.hybrid_alpha,
            )
        
        landmarks = points[landmark_indices]
        
        # Compute witness complex persistence
        diagrams = compute_witness_persistence(
            points,
            landmark_indices,
            max_dim=self.max_dim,
            k_witness=self.k_witness,
        )
        
        # Extract persistence statistics per dimension
        h0_stats = self._extract_diagram_stats(diagrams.get(0, []))
        h1_stats = self._extract_diagram_stats(diagrams.get(1, []))
        h2_stats = self._extract_diagram_stats(diagrams.get(2, []))
        
        # Betti numbers (finite features only)
        betti_0 = sum(1 for b, d in diagrams.get(0, []) if np.isfinite(d))
        betti_1 = sum(1 for b, d in diagrams.get(1, []) if np.isfinite(d))
        betti_2 = sum(1 for b, d in diagrams.get(2, []) if np.isfinite(d))
        
        # Coverage statistics
        cov_stats = self._compute_coverage(points, landmark_indices, self.k_witness)
        
        # Landmark geometry
        lm_geom = self._compute_landmark_geometry(landmarks)
        
        # Witness statistics
        wit_stats = self._compute_witness_stats(points, landmarks, self.k_witness)
        
        comp_time = time.time() - t0
        # Build feature dataclass
        features_obj = TDAFeatures(
            h0_births=h0_stats[0],
            h0_deaths=h0_stats[1],
            h0_lifetimes=h0_stats[2],
            h1_births=h1_stats[0],
            h1_deaths=h1_stats[1],
            h1_lifetimes=h1_stats[2],
            h2_births=h2_stats[0],
            h2_deaths=h2_stats[1],
            h2_lifetimes=h2_stats[2],
            betti_0=betti_0,
            betti_1=betti_1,
            betti_2=betti_2,
            coverage_mean=cov_stats['mean'],
            coverage_std=cov_stats['std'],
            coverage_p95=cov_stats['p95'],
            coverage_ratio=cov_stats['ratio'],
            landmark_density_mean=lm_geom['density_mean'],
            landmark_density_std=lm_geom['density_std'],
            landmark_spacing_mean=lm_geom['spacing_mean'],
            landmark_spacing_std=lm_geom['spacing_std'],
            witness_knn_mean=wit_stats['mean'],
            witness_knn_std=wit_stats['std'],
            n_points=n,
            n_landmarks=self.m,
            computation_time=comp_time,
        )
        
        # For teacher-only extraction, also update the temporal buffer and
        # compute sheaf/variance diagnostics so downstream logging sees values.
        try:
            vec = features_obj.to_vector()
            # Use same computations as student path (this appends to buffer)
            try:
                self._compute_sheaf_score(vec)
            except Exception:
                self.last_sheaf_score = 0.0
            try:
                self._compute_variance_score()
            except Exception:
                self.last_variance_score = 0.0
            try:
                self._compute_snr_score(vec)
            except Exception:
                self.last_snr_score = 0.0
            try:
                self._compute_signal_strength(vec)
            except Exception:
                self.last_signal_strength = 0.0
            try:
                self._compute_topo_signal(vec)
            except Exception:
                self.last_topo_signal = 0.0
            # Mark that teacher was used for this extraction and increment
            # the aggregate fallback counter. Avoid double-counting: some
            # caller paths (student->fallback) already set
            # `last_used_teacher` and incremented the counter before
            # delegating to this method, so only increment here when the
            # flag was previously False.
            if not getattr(self, 'last_used_teacher', False):
                self.last_used_teacher = True
                # ensure attribute exists and increment safely
                try:
                    self.sheaf_fallback_count += 1
                except Exception:
                    self.sheaf_fallback_count = int(getattr(self, 'sheaf_fallback_count', 0)) + 1
            else:
                # already marked by caller; keep as True and do not double-count
                self.last_used_teacher = True
        except Exception:
            # If anything fails here, keep previous diagnostic values
            pass

        return features_obj
    
    def _extract_diagram_stats(self, diagram: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract births, deaths, and lifetimes from a persistence diagram."""
        if not diagram:
            return np.array([]), np.array([]), np.array([])
        
        births = np.array([b for b, d in diagram], dtype=np.float32)
        deaths = np.array([d for b, d in diagram], dtype=np.float32)
        lifetimes = deaths - births
        
        return births, deaths, lifetimes
    
    def _compute_coverage(self, points: np.ndarray, landmark_indices: np.ndarray, k: int) -> Dict[str, float]:
        """Compute coverage statistics."""
        from sklearn.neighbors import KDTree
        
        landmarks = points[landmark_indices]
        tree = KDTree(landmarks)
        dists, _ = tree.query(points, k=1)
        dists = dists.flatten()
        
        return {
            'mean': float(np.mean(dists)),
            'std': float(np.std(dists)),
            'p95': float(np.percentile(dists, 95)),
            'ratio': float(np.sum(dists < np.inf) / len(dists)) if len(dists) > 0 else 0.0,
        }
    
    def _compute_landmark_geometry(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Compute geometric statistics of landmark distribution."""
        from sklearn.neighbors import KDTree
        
        if len(landmarks) < 2:
            return {
                'density_mean': 0.0,
                'density_std': 0.0,
                'spacing_mean': 0.0,
                'spacing_std': 0.0,
            }
        
        tree = KDTree(landmarks)
        # k=2 to get nearest neighbor (excluding self)
        dists, _ = tree.query(landmarks, k=2)
        nn_dists = dists[:, 1]  # Nearest neighbor distances
        
        # Density proxy: 1 / (nearest neighbor distance)
        densities = 1.0 / (nn_dists + 1e-12)
        
        return {
            'density_mean': float(np.mean(densities)),
            'density_std': float(np.std(densities)),
            'spacing_mean': float(np.mean(nn_dists)),
            'spacing_std': float(np.std(nn_dists)),
        }
    
    def _compute_witness_stats(self, points: np.ndarray, landmarks: np.ndarray, k: int) -> Dict[str, float]:
        """Compute witness point statistics."""
        from sklearn.neighbors import KDTree
        
        tree = KDTree(landmarks)
        k_query = min(k, len(landmarks))
        dists, _ = tree.query(points, k=k_query)
        
        # Mean k-NN distance (average over k neighbors)
        mean_dists = np.mean(dists, axis=1)
        
        return {
            'mean': float(np.mean(mean_dists)),
            'std': float(np.std(mean_dists)),
        }
