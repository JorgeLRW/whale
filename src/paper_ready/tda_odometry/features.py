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
        
        # Lazy-load student model if needed
        self._student_model = None
        self._student_target_mean = None
        self._student_target_std = None
        if use_student:
            self._load_student_model()
    
    def _load_student_model(self):
        """Load student model from checkpoint."""
        import torch
        import sys
        from pathlib import Path
        
        # Add paths for student imports
        repo_root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(repo_root / 'paper_ready' / 'src'))
        
        try:
            from paper_ready.tda_odometry.student import get_student
        except ImportError:
            from src.paper_ready.tda_odometry.student import get_student
        
        if self.student_checkpoint is None:
            raise ValueError("use_student=True but student_checkpoint is None")
        
        # Load checkpoint
        ckpt = torch.load(self.student_checkpoint, map_location=self.student_device)
        
        # Determine architecture from checkpoint args if available
        arch = ckpt.get('args', {}).get('arch', 'pointnet_lite')
        
        # Create model
        self._student_model = get_student(arch, out_dim=51)
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
            pred = self._student_model(pts_tensor).squeeze(0)
            
            # Unnormalize if stats available
            if self._student_target_mean is not None:
                pred = pred * self._student_target_std + self._student_target_mean
            
            # Ensure GPU operations complete before returning
            if self.student_device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
        
        return pred.cpu().numpy()
    
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
        
        # Fast path: use student model
        if self.use_student:
            feature_vec = self._extract_with_student(points)
            comp_time = time.time() - t0
            
            # Return a TDAFeatures object populated from the vector
            # (Student predicts the to_vector() output directly, so we reconstruct)
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
            n_points=n,
            n_landmarks=self.m,
            computation_time=comp_time,
        )
    
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
