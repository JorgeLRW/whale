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
    
    def extract(self, points: np.ndarray) -> TDAFeatures:
        """Extract TDA features from a point cloud.
        
        Args:
            points: (N, 3) array of point coordinates
            
        Returns:
            TDAFeatures object
        """
        t0 = time.time()
        n = points.shape[0]
        
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
