"""Training pipeline for TDA-based odometry.

Implements supervised training on KITTI odometry sequences with ground truth poses.
Includes data loading, augmentation, training loop, and checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from tqdm import tqdm

from .features import TDAFeatureExtractor, TDAFeatures
from .models import TDAOdometryNet, pose_loss, rotation_matrix_to_quaternion
from paper_ready.kitti_utils import load_velodyne_bin


class KITTIOdometryDataset(Dataset):
    """KITTI Odometry dataset for TDA-based learning.
    
    Loads consecutive frame pairs and their ground truth relative poses.
    """
    
    def __init__(
        self,
        data_root: Path,
        sequences: List[str],
        feature_extractor: TDAFeatureExtractor,
        max_points: int = 200000,
        stride: int = 1,
        cache_features: bool = True,
    ):
        """Initialize dataset.
        
        Args:
            data_root: Path to KITTI data root (e.g., paper_ready/data/lidar)
            sequences: List of sequence IDs (e.g., ['00', '01'])
            feature_extractor: TDA feature extractor
            max_points: Maximum number of points per frame (downsample if exceeded)
            stride: Frame stride (1 = consecutive, 2 = every other frame, etc.)
            cache_features: Whether to cache extracted TDA features
        """
        self.data_root = Path(data_root)
        self.sequences = sequences
        self.feature_extractor = feature_extractor
        self.max_points = max_points
        self.stride = stride
        self.cache_features = cache_features
        
        # Build frame pair index
        self.frame_pairs = []
        self.poses = {}
        
        for seq in sequences:
            seq_dir = self.data_root / 'sequences' / seq
            velo_dir = seq_dir / 'velodyne'
            pose_file = self.data_root / 'poses' / f'{seq}.txt'
            
            if not velo_dir.exists():
                print(f'Warning: {velo_dir} not found, skipping sequence {seq}')
                continue
            
            if not pose_file.exists():
                print(f'Warning: {pose_file} not found, skipping sequence {seq}')
                continue
            
            # Load poses
            seq_poses = self._load_poses(pose_file)
            self.poses[seq] = seq_poses
            
            # Build frame pairs
            frame_files = sorted(velo_dir.glob('*.bin'))
            for i in range(0, len(frame_files) - stride, stride):
                if i + stride < len(seq_poses):
                    self.frame_pairs.append((seq, i, i + stride))
        
        print(f'Loaded {len(self.frame_pairs)} frame pairs from {len(sequences)} sequences')
        
        # Feature cache
        self._feature_cache = {} if cache_features else None
    
    def _load_poses(self, pose_file: Path) -> np.ndarray:
        """Load ground truth poses from KITTI format (3x4 matrices per line)."""
        poses = []
        with open(pose_file, 'r') as f:
            for line in f:
                vals = [float(x) for x in line.strip().split()]
                if len(vals) == 12:
                    # Reshape to 3x4 matrix
                    T = np.array(vals, dtype=np.float32).reshape(3, 4)
                    # Convert to 4x4 homogeneous
                    T_hom = np.eye(4, dtype=np.float32)
                    T_hom[:3, :] = T
                    poses.append(T_hom)
        return np.array(poses)
    
    def _compute_relative_pose(self, T_world_t: np.ndarray, T_world_t1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute relative pose from t to t+1.
        
        Args:
            T_world_t: 4x4 world pose at time t
            T_world_t1: 4x4 world pose at time t+1
            
        Returns:
            translation: (3,) relative translation
            rotation_quat: (4,) relative rotation as quaternion (w, x, y, z)
        """
        # Relative transform: T_t_t1 = T_t^{-1} * T_t1
        T_t_world = np.linalg.inv(T_world_t)
        T_t_t1 = T_t_world @ T_world_t1
        
        translation = T_t_t1[:3, 3]
        rotation_mat = T_t_t1[:3, :3]
        rotation_quat = rotation_matrix_to_quaternion(rotation_mat)
        
        return translation, rotation_quat
    
    def _extract_features(self, seq: str, frame_idx: int) -> TDAFeatures:
        """Extract TDA features for a frame (with caching)."""
        cache_key = (seq, frame_idx)
        
        if self._feature_cache is not None and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        # Load point cloud
        velo_file = self.data_root / 'sequences' / seq / 'velodyne' / f'{frame_idx:06d}.bin'
        points = load_velodyne_bin(velo_file)
        
        # Downsample if needed
        if len(points) > self.max_points:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(points), size=self.max_points, replace=False)
            points = points[indices]
        
        # Extract features
        features = self.feature_extractor.extract(points)
        
        if self._feature_cache is not None:
            self._feature_cache[cache_key] = features
        
        return features
    
    def __len__(self) -> int:
        return len(self.frame_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample (frame pair + ground truth relative pose)."""
        seq, frame_t, frame_t1 = self.frame_pairs[idx]
        
        # Extract TDA features
        features_t = self._extract_features(seq, frame_t)
        features_t1 = self._extract_features(seq, frame_t1)
        
        # Get ground truth poses
        T_world_t = self.poses[seq][frame_t]
        T_world_t1 = self.poses[seq][frame_t1]
        
        # Compute relative pose
        translation, rotation_quat = self._compute_relative_pose(T_world_t, T_world_t1)
        
        return {
            'features_t': torch.from_numpy(features_t.to_vector()),
            'features_t1': torch.from_numpy(features_t1.to_vector()),
            'translation': torch.from_numpy(translation),
            'rotation_quat': torch.from_numpy(rotation_quat),
            'seq': seq,
            'frame_t': frame_t,
            'frame_t1': frame_t1,
        }


class OdometryTrainer:
    """Trainer for TDA-based odometry models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr: float = 1e-4,
        trans_weight: float = 1.0,
        rot_weight: float = 1.0,
    ):
        """Initialize trainer.
        
        Args:
            model: Odometry model (e.g., TDAOdometryNet)
            device: Device to train on
            lr: Learning rate
            trans_weight: Weight for translation loss
            rot_weight: Weight for rotation loss
        """
        self.model = model.to(device)
        self.device = device
        self.trans_weight = trans_weight
        self.rot_weight = rot_weight
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.history = {
            'train_loss': [],
            'train_trans_loss': [],
            'train_rot_loss': [],
            'val_loss': [],
            'val_trans_loss': [],
            'val_rot_loss': [],
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_trans_loss = 0.0
        epoch_rot_loss = 0.0
        
        for batch in tqdm(dataloader, desc='Training'):
            features_t = batch['features_t'].to(self.device)
            features_t1 = batch['features_t1'].to(self.device)
            trans_gt = batch['translation'].to(self.device)
            rot_gt = batch['rotation_quat'].to(self.device)
            
            # Forward pass
            trans_pred, rot_pred = self.model(features_t, features_t1)
            
            # Compute loss
            loss, trans_loss, rot_loss = pose_loss(
                trans_pred, rot_pred, trans_gt, rot_gt,
                trans_weight=self.trans_weight,
                rot_weight=self.rot_weight,
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_trans_loss += trans_loss.item()
            epoch_rot_loss += rot_loss.item()
        
        n_batches = len(dataloader)
        return {
            'loss': epoch_loss / n_batches,
            'trans_loss': epoch_trans_loss / n_batches,
            'rot_loss': epoch_rot_loss / n_batches,
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        
        val_loss = 0.0
        val_trans_loss = 0.0
        val_rot_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                features_t = batch['features_t'].to(self.device)
                features_t1 = batch['features_t1'].to(self.device)
                trans_gt = batch['translation'].to(self.device)
                rot_gt = batch['rotation_quat'].to(self.device)
                
                # Forward pass
                trans_pred, rot_pred = self.model(features_t, features_t1)
                
                # Compute loss
                loss, trans_loss, rot_loss = pose_loss(
                    trans_pred, rot_pred, trans_gt, rot_gt,
                    trans_weight=self.trans_weight,
                    rot_weight=self.rot_weight,
                )
                
                # Accumulate metrics
                val_loss += loss.item()
                val_trans_loss += trans_loss.item()
                val_rot_loss += rot_loss.item()
        
        n_batches = len(dataloader)
        return {
            'loss': val_loss / n_batches,
            'trans_loss': val_trans_loss / n_batches,
            'rot_loss': val_rot_loss / n_batches,
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_dir: Optional[Path] = None,
    ):
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints (optional)
        """
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f'\n=== Epoch {epoch+1}/{epochs} ===')
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Trans: {train_metrics['trans_loss']:.4f}, "
                  f"Rot: {train_metrics['rot_loss']:.4f}")
            
            # Validate
            val_metrics = self.validate(val_loader)
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Trans: {val_metrics['trans_loss']:.4f}, "
                  f"Rot: {val_metrics['rot_loss']:.4f}")
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_trans_loss'].append(train_metrics['trans_loss'])
            self.history['train_rot_loss'].append(train_metrics['rot_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_trans_loss'].append(val_metrics['trans_loss'])
            self.history['val_rot_loss'].append(val_metrics['rot_loss'])
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Save checkpoint
            if checkpoint_dir is not None:
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    checkpoint_path = checkpoint_dir / 'best_model.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_metrics['loss'],
                    }, checkpoint_path)
                    print(f'Saved best model to {checkpoint_path}')
                
                # Save latest
                latest_path = checkpoint_dir / 'latest_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                }, latest_path)
