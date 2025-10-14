"""Neural odometry model using TDA features.

Implements a learned odometry estimator that takes TDA features from consecutive
point cloud frames and predicts relative pose (translation + rotation).

Architecture options:
1. TDAOdometryNet: MLP on concatenated frame-pair TDA features
2. TDAOdometryRNN: LSTM/GRU for temporal sequence modeling
3. TDAOdometryTransformer: Attention-based architecture for long-range dependencies

Loss functions:
- Translation loss: MSE on (dx, dy, dz)
- Rotation loss: Geodesic distance on SO(3) or quaternion L2
- Combined loss: Weighted sum with tunab
le hyperparameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class TDAOdometryNet(nn.Module):
    """MLP-based odometry estimator using TDA features from frame pairs.
    
    Input: Concatenated TDA feature vectors from frame_t and frame_{t+1}
    Output: Relative pose (translation + rotation quaternion)
    """
    
    def __init__(
        self,
        feature_dim: int = 51,
        hidden_dims: Tuple[int, ...] = (256, 512, 512, 256),
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        """Initialize odometry network.
        
        Args:
            feature_dim: Dimension of TDA feature vector per frame
            hidden_dims: Tuple of hidden layer dimensions
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.input_dim = feature_dim * 2  # Concatenate frame_t and frame_{t+1}
        
        # Build encoder
        layers = []
        in_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output heads
        self.translation_head = nn.Linear(hidden_dims[-1], 3)  # (dx, dy, dz)
        self.rotation_head = nn.Linear(hidden_dims[-1], 4)  # Quaternion (qw, qx, qy, qz)
        
    def forward(self, features_t: torch.Tensor, features_t1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            features_t: (B, feature_dim) TDA features at time t
            features_t1: (B, feature_dim) TDA features at time t+1
            
        Returns:
            translation: (B, 3) relative translation
            rotation_quat: (B, 4) relative rotation as quaternion (normalized)
        """
        # Concatenate frame pair features
        x = torch.cat([features_t, features_t1], dim=1)  # (B, feature_dim*2)
        
        # Encode
        h = self.encoder(x)  # (B, hidden_dims[-1])
        
        # Predict translation
        translation = self.translation_head(h)  # (B, 3)
        
        # Predict rotation (quaternion, normalized)
        rotation_quat = self.rotation_head(h)  # (B, 4)
        rotation_quat = F.normalize(rotation_quat, p=2, dim=1)  # Normalize to unit quaternion
        
        return translation, rotation_quat


class TDAOdometryRNN(nn.Module):
    """LSTM-based odometry estimator for temporal sequence modeling.
    
    Input: Sequence of TDA feature vectors
    Output: Sequence of relative poses
    """
    
    def __init__(
        self,
        feature_dim: int = 51,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        """Initialize RNN odometry network.
        
        Args:
            feature_dim: Dimension of TDA feature vector per frame
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        # Output dimension after LSTM
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Output heads
        self.translation_head = nn.Linear(lstm_output_dim, 3)
        self.rotation_head = nn.Linear(lstm_output_dim, 4)
        
    def forward(self, features_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            features_seq: (B, T, feature_dim) sequence of TDA features
            
        Returns:
            translations: (B, T-1, 3) relative translations
            rotations_quat: (B, T-1, 4) relative rotations as quaternions
        """
        # Encode sequence
        h, _ = self.lstm(features_seq)  # (B, T, lstm_output_dim)
        
        # Predict pairwise odometry
        # Use consecutive frame pairs: (h_t, h_{t+1}) for each t
        h_t = h[:, :-1, :]  # (B, T-1, lstm_output_dim)
        h_t1 = h[:, 1:, :]  # (B, T-1, lstm_output_dim)
        
        # Concatenate or use difference
        delta_h = h_t1 - h_t  # (B, T-1, lstm_output_dim)
        
        # Predict
        translations = self.translation_head(delta_h)  # (B, T-1, 3)
        rotations_quat = self.rotation_head(delta_h)  # (B, T-1, 4)
        rotations_quat = F.normalize(rotations_quat, p=2, dim=2)
        
        return translations, rotations_quat


def quaternion_geodesic_loss(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """Geodesic distance loss on SO(3) via quaternion dot product.
    
    Args:
        q_pred: (B, 4) predicted quaternions (normalized)
        q_gt: (B, 4) ground truth quaternions (normalized)
        
    Returns:
        loss: Scalar geodesic distance loss
    """
    # Clamp dot product to avoid numerical issues with acos
    dot = torch.sum(q_pred * q_gt, dim=1)
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # Geodesic distance: arccos(|q_pred · q_gt|)
    # Take absolute value because q and -q represent the same rotation
    angle = torch.acos(torch.abs(dot))
    
    return torch.mean(angle)


def pose_loss(
    trans_pred: torch.Tensor,
    rot_pred: torch.Tensor,
    trans_gt: torch.Tensor,
    rot_gt: torch.Tensor,
    trans_weight: float = 1.0,
    rot_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined pose loss (translation + rotation).
    
    Args:
        trans_pred: (B, 3) predicted translations
        rot_pred: (B, 4) predicted rotations (quaternions)
        trans_gt: (B, 3) ground truth translations
        rot_gt: (B, 4) ground truth rotations (quaternions)
        trans_weight: Weight for translation loss
        rot_weight: Weight for rotation loss
        
    Returns:
        total_loss: Combined weighted loss
        trans_loss: Translation MSE loss
        rot_loss: Rotation geodesic loss
    """
    # Translation loss (MSE)
    trans_loss = F.mse_loss(trans_pred, trans_gt)
    
    # Rotation loss (geodesic distance)
    rot_loss = quaternion_geodesic_loss(rot_pred, rot_gt)
    
    # Combined loss
    total_loss = trans_weight * trans_loss + rot_weight * rot_loss
    
    return total_loss, trans_loss, rot_loss


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (w, x, y, z).
    
    Args:
        R: (3, 3) rotation matrix or (B, 3, 3) batch of rotation matrices
        
    Returns:
        q: (4,) quaternion or (B, 4) batch of quaternions
    """
    if R.ndim == 2:
        R = R[None, :, :]
        squeeze = True
    else:
        squeeze = False
    
    batch_size = R.shape[0]
    q = np.zeros((batch_size, 4), dtype=np.float32)
    
    for i in range(batch_size):
        Ri = R[i]
        trace = np.trace(Ri)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q[i, 0] = 0.25 / s  # w
            q[i, 1] = (Ri[2, 1] - Ri[1, 2]) * s  # x
            q[i, 2] = (Ri[0, 2] - Ri[2, 0]) * s  # y
            q[i, 3] = (Ri[1, 0] - Ri[0, 1]) * s  # z
        elif Ri[0, 0] > Ri[1, 1] and Ri[0, 0] > Ri[2, 2]:
            s = 2.0 * np.sqrt(1.0 + Ri[0, 0] - Ri[1, 1] - Ri[2, 2])
            q[i, 0] = (Ri[2, 1] - Ri[1, 2]) / s
            q[i, 1] = 0.25 * s
            q[i, 2] = (Ri[0, 1] + Ri[1, 0]) / s
            q[i, 3] = (Ri[0, 2] + Ri[2, 0]) / s
        elif Ri[1, 1] > Ri[2, 2]:
            s = 2.0 * np.sqrt(1.0 + Ri[1, 1] - Ri[0, 0] - Ri[2, 2])
            q[i, 0] = (Ri[0, 2] - Ri[2, 0]) / s
            q[i, 1] = (Ri[0, 1] + Ri[1, 0]) / s
            q[i, 2] = 0.25 * s
            q[i, 3] = (Ri[1, 2] + Ri[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + Ri[2, 2] - Ri[0, 0] - Ri[1, 1])
            q[i, 0] = (Ri[1, 0] - Ri[0, 1]) / s
            q[i, 1] = (Ri[0, 2] + Ri[2, 0]) / s
            q[i, 2] = (Ri[1, 2] + Ri[2, 1]) / s
            q[i, 3] = 0.25 * s
    
    # Normalize
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    
    if squeeze:
        q = q[0]
    
    return q
