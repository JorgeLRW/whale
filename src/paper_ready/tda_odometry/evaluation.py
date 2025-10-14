"""Evaluation utilities for TDA-based odometry.

Compute standard odometry metrics on KITTI sequences and compare against baselines.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from .features import TDAFeatureExtractor
from .models import TDAOdometryNet
from .training import KITTIOdometryDataset


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to rotation matrix.
    
    Args:
        q: (4,) quaternion or (B, 4) batch of quaternions
        
    Returns:
        R: (3, 3) rotation matrix or (B, 3, 3) batch
    """
    if q.ndim == 1:
        q = q[None, :]
        squeeze = True
    else:
        squeeze = False
    
    batch_size = q.shape[0]
    R = np.zeros((batch_size, 3, 3), dtype=np.float32)
    
    for i in range(batch_size):
        w, x, y, z = q[i]
        
        R[i, 0, 0] = 1 - 2*y*y - 2*z*z
        R[i, 0, 1] = 2*x*y - 2*w*z
        R[i, 0, 2] = 2*x*z + 2*w*y
        
        R[i, 1, 0] = 2*x*y + 2*w*z
        R[i, 1, 1] = 1 - 2*x*x - 2*z*z
        R[i, 1, 2] = 2*y*z - 2*w*x
        
        R[i, 2, 0] = 2*x*z - 2*w*y
        R[i, 2, 1] = 2*y*z + 2*w*x
        R[i, 2, 2] = 1 - 2*x*x - 2*y*y
    
    if squeeze:
        R = R[0]
    
    return R


def compute_trajectory(
    relative_poses: List[Tuple[np.ndarray, np.ndarray]],
    init_pose: np.ndarray = None,
) -> np.ndarray:
    """Accumulate relative poses into absolute trajectory.
    
    Args:
        relative_poses: List of (translation, rotation_quat) tuples
        init_pose: Initial 4x4 pose (default: identity)
        
    Returns:
        trajectory: (N+1, 4, 4) array of absolute poses
    """
    if init_pose is None:
        init_pose = np.eye(4, dtype=np.float32)
    
    trajectory = [init_pose]
    current_pose = init_pose.copy()
    
    for trans, rot_quat in relative_poses:
        # Build relative transform
        T_rel = np.eye(4, dtype=np.float32)
        T_rel[:3, :3] = quaternion_to_rotation_matrix(rot_quat)
        T_rel[:3, 3] = trans
        
        # Accumulate
        current_pose = current_pose @ T_rel
        trajectory.append(current_pose.copy())
    
    return np.array(trajectory)


def compute_ate(predicted_poses: np.ndarray, gt_poses: np.ndarray) -> float:
    """Compute Absolute Trajectory Error (ATE).
    
    Args:
        predicted_poses: (N, 4, 4) predicted poses
        gt_poses: (N, 4, 4) ground truth poses
        
    Returns:
        ate_rmse: RMSE of translational errors
    """
    n = min(len(predicted_poses), len(gt_poses))
    
    errors = []
    for i in range(n):
        pred_trans = predicted_poses[i, :3, 3]
        gt_trans = gt_poses[i, :3, 3]
        error = np.linalg.norm(pred_trans - gt_trans)
        errors.append(error)
    
    ate_rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    return ate_rmse


def compute_rpe(predicted_poses: np.ndarray, gt_poses: np.ndarray, delta: int = 1) -> Tuple[float, float]:
    """Compute Relative Pose Error (RPE) for translation and rotation.
    
    Args:
        predicted_poses: (N, 4, 4) predicted poses
        gt_poses: (N, 4, 4) ground truth poses
        delta: Frame delta for relative pose (default: 1 = consecutive frames)
        
    Returns:
        rpe_trans: RMSE of relative translation errors
        rpe_rot: RMSE of relative rotation errors (degrees)
    """
    n = min(len(predicted_poses), len(gt_poses))
    
    trans_errors = []
    rot_errors = []
    
    for i in range(n - delta):
        # Predicted relative pose
        T_pred_i = predicted_poses[i]
        T_pred_j = predicted_poses[i + delta]
        T_pred_rel = np.linalg.inv(T_pred_i) @ T_pred_j
        
        # Ground truth relative pose
        T_gt_i = gt_poses[i]
        T_gt_j = gt_poses[i + delta]
        T_gt_rel = np.linalg.inv(T_gt_i) @ T_gt_j
        
        # Error transform
        T_error = np.linalg.inv(T_gt_rel) @ T_pred_rel
        
        # Translation error
        trans_error = np.linalg.norm(T_error[:3, 3])
        trans_errors.append(trans_error)
        
        # Rotation error (angle of rotation matrix)
        R_error = T_error[:3, :3]
        trace = np.trace(R_error)
        angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
        rot_errors.append(np.degrees(angle))
    
    rpe_trans = np.sqrt(np.mean(np.array(trans_errors) ** 2))
    rpe_rot = np.sqrt(np.mean(np.array(rot_errors) ** 2))
    
    return rpe_trans, rpe_rot


def evaluate_sequence(
    model: TDAOdometryNet,
    sequence: str,
    data_root: Path,
    feature_extractor: TDAFeatureExtractor,
    max_points: int = 200000,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> Dict[str, float]:
    """Evaluate model on a single KITTI sequence.
    
    Args:
        model: Trained odometry model
        sequence: Sequence ID (e.g., '00')
        data_root: Path to KITTI data root
        feature_extractor: TDA feature extractor
        max_points: Max points per frame
        device: Device to run on
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    model.to(device)
    
    # Create dataset
    dataset = KITTIOdometryDataset(
        data_root=data_root,
        sequences=[sequence],
        feature_extractor=feature_extractor,
        max_points=max_points,
        stride=1,
        cache_features=True,
    )
    
    # Predict relative poses
    relative_poses_pred = []
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f'Evaluating seq {sequence}'):
            sample = dataset[i]
            
            features_t = sample['features_t'].unsqueeze(0).to(device)
            features_t1 = sample['features_t1'].unsqueeze(0).to(device)
            
            trans_pred, rot_pred = model(features_t, features_t1)
            
            trans_pred = trans_pred.cpu().numpy()[0]
            rot_pred = rot_pred.cpu().numpy()[0]
            
            relative_poses_pred.append((trans_pred, rot_pred))
    
    # Compute predicted trajectory
    gt_poses = dataset.poses[sequence]
    init_pose = gt_poses[0]
    pred_trajectory = compute_trajectory(relative_poses_pred, init_pose)
    
    # Compute metrics
    ate = compute_ate(pred_trajectory, gt_poses)
    rpe_trans, rpe_rot = compute_rpe(pred_trajectory, gt_poses, delta=1)
    
    return {
        'ate_rmse': ate,
        'rpe_trans_rmse': rpe_trans,
        'rpe_rot_rmse': rpe_rot,
        'sequence': sequence,
        'n_frames': len(gt_poses),
    }


def evaluate_all_sequences(
    model: TDAOdometryNet,
    sequences: List[str],
    data_root: Path,
    feature_extractor: TDAFeatureExtractor,
    max_points: int = 200000,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> Dict[str, Dict[str, float]]:
    """Evaluate model on multiple KITTI sequences.
    
    Args:
        model: Trained odometry model
        sequences: List of sequence IDs
        data_root: Path to KITTI data root
        feature_extractor: TDA feature extractor
        max_points: Max points per frame
        device: Device to run on
        
    Returns:
        results: Dictionary mapping sequence ID to metrics
    """
    results = {}
    
    for seq in sequences:
        print(f'\n=== Evaluating sequence {seq} ===')
        metrics = evaluate_sequence(
            model=model,
            sequence=seq,
            data_root=data_root,
            feature_extractor=feature_extractor,
            max_points=max_points,
            device=device,
        )
        results[seq] = metrics
        
        print(f"Sequence {seq}: ATE={metrics['ate_rmse']:.3f}m, "
              f"RPE_trans={metrics['rpe_trans_rmse']:.3f}m, "
              f"RPE_rot={metrics['rpe_rot_rmse']:.3f}°")
    
    # Compute aggregate statistics
    ate_values = [metrics['ate_rmse'] for metrics in results.values()]
    rpe_trans_values = [metrics['rpe_trans_rmse'] for metrics in results.values()]
    rpe_rot_values = [metrics['rpe_rot_rmse'] for metrics in results.values()]
    
    print(f'\n=== Aggregate Results ===')
    print(f'ATE:      mean={np.mean(ate_values):.3f}m, std={np.std(ate_values):.3f}m')
    print(f'RPE_trans: mean={np.mean(rpe_trans_values):.3f}m, std={np.std(rpe_trans_values):.3f}m')
    print(f'RPE_rot:   mean={np.mean(rpe_rot_values):.3f}°, std={np.std(rpe_rot_values):.3f}°')
    
    results['aggregate'] = {
        'ate_mean': np.mean(ate_values),
        'ate_std': np.std(ate_values),
        'rpe_trans_mean': np.mean(rpe_trans_values),
        'rpe_trans_std': np.std(rpe_trans_values),
        'rpe_rot_mean': np.mean(rpe_rot_values),
        'rpe_rot_std': np.std(rpe_rot_values),
    }
    
    return results
