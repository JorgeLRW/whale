"""Hybrid TDA+Neural Odometry System.

This module implements a learned odometry estimator that uses topological and geometric
features extracted from point clouds via witness complex persistence.

Architecture:
- TDA Feature Extractor: Converts point clouds → persistence diagrams + geometric features
- Neural Odometry Model: Estimates relative pose (translation + rotation) between frames
- Training Pipeline: Supervised learning on KITTI odometry ground truth
- Evaluation: Compare against ICP, visual odometry, and learned baselines

Key insight: Persistent homology provides robust, rotation-invariant geometric features
that are complementary to raw point coordinates or learned CNN features.
"""

from .features import TDAFeatureExtractor
from .models import TDAOdometryNet
from .training import OdometryTrainer
from .evaluation import evaluate_sequence

__all__ = [
    "TDAFeatureExtractor",
    "TDAOdometryNet",
    "OdometryTrainer",
    "evaluate_sequence",
]
