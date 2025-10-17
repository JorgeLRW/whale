"""Prototype utilities for directed / temporal witness graphs.

These functions are intentionally small, dependency-free, and meant for
fast experiments. They produce a directed adjacency list representing
temporal witness relations between landmarks across consecutive frames.
"""
from typing import List, Tuple
import numpy as np


def build_temporal_witness_graph(frames: List[np.ndarray], m_landmarks: int = 16, k_nn: int = 4) -> Tuple[List[Tuple[int,int]], np.ndarray]:
    """Build a directed temporal witness graph from a list of point clouds.

    Args:
        frames: list of (N_i, D) point clouds for each time step.
        m_landmarks: number of landmarks to sample per frame (random subset).
        k_nn: number of nearest edges from each landmark at time t to landmarks at t+1.

    Returns:
        edges: list of directed edges (u,v) where u and v are global landmark ids.
        landmarks: array of shape (M, D) with concatenated landmarks for all frames.

    Notes:
        - Landmark ids are laid out as frame_index * m_landmarks + local_index.
        - Edges only go from frame t -> t+1 (temporal direction enforced).
    """
    landmarks = []
    for pts in frames:
        n = pts.shape[0]
        if n <= m_landmarks:
            idx = np.arange(n)
        else:
            idx = np.random.choice(n, m_landmarks, replace=False)
        landmarks.append(pts[idx])

    landmarks = np.vstack(landmarks)
    M, D = landmarks.shape
    edges = []

    # connect landmarks from frame t to t+1 using nearest neighbors
    frames_count = len(frames)
    for t in range(frames_count - 1):
        base_t = t * m_landmarks
        base_tp = (t + 1) * m_landmarks
        A = landmarks[base_t:base_t + m_landmarks]  # (m, D)
        B = landmarks[base_tp:base_tp + m_landmarks]
        # pairwise distances
        d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
        for i in range(A.shape[0]):
            nn = np.argsort(d[i])[:k_nn]
            for j in nn:
                edges.append((base_t + i, base_tp + j))

    return edges, landmarks


def edges_to_adjacency(edges: List[Tuple[int,int]], num_nodes: int) -> np.ndarray:
    """Return adjacency matrix (directed) for a list of edges."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.int32)
    for u, v in edges:
        A[u, v] += 1
    return A
