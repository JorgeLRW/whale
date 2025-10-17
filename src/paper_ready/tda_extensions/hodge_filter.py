"""Small Hodge Laplacian utilities for topological signal processing.

This module computes the graph (0-form) Laplacian and performs a simple
low-pass filter in the Laplacian eigenbasis. It's intentionally minimal
and uses numpy so it runs quickly on small graphs.
"""
import numpy as np
from typing import Tuple


def graph_laplacian(adj: np.ndarray) -> np.ndarray:
    """Compute symmetric (combinatorial) graph Laplacian L = D - (A+A^T)/2."""
    sym = 0.5 * (adj + adj.T)
    deg = np.sum(sym, axis=1)
    L = np.diag(deg) - sym
    return L


def low_pass_filter_signal(signal: np.ndarray, L: np.ndarray, keep_frac: float = 0.2) -> np.ndarray:
    """Project signal onto Laplacian eigenbasis and keep low-frequency components.

    Args:
        signal: (N,) or (N, C) array of signals on nodes.
        L: (N,N) Laplacian matrix.
        keep_frac: fraction of lowest-frequency eigenvectors to keep.
    Returns:
        filtered signal with same shape as input.
    """
    w, V = np.linalg.eigh(L)
    idx = np.argsort(w)
    k = max(1, int(np.ceil(keep_frac * len(w))))
    keep = idx[:k]
    coeffs = V[:, keep].T @ signal
    recon = V[:, keep] @ coeffs
    return recon
