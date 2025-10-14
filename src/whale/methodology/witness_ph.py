"""Witness complex persistence without relying on external TDA libraries.

This module now includes an optional GPU acceleration path for the dense
pairwise distance computation required when constructing the witness complex.
If PyTorch with CUDA support is available, the n×m distance matrix and k-NN
queries are computed on the GPU and only the top-k results are materialised on
the host. When CUDA is unavailable or a RuntimeError (e.g. OOM) occurs, the
implementation gracefully falls back to the original NumPy code path.
"""

from itertools import combinations
import math
from typing import Dict

import numpy as np

try:  # Optional GPU acceleration via PyTorch if available
    import torch

    _HAS_TORCH = torch.cuda.is_available()
except Exception:  # pragma: no cover - torch might not be installed
    torch = None
    _HAS_TORCH = False


def _compute_knn_on_gpu(X: np.ndarray, L: np.ndarray, k: int):
    """Compute k-NN indices/distances using torch on CUDA, if possible.

    Returns ``(indices, distances)`` as numpy arrays or ``None`` if the GPU
    path is unavailable (missing torch/torch.cuda) or raises a RuntimeError
    such as CUDA OOM. Only the top-k neighbours per point are copied back to
    host memory to minimise PCIe transfer overhead.
    """

    if not _HAS_TORCH:
        return None

    device = torch.device("cuda")

    try:
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        L_t = torch.from_numpy(np.asarray(L, dtype=np.float32)).to(device)

        # torch.cdist computes the full pairwise distance matrix efficiently on GPU.
        dists = torch.cdist(X_t, L_t)
        # Retrieve the smallest k distances (sorted ascending) and their indices.
        neigh_dists, neigh_idx = torch.topk(
            dists, k=k, dim=1, largest=False, sorted=True
        )

        # Move only the top-k subset back to CPU; this keeps transfer small.
        return neigh_idx.cpu().numpy(), neigh_dists.cpu().numpy()
    except RuntimeError:
        # Likely CUDA out-of-memory; fall back to CPU implementation.
        return None


def _compute_knn_indices(X, L, k):
    """Return (indices, distances) of the k nearest landmarks for each point."""

    gpu_result = _compute_knn_on_gpu(X, L, k)
    if gpu_result is not None:
        return gpu_result

    # CPU fallback mirrors the previous NumPy implementation.
    dists = np.linalg.norm(X[:, None, :] - L[None, :, :], axis=2)
    neigh_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
    neigh_dists = np.take_along_axis(dists, neigh_idx, axis=1)
    order = np.argsort(neigh_dists, axis=1)
    neigh_idx = np.take_along_axis(neigh_idx, order, axis=1)
    neigh_dists = np.take_along_axis(neigh_dists, order, axis=1)
    return neigh_idx, neigh_dists


def build_witness_complex(X, L_indices, max_dim=2, k_witness=8, max_filtration=None, max_witnesses=10000):
    """Build witness complex simplices and their min-filtration values.
    
    Args:
        X: Point cloud (n×d array)
        L_indices: Landmark indices
        max_dim: Maximum simplex dimension
        k_witness: Number of nearest landmarks per witness
        max_filtration: Optional filtration cutoff
        max_witnesses: Maximum number of witnesses to use (subsamples if n > max_witnesses).
                      Using fewer witnesses dramatically speeds up simplex construction
                      while preserving topological features. Default 10000.
    """
    L = X[L_indices]
    n, _ = X.shape
    m = L.shape[0]

    # Subsample witnesses if too many (major speedup with minimal accuracy loss)
    if n > max_witnesses:
        witness_indices = np.random.RandomState(42).choice(n, size=max_witnesses, replace=False)
        X_witnesses = X[witness_indices]
    else:
        witness_indices = np.arange(n)
        X_witnesses = X

    n_witnesses = len(witness_indices)

    k = min(k_witness, m)
    neigh_idx, neigh_dists = _compute_knn_indices(X_witnesses, L, k)

    simplex_filtration = {}

    for wi in range(n_witnesses):
        inds = neigh_idx[wi]
        ds = neigh_dists[wi]
        global_inds = [int(L_indices[int(i)]) for i in inds]
        for dim in range(0, max_dim + 1):
            if dim == 0:
                for v, g in enumerate(global_inds):
                    key = (g,)
                    val = 0.0
                    if (max_filtration is None) or (val <= max_filtration):
                        if key not in simplex_filtration or val < simplex_filtration[key]:
                            simplex_filtration[key] = val
                continue

            for comb in combinations(range(len(global_inds)), dim + 1):
                simp = tuple(sorted(global_inds[i] for i in comb))
                vals = [ds[i] for i in comb]
                val = float(max(vals))
                if (max_filtration is None) or (val <= max_filtration):
                    if simp not in simplex_filtration or val < simplex_filtration[simp]:
                        simplex_filtration[simp] = val

    simplices = []
    for simp, val in simplex_filtration.items():
        simplices.append((float(val), len(simp) - 1, tuple(simp)))

    simplices.sort(key=lambda t: (t[0], t[1]))
    return simplices


def compute_persistence_from_simplices(simplices, max_dim=2):
    """Compute persistence diagrams from an ordered list of simplices."""
    simplices_by_dim = {}
    for global_idx, (f, d, s) in enumerate(simplices):
        simplices_by_dim.setdefault(d, []).append((global_idx, f, s))

    index_in_dim = {}
    for d, lst in simplices_by_dim.items():
        for local_idx, (global_idx, f, s) in enumerate(lst):
            index_in_dim[(d, s)] = local_idx

    filt_by_global = [item[0] for item in simplices]
    dim_by_global = [item[1] for item in simplices]

    diagrams = {d: [] for d in range(0, max_dim + 1)}

    columns = []
    column_filts = []
    low_map = {}
    creators = {d: [] for d in range(0, max_dim + 1)}

    local_filt_map: Dict[tuple[int, int], float] = {}
    for d, lst in simplices_by_dim.items():
        for local_idx, (global_idx, f, s) in enumerate(lst):
            local_filt_map[(d, local_idx)] = f

    for global_idx, (filt, dim, s) in enumerate(simplices):
        if dim == 0:
            columns.append(0)
            column_filts.append(filt)
            creators[0].append(global_idx)
            continue

        faces = [tuple(sorted(face)) for face in combinations(s, dim)]
        col = 0
        for face in faces:
            key = (dim - 1, face)
            if key not in index_in_dim:
                continue
            local_idx = index_in_dim[key]
            col |= (1 << local_idx)

        while col:
            pivot = col.bit_length() - 1
            lm_key = (dim - 1, pivot)
            if lm_key not in low_map:
                break
            other_col_idx = low_map[lm_key]
            col ^= columns[other_col_idx]

        if col == 0:
            columns.append(0)
            column_filts.append(filt)
            creators[dim].append(global_idx)
        else:
            pivot = col.bit_length() - 1
            birth_f = local_filt_map[(dim - 1, pivot)]
            death_f = filt
            diagrams[dim - 1].append((birth_f, death_f))
            col_idx = len(columns)
            columns.append(col)
            column_filts.append(filt)
            low_map[(dim - 1, pivot)] = col_idx

    for d, globs in creators.items():
        for global_idx in globs:
            filt = simplices[global_idx][0]
            diagrams[d].append((filt, math.inf))

    return diagrams


def compute_witness_persistence(X, L_indices, max_dim=2, k_witness=8, max_filtration=None, max_witnesses=10000):
    """Compute witness complex persistence diagrams.
    
    Args:
        X: Point cloud (n×d array)
        L_indices: Landmark indices
        max_dim: Maximum homology dimension
        k_witness: Number of nearest landmarks per witness
        max_filtration: Optional filtration cutoff
        max_witnesses: Maximum witnesses to use (subsamples if needed for speed)
    """
    simplices = build_witness_complex(X, L_indices, max_dim=max_dim, k_witness=k_witness, 
                                     max_filtration=max_filtration, max_witnesses=max_witnesses)
    diagrams = compute_persistence_from_simplices(simplices, max_dim=max_dim)
    return diagrams


def bottleneck_distance(diagA, diagB, tol=1e-6, top_k=None):
    """Pure-Python bottleneck distance between two diagrams."""
    A = [tuple(map(float, p)) for p in diagA]
    B = [tuple(map(float, p)) for p in diagB]

    total_pts = len(A) + len(B)
    if total_pts > 2000:
        K = 200 if top_k is None else int(top_k)

        def top_k_fn(lst, k):
            return sorted(
                lst,
                key=lambda p: (p[1] - p[0]) if p[1] != math.inf else float('inf'),
                reverse=True,
            )[:k]

        A = top_k_fn(A, K)
        B = top_k_fn(B, K)

    def pers(p):
        return (p[1] - p[0]) if p[1] != math.inf else float('inf')

    if not A and not B:
        return 0.0

    def linf(p, q, inf_rep):
        pa = p[1] if p[1] != math.inf else inf_rep
        qa = q[1] if q[1] != math.inf else inf_rep
        return max(abs(p[0] - q[0]), abs(pa - qa))

    finite_vals = [v for p in (A + B) for v in p if v != math.inf]
    if finite_vals:
        span = max(finite_vals) - min(finite_vals)
        inf_rep = max(finite_vals) + span * 2.0
    else:
        span = 1.0
        inf_rep = 1.0

    lo = 0.0
    max_pers = 0.0
    for p in (A + B):
        if p[1] != math.inf:
            max_pers = max(max_pers, p[1] - p[0])
    hi = max(span * 2.0, max_pers * 1.5, 1e-12)

    def feasible(eps):
        A_big = [p for p in A if pers(p) > 2 * eps]
        B_big = [p for p in B if pers(p) > 2 * eps]

        if len(A_big) > len(B_big) + len([p for p in B if pers(p) / 2.0 <= eps + 1e-12]):
            return False
        if len(B_big) > len(A_big) + len([p for p in A if pers(p) / 2.0 <= eps + 1e-12]):
            return False

        nA = len(A_big)
        nB = len(B_big)
        N = nA + nB

        adj = [[] for _ in range(N)]

        for i, p in enumerate(A_big):
            for j, q in enumerate(B_big):
                if linf(p, q, inf_rep) <= eps + 1e-12:
                    adj[i].append(j)

        for i, p in enumerate(A_big):
            if pers(p) / 2.0 <= eps + 1e-12:
                for rd in range(nB, N):
                    adj[i].append(rd)

        for li in range(nA, N):
            for j, q in enumerate(B_big):
                if pers(q) / 2.0 <= eps + 1e-12:
                    adj[li].append(j)

        for li in range(nA, N):
            for rd in range(nB, N):
                adj[li].append(rd)

        matchR = [-1] * N

        def dfs_iter(u, seen):
            stack = [(u, iter(adj[u]))]
            parent_left = {u: None}
            parent_right: Dict[int, int] = {}
            while stack:
                curr, it = stack[-1]
                try:
                    v = next(it)
                except StopIteration:
                    stack.pop()
                    continue
                if seen[v]:
                    continue
                seen[v] = True
                parent_right[v] = curr
                if matchR[v] == -1:
                    left = curr
                    rv = v
                    while True:
                        matchR[rv] = left
                        prev_right = parent_left[left]
                        if prev_right is None:
                            break
                        left = parent_right[prev_right]
                        rv = prev_right
                    return True
                else:
                    next_left = matchR[v]
                    if next_left not in parent_left:
                        parent_left[next_left] = v
                        stack.append((next_left, iter(adj[next_left])))
            return False

        matched = 0
        for u in range(N):
            seen = [False] * N
            if dfs_iter(u, seen):
                matched += 1

        return matched == N

    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return hi
