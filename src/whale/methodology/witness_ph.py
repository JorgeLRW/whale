"""Witness complex persistence without relying on external TDA libraries."""

from itertools import combinations
import math
from typing import Dict


def build_witness_complex(X, L_indices, max_dim=2, k_witness=8, max_filtration=None):
    """Build witness complex simplices and their min-filtration values."""
    import numpy as np

    L = X[L_indices]
    n, _ = X.shape
    m = L.shape[0]

    dists = np.linalg.norm(X[:, None, :] - L[None, :, :], axis=2)
    k = min(k_witness, m)
    neigh_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
    neigh_dists = np.take_along_axis(dists, neigh_idx, axis=1)
    order = np.argsort(neigh_dists, axis=1)
    neigh_idx = np.take_along_axis(neigh_idx, order, axis=1)
    neigh_dists = np.take_along_axis(neigh_dists, order, axis=1)

    simplex_filtration = {}

    for wi in range(n):
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


def compute_witness_persistence(X, L_indices, max_dim=2, k_witness=8, max_filtration=None):
    simplices = build_witness_complex(X, L_indices, max_dim=max_dim, k_witness=k_witness, max_filtration=max_filtration)
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
