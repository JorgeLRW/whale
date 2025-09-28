import numpy as np
from sklearn.neighbors import KDTree
import time
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Union

try:
    from typing import Literal
except ImportError:  # Python <3.8 fallback
    from typing_extensions import Literal  # type: ignore

# Optional Numba JIT for the MaxMin inner-loop update
try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False
    njit = None  # type: ignore[assignment]

# Optional ANN backends
HAS_HNSW = False
HAS_FAISS = False
try:
    import hnswlib
    HAS_HNSW = True
except Exception:
    HAS_HNSW = False
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


class KDTreeIndex:
    def __init__(self, X: np.ndarray):
        self.tree = KDTree(X)

    def query(self, q: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        d, idx = self.tree.query(np.atleast_2d(q), k=k)
        return d[0], idx[0]


if HAS_HNSW:
    class HNSWIndex:
        def __init__(self, X: np.ndarray, space: Literal['l2', 'ip', 'cosine'] = 'l2'):
            import hnswlib
            d = X.shape[1]
            self.p = hnswlib.Index(space=space, dim=d)
            self.p.init_index(max_elements=X.shape[0], ef_construction=200, M=16)
            self.p.add_items(X.astype('float32'))
            try:
                self.p.set_ef(200)
            except Exception:
                pass

        def query(self, q, k=1):
            labels, dists = self.p.knn_query(np.atleast_2d(q).astype('float32'), k=k)
            return dists[0].tolist(), labels[0].tolist()

else:
    HNSWIndex = None  # type: ignore[assignment]


def compute_kth_nn_radii(X: np.ndarray, k: int = 10, leaf_size: int = 40, ann_method: Optional[str] = None) -> np.ndarray:
    """Return the distance to the k-th nearest neighbor for each point."""
    n, d = X.shape
    k_q = min(k + 1, n)
    if ann_method is None:
        if HAS_FAISS and n >= 5000:
            ann_method = 'faiss'
        elif HAS_HNSW:
            ann_method = 'hnsw'
        else:
            ann_method = 'kdtree'
    if ann_method == 'hnsw' and HAS_HNSW:
        idx = HNSWIndex(X)
        radii = np.empty(n)
        for i in range(n):
            dists, ids = idx.query(X[i], k=k_q)
            radii[i] = dists[-1]
        return radii
    elif ann_method == 'faiss' and HAS_FAISS:
        import faiss
        xb = X.astype('float32')
        try:
            index = faiss.IndexFlatL2(d)
            index.add(xb)  # type: ignore[misc]
            D2, I = index.search(xb, k_q)  # type: ignore[misc]
            D2 = np.maximum(D2, 0.0)
            radii = np.sqrt(D2[:, -1])
            return radii
        except Exception:
            xb2 = X.astype('float32')
            index = faiss.IndexFlatL2(d)
            index.add(xb2)  # type: ignore[misc]
            radii = np.empty(n, dtype=float)
            for i in range(n):
                D2, I = index.search(xb2[i:i+1], k_q)  # type: ignore[misc]
                radii[i] = float(np.sqrt(max(0.0, D2[0, -1])))
            return radii
    else:
        tree = KDTree(X, leaf_size=leaf_size)
        dists, idxs = tree.query(X, k=k_q)
        radii = dists[:, -1]
        return radii

def density_proxy_from_radii(radii: np.ndarray, d: int) -> np.ndarray:
    eps = 1e-12
    rho = 1.0 / (np.power(radii + eps, float(d)) + eps)
    return rho

def sample_candidates_by_importance(n: int, weights: np.ndarray, C: int, rng: np.random.RandomState) -> np.ndarray:
    probs = weights / (weights.sum() + 1e-16)
    if C >= n:
        return np.arange(n)
    return rng.choice(n, size=C, replace=False, p=probs)

def pairwise_distances(S: np.ndarray) -> np.ndarray:
    XX = np.sum(S * S, axis=1)
    D2 = XX[:, None] + XX[None, :] - 2.0 * (S @ S.T)
    D2 = np.maximum(D2, 0.0)
    return np.sqrt(D2)

def density_weighted_maxmin(S: np.ndarray, rho_S: np.ndarray, m: int, alpha: float = 0.8, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    C = S.shape[0]
    if rng is None:
        rng = np.random.RandomState(42)

    if m >= C:
        return np.arange(C)

    seed = int(np.argmax((1.0 / (rho_S + 1e-12)) ** alpha))
    selected = [seed]

    diff = S - S[seed]
    d_s = np.linalg.norm(diff, axis=1)

    if HAS_NUMBA and njit is not None:
        @njit
        def _update_dists_numba(S_local, sel_idx, d_s_local):
            CL, D = S_local.shape
            for i in range(CL):
                sdist = 0.0
                for j in range(D):
                    dv = S_local[i, j] - S_local[sel_idx, j]
                    sdist += dv * dv
                sdist = np.sqrt(sdist)
                if sdist < d_s_local[i]:
                    d_s_local[i] = sdist
            return d_s_local

        for t in range(1, m):
            score = d_s * (1.0 / (rho_S + 1e-12)) ** alpha
            score[selected] = -1.0
            nxt = int(np.argmax(score))
            selected.append(nxt)
            d_s = _update_dists_numba(S, nxt, d_s)

    else:
        for t in range(1, m):
            score = d_s * (1.0 / (rho_S + 1e-12)) ** alpha
            score[selected] = -1.0
            nxt = int(np.argmax(score))
            selected.append(nxt)

            diff = S - S[nxt]
            new_d = np.linalg.norm(diff, axis=1)
            d_s = np.minimum(d_s, new_d)

    return np.array(selected, dtype=int)


def density_weighted_maxmin_faiss(S: np.ndarray, rho_S: np.ndarray, m: int, alpha: float = 0.8,
                                   rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    if not HAS_FAISS:
        return density_weighted_maxmin(S, rho_S, m, alpha=alpha, rng=rng)

    import faiss
    C, D = S.shape
    if rng is None:
        rng = np.random.RandomState(42)

    if m >= C:
        return np.arange(C, dtype=int)

    xb = S.astype('float32')
    index = faiss.IndexFlatL2(D)
    index.add(xb)  # type: ignore[misc]

    weights = (1.0 / (rho_S + 1e-12)) ** alpha
    seed = int(np.argmax(weights))
    selected = [seed]

    D2_seed, I_seed = index.search(xb[seed:seed+1], C)  # type: ignore[misc]
    diff = S - S[seed]
    d_s = np.linalg.norm(diff, axis=1)

    for t in range(1, m):
        score = d_s * weights
        score[selected] = -1.0
        nxt = int(np.argmax(score))
        selected.append(nxt)

        q = xb[nxt:nxt+1]
        D2_new, I_new = index.search(q, C)  # type: ignore[misc]
        diff = S - S[nxt]
        new_d = np.linalg.norm(diff, axis=1)
        d_s = np.minimum(d_s, new_d)

    return np.array(selected, dtype=int)


def density_weighted_maxmin_lazy(S: np.ndarray, rho_S: np.ndarray, m: int, alpha: float = 0.8, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    import heapq

    C = S.shape[0]
    if rng is None:
        rng = np.random.RandomState(42)

    if m >= C:
        return np.arange(C)

    seed = int(np.argmax((1.0 / (rho_S + 1e-12)) ** alpha))
    selected = [seed]

    diff = S - S[seed]
    d_s = np.linalg.norm(diff, axis=1)

    weights = (1.0 / (rho_S + 1e-12)) ** alpha
    scores = d_s * weights

    heap = [(-scores[i], i) for i in range(C) if i != seed]
    heapq.heapify(heap)

    in_selected = set([seed])

    ann_idx = None
    try:
        if HAS_HNSW:
            ann_idx = HNSWIndex(S)
    except Exception:
        ann_idx = None

    while len(selected) < m and heap:
        neg_score, idx = heapq.heappop(heap)
        if idx in in_selected:
            continue
        approx_dist = None
        if ann_idx is not None:
            try:
                ds, ids = ann_idx.query(S[idx], k=1)
                approx_dist = float(ds[0])
            except Exception:
                approx_dist = None

        popped_score = -neg_score
        if approx_dist is not None and approx_dist * weights[idx] >= popped_score * 0.95:
            cur_dist = approx_dist
        else:
            if len(selected) > 0:
                dists_to_sel = np.linalg.norm(S[idx] - S[selected], axis=1)
                cur_dist = float(dists_to_sel.min())
            else:
                cur_dist = np.inf
        true_score = cur_dist * weights[idx]
        if -neg_score <= true_score * 1.0000001:
            if idx in in_selected:
                continue
            selected.append(idx)
            in_selected.add(idx)
        else:
            heapq.heappush(heap, (-true_score, idx))

    return np.array(selected, dtype=int)

def select_landmarks(X: np.ndarray, m: int, c: int = 8, k_density: int = 15, alpha: float = 0.8,
                     rng_seed: int = 42, strategy: str = 'incremental') -> Tuple[np.ndarray, Dict[str, Dict[str, Union[float, int]]]]:
    t0 = time.time()
    rng = np.random.RandomState(rng_seed)
    n, d = X.shape

    t = time.time()
    radii = compute_kth_nn_radii(X, k=k_density)
    rho = density_proxy_from_radii(radii, d)
    tA = time.time() - t

    t = time.time()
    C = int(np.ceil(c * m * max(1.0, np.log(max(2, n)))))
    C = min(C, n)
    cand_idx = sample_candidates_by_importance(n, (1.0 / (rho + 1e-12)) ** alpha, C, rng)
    S = X[cand_idx]
    rho_S = rho[cand_idx]
    tB = time.time() - t

    t = time.time()
    if strategy == 'auto':
        if HAS_FAISS and C >= 2000:
            sel_in_S = density_weighted_maxmin_faiss(S, rho_S, m, alpha=alpha, rng=rng)
        elif C < 10000:
            sel_in_S = density_weighted_maxmin(S, rho_S, m, alpha=alpha, rng=rng)
        else:
            sel_in_S = density_weighted_maxmin_lazy(S, rho_S, m, alpha=alpha, rng=rng)
    elif strategy == 'lazy':
        sel_in_S = density_weighted_maxmin_lazy(S, rho_S, m, alpha=alpha, rng=rng)
    else:
        sel_in_S = density_weighted_maxmin(S, rho_S, m, alpha=alpha, rng=rng)
    tC = time.time() - t

    landmark_idx = cand_idx[sel_in_S]
    timings: Dict[str, Union[float, int]] = {
        'total': time.time() - t0,
        'stageA': tA,
        'stageB': tB,
        'stageC': tC,
        'C': int(C),
    }
    meta: Dict[str, Union[float, int]] = {
        'radii_mean': float(np.mean(radii)),
        'radii_median': float(np.median(radii)),
    }
    return landmark_idx, {'timings': timings, 'meta': meta}
