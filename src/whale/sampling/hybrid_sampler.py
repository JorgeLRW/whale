import numpy as np
from typing import List, Optional

try:
    from sklearn.neighbors import NearestNeighbors, KernelDensity
    SKLEARN = True
except Exception:
    SKLEARN = False
    NearestNeighbors = None  # type: ignore[assignment]
    KernelDensity = None  # type: ignore[assignment]


def kde_density_scores(X: np.ndarray, bandwidth: Optional[float] = None, k: int = 10):
    """Return a normalised density score per point using KDE or k-NN inverse distance."""
    n = X.shape[0]
    if SKLEARN and KernelDensity is not None and bandwidth is not None:
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(X)
        logdens = kde.score_samples(X)
        dens = np.exp(logdens)
        dens = (dens - dens.min()) / (dens.max() - dens.min() + 1e-12)
        return dens

    k = min(k, max(1, n - 1))
    if SKLEARN and NearestNeighbors is not None and n > 1:
        nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(X)
        d, _ = nn.kneighbors(X)
        if d.shape[1] > 1:
            avg = d[:, 1:].mean(axis=1)
        else:
            avg = np.full(n, d[:, 0].mean())
    else:
        if n > 1:
            D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
            idx = np.argsort(D, axis=1)[:, 1:min(k + 1, n)]
            avg = np.mean(np.take_along_axis(D, idx, axis=1), axis=1)
        else:
            avg = np.zeros(n)
    with np.errstate(divide='ignore'):
        score = 1.0 / (avg + 1e-12)
    score = (score - score.min()) / (score.max() - score.min() + 1e-12)
    return score


def hybrid_maxmin_kde(X: np.ndarray, m: int, alpha: float = 0.5, bandwidth: Optional[float] = None, seed: Optional[int] = None) -> List[int]:
    """Hybrid sampler combining density estimates and farthest-point coverage."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    n = X.shape[0]
    if m <= 0:
        return []

    importance = kde_density_scores(X, bandwidth=bandwidth)

    dist = np.full(n, np.inf)
    selected = []

    first = int(np.argmax(importance))
    selected.append(first)
    if n > 1:
        dnew = np.linalg.norm(X - X[first], axis=1)
        dist = np.minimum(dist, dnew)
    else:
        dist[first] = 0.0

    while len(selected) < m:
        finite_mask = np.isfinite(dist)
        nd = np.zeros_like(dist)
        if finite_mask.any():
            max_dist = dist[finite_mask].max()
            if max_dist > 1e-12:
                nd[finite_mask] = dist[finite_mask] / (max_dist + 1e-12)

        combined = alpha * importance + (1.0 - alpha) * nd
        combined[selected] = -np.inf
        idx = int(np.argmax(combined))
        if not np.isfinite(combined[idx]):
            break
        selected.append(idx)
        if n > 1:
            dnew = np.linalg.norm(X - X[idx], axis=1)
            dist = np.minimum(dist, dnew)
        else:
            dist[idx] = 0.0

    uniq = []
    seen = set()
    for i in selected:
        if i not in seen:
            uniq.append(int(i))
            seen.add(i)
        if len(uniq) >= m:
            break
    return uniq
