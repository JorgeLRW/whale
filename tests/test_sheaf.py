import numpy as np
from paper_ready.tda_odometry.features import TDAFeatureExtractor


def test_sheaf_score_basic():
    ext = TDAFeatureExtractor(use_student=False, sheaf_enabled=True, sheaf_window=4, sheaf_alpha=0.8, sheaf_threshold=0.25)

    # Create deterministic synthetic vectors
    rng = np.random.RandomState(0)
    vecs = [rng.randn(51).astype(np.float32) for _ in range(6)]

    scores = []
    for v in vecs:
        s = ext._compute_sheaf_score(v)
        scores.append(s)

    # After first vector, score should be 0.0 (no pair)
    assert isinstance(scores[0], float)
    assert scores[0] == 0.0

    # Later scores should be finite and within a reasonable bound [0, 2]
    for s in scores[1:]:
        assert np.isfinite(s)
        assert 0.0 <= s <= 2.0
