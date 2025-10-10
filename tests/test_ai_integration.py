import unittest

import numpy as np

from whale.ai import (
    batch_witness_summaries,
    embedding_to_point_cloud,
    embedding_witness_summary,
)


def _generate_embeddings(n_points: int = 64, dims: int = 4) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(size=(n_points, dims)).astype(np.float64)


class AiIntegrationTests(unittest.TestCase):
    def test_embedding_to_point_cloud_basic(self) -> None:
        embeddings = _generate_embeddings(32, 3)
        cloud = embedding_to_point_cloud(embeddings)
        self.assertEqual(cloud.points.shape, embeddings.shape)
        self.assertEqual(cloud.intensities.shape[0], embeddings.shape[0])

    def test_embedding_witness_summary_outputs(self) -> None:
        embeddings = _generate_embeddings(48, 3)
        summary, diagrams, keys = embedding_witness_summary(
            embeddings,
            method="random",
            max_dim=1,
            k_witness=16,
            m=12,
            seed=123,
            selection_c=4,
        )
        self.assertIn("coverage_mean", summary)
        self.assertEqual(len(diagrams), 2)
        self.assertTrue(all(isinstance(key, str) for key in keys))

    def test_batch_witness_summaries_vector_output(self) -> None:
        embeddings = np.stack([_generate_embeddings(40, 3), _generate_embeddings(40, 3)])
        summaries, keys, vectors = batch_witness_summaries(
            embeddings,
            method="random",
            max_dim=1,
            k_witness=16,
            m=10,
            seed=999,
            selection_c=4,
            return_vectors=True,
            tda_mode="fast",
        )
        self.assertEqual(len(summaries), 2)
        self.assertTrue(keys)
        self.assertIsNotNone(vectors)
        assert vectors is not None  # for type checkers
        self.assertEqual(len(vectors), 2)
        for vec in vectors:
            self.assertEqual(vec.shape[0], len(keys))

    def test_regular_mode_adds_dim_two(self) -> None:
        embeddings = _generate_embeddings(56, 3)
        _summary, diagrams, _keys = embedding_witness_summary(
            embeddings,
            method="random",
            seed=7,
            selection_c=4,
            tda_mode="regular",
        )
        self.assertEqual(len(diagrams), 3)
