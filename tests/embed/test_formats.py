from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from embed.encoders import clean_embedding
from embed.formats import (
    StreamingPCSWriter,
    iter_pcs,
    read_exact_gt,
    read_gold_gt,
    read_pcs_layout,
    validate_exact_gt,
    validate_pcs,
    write_gold_gt,
)
from embed.ground_truth import compute_ground_truth


class FormatTest(unittest.TestCase):
    def test_pcs_round_trip_and_offsets_are_float_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pcs"
            clouds = [
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                np.asarray([[-1.0, 0.0]], dtype=np.float32),
            ]
            with StreamingPCSWriter(path, expected_count=2) as writer:
                for cloud in clouds:
                    writer.append(cloud)

            layout = read_pcs_layout(path)
            self.assertEqual((layout.dim, layout.count, layout.num_vectors), (2, 2, 3))
            actual = list(iter_pcs(path))
            np.testing.assert_array_equal(actual[0], clouds[0])
            np.testing.assert_array_equal(actual[1], clouds[1])
            self.assertEqual(validate_pcs(path)["min_vectors_per_cloud"], 1)

    def test_padding_cleanup_and_normalization(self):
        raw = np.asarray([[0.0, 0.0], [3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
        cleaned = clean_embedding(raw)
        self.assertEqual(cleaned.shape, (2, 2))
        np.testing.assert_allclose(np.linalg.norm(cleaned, axis=1), 1.0, atol=1e-6)

    def test_gold_csr_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.csr"
            write_gold_gt(path, [[3, 8], [1], [0, 4, 9]])
            rows = read_gold_gt(path, 3)
            self.assertEqual([row.tolist() for row in rows], [[3, 8], [1], [0, 4, 9]])

    def test_exact_gt_matches_asymmetric_chamfer_ip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "points.pcs"
            query_path = root / "queries.pcs"
            gt_path = root / "neighbors.gt"
            corpus = [
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                np.asarray([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float32),
            ]
            queries = [
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                np.asarray([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float32),
            ]
            with StreamingPCSWriter(corpus_path, 2) as writer:
                for cloud in corpus:
                    writer.append(cloud)
            with StreamingPCSWriter(query_path, 2) as writer:
                for cloud in queries:
                    writer.append(cloud)

            compute_ground_truth(
                corpus_path,
                query_path,
                gt_path,
                k=2,
                metric="ip",
                chunk_clouds=1,
                batch_queries=1,
                threads=1,
                quiet=True,
            )
            self.assertEqual(validate_exact_gt(gt_path, 2), {"num_queries": 2, "k": 2})
            distances, ids = read_exact_gt(gt_path, 2)
            np.testing.assert_array_equal(ids, [[0, 1], [1, 0]])
            np.testing.assert_allclose(distances, [[-1.0, 0.0], [-1.0, 0.0]])

    def test_exact_gt_matches_squared_chamfer_l2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "points.pcs"
            query_path = root / "queries.pcs"
            gt_path = root / "neighbors.gt"
            with StreamingPCSWriter(corpus_path, 2) as writer:
                writer.append(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
                writer.append(np.asarray([[-1.0, 0.0]], dtype=np.float32))
            with StreamingPCSWriter(query_path, 1) as writer:
                writer.append(np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32))

            compute_ground_truth(
                corpus_path,
                query_path,
                gt_path,
                k=2,
                metric="l2",
                chunk_clouds=2,
                batch_queries=1,
                quiet=True,
            )
            distances, ids = read_exact_gt(gt_path, 1)
            np.testing.assert_array_equal(ids, [[0, 1]])
            # doc 0: mean(0, 2); doc 1: mean(4, 2)
            np.testing.assert_allclose(distances, [[1.0, 3.0]])


if __name__ == "__main__":
    unittest.main()
