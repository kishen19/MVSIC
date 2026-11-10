import unittest

import numpy as np

from emdivf_experiments.embedding.corpus_prototypes import (
    diverse_vector_indices,
    pseudoquery,
    sample_rows,
)


class CorpusPrototypeTest(unittest.TestCase):
    def test_sampling_is_reproducible_unique_and_eligible(self):
        rows = [{"title": "title" if i % 2 else "", "text": "document"}
                for i in range(100)]
        sizes = np.arange(100)
        first, info = sample_rows(rows, sizes, count=25, seed=42)
        second, repeated = sample_rows(rows, sizes, count=25, seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(info, repeated)
        self.assertEqual(len(np.unique(first)), 25)
        self.assertTrue(np.all(sizes[first] >= 32))
        self.assertEqual(sum(s["sampled"] for s in info["strata"]), 25)
        self.assertEqual(info["excluded_documents"], 32)
        with self.assertRaises(ValueError):
            sample_rows(rows, sizes, count=69, seed=42)

    def test_pseudoquery_is_deterministic_and_bounded(self):
        row = {"title": "one two three four five six seven eight nine", "text":
               " ".join(f"word{i}" for i in range(100))}
        text = pseudoquery(row, seed=42, document_row=7)
        self.assertEqual(text, pseudoquery(row, seed=42, document_row=7))
        self.assertEqual(len(text.split()), 32)
        self.assertTrue(text.startswith("one two three four five six seven eight "))

    def test_vector_subset_uses_distinct_original_rows_even_on_ties(self):
        tied = np.ones((40, 8), dtype=np.float32)
        indices = diverse_vector_indices(tied)
        np.testing.assert_array_equal(indices, np.arange(32))
        cloud = np.random.default_rng(7).normal(size=(40, 8)).astype(np.float32)
        first = diverse_vector_indices(cloud)
        second = diverse_vector_indices(cloud)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 32)
        self.assertTrue(np.all(np.diff(first) > 0))

    def test_vector_subset_rejects_short_or_nonfinite_clouds(self):
        with self.assertRaises(ValueError):
            diverse_vector_indices(np.ones((31, 8), dtype=np.float32))
        with self.assertRaises(ValueError):
            diverse_vector_indices(np.full((40, 8), np.nan, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
