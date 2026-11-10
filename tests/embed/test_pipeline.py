from __future__ import annotations

import unittest
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from embed.datasets import DatasetBundle, QuerySet, Record, RecordCollection
from embed.encoders import Encoder
from embed.formats import read_exact_gt, read_gold_gt, read_ids, read_pcs_layout
from embed.pipeline import GenerationOptions, _prepare_queries, _query_stem, generate


class FakeEncoder(Encoder):
    def encode_documents(self, values):
        return [np.asarray(value, dtype=np.float32) for value in values]

    def encode_queries(self, values):
        return [np.asarray(value, dtype=np.float32) for value in values]

    def metadata(self):
        return {
            "type": "fake",
            "checkpoint": "fake",
            "doc_maxlen": 300,
            "query_maxlen": 32,
        }


class PipelineTest(unittest.TestCase):
    def test_gold_filtering_maps_external_ids_to_rows(self):
        query_set = QuerySet(
            name="search",
            records=[Record("q0", "zero"), Record("q1", "one"), Record("q2", "two")],
            qrels={"q0": ["missing"], "q1": ["d2", "d1", "d2"], "q2": []},
        )
        prepared = _prepare_queries(query_set, {"d1": 7, "d2": 3})
        self.assertEqual([record.id for record in prepared.records], ["q1"])
        self.assertEqual(prepared.gold_rows, [[3, 7]])
        self.assertEqual(prepared.filtered_count, 2)
        self.assertEqual(prepared.missing_references, 1)

    def test_lotte_suffixes(self):
        self.assertEqual(_query_stem("pooled", "search"), "pooled")
        self.assertEqual(_query_stem("pooled", "forum"), "pooled-forum")
        self.assertEqual(_query_stem("pooled-test", "forum"), "pooled-test-forum")

    def test_end_to_end_generation_keeps_artifacts_aligned(self):
        documents = [
            Record("d0", [[1.0, 0.0]]),
            Record("d1", [[0.0, 1.0]]),
        ]
        bundle = DatasetBundle(
            suite="beir",
            name="tiny",
            source="test/tiny",
            split="test",
            corpus=RecordCollection(2, lambda: iter(documents)),
            query_sets=[
                QuerySet(
                    "search",
                    [Record("q0", [[1.0, 0.0]]), Record("q1", [[-1.0, 0.0]])],
                    {"q0": ["d0"], "q1": ["missing"]},
                )
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            options = GenerationOptions(
                suite="beir",
                name="tiny",
                data_root=Path(directory),
                source=None,
                split="test",
                checkpoint="fake",
                batch_size=2,
                query_batch_size=2,
                doc_maxlen=300,
                query_maxlen=32,
                dtype="fp32",
                threads=None,
                zero_atol=1e-9,
                limit=None,
                overwrite=False,
                compute_gt=True,
                gt_k=2,
                metric="ip",
                chunk_clouds=1,
                batch_queries=1,
                quiet=True,
            )
            with patch("embed.pipeline.load_bundle", return_value=bundle), patch(
                "embed.pipeline.create_encoder", return_value=FakeEncoder()
            ):
                manifest = generate(options)

            root = Path(directory) / "beir" / "tiny"
            self.assertEqual(manifest, root / "tiny_manifest.json")
            self.assertEqual(read_ids(root / "tiny_query_ids.txt"), ["q0"])
            self.assertEqual(read_pcs_layout(root / "tiny_queries.pcs").count, 1)
            self.assertEqual(
                [row.tolist() for row in read_gold_gt(root / "tiny_gold_neighbors.csr", 1)],
                [[0]],
            )
            _, ids = read_exact_gt(root / "tiny_chamfer_neighbors.gt", 1)
            self.assertEqual(ids.tolist(), [[0, 1]])

            with patch("embed.pipeline.load_bundle", return_value=bundle):
                with self.assertRaisesRegex(ValueError, "existing metric"):
                    generate(replace(options, metric="l2"))


if __name__ == "__main__":
    unittest.main()
