"""Bounded ordering and honest comparison scope without running native work."""

import importlib.util
from pathlib import Path
import sys
import unittest

DIRECTORY = Path(__file__).parent / "refactor"
sys.path.insert(0, str(DIRECTORY))
try:
    spec = importlib.util.spec_from_file_location("cross_tree", DIRECTORY / "cross_tree.py")
    cross_tree = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cross_tree)
finally:
    sys.path.pop(0)


class CrossTreeTests(unittest.TestCase):
    def test_batch_scaling_schedule_keeps_tree_fixed_and_reverses_complete_matrix(self):
        plan = cross_tree.batch_scaling_schedule()
        self.assertEqual(len(plan), 96)
        self.assertTrue(all(row["tree"] == cross_tree.TREES[1] for row in plan))
        forward = [(plan[i]["batch_size"], plan[i]["threads"]) for i in range(0, 48, 4)]
        reverse = [(plan[i]["batch_size"], plan[i]["threads"]) for i in range(48, 96, 4)]
        self.assertEqual(reverse, list(reversed(forward)))
        for begin in range(0, 96, 4):
            block = plan[begin:begin+4]
            self.assertEqual([row["implementation"] for row in block], ["original", "refactor", "refactor", "original"])
            self.assertTrue(all(len(row["cpus"]) == row["threads"] and len(set(row["cpus"])) == row["threads"] for row in block))
        self.assertEqual(len(cross_tree.batch_scaling_schedule(blocks=1)), 48)

    def test_batch_scaling_topology_rejects_smt_duplicates_and_changed_numbering(self):
        topology = {cpu: (cpu % 4, cpu // 4, {cpu, cpu + 96}) for cpu in range(96)}
        cross_tree.validate_scaling_topology(topology)
        topology[84] = topology[80]
        with self.assertRaises(ValueError):
            cross_tree.validate_scaling_topology(topology)

    def test_scaling_summary_separates_batches_and_uses_each_engine_single_worker_baseline(self):
        rows = []
        for batch in (128, 1024):
            for threads in (1, 4):
                for kind in ("original", "refactor"):
                    qps = (100 if kind == "original" else 150) if threads == 1 else 400
                    rows.append({"batch_size": batch, "threads": threads, "tree": cross_tree.TREES[1],
                                 "implementation": kind, "recall": .9,
                                 "native": {"qps": qps * batch / 128, "peak_rss_bytes": 1024}})
        result = cross_tree.summarize_scaling(rows)
        self.assertEqual(len(result), 8)
        for row in result:
            if row["threads"] == 4 and row["implementation"] == "refactor":
                self.assertAlmostEqual(row["speedup_from_one_worker"], 400 / 150)
                self.assertAlmostEqual(row["parallel_efficiency"], 2 / 3)
                self.assertEqual(row["qps_ratio_to_original"], 1)

    def test_fine_probes_pair_each_budget_with_its_own_baseline_and_reverse_order(self):
        plan = cross_tree.fine_probe_schedule()
        self.assertEqual(len(plan), 32)
        self.assertEqual([plan[i]["comparison_probe"] for i in (0, 4, 8, 12)], [14, 15, 16, 17])
        self.assertEqual([plan[i]["comparison_probe"] for i in (16, 20, 24, 28)], [17, 16, 15, 14])
        for row in plan:
            old = row["implementation"] == "original"
            self.assertEqual(row["tree"], cross_tree.TREES[1 if old else 2])
            self.assertEqual(row["nprobes"], 16 if old else row["comparison_probe"])
        self.assertEqual(len(cross_tree.fine_probe_schedule((15,), 2)), 8)
        with self.assertRaises(ValueError):
            cross_tree.fine_probe_schedule((14, 14))

    def test_fine_probe_summary_preserves_lower_recall_even_when_faster(self):
        rows = []
        for probe, recall in [(14, .89), (15, .9), (16, .91)]:
            for kind in ("original", "refactor"):
                old = kind == "original"
                rows.append({"comparison_probe": probe, "tree": cross_tree.TREES[1 if old else 2],
                             "implementation": kind, "recall": .9 if old else recall,
                             "native": {"qps": 100 if old else 110, "peak_rss_bytes": 1024}})
        results = cross_tree.summarize_fine_probes(rows)
        self.assertFalse(results[0]["meets_baseline_recall_without_tolerance"])
        self.assertIsNone(results[0]["qps_ratio_at_equal_or_higher_recall"])
        self.assertEqual(results[0]["observed_qps_ratio"], 1.1)
        self.assertEqual(results[1]["qps_ratio_at_equal_recall"], 1.1)
        self.assertTrue(results[2]["meets_baseline_recall_without_tolerance"])
        self.assertIsNone(results[2]["qps_ratio_at_equal_recall"])
        self.assertEqual(results[2]["qps_ratio_at_equal_or_higher_recall"], 1.1)

    def test_schedule_uses_every_tree_and_reverses_tree_order(self):
        plan = cross_tree.schedule()
        self.assertEqual(len(plan), 24)
        self.assertEqual([plan[i]["tree"] for i in (0, 4, 8)], list(cross_tree.TREES))
        self.assertEqual([plan[i]["tree"] for i in (12, 16, 20)], list(reversed(cross_tree.TREES)))
        for begin in range(0, len(plan), 4):
            self.assertEqual([row["implementation"] for row in plan[begin:begin + 4]],
                             ["original", "refactor", "refactor", "original"])
        with self.assertRaises(ValueError):
            cross_tree.schedule(3)

    def test_equal_recall_comparison_does_not_pool_the_weaker_original_tree(self):
        rows = [{"tree": cross_tree.TREES[0], "implementation": "original", "recall_min": .89,
                 "recall_max": .89, "qps_median": 999},
                {"tree": cross_tree.TREES[1], "implementation": "original", "recall_min": .9,
                 "recall_max": .9, "qps_median": 100},
                {"tree": cross_tree.TREES[2], "implementation": "refactor", "recall_min": .9,
                 "recall_max": .9, "qps_median": 97}]
        self.assertEqual(cross_tree.equal_recall_comparison(rows)["qps_ratio_at_equal_recall"], .97)
        rows[2]["recall_min"] = .899
        self.assertIsNone(cross_tree.equal_recall_comparison(rows)["qps_ratio_at_equal_recall"])

    def test_prediction_differences_are_retained_and_not_assigned_to_known_bug(self):
        import numpy as np
        old = (np.array([[1, 2]], dtype=np.uint32), np.array([[-.4, -.2]], dtype=np.float32))
        new = (np.array([[1, 3]], dtype=np.uint32), np.array([[-.4, -.1]], dtype=np.float32))
        difference = cross_tree.prediction_difference(old, new)
        self.assertFalse(difference["matching"])
        self.assertFalse(difference["attributed_to_known_original_bug"])
        self.assertEqual(difference["queries_with_differences"][0]["original_only_ids"], [2])
        self.assertEqual(difference["queries_with_differences"][0]["refactor_only_ids"], [3])
        reordered = (old[0][:, ::-1], old[1][:, ::-1])
        self.assertTrue(cross_tree.prediction_difference(old, reordered)["matching"])


if __name__ == "__main__":
    unittest.main()
