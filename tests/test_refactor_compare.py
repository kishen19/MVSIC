"""Lightweight validation of comparison provenance and orchestration helpers."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


_spec = importlib.util.spec_from_file_location("refactor_compare", Path(__file__).parent / "refactor" / "compare.py")
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)


class ComparisonTests(unittest.TestCase):
    def test_retired_l2_retrieval_is_rejected_before_execution(self):
        config = {"original": "old", "refactor": "new", "points": "points.pcs",
                  "queries": "queries.pcs", "ground_truth": "truth.gt",
                  "cpus": [0], "build_cpus": [0], "metric": "ip"}
        compare.validate_config(config)
        with self.assertRaisesRegex(ValueError, "supports only metric ip"):
            compare.validate_config({**config, "metric": "l2"})

    def test_active_marker_blocks_before_spawning(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "marker.json"
            marker.write_text(json.dumps({"active": True, "phase": "latency"}))
            with self.assertRaisesRegex(RuntimeError, "active"):
                compare.measured_process(["this-command-must-not-run"], root / "process", marker=marker, threads=1)
            self.assertFalse((root / "process").exists())
            marker.write_text(json.dumps({"active": False}))
            self.assertFalse(compare.check_marker(marker)["active"])

    def test_binary_pin_does_not_follow_mutable_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "binary"
            source.write_bytes(b"original executable")
            pinned, metadata = compare.pin_binary(source, root / "pinned")
            source.write_bytes(b"rebuilt executable")
            self.assertEqual(pinned.read_bytes(), b"original executable")
            self.assertEqual(compare.identity(pinned)["sha256"], metadata["source"]["sha256"])
            self.assertFalse(pinned.stat().st_mode & 0o222)

    def test_controller_interruption_stops_the_child_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "marker.json"
            marker.write_text(json.dumps({"active": False}))
            process = Mock(pid=123456, returncode=-15)
            process.poll.return_value = None
            with patch.object(compare.subprocess, "Popen", return_value=process), \
                 patch.object(compare.time, "sleep", side_effect=KeyboardInterrupt), \
                 patch.object(compare.os, "killpg") as stop:
                with self.assertRaises(KeyboardInterrupt):
                    compare.measured_process(["placeholder"], root / "run", marker=marker, threads=1)
            stop.assert_called_once_with(123456, compare.signal.SIGTERM)
            self.assertTrue(json.loads((root / "run/process.json").read_text())["controller_interrupted"])

    def test_shared_libraries_are_copied_and_loader_resolution_is_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "libmvsic.so"
            library.write_bytes(b"library implementation")
            listing = SimpleNamespace(returncode=0, stderr="", stdout=f"libmvsic.so => {library} (0x1234)\n")
            with patch.object(compare.subprocess, "run", return_value=listing):
                pinned, metadata = compare.pin_libraries("binary", root / "pinned")
            self.assertEqual((pinned / "libmvsic.so").read_bytes(), b"library implementation")
            self.assertIn(str(pinned), compare.library_environment(pinned)["LD_LIBRARY_PATH"])
            with patch.object(compare.subprocess, "run", return_value=listing):
                with self.assertRaisesRegex(RuntimeError, "does not use pinned"):
                    compare.verify_pinned_libraries("binary", pinned, metadata)
            listing.stdout = f"libmvsic.so => {pinned / 'libmvsic.so'} (0x1234)\n"
            with patch.object(compare.subprocess, "run", return_value=listing):
                compare.verify_pinned_libraries("binary", pinned, metadata)

    def test_recall_matching_does_not_accept_lower_recall(self):
        params = {"mode": "batch", "threads": 4, "nprobes": 4, "rerank": 20, "k": 10}
        old = {"implementation": "original", "params": params, "recall_min": 0.95,
               "recall_max": 0.95, "qps_median": 100}
        lower = {"implementation": "refactor", "params": {**params, "nprobes": 2},
                 "recall_min": 0.9, "recall_max": 0.9, "qps_median": 1000}
        matched = {"implementation": "refactor", "params": {**params, "nprobes": 8},
                   "recall_min": 0.951, "recall_max": 0.951, "qps_median": 110}
        result = compare.match_recall([old, lower, matched], 0.001)
        self.assertEqual(result[0]["refactor"], matched)
        self.assertEqual(result[0]["speed_ratio"], 1.1)
        self.assertIsNone(compare.match_recall([old, lower], 0.001)[0]["speed_ratio"])

    def test_search_scopes_use_shared_or_matching_independent_artifacts(self):
        builds = [{"implementation": kind, "artifact": {"path": f"/build-{number}-{kind}.idx", "sha256": str(number)}}
                  for number, kind in enumerate(("original", "refactor", "refactor", "original"))]
        trials = compare.search_trials({"search_blocks": 2}, builds)
        shared = [row for row in trials if row["scope"] == "shared_original"]
        independent = [row for row in trials if row["scope"] == "independent_build"]
        self.assertEqual(len(shared), 8)
        self.assertEqual(len(independent), 8)
        self.assertTrue(all(row["artifact"] == builds[0]["artifact"] for row in shared))
        self.assertTrue(all(row["build_number"] is None for row in shared))
        for number, row in enumerate(independent):
            expected = builds[number % len(builds)]
            self.assertEqual(row["artifact"], expected["artifact"])
            self.assertEqual(row["implementation"], expected["implementation"])
            self.assertEqual(row["build_number"], number % len(builds))
        self.assertEqual(compare.search_trials({"search_scopes": ["independent_build"], "search_blocks": 2}, builds), independent)

    def test_aggregation_and_recall_matching_never_mix_tree_scopes(self):
        params = {"mode": "batch", "threads": 1, "nprobes": 2, "rerank": 20}
        original = {"scope": "independent_build", "implementation": "original", "params": params,
                    "native": {"qps": 100, "peak_rss_bytes": 1024}, "recall": .95}
        refactor = {**original, "implementation": "refactor", "native": {"qps": 150, "peak_rss_bytes": 2048}}
        shared = {**refactor, "scope": "shared_original", "native": {"qps": 1000, "peak_rss_bytes": 2048}}
        rows = compare.aggregate([original, refactor, shared])
        self.assertEqual(len(rows), 3)
        self.assertEqual(compare.match_recall(rows, .001)[0]["speed_ratio"], 1.5)
        rows = compare.aggregate([original, shared])
        self.assertIsNone(compare.match_recall(rows, .001)[0]["refactor"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is needed for PCS sampling")
    def test_query_sample_matches_legacy_order_and_values(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "queries.pcs", root / "sample.pcs"
            values = np.arange(60, dtype=np.float32)
            offsets = np.arange(0, 61, 12, dtype=np.uint64)
            with source.open("wb") as stream:
                np.array([4, 5, 15], dtype="<u8").tofile(stream)
                values.tofile(stream)
                np.array([6], dtype="<u8").tofile(stream)
                offsets.tofile(stream)
            selected, count = compare.sample_queries(source, output, 3, 42)
            expected = np.sort(np.random.default_rng(42).choice(5, size=3, replace=False))
            np.testing.assert_array_equal(selected, expected)
            self.assertEqual(count, 5)
            with output.open("rb") as stream:
                header = np.fromfile(stream, dtype="<u8", count=3)
                sampled = np.fromfile(stream, dtype="<f4", count=36)
            np.testing.assert_array_equal(header, [4, 3, 9])
            np.testing.assert_array_equal(sampled.reshape(3, 12), values.reshape(5, 12)[expected])


if __name__ == "__main__":
    unittest.main()
