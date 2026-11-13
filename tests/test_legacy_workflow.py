"""Existing YAML sweeps, artifact boundaries and CSV semantics without native work."""

import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mvsic.benchmark.cli import main
from mvsic.benchmark.config import read_config
from mvsic.benchmark.runner import file_identity
from mvsic.benchmark.worker import LegacySweep, recall
from mvsic.benchmark.workflow import _check_artifact, prepare, run


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = {
            "name": "original YAML", "datasets": [{"name": "tiny", "path": "data/tiny", "is_mmap": True,
                                                      "index_dir": "indices/tiny", "results_dir": "results/tiny"}],
            "indices": [{"name": "mvivf", "metric": "ip", "builds": [{
                "build_name": "tree", "build_params": {"max_leaf_size": 8, "inner_init": "uniform"},
                "variants": [{"name": "tq1", "compress": True, "quantizer": "OneBitTQ"},
                             {"name": "tq4", "compress": False, "quantizer": "TQ"}],
                "search_configs": [{"name": "k=2", "params": {
                    "k": [2], "nprobes": [8, 1, 4], "query_compression": ["none", "WARDS"],
                    "config": [{"name": "nr0", "num_rerank": 0},
                               {"name": "nr4", "num_rerank": 4, "nprobes": [2, 16]}]}}]}]}]}

    def jobs(self, action="search", **options):
        return prepare(self.raw, self.root / "config.yaml", action, base_dir=self.root, mode="latency", **options)

    def test_split_sweeps_preserve_correlated_names_paths_and_order(self):
        jobs = self.jobs()
        self.assertEqual(len(jobs), 4)
        self.assertEqual([point["nprobes"] for point in jobs[0]["points"]], [1, 1, 4, 4, 8, 8])
        self.assertEqual([point["nprobes"] for point in jobs[1]["points"]], [2, 2, 16, 16])
        self.assertTrue(all(point["num_rerank"] == 4 for point in jobs[1]["points"]))
        self.assertEqual(jobs[0]["legacy"]["csv"], str(self.root / "results/tiny/mvivf/tree/tq1/k=2/latency_results_nr0.csv"))
        self.assertEqual(jobs[0]["config"]["build"]["output"], str(self.root / "indices/tiny/mvivf/tree/index.bin"))
        self.assertEqual(jobs[0]["config"]["index"]["params"]["inner_init"], "uniform")
        self.assertEqual(jobs[0]["config"]["measurement"]["repeats"], 1)
        build = self.jobs("build", build_threads=2)
        self.assertEqual(len(build), 1)
        self.assertEqual(build[0]["config"]["index"]["quantizer"], "None")
        self.assertFalse(build[0]["config"]["index"]["quantize_centers"])

    def test_combined_schema_defaults_and_unprefixed_legacy_mode(self):
        ds = self.raw["datasets"][0]
        ds.pop("index_dir"); ds["results"] = ds.pop("results_dir")
        index = self.raw["indices"][0]
        build = index.pop("builds")[0]
        build["name"] = build.pop("build_name")
        build["params"] = build.pop("build_params")
        build["params"]["pq_method"] = 4
        build.pop("variants")
        index["build_configs"] = [build]
        jobs = prepare(self.raw, self.root / "a.yaml", "search", base_dir=self.root, num_threads=None)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["config"]["index"]["quantizer"], "TQ")
        self.assertFalse(jobs[0]["config"]["index"]["quantize_centers"])
        self.assertTrue(jobs[0]["legacy"]["both_modes"])
        self.assertTrue(jobs[0]["legacy"]["csv"].endswith("tree/k=2/results_nr0.csv"))

    def test_yaml_merge_overrides_work_but_explicit_duplicates_fail(self):
        path = self.root / "anchors.yaml"
        path.write_text("first: &base {max_leaf_size: 100, seed: 0}\nsecond: {<<: *base, seed: 7}\n")
        self.assertEqual(read_config(path)["second"], {"max_leaf_size": 100, "seed": 7})
        path.write_text("a: 1\na: 2\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_config(path)

    def test_retired_methods_codecs_metrics_and_output_collisions_fail(self):
        for mutate in (lambda raw: raw["indices"][0].update(name="muvera"),
                       lambda raw: raw["indices"][0].update(metric="l2"),
                       lambda raw: raw["indices"][0]["builds"][0]["build_params"].update(inner_metric="l2"),
                       lambda raw: raw["indices"][0]["builds"][0]["variants"][0].update(quantizer="FastScan"),
                       lambda raw: raw["indices"][0]["builds"][0]["variants"][1].update(name="tq1")):
            with self.subTest(mutate=mutate):
                raw = copy.deepcopy(self.raw); mutate(raw)
                with self.assertRaises(ValueError):
                    prepare(raw, self.root / "a.yaml", "search", base_dir=self.root)

    def test_cli_accepts_original_yaml_without_native_import_or_conversion(self):
        path = self.root / "search.yaml"
        import yaml
        path.write_text(yaml.safe_dump(self.raw))
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch("mvsic.benchmark.workflow.runtime_identity", side_effect=AssertionError("native import")):
            code = main(["validate", "--config", str(path), "--base-dir", str(self.root), "--latency"])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["workflow"], "original_yaml")
        self.assertEqual(result["operating_points"], 20)

    def test_shared_raw_skeleton_accepts_serving_variants_but_not_changed_build(self):
        job = self.jobs("build")[0]
        value = job["config"]
        artifact = Path(value["build"]["output"])
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"saved raw skeleton")
        inputs = {"points": {"sha256": "corpus"}}
        record = {"status": "complete", "fingerprint": "structural", "artifact": file_identity(artifact),
                  "index": value["index"], "inputs": inputs}
        Path(str(artifact) + ".json").write_text(json.dumps(record))
        for search in self.jobs():
            self.assertEqual(_check_artifact(search["config"], inputs), ("structural", None))
        search["config"]["index"]["params"]["seed"] = 1
        with self.assertRaisesRegex(ValueError, "parameters"):
            _check_artifact(search["config"], inputs)

    def test_named_sweeps_share_loaded_variant_and_grid_extension_keeps_raw_cache(self):
        data = self.root / "data/tiny"
        data.mkdir(parents=True)
        for suffix in ("points.pcs", "queries.pcs", "chamfer_neighbors.gt"):
            (data / f"tiny_{suffix}").write_bytes(b"identity-only fixture")
        runtime = {"native": {"sha256": "binary"}, "package_version": "test"}
        with patch("mvsic.benchmark.workflow.runtime_identity", return_value=runtime), \
             patch("mvsic.benchmark.workflow._check_artifact", return_value=("tree", None)), \
             patch("mvsic.benchmark.workflow._worker") as worker, patch("builtins.print"):
            first = run("search", self.raw, self.root / "a.yaml", base_dir=self.root, mode="latency")
            self.assertEqual(worker.call_count, 2, "two serving variants, not four named rerank rows")
            self.assertTrue(all(len(call.kwargs["groups"]) == 2 for call in worker.call_args_list))
            self.raw["indices"][0]["builds"][0]["search_configs"][0]["params"]["nprobes"].append(32)
            second = run("search", self.raw, self.root / "a.yaml", base_dir=self.root, mode="latency")
        self.assertEqual(first, second, "adding sweep points must preserve completed raw records")

    def test_original_yaml_workers_resolve_independent_parlay_and_library_limits(self):
        data = self.root / "data/tiny"
        data.mkdir(parents=True)
        for suffix in ("points.pcs", "queries.pcs", "chamfer_neighbors.gt"):
            (data / f"tiny_{suffix}").write_bytes(b"identity-only fixture")
        runtime = {"native": {"sha256": "binary"}, "package_version": "test"}
        libraries = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS")
        cases = [("batch", {}, 16, {}),
                 ("multi_latency", {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "2"}, 16, {"MKL_NUM_THREADS": "2"}),
                 ("latency", {name: "8" for name in libraries}, 1, {})]
        for mode, provided, threads, expected in cases:
            with self.subTest(mode=mode), patch.dict(os.environ, {"PARLAY_NUM_THREADS": "99", **provided}, clear=True), \
                 patch("mvsic.benchmark.workflow.runtime_identity", return_value=runtime), \
                 patch("mvsic.benchmark.workflow._check_artifact", return_value=("tree", None)), \
                 patch("mvsic.benchmark.runner.subprocess.run", return_value=SimpleNamespace(returncode=0)) as process, \
                 patch("builtins.print"):
                directories = run("search", self.raw, self.root / "a.yaml", base_dir=self.root,
                                  mode=mode, num_threads=16, build_threads=16)
                self.assertEqual(len(directories), 2, "named sweeps still share one process per serving variant")
                for call in process.call_args_list:
                    env = call.kwargs["env"]
                    self.assertEqual(env["PARLAY_NUM_THREADS"], str(threads))
                    for name in libraries:
                        self.assertEqual(env[name], expected.get(name, "1"))
                for directory in directories:
                    metadata = json.loads((directory / "search-process.json").read_text())["thread_environment"]
                    self.assertEqual(metadata["provided"]["OMP_NUM_THREADS"], provided.get("OMP_NUM_THREADS"))
                    limits = json.loads((directory / "manifest.json").read_text())["runtime"]["thread_environment"]
                    self.assertEqual(limits["search"], metadata["effective"])
                    self.assertEqual(limits["build"]["PARLAY_NUM_THREADS"], "16")

    def test_csv_precision_resume_tampering_and_stopping(self):
        options = {"csv": str(self.root / "latency_results_nr4.csv"), "mode": "latency", "both_modes": False,
                   "labels": [], "qps_threshold": 10}
        def record(probes, score, qps):
            return {"params": {"k": 2, "nprobes": probes, "num_rerank": 4}, "legacy_stats": {},
                    "recall_at_k": score, "recall_at_one": 1., "qps": qps,
                    "measurement": {"threads": 1}, "queries": 13, "fingerprint": str(probes)}
        precise = 123.12345678901234
        with patch("builtins.print"):
            sweep = LegacySweep(options, False, {"native": "a"})
            self.assertFalse(sweep.append(record(1, .8, precise)))
            resumed = LegacySweep(options, False, {"native": "a"})
            self.assertFalse(resumed.append(record(1, .8, precise)))
            self.assertTrue(resumed.append(record(2, .9, 5.)))
        with Path(options["csv"]).open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(float(rows[0]["QPS_seq"]), precise)
        self.assertEqual(rows[0]["QPS_par"], "")
        self.assertTrue(Path(options["csv"] + ".stop.json").exists())
        Path(options["csv"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "provenance"):
            LegacySweep(options, False, {"native": "a"})

    def test_recall_one_in_k_looks_at_all_returned_ids(self):
        import numpy as np
        ids = np.array([[7, 3]], dtype=np.uint32)
        truth = np.array([[3, 2, 7]], dtype=np.uint32)
        distances = np.array([[0., 1., 2.]], dtype=np.float32)
        self.assertEqual(recall(ids, truth, distances, 2, truth_k=1), 1.)
        self.assertEqual(recall(ids, truth, distances, 2), .5)


if __name__ == "__main__":
    unittest.main()
