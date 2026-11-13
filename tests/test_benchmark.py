"""Schema, provenance, recall and failure handling without a compiled extension."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mvsic.benchmark.cli import main
from mvsic.benchmark.config import BUILD_DEFAULTS, load_config, resolve_config
from mvsic.benchmark.runner import (atomic_json, build_identity, completed, execution_runtime, file_identity,
                                    run as run_benchmark, runtime_identity, writer_lock)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "plan.json"
        self.raw = {"version": 1, "dataset": {"points": "points.pcs"},
                    "search": {"k": 2, "num_rerank": [0, 3]}}

    def resolve(self, raw=None):
        return resolve_config(raw or self.raw, self.source)

    def test_resolved_defaults_and_grid(self):
        config = self.resolve()
        self.assertEqual(config.resolved["dataset"]["points"], str(self.root / "points.pcs"))
        self.assertEqual(config.resolved["index"]["params"], BUILD_DEFAULTS)
        self.assertEqual(len(config.points()), 12)
        explicit = resolve_config(config.resolved, self.source)
        self.assertEqual(config.fingerprint, explicit.fingerprint)

    def test_aliases_canonicalize(self):
        self.raw["index"] = {"quantizer": "TQ1", "params": {"build_with_8btq": True}}
        first = self.resolve()
        self.raw["index"] = {"quantizer": "OneBitTQ", "params": {"assignment_quantizer": "TQ8"}}
        self.assertEqual(first.fingerprint, self.resolve().fingerprint)

    def test_invalid_types_and_combinations(self):
        cases = [
            {"version": True}, {"unexpected": 4}, {"index": {"metric": "cosine"}},
            {"index": {"metric": "l2"}},
            {"index": {"quantize_centers": "false"}},
            {"index": {"params": {"max_leaf_size": True}}},
            {"index": {"params": {"seed": 2**32}}},
            {"index": {"params": {"inner_metric": "cosine"}}},
            {"index": {"params": {"inner_metric": "l2"}}},
            {"index": {"params": {"inner_init": "Random"}}},
            {"index": {"params": {"inner_backend": "unknown"}}},
            {"index": {"params": {"build_with_8btq": True, "assignment_quantizer": "TQ1"}}},
            {"search": {"k": 2, "num_rerank": 1}}, {"search": {"root_m2m": "false"}},
            {"search": {"nprobes": []}}, {"search": {"nprobes": [False]}},
            {"search": {"query_compression": "typo"}},
            {"search": {"query_compression_threshold": float("nan")}},
            {"search": {"query_compression": "Wards", "query_compression_threshold": -1}},
            {"search": {"query_alignment_strict": True}},
            {"measurement": {"threads": 2}}, {"measurement": {"repeats": 0}},
            {"dataset": {"points": "points.pcs", "mmap": 1}},
            {"build": {"output": "points.pcs"}},
        ]
        for update in cases:
            with self.subTest(update=update), self.assertRaises(ValueError):
                raw = copy.deepcopy(self.raw)
                raw.update(update)
                self.resolve(raw)

    def test_duplicate_json_rejected(self):
        self.source.write_text('{"version":1,"version":1,"dataset":{"points":"x"}}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_config(self.source)

    def test_inner_options_set_config_without_changing_factory_signature(self):
        from mvsic.benchmark.worker import _index
        self.raw["index"] = {"params": {"inner_init": "kmeans++", "inner_backend": "adaptive"}}
        params = SimpleNamespace(mvclus=SimpleNamespace())
        seen = {}
        def factory(**kwargs):
            seen.update(kwargs)
            return params
        native = SimpleNamespace(IndexParams=SimpleNamespace(mvivf=factory),
                                 ClusteringQuantizer=SimpleNamespace(None_=0),
                                 QuantizerType=SimpleNamespace(OneBitTQ=1),
                                 IndexMVIVF=lambda *args: args)
        result = _index(native, SimpleNamespace(dims=128), self.resolve().resolved)
        self.assertEqual(len(seen), 13)
        self.assertFalse(hasattr(params.mvclus, "inner_metric"))
        self.assertEqual(params.mvclus.inner_init, "kmeans++")
        self.assertEqual(params.mvclus.inner_backend, "adaptive")
        self.assertEqual(result, (128, params))

    def test_retired_metric_compatibility_canonicalizes_to_ip(self):
        reference = self.resolve()
        for metric in ("same", "ip"):
            self.raw["index"] = {"metric": "ip", "params": {"inner_metric": metric}}
            self.assertEqual(self.resolve().fingerprint, reference.fingerprint)
            self.assertNotIn("inner_metric", self.resolve().resolved["index"]["params"])
        self.raw["index"]["params"]["inner_metric"] = "l2"
        with self.assertRaisesRegex(ValueError, "L2 is retired"):
            self.resolve()

    def test_cli_validation_does_not_need_native(self):
        self.source.write_text(json.dumps(self.raw))
        with patch("builtins.print"):
            self.assertEqual(main(["validate", "--config", str(self.source)]), 0)
        self.raw["search"]["k"] = False
        self.source.write_text(json.dumps(self.raw))
        with patch("builtins.print"):
            self.assertEqual(main(["validate", "--config", str(self.source)]), 1)



class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_resume_verifies_bytes(self):
        artifact = self.root / "index.bin"
        metadata = self.root / "index.bin.json"
        artifact.write_bytes(b"original")
        atomic_json(metadata, {"status": "complete", "fingerprint": "f", "artifact": file_identity(artifact)})
        self.assertTrue(completed(metadata, "f", artifact))
        self.assertFalse(completed(metadata, "changed", artifact))
        stat = artifact.stat()
        artifact.write_bytes(b"modified")
        os.utime(artifact, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertFalse(completed(metadata, "f", artifact))
        metadata.write_text('{"status":')
        self.assertFalse(completed(metadata, "f", artifact))

    def test_build_fingerprint_ignores_search_but_tracks_binary_and_inputs(self):
        config = resolve_config({"version": 1, "dataset": {"points": "points.pcs"}}, self.root / "p.json").resolved
        inputs = {"points": {"sha256": "a"}}
        runtime = {"native": {"sha256": "binary"}, "package_version": "0"}
        first = build_identity(config, inputs, runtime)
        config["search"]["nprobes"] = 100
        self.assertEqual(first, build_identity(config, inputs, runtime))
        inputs["points"]["sha256"] = "b"
        self.assertNotEqual(first, build_identity(config, inputs, runtime))
        inputs["points"]["sha256"] = "a"
        runtime["native"]["sha256"] = "new"
        self.assertNotEqual(first, build_identity(config, inputs, runtime))

    def test_build_fingerprint_tracks_effective_build_limits_independently_of_search(self):
        config = resolve_config({"version": 1, "dataset": {"points": "points.pcs"},
                                 "build": {"threads": 16},
                                 "measurement": {"mode": "throughput_batch", "threads": 16}}, self.root / "p.json").resolved
        inputs = {"points": {"sha256": "corpus"}}
        runtime = {"native": {"sha256": "binary"}, "package_version": "test"}
        first = execution_runtime(runtime, config, {})
        fingerprint = build_identity(config, inputs, first)
        self.assertNotEqual(fingerprint, build_identity(config, inputs, runtime), "old metadata lacks effective limits")
        self.assertEqual(fingerprint, build_identity(config, inputs, execution_runtime(runtime, config, {"OMP_NUM_THREADS": "1"})))
        self.assertNotEqual(fingerprint, build_identity(config, inputs, execution_runtime(runtime, config, {"OMP_NUM_THREADS": "2"})))
        config["measurement"].update(mode="latency_single", threads=1)
        second = execution_runtime(runtime, config, {})
        self.assertEqual(fingerprint, build_identity(config, inputs, second))
        self.assertNotEqual(first["thread_environment"]["search"], second["thread_environment"]["search"])
        self.assertNotIn("thread_environment", runtime, "resolving a job must not mutate the shared runtime")

    def test_writer_lock_rejects_concurrent_owner(self):
        path = self.root / ".lock"
        with writer_lock(path):
            with self.assertRaisesRegex(RuntimeError, "another benchmark"):
                with writer_lock(path):
                    pass
        with writer_lock(path):
            pass

    def test_shared_library_and_backend_changes_invalidate_build_identity(self):
        extension = self.root / "_mvsic.test.so"
        extension.write_bytes(b"unchanged extension")
        libraries = self.root / ".libs"
        libraries.mkdir()
        library = libraries / "libmvsic.so"
        library.write_bytes(b"old implementation")
        config = resolve_config({"version": 1, "dataset": {"points": "p.pcs"}}, self.root / "p.json").resolved
        output = SimpleNamespace(returncode=0, stderr="", stdout=f"libmvsic.so => {library} (0x1234)\n")
        with patch("mvsic.benchmark.runner.importlib.util.find_spec", return_value=SimpleNamespace(origin=str(extension))), \
                patch("mvsic.benchmark.runner.subprocess.run", return_value=output), \
                patch.dict(os.environ, {"MVSIC_CPU_ISA": "avx2"}):
            first = runtime_identity()
            before = build_identity(config, {"points": {}}, first)
            library.write_bytes(b"new implementation")
            second = runtime_identity()
            self.assertEqual(first["native"], second["native"])
            self.assertNotEqual(before, build_identity(config, {"points": {}}, second))
            with patch.dict(os.environ, {"MVSIC_CPU_ISA": "avx512"}):
                self.assertNotEqual(build_identity(config, {"points": {}}, second),
                                    build_identity(config, {"points": {}}, runtime_identity()))


class ThreadEnvironmentTests(unittest.TestCase):
    def test_worker_limits_and_provenance_for_build_parallel_and_serial(self):
        libraries = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS")
        cases = [
            ("build", "throughput_batch", 16, {}, {}),
            ("search", "throughput_batch", 16, {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "2",
                                               "OPENBLAS_NUM_THREADS": "3", "RAYON_NUM_THREADS": "4"},
             {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "3", "RAYON_NUM_THREADS": "4"}),
            ("search", "latency_parallel", 16, {"OMP_NUM_THREADS": "1"}, {}),
            ("search", "latency_single", 1, {name: "8" for name in libraries}, {}),
        ]
        for action, mode, threads, provided, expected in cases:
            with self.subTest(action=action, mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "points.pcs").write_bytes(b"points")
                (root / "queries.pcs").write_bytes(b"queries")
                config = resolve_config({"version": 1, "dataset": {"points": "points.pcs", "queries": "queries.pcs"},
                                         "build": {"output": "tree.idx", "threads": 16},
                                         "measurement": {"mode": mode, "threads": threads}}, root / "plan.json")
                ambient = {"PARLAY_NUM_THREADS": "99", **provided}
                runtime = {"native": {"sha256": "binary"}, "package_version": "test"}
                with patch.dict(os.environ, ambient, clear=True), \
                     patch("mvsic.benchmark.runner.runtime_identity", return_value=runtime), \
                     patch("mvsic.benchmark.runner.subprocess.run", return_value=SimpleNamespace(returncode=0)) as process, \
                     patch("builtins.print"):
                    directory = run_benchmark(action, config)
                actual = process.call_args.kwargs["env"]
                requested = 16 if action == "build" else threads
                self.assertEqual(actual["PARLAY_NUM_THREADS"], str(requested))
                for name in libraries:
                    self.assertEqual(actual[name], expected.get(name, "1"))
                metadata = json.loads((directory / f"{action}-process.json").read_text())["thread_environment"]
                self.assertEqual(metadata["provided"]["PARLAY_NUM_THREADS"], "99")
                for name in libraries:
                    self.assertEqual(metadata["provided"][name], provided.get(name))
                effective = {name: actual[name] for name in ("PARLAY_NUM_THREADS", *libraries)}
                self.assertEqual(metadata["effective"], effective)
                manifest = json.loads((directory / "manifest.json").read_text())
                self.assertEqual(manifest["runtime"]["thread_environment"][action], effective)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is required for result tests")
class ResultTests(unittest.TestCase):
    def test_failed_point_is_recorded_and_other_points_resume(self):
        import numpy as np
        from mvsic.benchmark.worker import _search
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "tree.idx"
            artifact.write_bytes(b"test skeleton")
            config = resolve_config({"version": 1,
                                     "dataset": {"points": "points.pcs", "queries": "queries.pcs"},
                                     "build": {"output": str(artifact)},
                                     "search": {"k": 2, "nprobes": [1, 2], "num_rerank": 0},
                                     "measurement": {"python_timing": False, "repeats": 2}}, root / "p.json").resolved
            atomic_json(str(artifact) + ".json", {"status": "complete", "fingerprint": "build",
                                                  "artifact": file_identity(artifact)})
            context = {"build_fingerprint": "build", "inputs": {}, "runtime": {}}
            job = {"config": config, "context": context, "directory": str(root), "force": False}
            queries = [np.ones((1, 8), dtype=np.float32)]
            calls = []

            def benchmark(index, source, params, mode, warmup, repeats):
                calls.append(params["nprobes"])
                if params["nprobes"] == 1:
                    raise ValueError("intentional unsupported point")
                return {"ids": np.array([[42, 7]], dtype=np.uint32),
                        "distances": np.array([[0.12345678, 2]], dtype=np.float32),
                        "seconds": [0.00123456789, 0.000912345678], "selected_repeat": 1,
                        "warmup_seconds": 0.0123456789, "first_call_seconds": 0.00912345678,
                        "query_seconds": [[0.00123456789], [0.000912345678]]}

            native = SimpleNamespace(PointCloudSet=SimpleNamespace(from_file=lambda *args: object()),
                                     benchmark=benchmark)
            index = SimpleNamespace(load=lambda *args: None)
            with patch("mvsic.benchmark.worker._index", return_value=index), \
                    patch("mvsic.benchmark.worker._queries", return_value=(queries, np.array([0], dtype=np.uint32), 1)), \
                    patch("mvsic.benchmark.worker._params", side_effect=lambda native, point: point), \
                    patch("mvsic.benchmark.worker.traceback.print_exc"):
                with self.assertRaisesRegex(RuntimeError, "1 of 2"):
                    _search(job, native)
                records = [json.loads(path.read_text()) for path in root.glob("search-*.json")]
                self.assertEqual(sorted(record["status"] for record in records), ["complete", "failed"])
                successful = next(record for record in records if record["status"] == "complete")
                self.assertEqual(successful["seconds"], [0.00123456789, 0.000912345678])
                self.assertEqual(successful["selected_repeat"], 1)
                self.assertEqual(successful["warmup_seconds"], 0.0123456789)
                self.assertEqual(successful["first_call_seconds"], 0.00912345678)
                self.assertEqual(successful["first_call_scope"], "single_query")
                with self.assertRaisesRegex(RuntimeError, "1 of 2"):
                    _search(job, native)
                self.assertEqual(calls, [1, 2, 1])
                Path(successful["outputs"][0]["path"]).write_bytes(b"corrupt")
                with self.assertRaisesRegex(RuntimeError, "1 of 2"):
                    _search(job, native)
                self.assertEqual(calls, [1, 2, 1, 1, 2])

    def test_ground_truth_checks_length_and_order(self):
        import numpy as np
        from mvsic.benchmark.worker import read_ground_truth
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "truth.gt"
            path.write_bytes(struct.pack("<i", 2) + struct.pack("<fIfI", -2.5, 8, 0.2, 42))
            ids, distances = read_ground_truth(path, 1)
            np.testing.assert_array_equal(ids, [[8, 42]])
            self.assertAlmostEqual(float(distances[0, 0]), -2.5)
            with self.assertRaisesRegex(ValueError, "length"):
                read_ground_truth(path, 2)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "length"):
                read_ground_truth(path, 1)

    def test_recall_ties_duplicates_and_missing_results(self):
        import numpy as np
        from mvsic.benchmark.worker import recall
        truth = np.array([[10, 20, 30, 40], [10, 20, 30, 40]], dtype=np.uint32)
        distances = np.array([[0, 1, 1, 2], [0, 1, 1, 2]], dtype=np.float32)
        got = np.array([[10, 30], [20, 20]], dtype=np.uint32)
        self.assertEqual(recall(got, truth, distances, 2), 0.75)
        got[1] = [np.iinfo(np.uint32).max, np.iinfo(np.uint32).max]
        self.assertEqual(recall(got, truth, distances, 2), 0.5)
        with self.assertRaisesRegex(ValueError, "fewer"):
            recall(got, truth, distances, 5)

    def test_full_precision_survives_record_storage(self):
        value = 0.12345678901234567
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            atomic_json(path, {"recall": value, "seconds": [1.234567890123e-7]})
            record = json.loads(path.read_text())
            self.assertEqual(record["recall"], value)
            self.assertEqual(record["seconds"], [1.234567890123e-7])


if __name__ == "__main__":
    unittest.main()
