"""Independent-build experimental design checks; no native work."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

from mvsic.benchmark.config import resolve_config


_directory = Path(__file__).parent / "refactor"
sys.path.insert(0, str(_directory))
try:
    _spec = importlib.util.spec_from_file_location("build_quality", _directory / "build_quality.py")
    quality = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(quality)
finally:
    sys.path.pop(0)


class BuildQualityTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.gettempdir()) / "mvsic-quality-plan-test"
        self.base = resolve_config({"version": 1, "dataset": {"points": "points", "queries": "queries", "ground_truth": "gt"},
                                    "index": {"params": {"assignment_quantizer": "TQ8"}}}, self.root / "base.json").resolved
        self.spec = {"version": 1, "base_config": "base.json", "model_seeds": [0, 1], "build_repetitions": 2,
                     "build_cpus": [3], "build_threads": 1,
                     "schedules": [{"name": "single", "mode": "latency_single", "threads": 1, "cpus": [3]}],
                     "training_variants": [{"name": "adaptive", "changes": {"inner_backend": "adaptive"}},
                                           {"name": "adaptive_seed", "parent": "adaptive", "changes": {"inner_init": "kmeans++"}}]}

    def test_each_recipe_seed_repeat_gets_own_artifact(self):
        plan = quality.create_plan(self.spec, self.base, self.root / "out", "training")
        self.assertEqual(len(plan["trials"]), 12)
        artifacts = [trial["build_config"]["build"]["output"] for trial in plan["trials"]]
        self.assertEqual(len(set(artifacts)), 12)
        for trial in plan["trials"]:
            resolve_config(trial["build_config"], self.root / "check.json")
            self.assertEqual(trial["build_config"]["index"]["metric"], "ip")
            self.assertEqual(trial["build_config"]["index"]["quantizer"], self.base["index"]["quantizer"])
            self.assertEqual(trial["build_config"]["index"]["params"]["assignment_quantizer"], "None")
            for search in trial["searches"]:
                self.assertEqual(search["config"]["build"]["output"], trial["build_config"]["build"]["output"])
                self.assertEqual(search["config"]["measurement"]["query_seed"], self.base["measurement"]["query_seed"])
        self.assertEqual(self.base["index"]["params"]["assignment_quantizer"], "TQ8")
        self.assertEqual([trial["recipe"] for trial in plan["trials"][:6]],
                         ["reference", "adaptive", "adaptive_seed", "adaptive_seed", "adaptive", "reference"])

    def test_quantization_phase_only_changes_assignment_codec(self):
        with self.assertRaisesRegex(ValueError, "selected_training"):
            quality.recipes(self.spec, self.base, "quantization")
        self.spec["selected_training"] = "adaptive_seed"
        recipes = quality.recipes(self.spec, self.base, "quantization")
        self.assertEqual([row["name"] for row in recipes], ["float", "TQ8", "TQ4", "TQ1"])
        reference = recipes[0]["params"]
        for candidate in recipes[1:]:
            differing = {key for key in reference if reference[key] != candidate["params"][key]}
            self.assertEqual(differing, {"assignment_quantizer"})
            self.assertEqual(candidate["parent"], "float")

    def test_training_variants_cannot_change_serving_or_outer_fields(self):
        for changes in ({"seed": 4}, {"assignment_quantizer": "TQ1"},
                        {"inner_metric": "l2"},
                        {"inner_backend": "adaptive", "inner_init": "kmeans++"}):
            self.spec["training_variants"] = [{"name": "invalid", "changes": changes}]
            with self.assertRaises(ValueError):
                quality.recipes(self.spec, self.base, "training")

    def test_recipe_subset_keeps_independent_artifacts_without_adding_parent(self):
        self.spec["selected_training"] = "reference"
        full = quality.create_plan(self.spec, self.base, self.root / "out", "quantization")
        selected = quality.select_recipes(full, ["TQ4", "TQ1"])
        self.assertEqual([recipe["name"] for recipe in selected["recipes"]], ["TQ4", "TQ1"])
        self.assertEqual(selected["recipe_filter"], ["TQ4", "TQ1"])
        self.assertEqual(selected["omitted_parent_recipes"], ["float"])
        self.assertEqual(len(selected["trials"]), 8)
        self.assertEqual([trial["recipe"] for trial in selected["trials"][:4]], ["TQ4", "TQ1", "TQ1", "TQ4"])
        self.assertEqual(len(full["trials"]), 16)
        report = quality.summarize(selected, [], [])
        self.assertEqual(report["paired_builds"], [])
        self.assertEqual(report["paired_recall_matches"], [])
        for request in ([], ["unknown"], ["TQ4", "TQ4"]):
            with self.assertRaisesRegex(ValueError, "distinct recipes"):
                quality.select_recipes(full, request)

    def test_matching_is_paired_by_seed_and_recall(self):
        plan = {"phase": "training", "spec": {}, "recipes": [{"name": "reference", "parent": None},
                 {"name": "adaptive", "parent": "reference"}]}
        point = {"k": 10, "nprobes": 8, "num_rerank": 20}
        def row(recipe, seed, recall, qps):
            return {"recipe": recipe, "seed": seed, "repetition": 0, "schedule": "single",
                    "record": {"params": point, "recall_at_k": recall, "qps": qps}}
        report = quality.summarize(plan, [], [row("reference", 0, .95, 100), row("adaptive", 0, .9, 1000),
                                             row("adaptive", 1, .99, 1000), row("adaptive", 0, .951, 110)])
        self.assertEqual(report["paired_recall_matches"][0]["qps_ratio"], 1.1)
        self.assertFalse(report["automatic_selection"])

    def test_rejects_invalid_diagnostics_and_tolerances_before_builds(self):
        for value in (-1, float("nan"), float("inf"), True):
            self.spec["recall_tolerance"] = value
            with self.assertRaises(ValueError):
                quality.create_plan(self.spec, self.base, self.root / "out", "training")
        self.spec.pop("recall_tolerance")
        self.spec["diagnostics"] = {"sample_documents": 8, "sample_seed": 0, "k": 9}
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            quality.create_plan(self.spec, self.base, self.root / "out", "training")


if __name__ == "__main__":
    unittest.main()
