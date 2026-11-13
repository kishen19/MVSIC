"""Paired independent-build experiments through the installed benchmark interface.

Unlike compare.py, every training recipe serves its own newly built skeleton.
Use --plan-only while the shared machine is reserved by the EMD experiments.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

from compare import DEFAULT_MARKER, RESULT_ROOT, HostGuard, check_marker, identity, measured_process, write_json


NAME = re.compile(r"[a-zA-Z0-9_-]+\Z")
INNER_FIELDS = {"inner_init", "inner_backend"}


def recipes(spec, base, phase):
    if base["index"]["metric"] != "ip":
        raise ValueError("this experiment holds serving metric fixed at ip")
    initial = copy.deepcopy(base["index"]["params"])
    initial["assignment_quantizer"] = "None"
    initial["build_with_8btq"] = False
    known = {"reference": {"name": "reference", "parent": None, "changes": {}, "params": initial}}
    for variant in spec.get("training_variants", []):
        if set(variant) - {"name", "parent", "changes"}:
            raise ValueError("unknown training variant field")
        name = variant.get("name", "")
        parent = variant.get("parent", "reference")
        if not NAME.fullmatch(name) or name in known or parent not in known:
            raise ValueError("recipe names must be unique; each parent must precede its child")
        changes = variant.get("changes", {})
        if len(changes) != 1:
            raise ValueError("change exactly one inner training field per parent-child comparison")
        key, value = next(iter(changes.items()))
        if key not in INNER_FIELDS or key not in initial:
            raise ValueError(f"{key} is not an exposed inner-training option in this installed version")
        params = copy.deepcopy(known[parent]["params"])
        if params[key] == value:
            raise ValueError("recipe must change its parent's training setting")
        params[key] = value
        known[name] = {"name": name, "parent": parent, "changes": changes, "params": params}
    if phase == "training":
        return list(known.values())
    selected = spec.get("selected_training")
    if selected not in known:
        raise ValueError("quantization phase needs selected_training from a completed float-assignment study")
    selected_params = known[selected]["params"]
    result = [{"name": "float", "parent": None, "changes": {}, "params": copy.deepcopy(selected_params),
               "selected_training": selected}]
    for quantizer in spec.get("assignment_quantizers", ["TQ8", "TQ4", "TQ1"]):
        if quantizer not in ("TQ8", "TQ4", "TQ1") or any(row["name"] == quantizer for row in result):
            raise ValueError("assignment_quantizers must be distinct TQ8/TQ4/TQ1 choices")
        params = copy.deepcopy(selected_params)
        params["assignment_quantizer"] = quantizer
        result.append({"name": quantizer, "parent": "float", "changes": {"assignment_quantizer": quantizer},
                       "params": params, "selected_training": selected})
    return result


def create_plan(spec, base, output, phase):
    allowed = {"version", "base_config", "model_seeds", "build_repetitions", "build_threads", "build_cpus",
               "training_variants", "selected_training", "assignment_quantizers", "schedules", "recall_tolerance", "diagnostics"}
    if set(spec) - allowed or type(spec.get("version")) is not int or spec["version"] != 1:
        raise ValueError("experiment requires version 1 and known fields")
    seeds = spec.get("model_seeds", [0, 1, 2])
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("model_seeds must contain distinct uint32 values")
    repetitions = spec.get("build_repetitions", 1)
    if type(repetitions) is not int or not 1 <= repetitions <= 100:
        raise ValueError("build_repetitions must be between one and 100")
    build_threads = spec.get("build_threads", base["build"]["threads"])
    if type(build_threads) is not int or build_threads < 1:
        raise ValueError("build_threads must be positive")
    build_cpus = spec.get("build_cpus")
    _cpus(build_cpus, build_threads)
    tolerance = spec.get("recall_tolerance", 0.001)
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError("recall_tolerance must be finite and in [0,1)")
    diagnostics = spec.get("diagnostics")
    if diagnostics is not None:
        if not isinstance(diagnostics, dict) or set(diagnostics) != {"sample_documents", "sample_seed", "k"}:
            raise ValueError("diagnostics requires sample_documents, sample_seed and k")
        for key in ("sample_documents", "k"):
            if type(diagnostics[key]) is not int or not 1 <= diagnostics[key] < 2**32:
                raise ValueError(f"diagnostics.{key} must be a positive uint32")
        if diagnostics["k"] > diagnostics["sample_documents"]:
            raise ValueError("diagnostics.k cannot exceed sample_documents")
        if type(diagnostics["sample_seed"]) is not int or not 0 <= diagnostics["sample_seed"] < 2**32:
            raise ValueError("diagnostics.sample_seed must be uint32")
    schedules = spec.get("schedules", [])
    if not schedules:
        raise ValueError("declare at least one measurement schedule")
    schedule_names = set()
    for schedule in schedules:
        if set(schedule) != {"name", "mode", "threads", "cpus"} or not NAME.fullmatch(schedule["name"]):
            raise ValueError("each schedule requires name, mode, threads and cpus")
        if schedule["name"] in schedule_names:
            raise ValueError("schedule names must be unique")
        schedule_names.add(schedule["name"])
        if schedule["mode"] not in ("latency_single", "latency_parallel", "throughput_batch"):
            raise ValueError("invalid measurement schedule mode")
        if type(schedule["threads"]) is not int or schedule["threads"] < 1:
            raise ValueError("schedule threads must be positive")
        if schedule["mode"] == "latency_single" and schedule["threads"] != 1:
            raise ValueError("latency_single requires one thread")
        _cpus(schedule["cpus"], schedule["threads"])
    variants = recipes(spec, base, phase)
    trials = []
    for seed_index, seed in enumerate(seeds):
        for repetition in range(repetitions):
            ordered = variants if (seed_index + repetition) % 2 == 0 else list(reversed(variants))
            for variant in ordered:
                directory = Path(output) / f"seed-{seed}-repeat-{repetition}" / variant["name"]
                config = copy.deepcopy(base)
                config["name"] = f'{phase}/{variant["name"]}/seed={seed}/repeat={repetition}'
                config["index"]["params"] = copy.deepcopy(variant["params"])
                config["index"]["params"]["seed"] = seed
                config["build"] = {"output": str(directory / "index.mvivf"), "threads": build_threads, "cpu": None}
                config["output"]["directory"] = str(directory / "build-runs")
                config["measurement"]["cpu"] = None
                searches = []
                for schedule in schedules:
                    search = copy.deepcopy(config)
                    search["measurement"].update(mode=schedule["mode"], threads=schedule["threads"], cpu=None)
                    search["output"]["directory"] = str(directory / f'search-{schedule["name"]}')
                    searches.append({"name": schedule["name"], "cpus": schedule["cpus"], "config": search})
                trials.append({"recipe": variant["name"], "parent": variant["parent"], "changes": variant["changes"],
                               "seed": seed, "repetition": repetition, "directory": str(directory),
                               "build_cpus": build_cpus, "build_config": config, "searches": searches})
    return {"version": 1, "purpose": "independent MVIVF build quality at fixed IP serving", "phase": phase,
            "spec": spec, "recipes": variants, "trials": trials,
            "order": "forward and reverse recipe order on alternating seed/repetition blocks"}


def _cpus(cpus, threads):
    if not isinstance(cpus, list) or len(cpus) != threads or any(type(cpu) is not int or cpu < 0 for cpu in cpus) or len(set(cpus)) != len(cpus):
        raise ValueError("provide exactly one distinct pinned CPU per worker thread")


def select_recipes(plan, requested):
    if requested is None:
        return plan
    known = {recipe["name"] for recipe in plan["recipes"]}
    selected = set(requested)
    if not requested or len(selected) != len(requested) or not selected <= known:
        raise ValueError("--recipe must name distinct recipes present in this experiment")
    recipes = [recipe for recipe in plan["recipes"] if recipe["name"] in selected]
    return {**plan, "recipes": recipes,
            "trials": [trial for trial in plan["trials"] if trial["recipe"] in selected],
            "recipe_filter": requested,
            "omitted_parent_recipes": sorted({recipe["parent"] for recipe in recipes
                                               if recipe["parent"] is not None and recipe["parent"] not in selected})}


def runtime_key(runtime):
    # Affinity differs intentionally between build/search schedules. Implementation,
    # kernel environment and numerical-library versions must remain identical.
    return {key: runtime.get(key) for key in ("native", "native_libraries", "harness_sha256", "numpy_version",
                                            "python", "platform", "cpu", "environment")}


def summarize(plan, builds, searches):
    tolerance = plan["spec"].get("recall_tolerance", 0.001)
    if type(tolerance) not in (int, float) or not 0 <= tolerance < 1:
        raise ValueError("recall_tolerance must be in [0,1)")
    construction = []
    for recipe in plan["recipes"]:
        rows = [row for row in builds if row["recipe"] == recipe["name"]]
        if not rows:
            continue
        construction.append({"recipe": recipe["name"], "parent": recipe["parent"], "builds": len(rows),
                             "build_seconds_median": statistics.median(row["record"]["build_seconds"] for row in rows),
                             "build_seconds_min": min(row["record"]["build_seconds"] for row in rows),
                             "build_seconds_max": max(row["record"]["build_seconds"] for row in rows),
                             "peak_rss_bytes_median": statistics.median(row["record"]["memory_after"]["peak_rss_bytes"] for row in rows),
                             "artifact_bytes_median": statistics.median(row["record"]["artifact"]["bytes"] for row in rows)})
    matches, budget_matches, build_matches = [], [], []
    for recipe in plan["recipes"]:
        parent = recipe["parent"]
        if parent is None:
            continue
        for reference in (row for row in builds if row["recipe"] == parent):
            candidate = next((row for row in builds if row["recipe"] == recipe["name"]
                              and (row["seed"], row["repetition"]) ==
                                  (reference["seed"], reference["repetition"])), None)
            if candidate is not None:
                old, new = reference["record"], candidate["record"]
                build_matches.append({"recipe": recipe["name"], "parent": parent,
                                      "seed": reference["seed"], "repetition": reference["repetition"],
                                      "build_speed_ratio": old["build_seconds"] / new["build_seconds"],
                                      "peak_rss_ratio": new["memory_after"]["peak_rss_bytes"] / old["memory_after"]["peak_rss_bytes"],
                                      "artifact_bytes_ratio": new["artifact"]["bytes"] / old["artifact"]["bytes"]})
        for reference in (row for row in searches if row["recipe"] == parent):
            # Serving policy is fixed; only search budgets may vary for matching.
            fixed = {key: value for key, value in reference["record"]["params"].items() if key not in ("nprobes", "num_rerank")}
            candidates = [row for row in searches if row["recipe"] == recipe["name"]
                          and (row["seed"], row["repetition"], row["schedule"]) ==
                              (reference["seed"], reference["repetition"], reference["schedule"])
                          and {key: value for key, value in row["record"]["params"].items() if key not in ("nprobes", "num_rerank")} == fixed
                          and row["record"]["recall_at_k"] >= reference["record"]["recall_at_k"] - tolerance]
            best = max(candidates, key=lambda row: row["record"]["qps"]) if candidates else None
            matches.append({"recipe": recipe["name"], "parent": parent, "seed": reference["seed"],
                            "repetition": reference["repetition"], "schedule": reference["schedule"],
                            "reference_params": reference["record"]["params"],
                            "reference_recall": reference["record"]["recall_at_k"],
                            "candidate_params": None if best is None else best["record"]["params"],
                            "candidate_recall": None if best is None else best["record"]["recall_at_k"],
                            "qps_ratio": None if best is None else best["record"]["qps"] / reference["record"]["qps"]})
            exact = next((row for row in searches if row["recipe"] == recipe["name"]
                          and (row["seed"], row["repetition"], row["schedule"]) ==
                              (reference["seed"], reference["repetition"], reference["schedule"])
                          and row["record"]["params"] == reference["record"]["params"]), None)
            if exact is not None:
                budget_matches.append({"recipe": recipe["name"], "parent": parent,
                                       "seed": reference["seed"], "repetition": reference["repetition"],
                                       "schedule": reference["schedule"], "params": reference["record"]["params"],
                                       "recall_delta": exact["record"]["recall_at_k"] - reference["record"]["recall_at_k"],
                                       "qps_ratio": exact["record"]["qps"] / reference["record"]["qps"]})
    return {"schema_version": 1, "phase": plan["phase"], "construction": construction,
            "paired_builds": build_matches, "paired_recall_matches": matches,
            "paired_identical_budgets": budget_matches, "recall_tolerance": tolerance,
            "automatic_selection": False,
            "interpretation": "Each recipe serves its own independent build; inspect missing recall and slower matches before choosing a recipe."}


def execute(plan, output, marker):
    from mvsic.benchmark.config import resolve_config
    implementation = None
    query_selection = None
    inputs = {}
    builds, searches, profiles = [], [], []
    requested_cpus = set()
    for trial in plan["trials"]:
        requested_cpus.update(trial["build_cpus"])
        for search in trial["searches"]:
            requested_cpus.update(search["cpus"])
    if not requested_cpus <= os.sched_getaffinity(0):
        raise ValueError("experiment requests CPUs outside this process's available affinity")

    def launch(action, config, directory, cpus):
        nonlocal implementation
        filename = directory / f"{action}-config.json"
        validated = resolve_config(config, filename)
        write_json(filename, validated.resolved)
        execution = config["build"] if action == "build" else config["measurement"]
        process = directory / f"{action}-process"
        measured_process([sys.executable, "-m", "mvsic.cli", "benchmark", action, "--config", str(filename)],
                         process, marker=marker, threads=execution["threads"], cpus=cpus,
                         cwd=directory, extra_env={"PYTHONPATH": ""})
        lines = (process / "stdout.log").read_text().strip().splitlines()
        results = Path(lines[-1])
        manifest = json.loads((results / "manifest.json").read_text())
        key = runtime_key(manifest["runtime"])
        if implementation is None:
            implementation = key
        elif key != implementation:
            raise RuntimeError("native implementation, environment or numerical-library version changed during experiment")
        for name, item in manifest["inputs"].items():
            if name in inputs and inputs[name] != item:
                raise RuntimeError(f"dataset {name} changed during experiment")
            inputs[name] = item
        return results

    for trial in plan["trials"]:
        metadata = {key: trial[key] for key in ("recipe", "parent", "changes", "seed", "repetition")}
        directory = Path(trial["directory"])
        built = launch("build", trial["build_config"], directory, trial["build_cpus"])
        record = json.loads((built / "build.json").read_text())
        if record["resumed"]:
            raise RuntimeError("independent construction repetition unexpectedly reused an existing build")
        builds.append({**metadata, "record": record, "results": str(built)})
        write_json(output / "builds.json", builds)
        for search in trial["searches"]:
            measured = launch("search", search["config"], directory / search["name"], search["cpus"])
            point_plan = json.loads((measured / "plan.json").read_text())["points"]
            for point in point_plan:
                record = json.loads((measured / f'search-{point["fingerprint"]}.json').read_text())
                if record["status"] != "complete" or record["recall_at_k"] is None:
                    raise RuntimeError("all requested search points need complete timing and ground truth")
                if query_selection is None:
                    query_selection = record["query_selection_sha256"]
                elif query_selection != record["query_selection_sha256"]:
                    raise RuntimeError("query sample/order differs across recipes")
                if record["artifact"] != builds[-1]["record"]["artifact"]:
                    raise RuntimeError("search did not use this recipe's own newly built skeleton")
                searches.append({**metadata, "schedule": search["name"], "record": record, "results": str(measured)})
            write_json(output / "searches.json", searches)
        diagnostics = plan["spec"].get("diagnostics")
        if diagnostics is not None:
            filename = directory / "diagnostic-job.json"
            result_path = directory / "diagnostic.json"
            write_json(filename, {"config": trial["build_config"], "diagnostics": diagnostics,
                                  "runtime": implementation, "inputs": inputs, "output": str(result_path)})
            measured_process([sys.executable, str(Path(__file__).with_name("clustering_profile.py")),
                              "--job", str(filename)], directory / "diagnostic-process", marker=marker,
                             threads=trial["build_config"]["build"]["threads"], cpus=trial["build_cpus"],
                             cwd=directory, extra_env={"PYTHONPATH": ""})
            profiles.append({**metadata, "record": json.loads(result_path.read_text())})
            write_json(output / "profiles.json", profiles)
        write_json(output / "summary.json", summarize(plan, builds, searches))
    write_json(output / "implementation.json", {"runtime": implementation, "inputs": inputs,
                                                "query_selection_sha256": query_selection})
    return summarize(plan, builds, searches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("training", "quantization"), default="training")
    parser.add_argument("--output", required=True)
    parser.add_argument("--marker", default=DEFAULT_MARKER)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--recipe", action="append", help="run only this named recipe; repeat for a subset without rebuilding completed controls")
    host = parser.add_mutually_exclusive_group()
    host.add_argument("--quiet-host", action="store_true", help="stop if sustained unrelated CPU work appears")
    host.add_argument("--shared-host", action="store_true", help="allow background work confined away from configured physical cores")
    args = parser.parse_args()
    from mvsic.benchmark.config import load_config, resolve_config
    source = Path(args.config).resolve(strict=True)
    spec = json.loads(source.read_text())
    base_path = Path(spec["base_config"])
    base = load_config(base_path if base_path.is_absolute() else source.parent / base_path).resolved
    if "ground_truth" not in base["dataset"] or "queries" not in base["dataset"]:
        raise ValueError("build-quality comparisons require queries and ground truth")
    output = Path(args.output).expanduser().resolve()
    if not output.is_relative_to(RESULT_ROOT):
        raise ValueError(f"output must be under {RESULT_ROOT}")
    plan = select_recipes(create_plan(spec, base, output, args.phase), args.recipe)
    plan["provenance"] = {"spec": identity(source), "base_config": identity(base_path if base_path.is_absolute() else source.parent / base_path),
                          "controller": {name: identity(Path(__file__).with_name(name)) for name in
                                         ("build_quality.py", "clustering_profile.py", "compare.py")}}
    # Validate every generated config before any construction begins.
    for trial in plan["trials"]:
        resolve_config(trial["build_config"], Path(trial["directory"]) / "build-config.json")
        for search in trial["searches"]:
            resolve_config(search["config"], Path(trial["directory"]) / search["name"] / "search-config.json")
    if not args.plan_only:
        check_marker(args.marker)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    if not args.plan_only:
        cpus = {cpu for trial in plan["trials"] for cpu in trial["build_cpus"]}
        cpus.update(cpu for trial in plan["trials"] for search in trial["searches"] for cpu in search["cpus"])
        with HostGuard(output, enabled=args.quiet_host, shared_host=args.shared_host, reserved_cpus=cpus) as guard:
            result = execute(plan, output, args.marker)
            write_json(output / "summary.json", {**result, "measurement_scope": guard.scope})
    print(output)


if __name__ == "__main__":
    main()
