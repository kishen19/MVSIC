"""Bounded IP serving diagnostic: two implementations, three fixed pilot trees.

No index is built. Run only in a coordinated quiet window after validation.
Public search_with_stats diagnostics run after every timed ABBA trial finishes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from compare import (DEFAULT_MARKER, RESULT_ROOT, HostGuard, check_marker, identity,
                     library_environment, measured_process, pin_binary, pin_libraries,
                     read_predictions, recall, sample_queries, source_identity, truth_subset,
                     verify_pinned_libraries, write_json)


TREES = ("build-00-original.idx", "build-03-original.idx", "build-01-refactor.idx")
SCALING_WORKERS = (1, 4, 8, 24, 48, 96)
SCALING_CPUS = (80, 84, 88, 92, 64, 68, 72, 76, *range(0, 64, 4),
                *range(1, 96, 4), *range(2, 96, 4), *range(3, 96, 4))
STAGES = ("search_cmps", "probe_cmps", "center_distances_seconds", "beam_heap_seconds",
          "probe_heap_seconds", "root_seconds", "compression_seconds", "quantization_seconds",
          "leaf_seconds", "selection_seconds", "rerank_seconds", "greedy_seconds")
POLICY = {"metric": "ip", "mode": "latency", "k": 10, "nprobes": 16, "rerank": 50,
          "tq8-rerank": 1, "compression": "Wards", "threshold": 0.5, "root-m2m": 1,
          "leaf-size": 100, "branching": 0, "depth": 0, "niters": 5, "s": 0,
          "seed": 0, "assignment": "float", "warmup": 10, "repeats": 3}
KNOWN_ORIGINAL_ISSUES = [
    "Original AVX-512 TQ4 padding can let a zero padding lane beat real negative dots, changing IP traversal. "
    "Any observed cross-implementation difference is retained and remains unclassified until investigated.",
    "Original AVX2 TQ1 batch fallback returns FLT_MAX. This run uses original host-native AVX-512 single-query search, "
    "not that unsupported batch path. Strict correctness uses a separate AVX2 single-query reference.",
]


def schedule(blocks=2):
    if type(blocks) is not int or blocks not in (1, 2):
        raise ValueError("this focused diagnostic permits one or two ABBA blocks")
    return [{"block": block, "tree": tree, "implementation": kind}
            for block in range(blocks)
            for tree in (TREES if block % 2 == 0 else tuple(reversed(TREES)))
            for kind in ("original", "refactor", "refactor", "original")]


def fine_probe_schedule(probes=(14, 15, 16, 17), blocks=2):
    schedule(blocks)  # same bounded repetition policy as the cross-tree run
    if not probes or len(set(probes)) != len(probes) or any(p not in (14, 15, 16, 17) for p in probes):
        raise ValueError("fine-probe values must be distinct choices from 14,15,16,17")
    return [{"block": block, "comparison_probe": probe, "nprobes": 16 if kind == "original" else probe,
             "tree": TREES[1] if kind == "original" else TREES[2], "implementation": kind}
            for block in range(blocks)
            for probe in (probes if block % 2 == 0 else tuple(reversed(probes)))
            for kind in ("original", "refactor", "refactor", "original")]


def batch_scaling_schedule(workers=SCALING_WORKERS, blocks=2):
    schedule(blocks)
    if not workers or len(set(workers)) != len(workers) or any(n not in SCALING_WORKERS for n in workers):
        raise ValueError("scaling workers must be distinct choices from 1,4,8,24,48,96")
    return [{"block": block, "batch_size": batch, "threads": threads, "cpus": list(SCALING_CPUS[:threads]),
             "tree": TREES[1], "implementation": kind}
            for block in range(blocks)
            for batch in ((128, 1024) if block % 2 == 0 else (1024, 128))
            for threads in (workers if block % 2 == 0 else tuple(reversed(workers)))
            for kind in ("original", "refactor", "refactor", "original")]


def validate_scaling_topology(topology):
    """Reject a machine whose CPU numbering differs from this recorded plan."""
    cores = set()
    for cpu in SCALING_CPUS:
        package, core, siblings = topology[cpu]
        if package != cpu % 4 or set(siblings) != {cpu, cpu + 96} or (package, core) in cores:
            raise ValueError("batch scaling CPU topology differs from the prepared physical-core plan")
        cores.add((package, core))
    if len(cores) != 96:
        raise ValueError("batch scaling requires 96 distinct physical cores")


def summarize_scaling(rows):
    result = []
    for batch, threads in sorted({(row["batch_size"], row["threads"]) for row in rows}):
        group = [row for row in rows if (row["batch_size"], row["threads"]) == (batch, threads)]
        summaries = summarize(group)
        old = next(row for row in summaries if row["implementation"] == "original")
        for row in summaries:
            single = [r["native"]["qps"] for r in rows if r["batch_size"] == batch and r["threads"] == 1
                      and r["implementation"] == row["implementation"]]
            speedup = row["qps_median"] / statistics.median(single) if single else None
            result.append({**row, "batch_size": batch, "threads": threads,
                           "speedup_from_one_worker": speedup,
                           "parallel_efficiency": speedup / threads if speedup is not None else None,
                           "qps_ratio_to_original": row["qps_median"] / old["qps_median"],
                           "same_recall_as_original": row["recall_min"] == row["recall_max"] == old["recall_min"] == old["recall_max"]})
    return result


def summarize_fine_probes(rows):
    comparisons = []
    for probe in sorted({row["comparison_probe"] for row in rows}):
        # Pair each candidate with its adjacent baseline processes. Do not pool
        # the baselines across probe blocks and hide temporal variation.
        group = [row for row in rows if row["comparison_probe"] == probe]
        comparison = equal_recall_comparison(summarize(group))
        old, new = comparison["original"], comparison["refactor"]
        stable = old["recall_min"] == old["recall_max"] and new["recall_min"] == new["recall_max"]
        qualifies = stable and new["recall_min"] >= old["recall_max"]
        comparison.update(nprobes=probe, baseline_nprobes=16, stable_recall=stable,
                          meets_baseline_recall_without_tolerance=qualifies,
                          qps_ratio_at_equal_or_higher_recall=new["qps_median"] / old["qps_median"] if qualifies else None,
                          observed_qps_ratio=new["qps_median"] / old["qps_median"],
                          recall_delta=new["recall_min"] - old["recall_max"])
        comparisons.append(comparison)
    return comparisons


def prediction_difference(left, right, tolerance=1e-5):
    """Retain IDs and common-score disagreements; never waive baseline defects."""
    import numpy as np
    left_ids, left_distances = left
    right_ids, right_distances = right
    if left_ids.shape != right_ids.shape:
        raise ValueError("prediction shapes differ")
    output, maximum = [], 0.0
    for q, (a_ids, a_scores, b_ids, b_scores) in enumerate(zip(left_ids, left_distances, right_ids, right_distances)):
        a = {int(i): float(s) for i, s in zip(a_ids, a_scores) if i != np.iinfo(np.uint32).max}
        b = {int(i): float(s) for i, s in zip(b_ids, b_scores) if i != np.iinfo(np.uint32).max}
        common = set(a) & set(b)
        score_error = max((abs(a[i] - b[i]) for i in common), default=0.0)
        maximum = max(maximum, score_error)
        if set(a) != set(b) or score_error > tolerance:
            output.append({"query_row": q, "original_only_ids": sorted(set(a) - set(b)),
                           "refactor_only_ids": sorted(set(b) - set(a)),
                           "maximum_common_score_error": score_error})
    return {"matching": not output, "maximum_common_score_error": maximum,
            "score_tolerance": tolerance, "queries_with_differences": output,
            "attributed_to_known_original_bug": False}


def summarize(rows):
    result = []
    for tree in TREES:
        for kind in ("original", "refactor"):
            group = [row for row in rows if row["tree"] == tree and row["implementation"] == kind]
            if not group:
                continue
            result.append({"tree": tree, "implementation": kind, "trials": len(group),
                           "recall_min": min(row["recall"] for row in group),
                           "recall_max": max(row["recall"] for row in group),
                           "qps_median": statistics.median(row["native"]["qps"] for row in group),
                           "qps_min": min(row["native"]["qps"] for row in group),
                           "qps_max": max(row["native"]["qps"] for row in group),
                           "peak_rss_bytes_median": statistics.median(row["native"]["peak_rss_bytes"] for row in group)})
    return result


def equal_recall_comparison(summaries):
    old = next(row for row in summaries if row["tree"] == TREES[1] and row["implementation"] == "original")
    new = next(row for row in summaries if row["tree"] == TREES[2] and row["implementation"] == "refactor")
    equal = old["recall_min"] == old["recall_max"] == new["recall_min"] == new["recall_max"]
    return {"original": old, "refactor": new, "observed_recall_exactly_equal": equal,
            "qps_ratio_at_equal_recall": new["qps_median"] / old["qps_median"] if equal else None,
            "note": "Original build00 is not pooled into this originally equal-recall comparison."}


def pin_file(source, directory):
    before = identity(source)
    target = Path(directory) / before["sha256"] / Path(source).name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    target.chmod(0o444)
    pinned = identity(target)
    if before["sha256"] != pinned["sha256"]:
        raise RuntimeError("fixture changed while pinning")
    return target, {"source": before, "pinned": pinned}


def compiler_identity(path):
    path = Path(path).resolve(strict=True)
    commands = json.loads(path.read_text())
    selected = [row for row in commands if Path(row["file"]).name in
                ("native_bench.cc", "mvivf_search.cc", "mvivf_kernels.cc", "bindings.cpp")]
    compilers = {}
    import shlex
    for row in selected:
        command = row.get("arguments") or shlex.split(row["command"])
        executable = command[0]
        if executable not in compilers:
            compilers[executable] = {"binary": identity(executable),
                                    "version": subprocess.check_output([executable, "--version"], text=True)}
    return {"compile_commands": identity(path), "selected_commands": selected, "compilers": compilers,
            "scope": "original one IP TQ4-center/TQ1-leaf specialization; refactor supported codecs/ISAs; no compilation timed"}


def diagnostic_worker(job):
    import numpy as np
    from python_bench import load_extension
    native = load_extension(job["module"], "_mvsic")
    points = native.PointCloudSet.from_file(job["points"], True)
    queries = native.PointCloudSet.from_file(job["queries"], False)
    ip = native.IndexParams.mvivf(max_leaf_size=100, niters=5, seed=0)
    ip.quantizer = native.QuantizerType.OneBitTQ
    ip.quantize_centers = True
    params = native.SearchParams.mvivf(10, 16, 50, True, native.QueryCompression.Wards, 0.5)
    params.root_m2m = True
    index = native.IndexMVIVF(points.dims, ip)
    index.load(job["tree"], points)
    for q in range(min(10, len(queries))):
        index.search(queries[q], params)
    rows = []
    ids, distances = [], []
    for q in range(len(queries)):
        (found, scores), byte_count, stages = index.search_with_stats(queries[q], points, params)
        if len(stages) != len(STAGES):
            raise RuntimeError("unexpected public search_with_stats stage layout")
        ids.append(found); distances.append(scores)
        rows.append({"query_row": q, "raw_query_vectors": len(queries[q]), "bytes": byte_count,
                     **dict(zip(STAGES, stages))})
    output = Path(job["output"])
    np.savez(str(output) + ".npz", ids=np.asarray(ids, dtype=np.uint32), distances=np.asarray(distances, dtype=np.float32))
    write_json(output, {"purpose": "untimed diagnostic phase; public stage timers are not acceptance timings",
                        "tree": identity(job["tree"]), "queries": rows, "capabilities": native.capabilities(),
                        "aggregates": {key: {"mean": float(np.mean([row[key] for row in rows])),
                                               "median": float(np.median([row[key] for row in rows])),
                                               "p90": float(np.percentile([row[key] for row in rows], 90))}
                                       for key in ("bytes", *STAGES)}})


def run(args):
    import numpy as np
    check_marker(args.marker)
    output = Path(args.output).resolve()
    if not output.is_relative_to(RESULT_ROOT.resolve()):
        raise ValueError(f"output must be under {RESULT_ROOT}")
    if args.cpu not in os.sched_getaffinity(0):
        raise ValueError("requested physical CPU is unavailable")
    if args.batch_scaling:
        topology = {}
        for cpu in SCALING_CPUS:
            directory = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            topology[cpu] = (int((directory / "physical_package_id").read_text()),
                             int((directory / "core_id").read_text()),
                             tuple(map(int, (directory / "thread_siblings_list").read_text().split(","))))
        validate_scaling_topology(topology)
        if not set(SCALING_CPUS[:max(args.scaling_workers)]) <= os.sched_getaffinity(0):
            raise ValueError("scaling workers exceed the available CPU affinity")
    fixture = Path(args.fixture_dir).resolve(strict=True)
    old_provenance = json.loads((fixture / "provenance.json").read_text())
    config = json.loads((fixture / "config.json").read_text())
    if config.get("query_limit") != 128 or config.get("query_seed") != 42:
        raise ValueError("expected the original 128-query seed42 pilot fixture")
    if args.batch_scaling and config.get("build_threads") != 8:
        raise ValueError("scaling requires the fixed tree built with eight workers")
    output.mkdir(parents=True, exist_ok=False)
    args.created_output = True
    reserved = SCALING_CPUS[:max(args.scaling_workers)] if args.batch_scaling else [args.cpu]
    args.guard = HostGuard(output, enabled=args.quiet_host, shared_host=args.shared_host, reserved_cpus=reserved)
    args.guard.start()
    inputs = {key: identity(config[key]) for key in ("points", "queries", "ground_truth")}
    for key, current in inputs.items():
        if current["sha256"] != old_provenance["inputs"][key]["sha256"]:
            raise ValueError(f"pilot input changed: {key}")
    queries, query_id = pin_file(fixture / "queries.pcs", output / "fixtures")
    selection_file, selection_id = pin_file(fixture / "queries.pcs.selection.npy", output / "fixtures")
    if query_id["source"]["sha256"] != old_provenance["sampled_queries"]["sha256"] or selection_id["source"]["sha256"] != old_provenance["selection"]["sha256"]:
        raise ValueError("saved query sample differs from original pilot provenance")
    selection = np.load(selection_file, allow_pickle=False)
    with Path(config["queries"]).open("rb") as stream:
        header = np.fromfile(stream, dtype="<u8", count=3)
    expected = np.sort(np.random.default_rng(42).choice(int(header[1]), 128, replace=False))
    if not np.array_equal(selection, expected):
        raise ValueError("saved query selection is not the declared seed42 sample")
    if args.all_queries:
        queries, query_id = pin_file(config["queries"], output / "fixtures")
        selection = np.arange(int(header[1]), dtype=np.uint32)
        selection_file = output / "all-query-selection.npy"
        np.save(selection_file, selection, allow_pickle=False)
        selection_file.chmod(0o444)
        selection_id = {"pinned": identity(selection_file), "definition": "every original query in file order"}
    truth = truth_subset(config["ground_truth"], selection, int(header[1]))
    query_sets = {128: (queries, truth)}
    query_metadata = {128: {"queries": query_id, "selection": selection_id}}
    if args.batch_scaling:
        if int(header[1]) < 1024:
            raise ValueError("scaling requires at least 1024 real queries")
        sampled = output / "queries-1024.pcs"
        chosen, total = sample_queries(config["queries"], sampled, 1024, 42)
        sampled.chmod(0o444)
        query_sets[1024] = (sampled, truth_subset(config["ground_truth"], chosen, total))
        query_metadata[1024] = {"queries": identity(sampled), "selection": identity(str(sampled) + ".selection.npy")}
    trees, tree_ids = {}, {}
    builds = json.loads((fixture / "builds.json").read_text())
    for name in TREES:
        trees[name], tree_ids[name] = pin_file(fixture / name, output / "fixtures")
        expected_hash = next(row["artifact"]["sha256"] for row in builds if Path(row["artifact"]["path"]).name == name)
        if tree_ids[name]["source"]["sha256"] != expected_hash:
            raise ValueError(f"saved tree changed: {name}")
    binaries, libraries, binary_ids = {}, {}, {}
    for kind in ("original", "refactor"):
        binaries[kind], binary_ids[kind] = pin_binary(getattr(args, kind), output / "bin")
        libraries[kind], binary_ids[kind]["libraries"] = pin_libraries(getattr(args, kind), output / "libraries")
        verify_pinned_libraries(binaries[kind], libraries[kind], binary_ids[kind]["libraries"])
    module, module_id = pin_binary(args.module, output / "python")
    module_libraries, module_id["libraries"] = pin_libraries(args.module, output / "libraries")
    verify_pinned_libraries(module, module_libraries, module_id["libraries"])
    provenance = {"purpose": "fixed-tree IP regression diagnostic, not independent-build acceptance",
                  "experiment": "batch_scaling" if args.batch_scaling else "fine_probe" if args.fine_probes else "cross_tree",
                  "query_scope": "fixed_batches_128_1024" if args.batch_scaling else "full_confirmation" if args.all_queries else "pilot_128",
                  "query_count": [128, 1024] if args.batch_scaling else len(selection), "candidate_probes": args.candidate_probes if args.fine_probes else None,
                  "batch_query_sets": query_metadata if args.batch_scaling else None,
                  "tree_build_workers": config.get("build_threads"),
                  "scaling_affinities": {n: list(SCALING_CPUS[:n]) for n in args.scaling_workers} if args.batch_scaling else None,
                  "memory_policy": "Inherited Linux default/first-touch; process affinity changes only, no cache dropping or numactl override",
                  "fixtures": {"origin": str(fixture), "trees": tree_ids, "queries": query_id, "selection": selection_id},
                  "inputs": inputs, "binaries": binary_ids, "diagnostic_module": module_id,
                  "diagnostic_interpreter": identity(sys.executable), "numpy_version": np.__version__,
                  "compiler": compiler_identity(args.compile_commands), "policy": {**POLICY, **({"mode": "batch", "repeats": 5} if args.batch_scaling else {})}, "cpu": args.cpu,
                  "threads": args.scaling_workers if args.batch_scaling else 1, "refactor_isa": "avx512", "quiet_window_note": args.quiet_window_note,
                  "measurement_scope": args.guard.scope,
                  "marker": check_marker(args.marker), "known_original_issues": KNOWN_ORIGINAL_ISSUES,
                  "sources": {"original": source_identity(args.original_source), "refactor": source_identity(args.refactor_source)},
                  "environment": {key: os.environ.get(key) for key in ("MVSIC_CPU_ISA", "MVIVF_SUBBATCH", "LD_LIBRARY_PATH", "LD_PRELOAD", "MALLOC_CONF", "GLIBC_TUNABLES")},
                  "host": {"uname": list(os.uname()), "affinity": sorted(os.sched_getaffinity(0)),
                           "lscpu": json.loads(subprocess.check_output(["lscpu", "-J"], text=True))}}
    write_json(output / "provenance.json", provenance)
    plan = batch_scaling_schedule(args.scaling_workers, args.blocks) if args.batch_scaling else fine_probe_schedule(args.candidate_probes, args.blocks) if args.fine_probes else schedule(args.blocks)
    rows = []
    write_json(output / "plan.json", plan)
    for number, trial in enumerate(plan):
        label = f"timing-{number:02d}-{trial['implementation']}-{trial['tree'].removesuffix('.idx')}"
        result_path = output / (label + ".json")
        selected_queries, selected_truth = query_sets[trial["batch_size"]] if args.batch_scaling else (queries, truth)
        flags = {**POLICY, **({"mode": "batch", "repeats": 5} if args.batch_scaling else {}),
                 "nprobes": trial.get("nprobes", 16), "points": config["points"], "queries": selected_queries,
                 "index": trees[trial["tree"]], "output": result_path}
        command = [str(binaries[trial["implementation"]])]
        for key, value in flags.items():
            command += ["--" + key, str(value)]
        env = {**library_environment(libraries[trial["implementation"]]), "MVSIC_CPU_ISA": "avx512"}
        process = measured_process(command, output / label, marker=args.marker, threads=trial.get("threads", 1),
                                   cpus=trial.get("cpus", [args.cpu]), extra_env=env)
        native = json.loads(result_path.read_text())
        prepared_memory = {name: identity(str(result_path) + suffix) for name, suffix in
                           (("numa_maps", ".numa_maps"), ("proc_status", ".proc_status"))} if native.get("prepared_memory_snapshot") else None
        if args.batch_scaling and prepared_memory is None:
            raise RuntimeError("rebuild comparison drivers with prepared NUMA metadata before scaling")
        predictions = identity(str(result_path) + ".predictions")
        ids, _ = read_predictions(predictions["path"])
        score = recall(ids, selected_truth, 10)
        rows.append({**trial, "native": native, "recall": score, "correct_at_10": round(score * len(ids) * 10),
                     "recall_denominator": len(ids) * 10, "predictions": predictions, "process": process,
                     "prepared_memory": prepared_memory})
        write_json(output / "timings.json", rows)
        print(f"{number + 1}/{len(plan)} {trial['tree']} {trial['implementation']} recall={rows[-1]['recall']} qps={native['qps']}", flush=True)
    if args.batch_scaling:
        variations, cross_engine = [], []
        for batch in (128, 1024):
            for kind in ("original", "refactor"):
                group = [row for row in rows if row["batch_size"] == batch and row["implementation"] == kind]
                for row in group[1:]:
                    difference = prediction_difference(read_predictions(group[0]["predictions"]["path"]), read_predictions(row["predictions"]["path"]))
                    if not difference["matching"]:
                        variations.append({"batch_size": batch, "implementation": kind, "threads": row["threads"], "block": row["block"], **difference})
            old = next(row for row in rows if row["batch_size"] == batch and row["implementation"] == "original")
            new = next(row for row in rows if row["batch_size"] == batch and row["implementation"] == "refactor")
            cross_engine.append({"batch_size": batch, **prediction_difference(read_predictions(old["predictions"]["path"]), read_predictions(new["predictions"]["path"]))})
        write_json(output / "result.json", {"status": "complete", "experiment": "batch_scaling", "measurement_scope": args.guard.scope,
                   "operating_points": summarize_scaling(rows), "within_engine_thread_repeat_variations": variations,
                   "cross_engine_predictions": cross_engine, "known_original_issues": KNOWN_ORIGINAL_ISSUES,
                   "acceptance": "Fixed original03 tree built with eight workers; investigate every prediction or recall difference. NUMA placement can affect multi-node scaling."})
        return
    if args.fine_probes:
        variability = []
        for probe in args.candidate_probes:
            for kind in ("original", "refactor"):
                group = [row for row in rows if row["comparison_probe"] == probe and row["implementation"] == kind]
                for row in group[1:]:
                    difference = prediction_difference(read_predictions(group[0]["predictions"]["path"]),
                                                       read_predictions(row["predictions"]["path"]))
                    if not difference["matching"]:
                        variability.append({"comparison_probe": probe, "implementation": kind, "block": row["block"], **difference})
        write_json(output / "result.json", {
            "status": "complete", "experiment": "fine_probe", "measurement_scope": args.guard.scope,
            "query_scope": "full_confirmation" if args.all_queries else "pilot_128", "query_count": len(selection),
            "comparisons": summarize_fine_probes(rows), "within_operating_point_prediction_variation": variability,
            "known_original_issues": KNOWN_ORIGINAL_ISSUES,
            "acceptance": "No automatic default change. Pilot candidates need separate full-query confirmation; preserve every recall regression.",
            "prediction_scope": "Different trees and probe budgets intentionally may return different neighbors; every raw prediction is retained."})
        return
    summaries = summarize(rows)
    differences, variability = [], []
    for tree in TREES:
        old = next(row for row in rows if row["tree"] == tree and row["implementation"] == "original")
        new = next(row for row in rows if row["tree"] == tree and row["implementation"] == "refactor")
        differences.append({"tree": tree, **prediction_difference(read_predictions(old["predictions"]["path"]), read_predictions(new["predictions"]["path"]))})
        for kind in ("original", "refactor"):
            group = [row for row in rows if row["tree"] == tree and row["implementation"] == kind]
            for row in group[1:]:
                difference = prediction_difference(read_predictions(group[0]["predictions"]["path"]), read_predictions(row["predictions"]["path"]))
                if not difference["matching"]:
                    variability.append({"tree": tree, "implementation": kind, "block": row["block"], **difference})
    report = {"status": "timings_complete_diagnostics_pending", "measurement_scope": args.guard.scope, "summaries": summaries,
              "equal_recall_comparison": equal_recall_comparison(summaries), "cross_implementation_predictions": differences,
              "within_implementation_prediction_variation": variability, "known_original_issues": KNOWN_ORIGINAL_ISSUES}
    write_json(output / "result.json", report)
    diagnostics = []
    for name in TREES:
        label = "diagnostic-" + name.removesuffix(".idx")
        result_path = output / (label + ".json")
        job = {"module": str(module), "points": config["points"], "queries": str(queries),
               "tree": str(trees[name]), "output": str(result_path)}
        job_path = output / (label + "-job.json")
        write_json(job_path, job)
        process = measured_process([sys.executable, str(Path(__file__).resolve()), "--diagnostic-job", str(job_path)],
                                   output / label, marker=args.marker, threads=1, cpus=[args.cpu],
                                   extra_env={**library_environment(module_libraries), "MVSIC_CPU_ISA": "avx512"})
        timed = next(row for row in rows if row["tree"] == name and row["implementation"] == "refactor")
        with np.load(str(result_path) + ".npz", allow_pickle=False) as arrays:
            match = prediction_difference(read_predictions(timed["predictions"]["path"]), (arrays["ids"], arrays["distances"]))
        diagnostics.append({"tree": name, "record": identity(result_path), "predictions": identity(str(result_path) + ".npz"),
                            "process": process, "matches_timed_refactor": match})
        write_json(output / "diagnostics.json", diagnostics)
    report.update(status="complete", diagnostics=diagnostics,
                  acceptance="diagnostic only; investigate every prediction difference and repeat independent builds before acceptance")
    write_json(output / "result.json", report)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--diagnostic-job":
        diagnostic_worker(json.loads(Path(sys.argv[2]).read_text()))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--refactor", required=True)
    parser.add_argument("--module", required=True, help="validated new Python extension for separate public stats diagnostics")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixture-dir", default=str(RESULT_ROOT / "refactor/arguana-native-01"))
    parser.add_argument("--compile-commands", default="build/refactor-clang/compile_commands.json")
    parser.add_argument("--original-source", default="/home/kishen/MVSIC")
    parser.add_argument("--refactor-source", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--blocks", type=int, choices=(1, 2), default=2)
    parser.add_argument("--quiet-window-note", required=True)
    parser.add_argument("--marker", default=DEFAULT_MARKER)
    parser.add_argument("--plan-only", action="store_true")
    experiment = parser.add_mutually_exclusive_group()
    experiment.add_argument("--fine-probes", action="store_true", help="compare original03/probe16 with new01/probes14–17")
    experiment.add_argument("--batch-scaling", action="store_true", help="fixed original03 tree, batches128/1024, physical workers1/4/8/24/48/96")
    parser.add_argument("--scaling-workers", default="1,4,8,24,48,96", help="bounded subset for --batch-scaling")
    parser.add_argument("--candidate-probes", help="explicit comma-separated subset of 14,15,16,17; required for full confirmation")
    parser.add_argument("--all-queries", action="store_true", help="separate fine-probe confirmation using every original query")
    host = parser.add_mutually_exclusive_group()
    host.add_argument("--quiet-host", action="store_true", help="stop if sustained unrelated CPU work appears")
    host.add_argument("--shared-host", action="store_true", help="allow background work confined away from this physical core")
    args = parser.parse_args()
    args.scaling_workers = tuple(int(value) for value in args.scaling_workers.split(","))
    if args.batch_scaling:
        if not args.quiet_host and not args.plan_only:
            parser.error("batch scaling requires --quiet-host; shared-host results are a separate experiment")
        args.cpu = 80
    if (args.all_queries or args.candidate_probes) and not args.fine_probes:
        parser.error("--all-queries and --candidate-probes require --fine-probes")
    if args.all_queries and not args.candidate_probes:
        parser.error("full confirmation requires explicit --candidate-probes from the pilot")
    args.candidate_probes = tuple(int(value) for value in args.candidate_probes.split(",")) if args.candidate_probes else (14, 15, 16, 17)
    plan = batch_scaling_schedule(args.scaling_workers, args.blocks) if args.batch_scaling else fine_probe_schedule(args.candidate_probes, args.blocks) if args.fine_probes else schedule(args.blocks)
    if args.plan_only:
        print(json.dumps({"policy": {**POLICY, **({"mode": "batch", "repeats": 5} if args.batch_scaling else {})}, "timed_processes": plan, "diagnostic_processes": 0 if args.fine_probes or args.batch_scaling else 3,
                          "query_scope": "fixed_batches_128_1024" if args.batch_scaling else "full_confirmation" if args.all_queries else "pilot_128"}, indent=2))
        return
    args.created_output = False
    args.guard = None
    try:
        run(args)
    except BaseException as error:
        if args.created_output:
            write_json(Path(args.output) / "interruption.json", {"status": "incomplete_unsuitable_for_acceptance",
                       "error": repr(error), "host_guard_reason": args.guard.reason if args.guard else None})
        raise
    finally:
        if args.guard:
            args.guard.stop()


if __name__ == "__main__":
    main()
