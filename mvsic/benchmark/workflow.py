"""The original dataset/build/variant YAML workflow, using compiled workers."""

from __future__ import annotations

import copy
import itertools
import json
import math
import os
from pathlib import Path

from .config import SEARCH_DEFAULTS, digest, resolve_config
from .runner import atomic_json, build_identity, execution_runtime, file_identity, runtime_identity, writer_lock, _worker


MODES = {"latency": "latency_single", "multi_latency": "latency_parallel", "batch": "throughput_batch"}
LABELS = ("search_cmps", "probe_cmps", "t_search_dists", "t_search_beam", "t_search_rest",
          "t_search_top_level", "t_compress", "t_quant", "t_leaf_dists", "t_leaf_rest", "t_rerank", "t_greedy")


def is_workflow(raw):
    return isinstance(raw, dict) and "datasets" in raw and "indices" in raw


def _fields(value, allowed, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    extra = set(value) - set(allowed)
    if extra:
        raise ValueError(f"unsupported {where} fields: {', '.join(sorted(map(str, extra)))}")


def _name(value, where, empty=False):
    if not isinstance(value, str) or (not value and not empty) or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError(f"{where} must be a single directory/file name")
    return value


def _rows(value, where):
    if not isinstance(value, list) or not value or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{where} must be a nonempty list of mappings")
    return value


def _bool(value, where):
    if type(value) is not bool:
        raise ValueError(f"{where} must be boolean")
    return value


def _quantizer(value):
    names = {"none": "None", "float": "None", "tq": "TQ", "turboquant": "TQ", "tq4": "TQ",
             "onebittq": "OneBitTQ", "tq1": "OneBitTQ", "eightbittq": "EightBitTQ", "tq8": "EightBitTQ"}
    if not isinstance(value, str) or value.lower() not in names:
        raise ValueError(f"unsupported MVIVF serving quantizer {value!r}; supported: None, TQ, OneBitTQ, EightBitTQ")
    return names[value.lower()]


def expand_search(search):
    """Retain named correlated rows, then expand each row's Cartesian axes."""
    _fields(search, {"name", "params", "append"}, "search config")
    _name(search.get("name"), "search name")
    _bool(search.get("append", True), "search append")
    params = search.get("params") or {}
    _fields(params, set(SEARCH_DEFAULTS) | {"config"}, "search params")
    variants = params.get("config", [{"name": ""}])
    result, seen = [], set()
    for variant in _rows(variants, "params.config"):
        _fields(variant, set(SEARCH_DEFAULTS) | {"name"}, "params.config entry")
        name = _name(variant.get("name", ""), "config name", empty=True)
        if name in seen:
            raise ValueError(f"duplicate search config row name: {name!r}")
        seen.add(name)
        axes = {key: value for key, value in params.items() if key != "config"}
        axes.update({key: value for key, value in variant.items() if key != "name"})
        axes.setdefault("num_rerank", 0)
        axes.setdefault("nprobes", 1)
        for key, value in axes.items():
            axes[key] = value if isinstance(value, list) else [value]
            if not axes[key]:
                raise ValueError(f"empty search axis: {key}")
        size = 1
        for axis in axes.values():
            size *= len(axis)
        if size > 100000:
            raise ValueError("search grid exceeds 100000 operating points")
        keys = list(axes)
        points = [dict(zip(keys, values)) for values in itertools.product(*(axes[key] for key in keys))]
        for point in points:
            if "query_compression" in point:
                names = {"NONE": "None", "CARVE": "Carve", "WARDS": "Wards"}
                value = point["query_compression"]
                if str(value).upper() not in names:
                    raise ValueError(f"unsupported query_compression: {value}")
                point["query_compression"] = names[str(value).upper()]
        points.sort(key=lambda point: point["nprobes"])
        result.append({"name": name, "points": points})
    return result


def _measurement(options):
    mode = options.get("mode")
    if mode is not None and mode not in MODES:
        raise ValueError("mode must be latency, multi_latency or batch")
    threads = options.get("num_threads")
    if options.get("latency") and mode is None and threads is None:
        mode = "latency"
    ambient = int(os.environ.get("PARLAY_NUM_THREADS", len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1))
    if mode == "latency":
        threads = 1
    else:
        threads = ambient if threads is None else threads
    legacy_both = mode is None and options.get("num_threads") is None
    selected = "latency_parallel" if mode is None else MODES[mode]
    if threads == 1 and selected == "latency_parallel":
        selected = "latency_single"
    repeats = options.get("repeats")
    if repeats is None:
        repeats = 3 if legacy_both or mode == "batch" else 1
    warmup = options.get("warmup_queries")
    if warmup is None:
        warmup = 100 if legacy_both else 10
    return {"mode": selected, "threads": threads, "repeats": repeats, "warmup_queries": warmup,
            "query_limit": max(0, options.get("query_subsample", 1000)),
            "query_seed": options.get("query_subsample_seed", 42), "python_timing": False}, mode, legacy_both


def prepare(raw, source, action, **options):
    """Validate the complete workflow without importing native code or reading data."""
    _fields(raw, {"name", "datasets", "indices"}, "workflow")
    if action not in ("build", "search"):
        raise ValueError("workflow action must be build or search")
    threshold = options.get("qps_thresh", 10.0)
    if type(threshold) not in (int, float) or not math.isfinite(threshold):
        raise ValueError("QPS_thresh must be finite")
    datasets, indices = _rows(raw.get("datasets"), "datasets"), _rows(raw.get("indices"), "indices")
    base = Path(options.get("base_dir") or Path.cwd()).expanduser().resolve()
    source = Path(source).resolve()
    measurement, mode, legacy_both = _measurement(options)
    build_threads = options.get("build_threads")
    if build_threads is None:
        build_threads = int(os.environ.get("PARLAY_NUM_THREADS", len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1))
    jobs, paths = [], set()

    def absolute(value):
        if not isinstance(value, str) or not value:
            raise ValueError("workflow paths must be nonempty strings")
        path = Path(value).expanduser()
        return (path if path.is_absolute() else base / path).resolve()

    for ds in datasets:
        _fields(ds, {"name", "path", "index_dir", "results_dir", "results", "is_mmap"}, "dataset")
        name = _name(ds.get("name"), "dataset name")
        data = absolute(ds.get("path"))
        # Earlier combined configs kept both the index and CSVs under results.
        results = ds.get("results_dir", ds.get("results"))
        index_dir = ds.get("index_dir", ds.get("results"))
        if index_dir is None:
            raise ValueError(f"dataset {name}: index_dir is required")
        index_dir = absolute(index_dir)
        if action == "search" and results is None:
            raise ValueError(f"dataset {name}: results_dir (or results) is required")
        results_dir = absolute(results) if results is not None else index_dir
        dataset = {"name": name, "points": str(data / f"{name}_points.pcs"),
                   "mmap": _bool(ds.get("is_mmap", False), "dataset is_mmap")}
        if action == "search":
            dataset.update(queries=str(data / f"{name}_queries.pcs"), ground_truth=str(data / f"{name}_chamfer_neighbors.gt"))
        for index in indices:
            _fields(index, {"name", "metric", "build_configs", "builds"}, "index")
            if index.get("name") != "mvivf":
                raise ValueError(f"installed benchmark supports mvivf; method {index.get('name')!r} is not installed")
            if str(index.get("metric", "ip")).lower() != "ip":
                raise ValueError("MVIVF supports only ip (MaxSim); L2 is retired")
            if "builds" in index and "build_configs" in index:
                raise ValueError("use builds or build_configs, not both in one index")
            combined = "build_configs" in index
            builds = _rows(index.get("build_configs" if combined else "builds"), "builds")
            for build in builds:
                _fields(build, {"name", "params", "rebuild", "build_name", "build_params", "variants", "search_configs"}, "build")
                build_name = _name(build.get("name" if combined else "build_name"), "build name")
                params = copy.deepcopy(build.get("params" if combined else "build_params") or {})
                _bool(build.get("rebuild", False), "build rebuild")
                pq_method = params.pop("pq_method", 0)
                old_quantizers = {0: "None", 4: "TQ"}
                if type(pq_method) is not int or pq_method not in old_quantizers:
                    raise ValueError(f"legacy pq_method={pq_method} is retired; supported MVIVF codecs are None/TQ/OneBitTQ/EightBitTQ")
                output = index_dir / "mvivf" / build_name / "index.bin"
                value = {"version": 1, "name": f"{name}/mvivf/{build_name}", "dataset": dataset,
                         "index": {"name": "mvivf", "metric": "ip", "quantizer": "None", "quantize_centers": False, "params": params},
                         "build": {"output": str(output), "threads": build_threads},
                         "search": {"nprobes": 1, "num_rerank": 0}, "measurement": measurement,
                         "output": {"directory": str(results_dir / "mvivf" / build_name)}}
                config = resolve_config(value, source)
                if action == "build":
                    if str(output) in paths:
                        raise ValueError(f"duplicate build output: {output}")
                    paths.add(str(output))
                    jobs.append({"config": config.resolved, "force": bool(options.get("force") or options.get("rebuild") or build.get("rebuild")),
                                 "legacy": {"build_name": build_name}})
                    continue
                variants = build.get("variants") or [{"name": "", "compress": False, "quantizer": old_quantizers[pq_method]}]
                for variant in _rows(variants, "variants"):
                    _fields(variant, {"name", "compress", "quantizer"}, "serving variant")
                    variant_name = _name(variant.get("name", ""), "variant name", empty=True)
                    representation = {"quantizer": _quantizer(variant.get("quantizer", "None")),
                                      "quantize_centers": _bool(variant.get("compress", False), "variant compress")}
                    for search in _rows(build.get("search_configs"), "search_configs"):
                        for group in expand_search(search):
                            value = copy.deepcopy(config.resolved)
                            value["index"].update(representation)
                            directory = results_dir / "mvivf" / build_name
                            if variant_name:
                                directory /= variant_name
                            directory /= search["name"]
                            prefix = f"{mode}_" if mode else ("latency_" if options.get("num_threads") else "")
                            suffix = "_" + group["name"] if group["name"] else ""
                            csv_path = directory / f"{prefix}results{suffix}.csv"
                            if str(csv_path) in paths:
                                raise ValueError(f"duplicate search CSV: {csv_path}")
                            paths.add(str(csv_path))
                            value["output"]["directory"] = str(directory)
                            points = []
                            for point in group["points"]:
                                value["search"] = point
                                points.append(resolve_config(value, source).points()[0])
                            jobs.append({"config": resolve_config(value, source).resolved, "points": points,
                                         "force": bool(options.get("force") or not search.get("append", True)),
                                         "legacy": {"csv": str(csv_path), "mode": mode, "both_modes": legacy_both,
                                                    "build_directory": str(results_dir / "mvivf" / build_name),
                                                    "qps_threshold": options.get("qps_thresh", 10.0),
                                                    "fail_fast": bool(options.get("fail_fast")), "labels": list(LABELS)}})
    return jobs


def _check_artifact(value, inputs):
    artifact = Path(value["build"]["output"])
    metadata = Path(str(artifact) + ".json")
    record = json.loads(metadata.read_text()) if metadata.exists() else None
    if record is not None:
        if record.get("status") != "complete" or record.get("artifact") != file_identity(artifact):
            raise ValueError(f"incomplete or changed index artifact: {artifact}")
        expected = copy.deepcopy(value["index"])
        expected.update(quantizer="None", quantize_centers=False)
        if record.get("index") != expected or record.get("inputs", {}).get("points") != inputs["points"]:
            raise ValueError(f"build parameters or corpus differ from saved skeleton: {artifact}")
        return record["fingerprint"], None
    # Original v4 skeletons predate content fingerprints. Preserve that fact and
    # verify their declared structural parameters before the native loader checks
    # topology, extents and document ownership. Never invent a measured build.
    params_path = artifact.with_name("index_params.json")
    if not artifact.is_file() or not params_path.is_file():
        raise ValueError(f"missing index/provenance: {artifact}; run benchmark build first")
    declared = json.loads(params_path.read_text())
    params = value["index"]["params"]
    for key in ("k_per_level", "max_leaf_size", "max_depth", "s", "compress_input", "build_with_8btq"):
        expected = params[key] if key != "build_with_8btq" else params["assignment_quantizer"] == "TQ8"
        if declared.get(key) != expected:
            raise ValueError(f"original index metadata disagrees with {key}: {params_path}")
    for key, actual in declared.get("mvclus", {}).items():
        if key == "init":
            actual = str(actual).split(".")[-1]
        if key in params and params[key] != actual:
            raise ValueError(f"original index metadata disagrees with {key}: {params_path}")
    external = {"artifact": file_identity(artifact), "declared_parameters": file_identity(params_path),
                "origin": "external_original_skeleton; original build inputs/runtime are unverified"}
    return None, external


def run(action, raw, source, **options):
    jobs = prepare(raw, source, action, **options)
    if action == "search":
        # Named rerank/config sweeps share the same loaded representation and
        # lazy reranker. Grouping changes no CSV paths or stop-policy boundaries.
        merged = {}
        for job in jobs:
            key = digest({k: v for k, v in job["config"].items() if k != "search"})
            if key not in merged:
                merged[key] = {"config": job["config"], "force": False, "groups": []}
            merged[key]["groups"].append({k: job[k] for k in ("points", "legacy", "force")})
        jobs = list(merged.values())
    runtime, cached_inputs, failures, directories = runtime_identity(), {}, [], []
    for job in jobs:
        value = job["config"]
        directory = None
        try:
            inputs = {}
            for key, filename in value["dataset"].items():
                if key not in ("points", "queries", "ground_truth"):
                    continue
                if filename not in cached_inputs:
                    cached_inputs[filename] = file_identity(filename)
                inputs[key] = cached_inputs[filename]
            job_runtime = execution_runtime(runtime, value, os.environ)
            context = {"schema_version": 1, "inputs": inputs, "runtime": job_runtime,
                       "config_fingerprint": digest(value), "build_fingerprint": build_identity(value, inputs, job_runtime)}
            stable = {key: val for key, val in value.items() if key != "search"}
            directory = Path(value["output"]["directory"]) / ".mvsic" / digest({"config": stable, "runtime": job_runtime, "inputs": inputs})
            directory.mkdir(parents=True, exist_ok=True)
            with writer_lock(directory / ".lock"), writer_lock(Path(value["build"]["output"] + ".lock")):
                if action == "search":
                    context["build_fingerprint"], external = _check_artifact(value, inputs)
                    if external:
                        context["external_artifact"] = external
                    source_stats = Path(value["build"]["output"]).with_name("build_stats.json")
                    if source_stats.is_file():
                        stats = json.loads(source_stats.read_text())
                        for group in job["groups"]:
                            atomic_json(Path(group["legacy"]["build_directory"]) / "build_stats.json", stats)
                atomic_json(directory / "manifest.json", context)
                atomic_json(directory / "resolved_config.json", value)
                _worker(action, value, context, directory, job["force"],
                        **{key: job[key] for key in ("legacy", "points", "groups") if key in job})
                if action == "build":
                    record = json.loads((directory / "build.json").read_text())
                    target = Path(value["build"]["output"]).parent
                    params = value["index"]["params"]
                    structural = {key: params[key] for key in ("k_per_level", "max_leaf_size", "max_depth", "s", "compress_input")}
                    structural.update(method="MVIVF", num_spill=1, num_spill_l2=1,
                                      build_with_8btq=params["assignment_quantizer"] == "TQ8",
                                      mvclus={key: params[key] for key in ("niters", "max_point_clouds_per_cluster", "max_points_per_centroid_inner_kmeans", "init", "seed", "use_weighted_inner_kmeans", "assignment_quantizer", "inner_init", "inner_backend")})
                    atomic_json(target / "index_params.json", structural)
                    atomic_json(target / "build_stats.json", {"build_time_sec": record["build_seconds"],
                                "index_size_mb": record["artifact"]["bytes"] / 2**20, "build_params": params,
                                "kmeans_tree_height": record["height"], "memory_after": record["memory_after"],
                                "provenance": str(directory / "manifest.json"), "resumed": record["resumed"]})
            directories.append(directory)
            print(directory, flush=True)
        except (ValueError, RuntimeError, OSError) as error:
            failures.append({"job": value["name"], "error": str(error)})
            failure_path = (directory / "workflow-failure.json" if directory is not None else
                            Path(value["output"]["directory"]) / ".mvsic/failures" / f"{digest(job)}.json")
            atomic_json(failure_path, {"status": "failed", "job": value["name"], "error": str(error)})
            print(f"[FAIL {value['name']}] {error}", flush=True)
            if options.get("fail_fast"):
                break
    if failures:
        raise RuntimeError(f"{len(failures)} benchmark jobs failed: " + "; ".join(row["error"] for row in failures))
    return directories
