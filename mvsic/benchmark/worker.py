"""Native worker. The controller sets thread limits before starting this process."""

from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

from .config import BenchmarkConfig, digest
from .runner import atomic_json, completed, file_identity


def _memory():
    # Linux ru_maxrss is KiB; macOS reports bytes.
    scale = 1 if sys.platform == "darwin" else 1024
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    rss = None
    path = Path("/proc/self/statm")
    if path.exists():
        rss = int(path.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return {"rss_bytes": rss, "peak_rss_bytes": peak}


def _index_params(native, config):
    params = dict(config["index"]["params"])
    assignment = params.pop("assignment_quantizer")
    inner = {key: params.pop(key) for key in ("inner_init", "inner_backend")}
    ip = native.IndexParams.mvivf(**params)
    for key, value in inner.items():
        setattr(ip.mvclus, key, value)
    ip.mvclus.assignment_quantizer = getattr(native.ClusteringQuantizer, "None_" if assignment == "None" else assignment)
    quantizer = config["index"]["quantizer"]
    ip.quantizer = getattr(native.QuantizerType, "None_" if quantizer == "None" else quantizer)
    ip.quantize_centers = config["index"]["quantize_centers"]
    return ip


def _index(native, points, config):
    return native.IndexMVIVF(points.dims, _index_params(native, config))


def _params(native, point):
    fields = dict(point)
    compression = fields.pop("query_compression")
    fields["query_compression"] = getattr(native.QueryCompression, "None_" if compression == "None" else compression)
    params = native.SearchParams.mvivf(fields.pop("k"), fields.pop("nprobes"))
    for key, value in fields.items():
        setattr(params, key, value)
    return params


def _build(job, native):
    config, context = job["config"], job["context"]
    directory = Path(job["directory"])
    output = Path(config["build"]["output"])
    metadata = Path(str(output) + ".json")
    fingerprint = context["build_fingerprint"]
    if not job["force"] and completed(metadata, fingerprint, output):
        record = json.loads(metadata.read_text())
        record["resumed"] = True
        atomic_json(directory / "build.json", record)
        return
    if output.exists() and not job["force"]:
        raise ValueError(f"existing artifact has no matching complete provenance: {output}; choose a new output or --force")
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    points = native.PointCloudSet.from_file(config["dataset"]["points"], config["dataset"]["mmap"])
    load_seconds = time.perf_counter() - start
    before = _memory()
    index = _index(native, points, config)
    start = time.perf_counter()
    index.build(points)
    build_seconds = time.perf_counter() - start
    after = _memory()
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        start = time.perf_counter()
        index.save(str(temporary))
        save_seconds = time.perf_counter() - start
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    record = {"schema_version": 1, "status": "complete", "fingerprint": fingerprint,
              "artifact": file_identity(output), "index": config["index"], "inputs": context["inputs"],
              "runtime": context["runtime"], "load_seconds": load_seconds,
              "execution": {key: config["build"][key] for key in ("threads", "cpu")},
              "build_seconds": build_seconds, "save_seconds": save_seconds,
              "memory_before": before, "memory_after": after,
              "documents": len(points), "dimensions": points.dims,
              "height": index.get_height(), "mean_cluster_size": index.mean_cluster_size(),
              "max_cluster_size": index.max_cluster_size(), "resumed": False}
    atomic_json(metadata, record)
    atomic_json(directory / "build.json", record)


def read_ground_truth(filename, nqueries):
    """Legacy little-endian .gt: int32 width, then (float32 distance, uint32 id)."""
    import numpy as np
    path = Path(filename)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"ids", "distances"}:
                raise ValueError("ground truth NPZ must contain exactly ids and distances")
            ids, distances = arrays["ids"], arrays["distances"]
        if ids.dtype != np.uint32 or distances.dtype != np.float32:
            raise ValueError("ground truth ids/distances must be uint32/float32")
    else:
        with path.open("rb") as stream:
            header = stream.read(4)
        if len(header) != 4:
            raise ValueError("truncated ground truth header")
        width = int.from_bytes(header, "little", signed=True)
        if width <= 0 or path.stat().st_size != 4 + nqueries * width * 8:
            raise ValueError("ground truth shape/file length does not match queries")
        records = np.memmap(path, dtype=[("distance", "<f4"), ("id", "<u4")],
                            offset=4, mode="r", shape=(nqueries, width))
        ids, distances = records["id"], records["distance"]
    if ids.ndim != 2 or ids.shape[0] != nqueries or ids.shape != distances.shape or ids.shape[1] == 0:
        raise ValueError("ground truth arrays must have shape (queries, neighbors)")
    if not np.isfinite(distances).all() or np.any(distances[:, 1:] < distances[:, :-1]):
        raise ValueError("ground truth distances must be finite and sorted ascending")
    if np.any(ids == np.iinfo(np.uint32).max):
        raise ValueError("ground truth contains the missing-result sentinel")
    return ids, distances


def recall(predicted, truth_ids, truth_distances, k, *, truth_k=None):
    """Top-k recall with legacy absolute 1e-6 boundary ties, capped per query."""
    import numpy as np
    if truth_ids.shape[1] < k:
        raise ValueError(f"ground truth has fewer than k={k} neighbors")
    if predicted.shape[0] != truth_ids.shape[0] or len(predicted) == 0:
        raise ValueError("prediction and ground truth query counts must match and be nonzero")
    truth_k = k if truth_k is None else truth_k
    total = 0.0
    for got, ids, distances in zip(predicted, truth_ids, truth_distances):
        # Only IDs count. Quantized predicted distances are not comparable to exact GT.
        accepted = set(map(int, ids[:truth_k]))
        boundary = float(distances[truth_k - 1])
        for offset in range(truth_k, len(ids)):
            if abs(float(distances[offset]) - boundary) >= 1e-6:
                break
            accepted.add(int(ids[offset]))
        returned = set(map(int, got[:k]))
        returned.discard(int(np.iinfo(np.uint32).max))
        total += min(truth_k, len(returned & accepted)) / truth_k
    return total / len(predicted)


def _queries(native, source, limit, seed):
    import numpy as np
    queries = native.PointCloudSet.from_file(source, False)
    nqueries = len(queries)
    if nqueries == 0:
        raise ValueError("query corpus is empty")
    selection = np.arange(nqueries, dtype=np.uint32)
    if limit and limit < nqueries:
        selection = np.sort(np.random.default_rng(seed).choice(nqueries, size=limit, replace=False)).astype(np.uint32)
        clouds = [queries[int(row)] for row in selection]
        values = np.concatenate(clouds)
        offsets = np.concatenate((np.array([0], dtype=np.uint64),
                                  np.cumsum([cloud.size for cloud in clouds], dtype=np.uint64)))
        queries = native.PointCloudSet(values, offsets)
    return queries, selection, nqueries


def _python_times(index, queries, params, mode, repeats):
    import numpy as np
    seconds, per_query = [], []
    if mode == "throughput_batch":
        for _ in range(repeats):
            start = time.perf_counter()
            result = index.search_all(queries, params)
            seconds.append(time.perf_counter() - start)
            del result
        return seconds, None
    # Array extraction is outside the timing, matching index.search(array, params)
    # in an application with already prepared queries. NumPy result conversion is inside.
    arrays = [queries[row] for row in range(len(queries))]
    for _ in range(repeats):
        times = []
        for query in arrays:
            start = time.perf_counter()
            result = index.search(query, params)
            times.append(time.perf_counter() - start)
            del result
        per_query.append(times)
        seconds.append(float(np.sum(times)))
    return seconds, per_query


class LegacySweep:
    """Full-precision CSV compatibility with content-verified resume and stops."""

    def __init__(self, options, force, base):
        import csv
        self.options, self.history = options, []
        self.path = Path(options["csv"])
        self.metadata = self.path.with_suffix(self.path.suffix + ".json")
        self.fingerprint = digest({"base": base, "mode": options["mode"], "both_modes": options["both_modes"]})
        self.rows = []
        if self.path.exists() and not force:
            metadata = json.loads(self.metadata.read_text()) if self.metadata.exists() else {}
            if metadata.get("fingerprint") != self.fingerprint or metadata.get("csv") != file_identity(self.path):
                raise ValueError(f"CSV has incompatible or missing provenance: {self.path}; use --force to replace it")
            with self.path.open(newline="") as stream:
                self.rows = list(csv.DictReader(stream))
        stop = self.path.with_suffix(self.path.suffix + ".stop.json")
        if stop.exists():
            stop.unlink()

    def append(self, record):
        import csv
        point, stats = record["params"], record.get("legacy_stats", {})
        parallel = self.options["mode"] in ("multi_latency", "batch")
        row = {"k": point["k"], "recall_1_k": record.get("recall_at_one"),
               "recall_k_k": record["recall_at_k"],
               "QPS_seq": None if parallel else record["qps"],
               "QPS_par": record["qps"] if parallel else stats.get("batch_qps"),
               "nprobes": point["nprobes"], "num_rerank": point["num_rerank"],
               "avg_cmps": stats.get("avg_cmps"), **point}
        row.update({name: value for name, value in zip(self.options["labels"], stats.get("timings", []))})
        row.update({name: None for name in self.options["labels"] if name not in row})
        if record.get("latency_seconds"):
            row.update({f"latency_{key}": value for key, value in record["latency_seconds"].items()})
        row.update(timing_scope="native_search; diagnostic stage timers collected separately",
                   threads=record["measurement"]["threads"], queries=record["queries"],
                   result_fingerprint=record["fingerprint"])
        # Match all parameters, not merely those coincidentally present in an old
        # row. The exact raw fingerprint remains the authority for computation.
        def same(previous):
            return all(str(previous.get(key, "")) == str(value) for key, value in point.items())
        self.rows = [previous for previous in self.rows if not same(previous)] + [row]
        self.rows.sort(key=lambda value: float(value["nprobes"]))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(row)
        for previous in self.rows:
            fields.extend(key for key in previous if key not in fields)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(temporary, self.path)
        atomic_json(self.metadata, {"fingerprint": self.fingerprint, "csv": file_identity(self.path),
                                   "precision": "full float precision", "mode": self.options["mode"]})
        current = record["recall_at_k"]
        self.history.append(current)
        reason = None
        if current is not None:
            if current >= 1:
                reason = "recall reached 1.0"
            elif len(self.history) > 3 and current == self.history[-2] == self.history[-3]:
                reason = "recall plateau"
            elif len(self.history) > 1 and current < self.history[-2]:
                reason = "recall decreased"
        threshold = self.options.get("qps_threshold")
        if threshold is not None and threshold > 0 and 0 < record["qps"] < threshold:
            reason = f"QPS below {threshold}"
        print(f"{point} | recall={current} QPS={record['qps']}", flush=True)
        if reason:
            atomic_json(self.path.with_suffix(self.path.suffix + ".stop.json"),
                        {"reason": reason, "params": point, "result_fingerprint": record["fingerprint"]})
            print(f"Stopping sweep: {reason}", flush=True)
        return reason is not None


def _search(job, native):
    import numpy as np
    config, context = job["config"], job["context"]
    directory = Path(job["directory"])
    artifact = config["build"]["output"]
    external = context.get("external_artifact")
    if external:
        if file_identity(artifact) != external["artifact"]:
            raise ValueError("external index artifact changed after validation")
    elif not completed(artifact + ".json", context["build_fingerprint"], artifact):
        raise ValueError("index artifact is missing, changed, or has incompatible provenance; run benchmark build first")
    artifact_identity = file_identity(artifact)
    measurement = config["measurement"]
    dataset = config["dataset"]
    start = time.perf_counter()
    points = native.PointCloudSet.from_file(dataset["points"], dataset["mmap"])
    queries, selection, nqueries = _queries(native, dataset["queries"], measurement["query_limit"], measurement["query_seed"])
    load_data_seconds = time.perf_counter() - start
    gt = None
    if "ground_truth" in dataset:
        gt = read_ground_truth(dataset["ground_truth"], nqueries)
        gt = (gt[0][selection], gt[1][selection])
    index = _index(native, points, config)
    start = time.perf_counter()
    index.load(artifact, points)
    prepare_seconds = time.perf_counter() - start
    memory_prepared = _memory()
    selection_path = directory / "query_selection.npy"
    np.save(selection_path, selection, allow_pickle=False)
    selection_hash = hashlib_bytes(selection.tobytes())
    base = {"config": config["index"], "inputs": context["inputs"], "artifact": artifact_identity,
            "measurement": measurement, "runtime": context["runtime"], "query_selection": selection_hash}
    groups = job.get("groups", [{"points": job.get("points", BenchmarkConfig(Path("resolved"), config).points()),
                                 "legacy": job.get("legacy"), "force": job["force"]}])
    plan = []
    for group_number, group in enumerate(groups):
        both = bool(group["legacy"] and group["legacy"].get("both_modes"))
        for point in group["points"]:
            fingerprint = digest({**base, "params": point, **({"legacy_both_modes": both} if group["legacy"] else {})})
            plan.append({"fingerprint": fingerprint, "params": point, "group": group_number})
    atomic_json(directory / "plan.json", {"schema_version": 1, "points": plan,
                                          "stop_policy": "legacy_recall_and_qps" if any(g["legacy"] for g in groups) else "full_grid"})
    failures, stopped, sweeps = [], set(), {}
    for task in plan:
        number = task["group"]
        if number in stopped:
            continue
        group = groups[number]
        legacy = group["legacy"]
        if number not in sweeps:
            sweeps[number] = LegacySweep(legacy, group["force"], base) if legacy else None
        sweep = sweeps[number]
        fingerprint, point = task["fingerprint"], task["params"]
        result_path = directory / f"search-{fingerprint}.json"
        if not group["force"] and completed(result_path, fingerprint):
            if sweep and sweep.append(json.loads(result_path.read_text())):
                stopped.add(number)
            continue
        try:
            params = _params(native, point)
            warmup = measurement["warmup_queries"]
            if measurement["mode"] == "throughput_batch":
                warmup = int(warmup > 0)
            result = native.benchmark(index, queries, params, measurement["mode"],
                                      warmup, measurement["repeats"])
            seconds = np.asarray(result["seconds"], dtype=np.float64)
            if seconds.shape != (measurement["repeats"],) or not np.isfinite(seconds).all() or np.any(seconds <= 0):
                raise RuntimeError("native benchmark returned invalid repetition times")
            setup_times = {key: result.get(key) for key in ("warmup_seconds", "first_call_seconds")}
            for key, value in setup_times.items():
                if value is not None and (not np.isfinite(value) or value < 0):
                    raise RuntimeError(f"native benchmark returned invalid {key}")
            selected = int(result.get("selected_repeat", int(np.argmin(seconds))))
            if selected != int(np.argmin(seconds)):
                raise RuntimeError("native benchmark selected_repeat must identify the minimum-time repetition")
            ids = np.asarray(result["ids"], dtype=np.uint32)
            distances = np.asarray(result["distances"], dtype=np.float32)
            if ids.shape != (len(queries), point["k"]) or distances.shape != ids.shape:
                raise RuntimeError("native benchmark returned unexpected result shape")
            arrays = {"ids": ids, "distances": distances, "seconds": seconds,
                      "valid_counts": np.sum(ids != np.iinfo(np.uint32).max, axis=1).astype(np.uint32)}
            arrays.update({key: np.asarray(value, dtype=np.float64) for key, value in setup_times.items()
                           if value is not None})
            query_seconds = result["query_seconds"]
            percentiles = None
            if query_seconds is not None:
                query_seconds = np.asarray(query_seconds, dtype=np.float64)
                if query_seconds.shape != (measurement["repeats"], len(queries)) or not np.isfinite(query_seconds).all():
                    raise RuntimeError("native benchmark returned invalid query times")
                arrays["query_seconds"] = query_seconds
                percentiles = dict(zip(("p50", "p95", "p99"), np.percentile(query_seconds[selected], [50, 95, 99]).tolist()))
            python_seconds = None
            if measurement["python_timing"]:
                python_seconds, python_queries = _python_times(index, queries, params, measurement["mode"], measurement["repeats"])
                arrays["python_seconds"] = np.asarray(python_seconds)
                if python_queries is not None:
                    arrays["python_query_seconds"] = np.asarray(python_queries)
            recall_at_k = None if gt is None else recall(ids, gt[0], gt[1], point["k"])
            recall_at_one = None if gt is None else recall(ids, gt[0], gt[1], point["k"], truth_k=1)
            legacy_stats = {}
            if legacy:
                if legacy.get("both_modes"):
                    batch = native.benchmark(index, queries, params, "throughput_batch", 0, measurement["repeats"])
                    arrays["batch_seconds"] = np.asarray(batch["seconds"], dtype=np.float64)
                    arrays["batch_ids"] = np.asarray(batch["ids"], dtype=np.uint32)
                    arrays["batch_distances"] = np.asarray(batch["distances"], dtype=np.float32)
                    legacy_stats["batch_qps"] = len(queries) / float(np.min(arrays["batch_seconds"]))
                if measurement["mode"] != "throughput_batch":
                    # Stage counters are diagnostic: collect them separately so
                    # Python calls/timers do not contaminate native latency.
                    counters, timings = [], []
                    for q in range(len(queries)):
                        _, count, times = index.search_with_stats(queries[q], points, params)
                        counters.append(count)
                        timings.append(times)
                    legacy_stats["avg_cmps"] = float(np.mean(counters))
                    legacy_stats["timings"] = np.sum(timings, axis=0).tolist()
            arrays_path = directory / f"search-{fingerprint}.npz"
            temporary = arrays_path.with_name(f".{arrays_path.name}.{os.getpid()}.tmp")
            with temporary.open("wb") as stream:
                np.savez(stream, **arrays)
            os.replace(temporary, arrays_path)
            record = {"schema_version": 1, "status": "complete", "fingerprint": fingerprint,
                      "params": point, "measurement": measurement, "timing_scope": "native_search",
                      "summary_statistic": "minimum_repetition", "selected_repeat": selected,
                      "seconds": seconds.tolist(), "queries": len(queries),
                      "qps": len(queries) / float(seconds[selected]),
                      "latency_seconds": percentiles, "recall_at_k": recall_at_k,
                      "recall_at_one": recall_at_one, "legacy_stats": legacy_stats,
                      "recall_policy": "unique_ids_with_absolute_1e-6_boundary_ties_capped_per_query",
                      "python_seconds": python_seconds, "load_data_seconds": load_data_seconds,
                      **setup_times,
                      "first_call_scope": "whole_batch" if measurement["mode"] == "throughput_batch" else "single_query",
                      "first_call_note": "first invocation for this operating point; may include lazy TQ8 preparation; included in first measured repetition when warmup is zero",
                      "prepare_seconds": prepare_seconds, "memory_prepared": memory_prepared,
                      "memory_after": _memory(), "artifact": artifact_identity,
                      "query_selection_sha256": selection_hash,
                      "outputs": [file_identity(arrays_path), file_identity(selection_path)]}
            atomic_json(result_path, record)
            if sweep and sweep.append(record):
                stopped.add(number)
        except Exception as error:
            failures.append(fingerprint)
            atomic_json(result_path, {"schema_version": 1, "status": "failed", "fingerprint": fingerprint,
                                     "params": point, "error": str(error), "traceback": traceback.format_exc()})
            traceback.print_exc()
            if legacy and legacy.get("fail_fast"):
                break
    atomic_json(directory / "search.json", {"schema_version": 1, "status": "failed" if failures else "complete",
                                            "requested": len(plan), "failed": failures})
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(plan)} search points failed")


def hashlib_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    execution = job["config"]["build"] if job["action"] == "build" else job["config"]["measurement"]
    cpu = execution["cpu"]
    try:
        if cpu is not None:
            if not hasattr(os, "sched_setaffinity"):
                raise RuntimeError("CPU pinning is unsupported on this platform")
            os.sched_setaffinity(0, {cpu})
        from mvsic import _mvsic as native
        atomic_json(Path(job["directory"]) / "capabilities.json", native.capabilities())
        if job["action"] == "build":
            _build(job, native)
        elif job["action"] == "search":
            _search(job, native)
        else:
            raise ValueError("unknown worker action")
    except Exception as error:
        atomic_json(Path(job["directory"]) / f'{job["action"]}-failure.json',
                    {"status": "failed", "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
