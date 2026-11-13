"""Python API timing worker for the original subset module or installed refactor."""

import argparse
import importlib.util
import json
from pathlib import Path
import resource
import struct
import sys
import time


def load_extension(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    config = job["config"]
    if config.get("metric", "ip") != "ip":
        raise ValueError("MVIVF retrieval supports only metric ip; L2 remains a clustering objective")
    if job["implementation"] == "original":
        native = load_extension(job["module"], "_mvsic_baseline")
    else:
        native = load_extension(job["module"], "_mvsic")
    import numpy as np
    params = job["params"]
    start = time.perf_counter()
    if job["implementation"] == "original":
        points = native.PointCloudSetIP(config["points"], True)
        queries = native.PointCloudSetIP(job["queries"], False)
        count = queries.size()
        with Path(config["points"]).open("rb") as stream:
            dims = struct.unpack("<Q", stream.read(8))[0]
    else:
        points = native.PointCloudSet.from_file(config["points"], True)
        queries = native.PointCloudSet.from_file(job["queries"], False)
        dims, count = points.dims, len(queries)
    load_seconds = time.perf_counter() - start
    ip = native.IndexParams.mvivf(k_per_level=config.get("branching", 0), max_leaf_size=config.get("leaf_size", 100),
                                  niters=config.get("niters", 5), seed=config.get("seed", 0), s=config.get("s", 0),
                                  max_depth=config.get("depth", 0), build_with_8btq=config.get("assignment", "float") == "tq8")
    if job["implementation"] == "original":
        index = native.IndexMVIVFCompressOneBitTQIP(dims, ip)
        compression = getattr(native.SearchParams.QueryCompression, params["compression"].upper())
    else:
        ip.quantizer = native.QuantizerType.OneBitTQ
        ip.quantize_centers = True
        index = native.IndexMVIVF(dims, ip)
        compression = getattr(native.QueryCompression, "None_" if params["compression"] == "None" else params["compression"])
    start = time.perf_counter()
    index.load(job["index"], points)
    prepare_seconds = time.perf_counter() - start
    sp = native.SearchParams.mvivf(params["k"], params["nprobes"], params["rerank"],
                                   bool(params["tq8-rerank"]), compression, params["threshold"])
    sp.root_m2m = bool(params["root-m2m"])
    # Query extraction is outside the timed API calls for both implementations.
    arrays = [queries[i] for i in range(count)] if params["mode"] == "latency" else None
    if job["implementation"] == "original":
        one = lambda q: index.search(q, points, sp)
        batch = lambda: index.search_all(queries, points, sp)
    else:
        one = lambda q: index.search(q, sp)
        batch = lambda: index.search_all(queries, sp)
    first_query = queries[0]
    start = time.perf_counter()
    one(first_query)
    first_query_seconds = time.perf_counter() - start
    if config.get("warmup", 10):
        if arrays is None:
            batch()
        else:
            for q in arrays[:config.get("warmup", 10)]:
                one(q)
    repeats = config.get("repeats", 5)
    times, query_times, best = [], [], None
    for r in range(repeats):
        if arrays is None:
            start = time.perf_counter()
            result = batch()
            elapsed = time.perf_counter() - start
        else:
            result, per_query = [], []
            for q in arrays:
                start = time.perf_counter()
                row = one(q)
                per_query.append(time.perf_counter() - start)
                result.append(row)
            elapsed = sum(per_query)
            query_times.append(per_query)
        times.append(elapsed)
        if elapsed == min(times):
            best = result
            selected = r
    ids = np.full((count, params["k"]), np.iinfo(np.uint32).max, dtype=np.uint32)
    distances = np.full(ids.shape, np.inf, dtype=np.float32)
    if job["implementation"] == "original":
        rows = best[0] if arrays is None else [row[0] for row in best]
        for q, row in enumerate(rows):
            for j, (doc, distance) in enumerate(row[:params["k"]]):
                ids[q, j], distances[q, j] = doc, distance
    else:
        if arrays is None:
            ids, distances = best
        else:
            for q, (row_ids, row_distances) in enumerate(best):
                ids[q, :len(row_ids)] = row_ids
                distances[q, :len(row_distances)] = row_distances
    output = Path(job["output"])
    np.savez(str(output) + ".npz", ids=ids, distances=distances, seconds=np.array(times),
             query_seconds=np.array(query_times))
    record = {"timing_scope": "python_api_including_native_and_result_conversion", "seconds": times,
              "numpy_version": np.__version__, "python_version": sys.version,
              "query_seconds": query_times if arrays is not None else None, "selected_repeat": selected,
              "qps": count / times[selected], "queries": count, "load_seconds": load_seconds,
              "prepare_seconds": prepare_seconds, "first_query_seconds": first_query_seconds,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
