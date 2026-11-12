"""Optional isolated stage diagnostics for a fixed standalone clustering workload.

Run through build_quality.py so thread limits, CPU affinity and the coordination
marker are enforced by its parent process before importing native code.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

from build_quality import runtime_key
from compare import write_json


def run(job):
    import numpy as np
    from mvsic import _mvsic as native
    from mvsic.benchmark.runner import file_identity, runtime_identity
    from mvsic.benchmark.worker import _index_params, _memory

    config, diagnostic = job["config"], job["diagnostics"]
    runtime = runtime_identity()
    if runtime_key(runtime) != job["runtime"]:
        raise RuntimeError("diagnostic implementation differs from measured index builds")
    if file_identity(config["dataset"]["points"]) != job["inputs"]["points"]:
        raise RuntimeError("diagnostic corpus changed after measured index builds")
    points = native.PointCloudSet.from_file(config["dataset"]["points"], config["dataset"]["mmap"])
    count = min(diagnostic["sample_documents"], len(points))
    if diagnostic["k"] > count:
        raise ValueError("diagnostic k exceeds available sampled documents")
    rows = np.sort(np.random.default_rng(diagnostic["sample_seed"]).choice(len(points), count, replace=False))
    clouds = [points[int(row)] for row in rows]
    values = np.ascontiguousarray(np.concatenate(clouds, axis=0), dtype=np.float32)
    offsets = np.concatenate([np.zeros(1, dtype=np.uint64),
                              np.cumsum([cloud.size for cloud in clouds], dtype=np.uint64)])
    sample = native.PointCloudSet(values, offsets)
    params = _index_params(native, config).mvclus
    # Float objective reporting incurs separate work outside the stage timers.
    # Diagnostic total time must not be substituted for production build time.
    params.verbose = 2
    before = _memory()
    start = time.perf_counter()
    centers, assignments, stats = native.cluster_point_clouds(
        sample, diagnostic["k"], config["index"]["params"]["s"], native.Metric.IP, params)
    seconds = time.perf_counter() - start
    after = _memory()
    iterations = list(stats["iterations"])
    if iterations:
        iterations[0]["float_cost"] = None  # Initialization has no assignment cost.
    initialization = iterations[0]["centroid_update_seconds"] if iterations else 0.0
    passes = iterations[1:]
    rows = rows.astype("<u8")
    output = Path(job["output"])
    np.savez(output.with_suffix(".npz"), sample_rows=rows, assignments=assignments)
    write_json(output, {
        "schema_version": 1, "status": "complete", "runtime": runtime,
        "scope": "standalone fixed sampled corpus; no recursive MVIVF root encoding reuse",
        "configuration": config["index"]["params"], "diagnostics": diagnostic,
        "sample_documents": count, "sample_vectors": int(values.shape[0]), "centers": len(centers),
        "sample_rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
        "assignment_implementation": stats["assignment_implementation"],
        "assignment_quantizer": config["index"]["params"]["assignment_quantizer"],
        "iterations": iterations, "initialization_seconds": initialization,
        "outer_assignment_seconds": sum(row["assignment_seconds"] for row in passes),
        "centroid_update_seconds": sum(row["centroid_update_seconds"] for row in passes),
        "centroid_update_scope": "grouping, sampling, inner Lloyd assignment and float centroid updates",
        "total_seconds": seconds, "float_cost_computed": True,
        "timing_note": "total includes validation, preparation, float objective reporting and final assignment outside iteration timers",
        "memory_before": before, "memory_after": after,
        "raw_results": str(output.with_suffix(".npz")),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.job).read_text()))


if __name__ == "__main__":
    main()
