"""Small-data native parity using the existing comparison drivers and controller.

These are correctness checks, not timing measurements. Outputs and native logs
are retained under the isolated results root; every process obeys the shared
experiment marker. No source or artifact in the original checkout is changed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from compare import (
    DEFAULT_MARKER, RESULT_ROOT, check_marker, identity, library_environment,
    measured_process, pin_binary, pin_libraries, read_predictions,
    verify_pinned_libraries, write_json,
)


def fixture(path, n, seed, minimum, spread):
    import numpy as np
    random = np.random.default_rng(seed)
    clouds = []
    offsets = [0]
    for i in range(n):
        values = random.normal(size=(minimum + i % spread, 128))
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        clouds.append(values.astype(np.float32))
        offsets.append(offsets[-1] + values.size)
    with Path(path).open("wb") as stream:
        np.asarray([128, n, offsets[-1] // 128], dtype="<u8").tofile(stream)
        for cloud in clouds:
            cloud.astype("<f4").tofile(stream)
        np.asarray([len(offsets), *offsets], dtype="<u8").tofile(stream)
    return clouds


def exact_scores(points, queries):
    import numpy as np
    result = []
    for query in queries:
        q = query.astype(np.float64)
        row = {}
        for i, document in enumerate(points):
            d = document.astype(np.float64)
            distances = -(q @ d.T)
            row[i] = float(distances.min(axis=1).mean())
        result.append(row)
    return result


def score_maps(path):
    import numpy as np
    ids, distances = read_predictions(path)
    output = []
    for row_ids, row_distances in zip(ids, distances):
        row = {}
        for document, distance in zip(row_ids, row_distances):
            document = int(document)
            if document == np.iinfo(np.uint32).max:
                if np.isfinite(distance):
                    raise AssertionError("padding has a finite score")
                continue
            if document in row or not np.isfinite(distance):
                raise AssertionError("duplicate document or nonfinite native score")
            row[document] = float(distance)
        output.append(row)
    return output


def compare_scores(actual, expected, tolerance=8e-6):
    if len(actual) != len(expected):
        raise AssertionError("query counts differ")
    maximum = 0.0
    for q, (got, want) in enumerate(zip(actual, expected)):
        if got.keys() != want.keys():
            raise AssertionError(f"query {q} visited document sets differ: extra={sorted(got.keys()-want.keys())}, missing={sorted(want.keys()-got.keys())}")
        for document, score in got.items():
            error = abs(score - want[document])
            maximum = max(maximum, error)
            if error > tolerance * max(1.0, abs(want[document])):
                raise AssertionError(f"query {q}, document {document}: {score} != {want[document]}")
    return maximum


def run(args):
    check_marker(args.marker)
    output = Path(args.output).resolve()
    if not output.is_relative_to(RESULT_ROOT.resolve()):
        raise ValueError(f"--output must be under {RESULT_ROOT}")
    output.mkdir(parents=True, exist_ok=False)
    args.created_output = True
    cpus = [int(value) for value in args.cpus.split(",")]
    if not cpus or len(cpus) != len(set(cpus)) or not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError("--cpus must be distinct available CPU IDs")
    threads = sorted({1, min(3, len(cpus))})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    binaries, libraries, provenance = {}, {}, {}
    for kind in ("original", "refactor"):
        source = getattr(args, kind)
        binaries[kind], executable = pin_binary(source, output / "bin")
        libraries[kind], dsos = pin_libraries(source, output / "libraries")
        verify_pinned_libraries(binaries[kind], libraries[kind], dsos)
        provenance[kind] = {"executable": executable, "libraries": dsos}
    points = fixture(output / "points.pcs", 96, 173, 1, 7)
    queries = fixture(output / "queries.pcs", 13, 839, 8, 19)
    write_json(output / "provenance.json", {
        "schema_version": 1, "purpose": "IP native correctness parity, no performance conclusions",
        "retrieval_metric": "ip",
        "binaries": provenance, "points": identity(output / "points.pcs"),
        "queries": identity(output / "queries.pcs"), "cpus": cpus,
        "isas": args.isa, "threads": threads, "marker": check_marker(args.marker),
        "reference_scope": "original_single_query_only" if args.reference_single_only else "original_single_and_batch",
        "original_batch_validation": not args.reference_single_only,
        "reference_limitation": (
            "Original AVX2 TQ1 batch score_one_db_cloud and score_one_db_cloud_grouped fallbacks return FLT_MAX; "
            "use its single-query scores as the strict reference for every refactor path. "
            "This excludes an unsupported original batch path, not any refactor score or visited-set comparison."
        ) if args.reference_single_only else None,
    })
    results = []

    def execute(kind, label, options, count=1, isa=None):
        result_path = output / f"{label}.json"
        options = {"points": output / "points.pcs", "queries": output / "queries.pcs",
                   "output": result_path, "repeats": 1, "warmup": 0, **options}
        command = [str(binaries[kind])]
        for key, value in options.items():
            command += ["--" + key, str(value)]
        environment = library_environment(libraries[kind])
        if isa is not None:
            environment["MVSIC_CPU_ISA"] = isa
        measured_process(command, output / label, marker=args.marker, threads=count,
                         cpus=cpus[:count], extra_env=environment)
        return score_maps(str(result_path) + ".predictions") if options["mode"] != "build" else None

    skeleton = output / "ip-original.idx"
    common = {"metric": "ip", "index": skeleton, "leaf-size": 8,
              "branching": 4, "niters": 2, "s": 3, "seed": 11, "assignment": "float"}
    execute("original", "ip-build", {**common, "mode": "build"})
    exact = exact_scores(points, queries)
    # k=N exposes every visited document. This avoids mistaking arbitrary
    # order among tied TQ1 candidates for an algorithm disagreement.
    cases = [
        ("full-coarse", 96, 0, 0, "None"),
        ("full-float", 96, 96, 0, "None"),
        ("full-tq8", 96, 96, 1, "None"),
        ("partial-coarse", 3, 0, 0, "None"),
        ("wards-float", 96, 96, 0, "Wards"),
        ("carve-float", 96, 96, 0, "Carve"),
        ("wards-partial", 3, 0, 0, "Wards"),
    ]
    for name, probes, rerank, tq8, compression in cases:
        options = {**common, "k": len(points), "nprobes": probes, "rerank": rerank,
                   "tq8-rerank": tq8, "compression": compression, "threshold": 0.5}
        expected = execute("original", f"ip-{name}-original",
                           {**options, "mode": "latency"})
        if probes >= len(points) and any(len(row) != len(points) for row in expected):
            raise AssertionError("baseline full-probe search omitted documents")
        if rerank and not tq8 and probes >= len(points):
            compare_scores(expected, exact)
        if not args.reference_single_only:
            for root in (0, 1):
                label = f"ip-{name}-original-batch-root{root}"
                batched = execute("original", label, {**options, "mode": "batch", "root-m2m": root})
                maximum = compare_scores(batched, expected)
                results.append({"case": label, "status": "passed", "maximum_score_error": maximum})
                write_json(output / "checks.json", results)
        for isa in args.isa:
            for count in threads:
                for mode, root in (("latency", 1), ("batch", 0), ("batch", 1)):
                    label = f"ip-{name}-{isa}-{count}-{mode}-root{root}"
                    actual = execute("refactor", label, {**options, "mode": mode, "root-m2m": root}, count, isa)
                    maximum = compare_scores(actual, expected)
                    if rerank and not tq8 and probes >= len(points):
                        compare_scores(actual, exact)
                    results.append({"case": label, "status": "passed", "maximum_score_error": maximum})
                    write_json(output / "checks.json", results)
    write_json(output / "result.json", {"status": "passed", "checks": len(results)})
    print(f"Passed {len(results)} native parity checks; artifacts: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--refactor", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpus", required=True)
    parser.add_argument("--isa", action="append", choices=("avx2", "avx512"))
    parser.add_argument("--reference-single-only", action="store_true",
                        help="use original single-query scores for every new path; skip the known unsupported original AVX2 TQ1 batch checks")
    parser.add_argument("--marker", default=DEFAULT_MARKER)
    args = parser.parse_args()
    args.isa = list(dict.fromkeys(args.isa or ["avx2"]))
    args.created_output = False
    try:
        run(args)
    except Exception as error:
        output = Path(args.output).resolve()
        if args.created_output and output.is_relative_to(RESULT_ROOT.resolve()) and output.is_dir():
            write_json(output / "result.json", {"status": "failed", "reason": str(error)})
        raise


if __name__ == "__main__":
    main()
