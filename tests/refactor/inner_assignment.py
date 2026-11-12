"""Guarded packed/unpacked inner-assignment measurements on fixed native targets.

Each executable checks decoded scores, padding, ties and weighted updates before
its timing loop. This measures inner assignment, not whole-index build or recall.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess

from compare import (
    DEFAULT_MARKER, RESULT_ROOT, ContentionGuard, check_marker, identity, library_environment,
    measured_process, pin_binary, pin_libraries, source_identity,
    verify_pinned_libraries, write_json,
)


VARIANTS = ("unpacked_kernel", "packed_kernel", "unpacked_assignment", "packed_assignment")


def summarize(raw, *, isa, codec, metric, n, k, d, repeats, loops):
    expected_isa = "avx512" if isa == "avx512" else "avx2 (packed argmin SIMD)"
    for key, value in {"checks": "passed", "isa": expected_isa, "codec": codec,
                       "metric": metric, "n": n, "k": k, "d": d, "loops": loops}.items():
        if raw.get(key) != value:
            raise ValueError(f"unexpected native {key}: {raw.get(key)!r}, expected {value!r}")
    samples = raw.get("seconds", {})
    if set(samples) != set(VARIANTS):
        raise ValueError("native timing variants differ")
    distributions = {}
    for variant in VARIANTS:
        values = samples[variant]
        if len(values) != repeats or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("native timing distribution is incomplete or invalid")
        distributions[variant] = {"samples_seconds": values, "median_seconds": statistics.median(values),
                                  "min_seconds": min(values), "max_seconds": max(values)}
    speedups = {}
    for kind in ("kernel", "assignment"):
        before, after = distributions[f"unpacked_{kind}"], distributions[f"packed_{kind}"]
        speedups[kind] = {"median": before["median_seconds"] / after["median_seconds"],
                          "min": before["min_seconds"] / after["min_seconds"],
                          "paired_samples": [a / b for a, b in zip(before["samples_seconds"], after["samples_seconds"])]}
    return {"isa": isa, "codec": codec, "metric": metric, "checks": "passed",
            "distributions": distributions, "packed_speedup": speedups}


def run(args):
    check_marker(args.marker)
    output = Path(args.output).resolve()
    if not output.is_relative_to(RESULT_ROOT.resolve()):
        raise ValueError(f"--output must be under {RESULT_ROOT}")
    if args.cpu not in os.sched_getaffinity(0):
        raise ValueError("requested CPU is unavailable")
    output.mkdir(parents=True, exist_ok=False)
    args.created_output = True
    build = Path(args.build).resolve(strict=True)
    previous = json.loads(Path(args.compiler_provenance).read_text()) if args.compiler_provenance else None
    compile_db = json.loads((build / "compile_commands.json").read_text()) if previous is None else None
    binaries, libraries, provenance = {}, {}, {}
    for isa in ("avx2", "avx512"):
        target = f"mvsic_inner_assignment_{isa}"
        source = build / target
        if previous is None:
            entries = [row for row in compile_db if f"{target}.dir/" in row.get("output", row.get("command", ""))]
        else:
            prior = previous["binaries"][isa]
            if identity(source)["sha256"] != prior["executable"]["pinned"]["sha256"]:
                raise ValueError(f"{target} differs from the recorded compiler provenance")
            entries = [prior["compile_command"]]
        if len(entries) != 1:
            raise ValueError(f"need exactly one compile command for {target}")
        compiler = shlex.split(entries[0]["command"])[0]
        binaries[isa], executable = pin_binary(source, output / "bin")
        libraries[isa], dsos = pin_libraries(source, output / "libraries")
        verify_pinned_libraries(binaries[isa], libraries[isa], dsos)
        if previous is not None:
            actual = {name: row["pinned"]["sha256"] for name, row in dsos.items()}
            expected = {name: row["pinned"]["sha256"] for name, row in prior["libraries"].items()}
            if actual != expected or identity(compiler)["sha256"] != prior["compiler"]["sha256"]:
                raise ValueError("compiler or native library differs from recorded provenance")
        provenance[isa] = {"executable": executable, "libraries": dsos, "compile_command": entries[0],
                           "compiler": identity(compiler),
                           "compiler_version": subprocess.check_output([compiler, "--version"], text=True)}
    topology = subprocess.check_output(["lscpu", "--json"], text=True)
    guard = ContentionGuard(output, shared_host=args.shared_host, reserved_cpus=[args.cpu])
    write_json(output / "provenance.json", {
        "schema_version": 1, "purpose": "packed versus singleton quantized inner assignment",
        "scope": "one physical CPU, prepared immutable inputs; assignment includes changed-center encoding",
        "exclusions": "not whole-tree build/recall; no VNNI/AMX validation",
        "created_utc": datetime.now(timezone.utc).isoformat(), "binaries": provenance,
        "compiler_provenance_reference": identity(args.compiler_provenance) if previous is not None else None,
        "source": source_identity(Path(__file__).resolve().parents[2]),
        "controller": identity(__file__), "marker": check_marker(args.marker),
        "cpu": args.cpu, "threads": 1, "cpu_topology": json.loads(topology),
        "n": args.n, "k": args.k, "d": args.d, "repeats": args.repeats, "loops": args.loops,
        "timeout_seconds_per_case": args.timeout,
        "contention_guard": ContentionGuard.policy,
        "measurement_scope": guard.scope,
        "variant_order": "forward on even repetitions, reverse on odd repetitions; one warmup per variant",
    })
    results = []
    try:
        guard.start()
        # Alternate fixed-ISA executables within each codec/objective pair.
        for codec in ("tq4", "tq8"):
            for metric in ("ip", "l2"):
                for isa in ("avx2", "avx512"):
                    case = f"{isa}-{codec}-{metric}"
                    command = ["timeout", "--signal=TERM", "--kill-after=5s", str(args.timeout),
                               str(binaries[isa]), codec, metric, str(args.n), str(args.k), str(args.d),
                               str(args.repeats), str(args.loops)]
                    measured_process(command, output / case, marker=args.marker, threads=1, cpus=[args.cpu],
                                     cwd="/tmp", extra_env=library_environment(libraries[isa]))
                    lines = (output / case / "stdout.log").read_text().splitlines()
                    raw = json.loads(next(line for line in reversed(lines) if line.strip()))
                    summary = summarize(raw, isa=isa, codec=codec, metric=metric, n=args.n,
                                        k=args.k, d=args.d, repeats=args.repeats, loops=args.loops)
                    write_json(output / case / "raw.json", raw)
                    write_json(output / case / "summary.json", summary)
                    results.append(summary)
                    write_json(output / "summary.json", {"status": "running", "cases": results})
                    print(f"{case}: kernel {summary['packed_speedup']['kernel']['median']:.3f}x; "
                          f"encode+assign {summary['packed_speedup']['assignment']['median']:.3f}x", flush=True)
    except BaseException as error:
        write_json(output / "summary.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                              "reason": guard.reason or str(error), "timings_accepted": False,
                                              "cases": results})
        raise
    finally:
        guard.stop()
    write_json(output / "summary.json", {"status": "completed", "timings_accepted": True,
                                          "measurement_scope": guard.scope, "cases": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compiler-provenance", help="reuse recorded compile commands only when executable, library and compiler hashes match")
    parser.add_argument("--marker", default=DEFAULT_MARKER)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--shared-host", action="store_true", help="allow background work confined away from this physical core")
    for key, value in (("n", 2800), ("k", 140), ("d", 128), ("repeats", 7), ("loops", 10), ("timeout", 600)):
        parser.add_argument("--" + key, type=int, default=value)
    args = parser.parse_args()
    if any(getattr(args, key) <= 0 for key in ("n", "k", "d", "repeats", "loops", "timeout")) or args.d > 65535:
        parser.error("sizes, repetitions and timeout must be positive; dimension must be <=65535")
    args.created_output = False
    run(args)


if __name__ == "__main__":
    main()
