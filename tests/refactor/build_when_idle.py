"""Run one diagnostic build when the shared experiment marker permits it."""

import argparse
from pathlib import Path
import time

from compare import DEFAULT_MARKER, RESULT_ROOT, check_marker, measured_process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(RESULT_ROOT.resolve()) or args.jobs < 1:
        raise ValueError("Use a positive job count and the isolated results root")
    if output.exists():
        raise ValueError("Use a new build-result directory")
    if not (args.build / "CMakeCache.txt").is_file():
        raise ValueError("Configure the build tree first")
    print(f"Waiting for {DEFAULT_MARKER} to become inactive", flush=True)
    while True:
        try:
            check_marker(DEFAULT_MARKER)
            break
        except (OSError, ValueError, RuntimeError):
            time.sleep(5)
    print("Marker inactive; starting diagnostic build", flush=True)
    targets = ["_mvsic", "mvivf_test", "mvclustering_test", "kernels_dispatch_test",
               "panels_test", "runtime_test", "mvsic_native_bench", "mvsic_baseline_bench",
               "_mvsic_baseline"]
    measured_process(["cmake", "--build", str(args.build.resolve()), "--parallel", str(args.jobs),
                      "--target", *targets], output, marker=DEFAULT_MARKER, threads=args.jobs)
    print(f"Build complete: {output}", flush=True)


if __name__ == "__main__":
    main()
