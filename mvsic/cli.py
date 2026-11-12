"""Command-line entry point for the installed package."""

import sys


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "benchmark":
        from .benchmark.cli import main as benchmark
        return benchmark(args[1:])
    print("Usage: mvsic benchmark {validate,build,search,plot} --config PATH")
    return 0 if not args or args[0] in ("--help", "-h") else 2


if __name__ == "__main__":
    raise SystemExit(main())
