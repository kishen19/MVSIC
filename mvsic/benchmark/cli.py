"""The installed ``mvsic benchmark`` command."""

import argparse
import json
import sys
from pathlib import Path

from . import load_config, run
from .config import read_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mvsic benchmark")
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("validate", "build", "search", "plot"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--base-dir", help="base for original YAML paths (default: current working directory)")
        command.add_argument("--methods", help="optional original methods.yaml metadata")
        if name in ("build", "search"):
            command.add_argument("--force", action="store_true", help="replace generated artifacts/results instead of resuming")
            command.add_argument("--fail-fast", action="store_true")
        if name == "build":
            command.add_argument("--rebuild", action="store_true", help="rebuild existing indices")
            command.add_argument("--build-threads", type=int)
        if name in ("search", "validate"):
            command.add_argument("--mode", choices=("latency", "multi_latency", "batch"))
            command.add_argument("--latency", action="store_true")
            command.add_argument("--num_threads", "--num-threads", type=int, dest="num_threads")
            command.add_argument("--QPS_thresh", "--qps-thresh", type=float, default=10.0, dest="qps_thresh")
            command.add_argument("--query_subsample", "--query-subsample", type=int, default=1000, dest="query_subsample")
            command.add_argument("--query_subsample_seed", "--query-subsample-seed", type=int, default=42, dest="query_subsample_seed")
            command.add_argument("--repeats", type=int)
            command.add_argument("--warmup-queries", type=int)
        if name == "plot":
            command.add_argument("--affix", default="")
            command.add_argument("--style", default="default")
            command.add_argument("--format", default="png", dest="file_format")
    args = parser.parse_args(argv)
    try:
        plan = load_config(args.config, base_dir=args.base_dir)
        options = vars(args).copy()
        for key in ("action", "config", "base_dir", "methods"):
            options.pop(key)
        if args.methods:
            methods = read_config(Path(args.methods).expanduser().resolve(strict=True))
            info = methods.get("mvivf", {})
            if info.get("class") != "MVIVF" or info.get("variable_param") != "nprobes":
                raise ValueError("methods.yaml must describe the supported MVIVF/nprobes method")
        if args.action == "validate":
            print(json.dumps(plan.validate(**options), indent=2, allow_nan=False, default=str))
        else:
            result = run(args.action, plan, **options)
            if args.action == "plot" and isinstance(result, list):
                for output in result:
                    print(output)
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f"mvsic benchmark: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
