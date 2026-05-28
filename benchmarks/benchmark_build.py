"""
Build-only benchmark harness.

Produces raw-skeleton indices (no quantization, raw centers, raw leaves).
Quantization/center-compression is applied at search time by
`benchmark_search.py` which picks a templated class variant; the same
skeleton file is reused across variants.

Config schema
-------------
```yaml
name: "MVIVF Build"
datasets:
  - name: nq500k
    path: data/beir/nq500k
    index_dir: experiments/mvivf_ablation/indices/nq500k
indices:
  - name: mvivf                  # method name -> class via methods.yaml
    metric: ip                   # ip | l2
    build_configs:
      - name: mvivf_k16_l500
        params:                  # only structural IndexParams fields
          k_per_level: 16
          max_leaf_size: 500
          max_depth: 0
          niters: 5
          max_point_clouds_per_cluster: 100
          max_points_per_centroid_inner_kmeans: 20
          s: 0
          # for mvivf_spill only:
          num_spill: 2           # a -> top-a spill at the root (level 0)
          num_spill_l2: 1        # b -> top-b spill at the second level (level 1)
          # opt-in MVIVF k-means assignment via the 8-bit TurboQuant
          # VPDPBUSD panel kernel (centroid update still float). Default false.
          # build_with_8btq: false
```

Layout on disk
--------------
```
<index_dir>/<index_name>/<build_name>/
    index.bin
    index_params.json
    build_stats.json
```

The build name is the stable identifier used by `benchmark_search.py`.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback

import yaml
from framework_utils import load_dataset

import mvsic

# FastPlaid is an optional baseline; only imported when the config asks for it.
_FastPlaidWrapper = None
_IGPWrapper = None
_GEMWrapper = None
_HnswlibWrapper = None
_load_point_clouds = None


def _ensure_fastplaid_imports():
    global _FastPlaidWrapper, _load_point_clouds
    if _FastPlaidWrapper is None:
        from framework_utils import FastPlaidWrapper as _FPW  # type: ignore
        from utils import load_point_clouds as _lpc  # type: ignore

        _FastPlaidWrapper = _FPW
        _load_point_clouds = _lpc


def _ensure_gem_imports():
    """Lazy import of GEMWrapper.

    Importing GEMWrapper itself is cheap (it's pure Python -- the C++
    runner is invoked via subprocess), so we don't gate on a try/except
    the way IGP does. We do still need utils.load_point_clouds to be
    available so that the build path can probe the document dimension.
    """
    global _GEMWrapper, _load_point_clouds
    if _GEMWrapper is None:
        from framework_utils import GEMWrapper as _GW  # type: ignore
        from utils import load_point_clouds as _lpc  # type: ignore
        _GEMWrapper = _GW
        _load_point_clouds = _lpc


def _ensure_hnswlib_imports():
    global _HnswlibWrapper, _load_point_clouds
    if _HnswlibWrapper is None:
        from framework_utils import HnswlibWrapper as _HW  # type: ignore
        from utils import load_point_clouds as _lpc  # type: ignore
        _HnswlibWrapper = _HW
        _load_point_clouds = _lpc


def _ensure_igp_imports():
    global _IGPWrapper, _load_point_clouds
    if _IGPWrapper is None:
        try:
            from framework_utils import IGPWrapper as _IWP  # type: ignore
            from utils import load_point_clouds as _lpc  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "IGP support is optional and only needed when running `index.name: igp`.\n"
                "If you want IGP, run `bash setup_external.sh --igp` in the repo root, then retry.\n"
                f"Original error: {e}"
            ) from e

        _IGPWrapper = _IWP
        _load_point_clouds = _lpc


def _dir_size_bytes(path: str) -> int:
    total = 0
    for r, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return total


def _signal_handler(sig, frame):
    print('\nCtrl+C detected. Exiting gracefully.')
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)


def _git_info(repo_dir: str):
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
        return sha, bool(dirty)
    except Exception:
        return None, None


def _resolve_base_class(method_class: str, metric: str):
    """
    The base (raw, no quantization) templated alias for a given family.
    e.g. ("MVIVF", "ip") -> IndexMVIVFIP
    """
    suffix = "IP" if metric.lower() == "ip" else "L2"
    cls_name = f"Index{method_class}{suffix}"
    if not hasattr(mvsic, cls_name):
        raise ValueError(f"Class {cls_name} not found in mvsic module.")
    return getattr(mvsic, cls_name)


def _resolve_factory(method_name: str):
    factory = getattr(mvsic.IndexParams, method_name, None)
    if factory is None:
        raise ValueError(f"IndexParams has no factory for method '{method_name}'")
    return factory


def _index_params_dict(p) -> dict:
    """Only the structural fields that matter for MVIVF-family builds."""
    out = {
        "method": str(p.method),
        "k_per_level": int(p.k_per_level),
        "max_leaf_size": int(p.max_leaf_size),
        "max_depth": int(p.max_depth),
        "num_spill": int(p.num_spill),
        "num_spill_l2": int(p.num_spill_l2),
        "s": int(p.s),
        "compress_input": bool(p.compress_input),
        "build_with_8btq": bool(p.build_with_8btq),
        "mvclus": {
            "niters": int(p.mvclus.niters),
            "max_point_clouds_per_cluster": int(p.mvclus.max_point_clouds_per_cluster),
            "max_points_per_centroid_inner_kmeans": int(
                p.mvclus.max_points_per_centroid_inner_kmeans
            ),
            "init": str(p.mvclus.init),
            "seed": int(p.mvclus.seed),
            "use_weighted_inner_kmeans": bool(p.mvclus.use_weighted_inner_kmeans),
        },
    }
    return out


def run_build(
    config: dict,
    methods: dict,
    *,
    rebuild_all: bool = False,
    fail_fast: bool = False,
) -> int:
    """
    Run all build jobs. Returns the number of failed jobs (0 = all ok).

    Python exceptions in one job are caught so the sweep continues; native
    crashes (segfault, etc.) still terminate the process.
    """
    failures: list[tuple[str, str]] = []

    for ds in config["datasets"]:
        ds_name = ds["name"]
        ds_path = ds["path"]
        index_dir = ds["index_dir"]
        print(f"\n=== Dataset: {ds_name} ({ds_path}) ===", flush=True)

        points = None             # mvsic.PointCloudSetIP, used by C++ indices.
        points_tensors = None     # list[torch.Tensor], used by FastPlaid.

        for index_details in config["indices"]:
            index_name = index_details["name"]
            metric = index_details.get("metric", "ip")
            method_info = methods[index_name]

            if index_name in {"fastplaid", "igp", "gem", "hnswlib"}:
                for bc in index_details["build_configs"]:
                    build_name = bc["name"]
                    build_params = bc.get("params") or {}
                    job_id = f"{ds_name}/{index_name}/{build_name}"

                    out_dir = os.path.join(index_dir, index_name, build_name)
                    os.makedirs(out_dir, exist_ok=True)
                    done_marker = os.path.join(out_dir, "_BUILD_OK")
                    stats_path = os.path.join(out_dir, "build_stats.json")

                    if os.path.exists(done_marker) and not (
                        rebuild_all or bc.get("rebuild", False)
                    ):
                        print(
                            f"  [{index_name}/{build_name}] exists at {out_dir}; skip.",
                            flush=True,
                        )
                        continue

                    try:
                        if index_name == "fastplaid":
                            _ensure_fastplaid_imports()
                        elif index_name == "igp":
                            _ensure_igp_imports()
                        elif index_name == "gem":
                            _ensure_gem_imports()
                        else:
                            _ensure_hnswlib_imports()
                        if points_tensors is None:
                            points_path = os.path.join(
                                ds_path, f"{ds_name}_points.pcs"
                            )
                            points_tensors = _load_point_clouds(points_path)
                        dim = points_tensors[0].shape[1]

                        print(
                            f"  [{index_name}/{build_name}] building (IP) -> {out_dir}",
                            flush=True,
                        )
                        if index_name == "fastplaid":
                            wrapper = _FastPlaidWrapper(dim, build_params, index_path=out_dir)
                        elif index_name == "igp":
                            wrapper = _IGPWrapper(dim, build_params, index_path=out_dir)
                        elif index_name == "gem":
                            # GEM build doesn't use the in-memory document
                            # tensors at all (the gem_runner subprocess
                            # streams from gem_data .npy shards), but it
                            # needs to know where to find the source .pcs
                            # files so the preprocessor can materialize the
                            # gem_data tree if it's missing.
                            wrapper = _GEMWrapper(dim, build_params, index_path=out_dir)
                            wrapper.ds_path = ds_path
                            wrapper.ds_name = ds_name
                        else:
                            bp = dict(build_params)
                            bp.setdefault("metric", metric)
                            wrapper = _HnswlibWrapper(dim, bp, index_path=out_dir)
                            wrapper.ds_path = ds_path
                            wrapper.ds_name = ds_name
                        t0 = time.time()
                        wrapper.build(points_tensors)
                        build_time = time.time() - t0

                        size_bytes = _dir_size_bytes(out_dir)
                        stats = {
                            "built_at": time.strftime(
                                "%Y-%m-%d %H:%M:%S %Z", time.localtime()
                            ),
                            "build_time_sec": build_time,
                            "index_size_mb": size_bytes / (1024 * 1024),
                            "build_params": build_params,
                        }
                        git_sha, git_dirty = _git_info(os.getcwd())
                        if git_sha:
                            stats["git_sha"] = git_sha
                            stats["git_dirty"] = git_dirty
                        with open(stats_path, "w") as f:
                            json.dump(stats, f, indent=2)
                        with open(done_marker, "w") as f:
                            f.write("ok\n")

                        print(
                            f"    build_time={build_time:.2f}s  "
                            f"size={stats['index_size_mb']:.2f}MB",
                            flush=True,
                        )
                    except Exception as e:
                        print(
                            f"  [FAIL {job_id}] {type(e).__name__}: {e}",
                            flush=True,
                        )
                        traceback.print_exc()
                        failures.append((job_id, str(e)))
                        if fail_fast:
                            break
                if fail_fast and failures:
                    break
                continue

            index_class = _resolve_base_class(method_info["class"], metric)
            factory = _resolve_factory(index_name)

            for bc in index_details["build_configs"]:
                build_name = bc["name"]
                build_params = bc.get("params") or {}
                job_id = f"{ds_name}/{index_name}/{build_name}"

                out_dir = os.path.join(index_dir, index_name, build_name)
                os.makedirs(out_dir, exist_ok=True)
                index_path = os.path.join(out_dir, "index.bin")
                params_path = os.path.join(out_dir, "index_params.json")
                stats_path = os.path.join(out_dir, "build_stats.json")

                if os.path.exists(index_path) and not (
                    rebuild_all or bc.get("rebuild", False)
                ):
                    print(
                        f"  [{index_name}/{build_name}] exists at {index_path}; skip.",
                        flush=True,
                    )
                    continue

                try:
                    if points is None:
                        points, _, _ = load_dataset(
                            ds_path, ds_name, is_mmap=bool(ds.get("is_mmap", False))
                        )
                    dim = points[0].get_dims()

                    ip = factory(**build_params)
                    with open(params_path, "w") as f:
                        json.dump(_index_params_dict(ip), f, indent=2, sort_keys=True)

                    print(
                        f"  [{index_name}/{build_name}] building ({metric.upper()}) -> {index_path}",
                        flush=True,
                    )
                    index = index_class(dim, ip)
                    t0 = time.time()
                    index.build(points)
                    build_time = time.time() - t0

                    index.save(index_path)
                    size_bytes = (
                        sum(
                            os.path.getsize(os.path.join(r, f))
                            for r, _, files in os.walk(index_path)
                            for f in files
                        )
                        if os.path.isdir(index_path)
                        else os.path.getsize(index_path)
                    )

                    stats = {
                        "built_at": time.strftime(
                            "%Y-%m-%d %H:%M:%S %Z", time.localtime()
                        ),
                        "build_time_sec": build_time,
                        "index_size_mb": size_bytes / (1024 * 1024),
                        "build_params": build_params,
                    }
                    git_sha, git_dirty = _git_info(os.getcwd())
                    if git_sha:
                        stats["git_sha"] = git_sha
                        stats["git_dirty"] = git_dirty
                    if index_name == "mvivf":
                        try:
                            stats["kmeans_tree_height"] = index.get_height()
                            stats.update(mvsic.get_mvivf_tree_stats(index))
                        except Exception as e:
                            print(f"    (skipping tree stats: {e})", flush=True)

                    with open(stats_path, "w") as f:
                        json.dump(stats, f, indent=2)

                    print(
                        f"    build_time={build_time:.2f}s  size={stats['index_size_mb']:.2f}MB",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"  [FAIL {job_id}] {type(e).__name__}: {e}",
                        flush=True,
                    )
                    traceback.print_exc()
                    failures.append((job_id, str(e)))
                    if fail_fast:
                        break
            if fail_fast and failures:
                break
        if fail_fast and failures:
            break

    if failures:
        print(f"\nBuild finished with {len(failures)} failed job(s):", flush=True)
        for jid, msg in failures:
            print(f"  {jid}: {msg}", flush=True)
    return len(failures)


def main():
    ap = argparse.ArgumentParser(description="MVSIC build-only harness")
    ap.add_argument("--config", required=True, help="Path to build YAML config")
    ap.add_argument(
        "--methods", default="benchmarks/methods.yaml", help="Path to methods.yaml"
    )
    ap.add_argument(
        "--rebuild", action="store_true", help="Rebuild even if index file exists"
    )
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the sweep after the first failed job.",
    )
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.methods) as f:
        methods = yaml.safe_load(f)

    n_fail = run_build(
        config,
        methods,
        rebuild_all=args.rebuild,
        fail_fast=args.fail_fast,
    )
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
