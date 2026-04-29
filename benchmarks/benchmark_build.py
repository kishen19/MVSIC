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

import yaml
from framework_utils import load_dataset

import mvsic


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


def run_build(config: dict, methods: dict, rebuild_all: bool = False):
    for ds in config['datasets']:
        ds_name = ds['name']
        ds_path = ds['path']
        index_dir = ds['index_dir']
        print(f"\n=== Dataset: {ds_name} ({ds_path}) ===", flush=True)

        points = None

        for index_details in config['indices']:
            index_name = index_details['name']
            metric = index_details.get('metric', 'ip')
            method_info = methods[index_name]

            index_class = _resolve_base_class(method_info['class'], metric)
            factory = _resolve_factory(index_name)

            for bc in index_details['build_configs']:
                build_name = bc['name']
                build_params = bc.get('params') or {}

                out_dir = os.path.join(index_dir, index_name, build_name)
                os.makedirs(out_dir, exist_ok=True)
                index_path = os.path.join(out_dir, "index.bin")
                params_path = os.path.join(out_dir, "index_params.json")
                stats_path = os.path.join(out_dir, "build_stats.json")

                if os.path.exists(index_path) and not (rebuild_all or bc.get('rebuild', False)):
                    print(f"  [{index_name}/{build_name}] exists at {index_path}; skip.", flush=True)
                    continue

                if points is None:
                    points, _, _ = load_dataset(ds_path, ds_name)
                dim = points[0].get_dims()

                ip = factory(**build_params)
                with open(params_path, 'w') as f:
                    json.dump(_index_params_dict(ip), f, indent=2, sort_keys=True)

                print(f"  [{index_name}/{build_name}] building ({metric.upper()}) -> {index_path}", flush=True)
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
                    'built_at': time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime()),
                    'build_time_sec': build_time,
                    'index_size_mb': size_bytes / (1024 * 1024),
                    'build_params': build_params,
                }
                git_sha, git_dirty = _git_info(os.getcwd())
                if git_sha:
                    stats['git_sha'] = git_sha
                    stats['git_dirty'] = git_dirty
                if index_name == 'mvivf':
                    try:
                        stats['kmeans_tree_height'] = index.get_height()
                        stats.update(mvsic.get_mvivf_tree_stats(index))
                    except Exception as e:
                        print(f"    (skipping tree stats: {e})", flush=True)

                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=2)

                print(
                    f"    build_time={build_time:.2f}s  size={stats['index_size_mb']:.2f}MB",
                    flush=True,
                )


def main():
    ap = argparse.ArgumentParser(description="MVSIC build-only harness")
    ap.add_argument("--config", required=True, help="Path to build YAML config")
    ap.add_argument(
        "--methods", default="benchmarks/methods.yaml", help="Path to methods.yaml"
    )
    ap.add_argument(
        "--rebuild", action="store_true", help="Rebuild even if index file exists"
    )
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.methods) as f:
        methods = yaml.safe_load(f)

    run_build(config, methods, rebuild_all=args.rebuild)


if __name__ == "__main__":
    main()
