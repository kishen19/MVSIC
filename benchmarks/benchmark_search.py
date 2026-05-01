"""
Search-only benchmark harness.

Loads a raw-skeleton index produced by `benchmark_build.py` and runs
search sweeps. The quantization/center-compression is applied purely
at load time by picking a templated class variant (e.g.
`IndexMVIVFCompressPQIP`); the same skeleton file is reused across
variants.

Config schema
-------------
```yaml
name: "MVIVF Search"
datasets:
  - name: nq500k
    path: data/beir/nq500k
    index_dir:    experiments/mvivf_ablation/indices/nq500k
    results_dir:  experiments/mvivf_ablation/results/nq500k
indices:
  - name: mvivf                   # method name -> class via methods.yaml
    metric: ip                    # ip | l2
    builds:
      - build_name: mvivf_k16_l500
        build_params:             # same params used to build (for re-constructing IndexParams)
          k_per_level: 16
          max_leaf_size: 500
          max_depth: 0
          niters: 5
          max_point_clouds_per_cluster: 100
          max_points_per_centroid_inner_kmeans: 20
          s: 0
        variants:                 # which class variants to evaluate
          - name: ""              # default (omit for a single raw run)
            compress: false       # (mvivf/mvivf_flat/mvivf_spill/svh_ivf only; svh_graph has no compress variant)
            quantizer: None       # None|PQ|FastScan|RaBitQ|TQ|SPQTQ|OneBitTQ
        search_configs:
          - name: k=10
            params:
              nprobes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
              k: [10]
              # optional: query compression (carved/wards). See SearchParams.
              # query_compression: ["None", "Wards"]
              # query_compression_threshold: [0.7]
              # compress_rerank: [false]
              config:
                - {name: "", num_rerank: 0}
```

Layout on disk
--------------
```
<index_dir>/<index_name>/<build_name>/index.bin        (from benchmark_build.py)
<results_dir>/<index_name>/<build_name>/<variant_name>/<search_name>/<csv>
```

If `variants` is omitted, a single variant named "" with
`compress=false, quantizer=None` is used and the variant directory is
collapsed.
"""

import argparse
import os
import signal
import sys
from itertools import product

import pandas as pd
import yaml
from framework_utils import StatsExtended, load_dataset

import mvsic

# FastPlaid is an optional baseline; only imported when the config asks for it.
_FastPlaidWrapper = None
_load_point_clouds = None


def _ensure_fastplaid_imports():
    global _FastPlaidWrapper, _load_point_clouds
    if _FastPlaidWrapper is None:
        from framework_utils import FastPlaidWrapper as _FPW  # type: ignore
        from utils import load_point_clouds as _lpc  # type: ignore
        _FastPlaidWrapper = _FPW
        _load_point_clouds = _lpc


def _signal_handler(sig, frame):
    print('\nCtrl+C detected. Exiting gracefully.')
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)


# ----- Families that support the `CompressCenters` template parameter. -----
_FAMILIES_WITH_COMPRESS = {"MVIVF", "MVIVFFlat", "MVIVFSpill", "SVHIVF"}

# Map user-facing quantizer names -> the C++ class suffix.
_QUANTIZER_SUFFIX = {
    "NONE": "",
    "PQ": "PQ",
    "FASTSCAN": "FastScan",
    "RABITQ": "RaBitQ",
    "TQ": "TQ",
    "TQ4BIT": "TQ",
    "TURBOQUANT": "TQ",
    "SPQTQ": "SPQTQ",
    "ONEBITTQ": "OneBitTQ",
    "TQ1BIT": "OneBitTQ",
    "ONEBIT": "OneBitTQ",
    "EIGHTBITTQ": "EightBitTQ",
    "TQ8BIT": "EightBitTQ",
    "EIGHTBIT": "EightBitTQ",
}


def _resolve_class(family: str, compress: bool, quantizer: str, metric: str,
                   method_info: dict = None):
    """
    Map (family, compress, quantizer, metric) -> mvsic.IndexXxx class.
    e.g. ("MVIVF", True, "PQ", "ip") -> mvsic.IndexMVIVFCompressPQIP

    If `method_info` is supplied (the per-method entry from methods.yaml),
    `compress` and `quantizer` are validated against the declared
    `supports_compress` / `quantizers` axes.
    """
    q_key = (quantizer or "None").upper()
    if q_key not in _QUANTIZER_SUFFIX:
        raise ValueError(
            f"Unknown quantizer '{quantizer}'. "
            f"Expected one of: {sorted(_QUANTIZER_SUFFIX)}"
        )
    q_suffix = _QUANTIZER_SUFFIX[q_key]

    if compress and family not in _FAMILIES_WITH_COMPRESS:
        raise ValueError(
            f"Family {family} does not support compress=True "
            f"(only {sorted(_FAMILIES_WITH_COMPRESS)} do)."
        )

    # Optional validation against methods.yaml declarations.
    if method_info is not None:
        if compress and not method_info.get('supports_compress', False):
            raise ValueError(
                f"methods.yaml: family {family} has supports_compress=false, "
                f"but variant requested compress=true."
            )
        declared = method_info.get('quantizers')
        if declared is not None:
            declared_upper = {str(q).upper() for q in declared}
            if q_key not in declared_upper:
                raise ValueError(
                    f"methods.yaml: family {family} does not declare quantizer "
                    f"'{quantizer}' in its `quantizers` list ({declared})."
                )

    parts = ["Index", family]
    if compress:
        parts.append("Compress")
    if q_suffix:
        parts.append(q_suffix)
    parts.append("IP" if metric.lower() == "ip" else "L2")

    cls_name = "".join(parts)
    if not hasattr(mvsic, cls_name):
        raise ValueError(f"Class {cls_name} not found in mvsic module.")
    return cls_name, getattr(mvsic, cls_name)


def _resolve_factory(method_name: str):
    factory = getattr(mvsic.IndexParams, method_name, None)
    if factory is None:
        raise ValueError(f"IndexParams has no factory for method '{method_name}'")
    return factory


def _search_factory(method_name: str):
    f = getattr(mvsic.SearchParams, method_name, None)
    if f is None:
        raise ValueError(f"SearchParams has no factory for method '{method_name}'")
    return f


class suppress_stdout_stderr:
    def __enter__(self):
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


def _format_scalar(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return x
    if isinstance(x, bool):
        return x
    try:
        import numpy as np
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            x = float(x)
    except Exception:
        pass
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x) if x.is_integer() else round(x, 3)
    return x


def _expand_timings(df: pd.DataFrame, labels):
    if not labels or "avg_timings" not in df.columns:
        return df

    def row_expand(row):
        t = row.get("avg_timings", None)
        # Some modes (e.g. batch/search_all) intentionally do not populate
        # per-stage timings yet; keep those rows and leave timing columns empty.
        if t is None or (isinstance(t, float) and pd.isna(t)):
            t = []
        elif not isinstance(t, (list, tuple)):
            raise ValueError(f"Expected list for avg_timings, got {type(t)}")
        out = {lab: (t[i] if i < len(t) else pd.NA) for i, lab in enumerate(labels)}
        rest = list(t[len(labels):])
        if rest:
            out["timings"] = rest
        return pd.Series(out)

    expanded = df.apply(row_expand, axis=1)
    return pd.concat([df.drop(columns=["avg_timings"]), expanded], axis=1)


def _reorder_columns(df: pd.DataFrame, variable_param: str, labels):
    common_front = ["k", "recall_1_k", "recall_k_k", "QPS_seq", "QPS_par"]
    if variable_param and variable_param in df.columns:
        common_front.append(variable_param)
    for c in ["num_rerank", "avg_cmps"]:
        if c in df.columns:
            common_front.append(c)
    remaining = [
        c for c in df.columns
        if c not in common_front and c not in labels and c != "timings"
    ]
    timing_cols = [c for c in labels if c in df.columns]
    if "timings" in df.columns:
        timing_cols.append("timings")
    ordered = common_front + remaining + timing_cols
    seen = set()
    ordered = [c for c in ordered if not (c in seen or seen.add(c))]
    return df[ordered]


def _expand_search_params(search_config: dict, variable_param: str):
    """Cartesian product of params across each named `config` variant."""
    params = search_config.get('params') or {}
    if not params:
        return []
    base_variable = params.get(variable_param, [])
    out = []
    for variant in params.get('config', []):
        vname = variant['name']
        var_values = variant.get(variable_param, base_variable)
        lists = {variable_param: var_values}
        for p, v in params.items():
            if p in (variable_param, 'config'):
                continue
            lists[p] = v
        for p, v in variant.items():
            if p == 'name' or p == variable_param:
                continue
            lists[p] = v if isinstance(v, list) else [v]
        keys, values = zip(*lists.items())
        for bundle in product(*values):
            d = dict(zip(keys, bundle))
            d['_variant_name'] = vname
            out.append(d)
    return out


def _build_search_params(method_name: str, p: dict):
    """
    Map a parameter dict to a SearchParams object.
    Pulls out the optional query-compression fields and applies them
    after construction.
    """
    p = dict(p)
    qc_name = p.pop('query_compression', None)
    qc_thr = p.pop('query_compression_threshold', None)
    qc_rerank = p.pop('compress_rerank', None)

    f = _search_factory(method_name)
    sp = f(**p)

    if qc_name is not None:
        qc_enum = {
            'NONE': mvsic.SearchParams.QueryCompression.None_
                if hasattr(mvsic.SearchParams.QueryCompression, 'None_')
                else getattr(mvsic.SearchParams.QueryCompression, 'None'),
            'CARVE': mvsic.SearchParams.QueryCompression.Carve,
            'WARDS': mvsic.SearchParams.QueryCompression.Wards,
        }[str(qc_name).upper()]
        sp.query_compression = qc_enum
    if qc_thr is not None:
        sp.query_compression_threshold = float(qc_thr)
    if qc_rerank is not None:
        sp.compress_rerank = bool(qc_rerank)
    return sp


def _run_fastplaid(ds, index_details, method_info, mode):
    """Run search-only sweeps for the FastPlaid baseline (BEIR5 only).

    Supports `--mode latency` and `--mode batch`. The `multi_latency` mode is
    not supported (FastPlaid has no per-query multi-thread path); we skip it
    with a warning.
    """
    if mode == "multi_latency":
        print("  [fastplaid] multi-latency unsupported; skipping.", flush=True)
        return

    _ensure_fastplaid_imports()

    ds_name = ds['name']
    ds_path = ds['path']
    index_dir = ds['index_dir']
    results_dir = ds['results_dir']

    variable_param = method_info.get('variable_param')

    points_path = os.path.join(ds_path, f"{ds_name}_points.pcs")
    queries_path = os.path.join(ds_path, f"{ds_name}_queries.pcs")
    gt_path = os.path.join(ds_path, f"{ds_name}_chamfer_neighbors.gt")

    points = None
    queries = None
    gt = None

    for build in index_details['builds']:
        build_name = build['build_name']
        build_params = build.get('build_params') or {}

        index_root = os.path.join(index_dir, "fastplaid", build_name)
        # FastPlaid persists a directory of artifacts; treat any non-empty dir
        # as a "loaded" index.
        if not (os.path.isdir(index_root) and os.listdir(index_root)):
            print(
                f"  [fastplaid/{build_name}] MISSING {index_root} -- "
                "run benchmark_build.py first (or copy your prebuilt index here).",
                flush=True,
            )
            continue

        if queries is None:
            queries = _load_point_clouds(queries_path)
            gt = mvsic.ReadGT(gt_path, len(queries))
        # Points only matter if compute_scores needs them; FastPlaidWrapper
        # doesn't, but we still load them lazily in case the wrapper API
        # changes upstream.
        # (Skipping points to save memory.)

        dim = queries[0].shape[1]
        index = _FastPlaidWrapper(dim, build_params, index_path=index_root)

        variant_results_dir = os.path.join(
            results_dir, "fastplaid", build_name
        )
        os.makedirs(variant_results_dir, exist_ok=True)

        for search_config in build.get('search_configs', []):
            search_name = search_config['name']
            search_out_dir = os.path.join(variant_results_dir, search_name)
            os.makedirs(search_out_dir, exist_ok=True)

            combos = _expand_search_params(search_config, variable_param)
            sv_buckets = {}
            for p in combos:
                vn = p.pop('_variant_name')
                sv_buckets.setdefault(vn, []).append(p)

            for sv_name, params_list in sv_buckets.items():
                prefix = _csv_prefix_for(mode) if mode else ""
                suffix = f"_{sv_name}" if sv_name else ""
                results_path = os.path.join(
                    search_out_dir, f"{prefix}results{suffix}.csv"
                )

                append = search_config.get('append', True)
                if os.path.exists(results_path) and not append:
                    os.remove(results_path)

                if variable_param:
                    params_list.sort(key=lambda p: p.get(variable_param, 0))

                rows = []
                for params in params_list:
                    if mode == "latency":
                        res = index.compute_stats_latency(queries, gt, params)
                    else:
                        res = index.compute_stats_batch(queries, gt, params)
                    print(
                        f"      {params} | "
                        f"R@{params['k']}={res.recall_k_k:.3f} "
                        f"QPS_seq={res.QPS_seq if res.QPS_seq is not None else 0:.1f} "
                        f"QPS_par={res.QPS_par if res.QPS_par is not None else 0:.1f}",
                        flush=True,
                    )
                    row = {
                        "k": res.k,
                        "recall_1_k": res.recall_1_k,
                        "recall_k_k": res.recall_k_k,
                        "QPS_seq": res.QPS_seq if res.QPS_seq is not None else 0.0,
                        "QPS_par": res.QPS_par if res.QPS_par is not None else 0.0,
                        "avg_cmps": res.avg_cmps,
                    }
                    row.update(params)
                    rows.append(row)

                df = pd.DataFrame(rows)
                if os.path.exists(results_path):
                    try:
                        prev = pd.read_csv(results_path)
                        df = pd.concat([prev, df], ignore_index=True)
                    except pd.errors.EmptyDataError:
                        pass
                if variable_param and variable_param in df.columns:
                    df = df.sort_values(by=variable_param)
                df.to_csv(results_path, index=False)


def _csv_prefix_for(mode):
    # Pick a per-mode CSV prefix so latency/multi_latency/batch coexist in the same dir.
    return {
        'latency':       'latency_',
        'multi_latency': 'multi_latency_',
        'batch':         'batch_',
        None:            '',
    }.get(mode, '')


def run_search(config: dict, methods: dict, num_threads=None, mode=None):
    for ds in config['datasets']:
        ds_name = ds['name']
        ds_path = ds['path']
        index_dir = ds['index_dir']
        results_dir = ds['results_dir']
        print(f"\n=== Dataset: {ds_name} ({ds_path}) ===", flush=True)

        points, queries, gt = None, None, None

        for index_details in config['indices']:
            index_name = index_details['name']
            metric = index_details.get('metric', 'ip')
            method_info = methods[index_name]
            family = method_info['class']
            variable_param = method_info.get('variable_param')
            labels = method_info.get('labels') or []

            if index_name == 'fastplaid':
                _run_fastplaid(ds, index_details, method_info, mode)
                continue

            factory = _resolve_factory(index_name)

            for build in index_details['builds']:
                build_name = build['build_name']
                build_params = build.get('build_params') or {}
                variants = build.get('variants') or [
                    {'name': '', 'compress': False, 'quantizer': 'None'}
                ]

                index_path = os.path.join(
                    index_dir, index_name, build_name, "index.bin"
                )
                if not os.path.exists(index_path):
                    print(
                        f"  [{index_name}/{build_name}] MISSING {index_path} -- run benchmark_build.py first",
                        flush=True,
                    )
                    continue

                if points is None or queries is None or gt is None:
                    points, queries, gt = load_dataset(ds_path, ds_name)
                dim = queries[0].get_dims()

                ip = factory(**build_params)

                for variant in variants:
                    v_name = variant.get('name', '')
                    v_compress = bool(variant.get('compress', False))
                    v_quantizer = variant.get('quantizer', 'None')

                    cls_name, index_cls = _resolve_class(
                        family, v_compress, v_quantizer, metric,
                        method_info=method_info,
                    )
                    print(
                        f"  [{index_name}/{build_name}/{v_name or 'raw'}] "
                        f"using {cls_name}",
                        flush=True,
                    )
                    index = index_cls(dim, ip)
                    index.load(index_path, points)

                    variant_results_dir = os.path.join(
                        results_dir, index_name, build_name
                    )
                    if v_name:
                        variant_results_dir = os.path.join(variant_results_dir, v_name)
                    os.makedirs(variant_results_dir, exist_ok=True)

                    for search_config in build.get('search_configs', []):
                        search_name = search_config['name']
                        search_out_dir = os.path.join(variant_results_dir, search_name)
                        os.makedirs(search_out_dir, exist_ok=True)

                        combos = _expand_search_params(search_config, variable_param)
                        variants_map = {}
                        for p in combos:
                            vn = p.pop('_variant_name')
                            variants_map.setdefault(vn, []).append(p)

                        for sv_name, params_list in variants_map.items():
                            if mode:
                                prefix = _csv_prefix_for(mode)
                            else:
                                # Backward compat: legacy --latency / --num_threads=1 path.
                                prefix = "latency_" if num_threads else ""
                            suffix = f"_{sv_name}" if sv_name else ""
                            results_path = os.path.join(
                                search_out_dir, f"{prefix}results{suffix}.csv"
                            )

                            append = search_config.get('append', True)
                            if os.path.exists(results_path) and not append:
                                os.remove(results_path)

                            existing = None
                            if os.path.exists(results_path) and append:
                                try:
                                    existing = pd.read_csv(results_path)
                                except pd.errors.EmptyDataError:
                                    existing = None

                            if variable_param:
                                params_list.sort(
                                    key=lambda p: p.get(variable_param, 0)
                                )

                            all_results = []
                            for params in params_list:
                                skip = False
                                recall_from_cache = None
                                if existing is not None:
                                    mask = pd.Series([True] * len(existing))
                                    for k, v in params.items():
                                        if k in existing.columns:
                                            mask &= existing[k] == v
                                    if mask.any():
                                        skip = True
                                        recall_from_cache = existing[mask].iloc[0][
                                            'recall_k_k'
                                        ]

                                if skip:
                                    class Mock:
                                        def __init__(self, r):
                                            self.recall_k_k = r
                                    current = Mock(recall_from_cache)
                                else:
                                    sp = _build_search_params(index_name, params)
                                    with suppress_stdout_stderr():
                                        if mode == 'latency':
                                            res = mvsic.compute_stats_latency(
                                                index, points, queries, gt, [sp]
                                            )
                                        elif mode == 'multi_latency':
                                            res = mvsic.compute_stats_multi_latency(
                                                index, points, queries, gt, [sp]
                                            )
                                        elif mode == 'batch':
                                            res = mvsic.compute_stats_batch(
                                                index, points, queries, gt, [sp]
                                            )
                                        elif num_threads:
                                            # Back-compat path for older callers.
                                            res = mvsic.compute_stats_extended_p_threaded(
                                                index, points, queries, gt,
                                                [sp], num_threads,
                                            )
                                        else:
                                            res = mvsic.compute_stats_extended(
                                                index, points, queries, gt, [sp]
                                            )
                                        current = StatsExtended(
                                            k=params['k'],
                                            recall_1_k=min(1.0, res[0].recall_1_k),
                                            recall_k_k=min(1.0, res[0].recall_k_k),
                                            QPS_seq=res[0].QPS_seq,
                                            QPS_par=res[0].QPS_par,
                                            avg_cmps=res[0].avg_cmps,
                                            avg_timings=res[0].avg_timings,
                                        )
                                    print(
                                        f"      {params} | "
                                        f"R@{params['k']}={current.recall_k_k:.3f} "
                                        f"QPS_seq={current.QPS_seq:.1f}",
                                        flush=True,
                                    )
                                    df = pd.concat(
                                        [
                                            pd.DataFrame([current]),
                                            pd.DataFrame([params]),
                                        ],
                                        axis=1,
                                    )
                                    df = df.loc[:, ~df.columns.duplicated()]
                                    df = _expand_timings(df, labels)
                                    df = _reorder_columns(df, variable_param, labels)
                                    df = df.map(_format_scalar)

                                    if os.path.exists(results_path):
                                        try:
                                            prev = pd.read_csv(results_path)
                                            df = pd.concat([prev, df], ignore_index=True)
                                        except pd.errors.EmptyDataError:
                                            pass
                                    if variable_param and variable_param in df.columns:
                                        df = df.sort_values(by=variable_param)
                                    df.to_csv(results_path, index=False)

                                all_results.append(current)

                                # Early-exit heuristics: perfect recall / plateau / drop.
                                if current.recall_k_k >= 1.0:
                                    print("      recall@k reached 1.0, stopping sweep", flush=True)
                                    break
                                if (
                                    len(all_results) > 3
                                    and current.recall_k_k == all_results[-2].recall_k_k
                                    and current.recall_k_k == all_results[-3].recall_k_k
                                ):
                                    print("      recall plateau, stopping sweep", flush=True)
                                    break
                                if (
                                    len(all_results) > 1
                                    and current.recall_k_k < all_results[-2].recall_k_k
                                ):
                                    print("      recall dropped, stopping sweep", flush=True)
                                    break


def main():
    ap = argparse.ArgumentParser(description="MVSIC search-only harness")
    ap.add_argument("--config", required=True, help="Path to search YAML config")
    ap.add_argument(
        "--methods", default="benchmarks/methods.yaml", help="Path to methods.yaml"
    )
    ap.add_argument(
        "--mode",
        choices=["latency", "multi_latency", "batch"],
        default=None,
        help=(
            "Pick the C++ measurement entry point: "
            "`latency` -> compute_stats_latency (single-thread per-query), "
            "`multi_latency` -> compute_stats_multi_latency (per-query loop with "
            "the ambient parlay pool), `batch` -> compute_stats_batch (search_all only). "
            "If unset, falls back to the legacy compute_stats_extended[_p_threaded] flow."
        ),
    )
    ap.add_argument(
        "--num_threads",
        type=int,
        default=None,
        help=(
            "Back-compat: if set (and --mode is not), use "
            "compute_stats_extended_p_threaded with this many threads."
        ),
    )
    ap.add_argument(
        "--latency",
        action="store_true",
        help="Back-compat alias for --mode latency.",
    )
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.methods) as f:
        methods = yaml.safe_load(f)

    mode = args.mode
    num_threads = args.num_threads
    if args.latency and mode is None and num_threads is None:
        mode = 'latency'

    run_search(config, methods, num_threads=num_threads, mode=mode)


if __name__ == "__main__":
    main()
