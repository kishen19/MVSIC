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
              # optional: 8-bit TurboQuant rerank gate. Defaults to true; set
              # to false to fall back to exact-float rerank for any method.
              # tq8_rerank: [true]
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
import struct
import sys
import tempfile
import traceback
from itertools import product

import numpy as np
import pandas as pd
import yaml
from framework_utils import StatsExtended, load_dataset

import mvsic

# Default query-subsampling cap. When the queries file has more than this many
# point clouds, we deterministically sample down to this size with the seed
# below so latency / batch / multi_latency runs cost a fixed amount per
# parameter combo regardless of dataset size.  Both the cap and the seed are
# tunable via the `--query_subsample` / `--query_subsample_seed` CLI flags.
_DEFAULT_QUERY_SUBSAMPLE = 1000
_DEFAULT_QUERY_SUBSAMPLE_SEED = 42

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


def _read_pcs_buffers(path):
    """Read a `.pcs` query file into raw numpy buffers.

    File layout (matches benchmarks/utils.py::load_point_clouds):

        uint64 dim, uint64 n, uint64 num_vectors,
        float32[num_vectors * dim] data,
        uint64 num_offsets,
        uint64[num_offsets] offsets,                # offsets are in floats
                                                    # (so cloud i has cloud_size = (offsets[i+1]-offsets[i]) // dim points)

    Some pcs writers append `n` per-cloud `uint32_t` ids after the offsets;
    others don't. We try to read them when present and fall back to
    ``arange(n)`` otherwise.
    """
    with open(path, 'rb') as f:
        dim = struct.unpack('Q', f.read(8))[0]
        n = struct.unpack('Q', f.read(8))[0]
        num_vectors = struct.unpack('Q', f.read(8))[0]
        data = np.fromfile(f, dtype=np.float32, count=num_vectors * dim)
        num_offsets = struct.unpack('Q', f.read(8))[0]
        offsets = np.fromfile(f, dtype=np.uint64, count=num_offsets)
        ids_buf = np.fromfile(f, dtype=np.uint32, count=n)
        if ids_buf.size != n:
            ids_buf = np.arange(n, dtype=np.uint32)
    return int(dim), int(n), data, offsets, ids_buf


def _subsample_pcs_buffers(dim, n, data, offsets, ids_buf, indices):
    """Select rows from a flat-pcs buffer by ``indices`` and return new
    ``(data, offsets, ids)`` arrays suitable for the
    ``mvsic.PointCloudSetIP(data, offsets, ids, dim)`` constructor.

    Offsets in the original file are float-element offsets; we preserve that
    convention in the rebuilt buffers.
    """
    indices = np.asarray(indices, dtype=np.int64)
    new_n = int(indices.size)
    # Cloud i in the original file occupies floats [offsets[i], offsets[i+1]).
    # Build the new flat data by gathering each selected slice in order.
    starts = offsets[indices].astype(np.int64)
    ends = offsets[indices + 1].astype(np.int64)
    sizes = ends - starts
    total = int(sizes.sum())
    new_data = np.empty(total, dtype=np.float32)
    new_offsets = np.empty(new_n + 1, dtype=np.uint64)
    new_offsets[0] = 0
    cursor = 0
    for i in range(new_n):
        s, e = int(starts[i]), int(ends[i])
        sz = e - s
        new_data[cursor:cursor + sz] = data[s:e]
        cursor += sz
        new_offsets[i + 1] = cursor
    new_ids = ids_buf[indices].astype(np.uint32)
    return new_data, new_offsets, new_ids


def _subsample_gt_via_temp(gt_path, indices, gt_loader=None):
    if gt_loader is None:
        gt_loader = mvsic.ReadGT
    """Slice rows out of a `.gt` file by ``indices``, write a temporary `.gt`
    file with the selected rows, and let ``gt_loader`` (default the C++
    ``mvsic.ReadGT``) parse it.

    GT file layout (matches mvsic/core/stats.h::ReadGT):

        int32 num_neighbors,
        for each query: num_neighbors * (float32 dist, uint32 docid)

    This avoids touching the C++ side of GT parsing.
    """
    indices = np.asarray(indices, dtype=np.int64)
    rec_size = 8  # (float, uint32) = 4 + 4 = 8 bytes
    with open(gt_path, 'rb') as f:
        header = f.read(4)
        num_neighbors = struct.unpack('i', header)[0]
        row_size = num_neighbors * rec_size
        # Read all rows into a flat byte buffer; per-row slicing is cheaper
        # than seeking many small times for the typical 7k-12k query files.
        body = f.read()
    rows = memoryview(body)
    out_path = tempfile.NamedTemporaryFile(
        prefix="mvsic_subgt_", suffix=".gt", delete=False
    ).name
    with open(out_path, 'wb') as f:
        f.write(header)
        for i in indices:
            base = int(i) * row_size
            f.write(rows[base:base + row_size])
    try:
        return gt_loader(out_path, int(indices.size))
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


def _maybe_subsample_queries(ds, points, queries, gt, n_max, seed,
                             *, fastplaid=False):
    """If ``queries`` has more than ``n_max`` clouds, deterministically sample
    ``n_max`` of them (seeded with ``seed``) and rebuild ``queries`` and
    ``gt`` to match.  ``points`` is left untouched (the GT doc-ids index into
    points, which is unchanged by query subsampling).

    Returns ``(queries, gt)`` ready to be passed to the per-mode entry
    points.  When subsampling is disabled (``n_max <= 0``) or the queries
    are already at or below the cap, returns the inputs verbatim.
    """
    if n_max is None or n_max <= 0:
        return queries, gt
    n_q = len(queries) if fastplaid else queries.size()
    if n_q <= n_max:
        return queries, gt

    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(n_q, size=int(n_max), replace=False))

    ds_path = ds["path"]
    ds_name = ds["name"]
    queries_path = os.path.join(ds_path, f"{ds_name}_queries.pcs")
    gt_path = os.path.join(ds_path, f"{ds_name}_chamfer_neighbors.gt")

    if fastplaid:
        # FastPlaid stores queries as a python list of torch tensors, so we
        # can slice directly without re-reading the pcs file.
        new_queries = [queries[int(i)] for i in indices]
    else:
        dim, _n, data, offsets, ids_buf = _read_pcs_buffers(queries_path)
        new_data, new_offsets, new_ids = _subsample_pcs_buffers(
            dim, _n, data, offsets, ids_buf, indices
        )
        new_queries = mvsic.PointCloudSetIP(
            data=new_data, offsets=new_offsets, ids=new_ids, dim=dim,
        )

    new_gt = _subsample_gt_via_temp(gt_path, indices)
    print(
        f"  [subsample] {ds_name}: queries {n_q} -> {int(n_max)} "
        f"(seed={int(seed)})",
        flush=True,
    )
    return new_queries, new_gt


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
    """Cartesian product of params across each named `config` variant.

    For each entry in ``params.config``, the sweep axis named by ``variable_param``
    (see ``benchmarks/methods.yaml``, e.g. ``L`` for ``svh_graph``) can be set
    **per variant**: ``variant.get(variable_param, params[variable_param])``.
    So each ``{name: nr..., num_rerank: N, L: [...]}`` row can use a different
    ``L`` list for that rerank bucket only.
    """
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
    Pulls out the optional query-compression and tq8_rerank fields and
    applies them after construction so the underlying factory signatures
    don't need to be method-name aware.
    """
    p = dict(p)
    qc_name = p.pop('query_compression', None)
    qc_thr = p.pop('query_compression_threshold', None)
    qc_rerank = p.pop('compress_rerank', None)
    tq8_rerank = p.pop('tq8_rerank', None)
    # `root_m2m` is the gated MVIVF / MVIVF Spill knob that fuses root-level
    # distance work across queries (see SearchParams::root_m2m). Default true
    # in C++; we accept it from YAML so configs can opt out for ablations.
    # Pop it pre-factory so the (k, nprobes, ...) signature isn't broken.
    root_m2m = p.pop('root_m2m', None)

    f = _search_factory(method_name)
    sp = f(**p)

    if qc_name is not None:
        # The pybind11 enum (see mvsic/python/register_common.cc) exposes
        # NONE / CARVE / WARDS (all caps); accept any case in the YAML.
        qc_enum = {
            'NONE':  mvsic.SearchParams.QueryCompression.NONE,
            'CARVE': mvsic.SearchParams.QueryCompression.CARVE,
            'WARDS': mvsic.SearchParams.QueryCompression.WARDS,
        }[str(qc_name).upper()]
        sp.query_compression = qc_enum
    if qc_thr is not None:
        sp.query_compression_threshold = float(qc_thr)
    if qc_rerank is not None:
        sp.compress_rerank = bool(qc_rerank)
    if tq8_rerank is not None:
        sp.tq8_rerank = bool(tq8_rerank)
    if root_m2m is not None:
        sp.root_m2m = bool(root_m2m)
    return sp


def _run_fastplaid(ds, index_details, method_info, mode, qps_thresh=None,
                   subsample_n=None, subsample_seed=None):
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
            queries, gt = _maybe_subsample_queries(
                ds, None, queries, gt, subsample_n, subsample_seed,
                fastplaid=True,
            )
        # Points only matter if compute_scores needs them; FastPlaidWrapper
        # doesn't, but we still load them lazily in case the wrapper API
        # changes upstream.
        # (Skipping points to save memory.)

        dim = queries[0].shape[1]
        # Mode-aware device selection (CPU-only setup):
        #   latency -> ``device="cpu"``  (single-device fast path; single-threaded
        #              kernel pinned via ``torch.set_num_threads(1)`` inside
        #              ``compute_stats_latency``).
        #   batch   -> ``device=None``   (FastPlaid default = ``["cpu"] *
        #              cpu_count`` -> one joblib worker per core; uses all
        #              threads for the single batched ``search`` call).
        # On a GPU box we let FastPlaid auto-select ``["cuda"]`` for both modes.
        try:
            import torch as _torch_for_device  # FastPlaid is already imported.
            _has_cuda = _torch_for_device.cuda.is_available()
        except Exception:
            _has_cuda = False
        if mode == "latency":
            fp_device = "cuda" if _has_cuda else "cpu"
        else:
            fp_device = None  # FastPlaid default (multi-CPU or single GPU)
        index = _FastPlaidWrapper(
            dim, build_params, index_path=index_root, device=fp_device
        )

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

                # Resume-aware QPS-thresh check: if the last row in the
                # cached CSV (sorted by variable_param) is already below
                # qps_thresh, skip the rest of the sweep.
                if (
                    qps_thresh is not None
                    and os.path.exists(results_path)
                    and append
                ):
                    try:
                        prev_df = pd.read_csv(results_path)
                    except pd.errors.EmptyDataError:
                        prev_df = None
                    if prev_df is not None and not prev_df.empty:
                        qps_col = _qps_attr_for_mode(mode)
                        if qps_col in prev_df.columns:
                            sort_col = (
                                variable_param
                                if variable_param
                                and variable_param in prev_df.columns
                                else None
                            )
                            ex = (
                                prev_df.sort_values(by=sort_col)
                                if sort_col
                                else prev_df
                            )
                            try:
                                last_qps = float(ex.iloc[-1][qps_col])
                            except (TypeError, ValueError):
                                last_qps = None
                            if (
                                last_qps is not None
                                and last_qps > 0
                                and last_qps < qps_thresh
                            ):
                                print(
                                    f"      [fastplaid] cached {qps_col}={last_qps:.2f} "
                                    f"< qps_thresh={qps_thresh}; "
                                    "skipping remaining sweep",
                                    flush=True,
                                )
                                params_list = []

                rows = []
                qps_attr = _qps_attr_for_mode(mode)
                qps_label = _qps_label_for_mode(mode)
                stop_sweep = False
                for params in params_list:
                    if stop_sweep:
                        break
                    try:
                        if mode == "latency":
                            res = index.compute_stats_latency(queries, gt, params)
                        else:
                            res = index.compute_stats_batch(queries, gt, params)
                        qps_val = getattr(res, qps_attr, None)
                        qps_str = (
                            f"{qps_val:.1f}"
                            if qps_val is not None
                            else "N/A"
                        )
                        print(
                            f"      {params} | "
                            f"R@{params['k']}={res.recall_k_k:.3f} "
                            f"{qps_label}={qps_str}",
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
                        if (
                            qps_thresh is not None
                            and qps_val is not None
                            and qps_val > 0
                            and qps_val < qps_thresh
                        ):
                            print(
                                f"      [fastplaid] {qps_attr}={qps_val:.2f} "
                                f"< qps_thresh={qps_thresh}; stopping sweep",
                                flush=True,
                            )
                            stop_sweep = True
                    except Exception as e:
                        jid = f"{ds_name}/fastplaid/{build_name}/{search_name}/{sv_name}"
                        print(
                            f"  [FAIL {jid}] {type(e).__name__}: {e}",
                            flush=True,
                        )
                        traceback.print_exc()

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


# The "QPS that matters" for each mode -- used by both the log line and the
# QPS_thresh early-exit. compute_stats_latency populates QPS_seq (single-thread
# per-query latency, so we report 1-thread QPS); compute_stats_multi_latency
# populates QPS_par (per-query latency at full parallelism); compute_stats_batch
# also reports QPS_par (single batched search_all). The legacy
# compute_stats_extended[_p_threaded] paths report QPS_seq.
def _qps_attr_for_mode(mode, num_threads=None):
    if mode == 'latency':
        return 'QPS_seq'
    if mode == 'multi_latency':
        return 'QPS_par'
    if mode == 'batch':
        return 'QPS_par'
    return 'QPS_seq'  # legacy compute_stats_extended path


def _qps_label_for_mode(mode, num_threads=None):
    if mode == 'latency':
        return 'QPS (1-thrd)'
    if mode == 'multi_latency':
        return 'QPS (all-thrd)'
    if mode == 'batch':
        return 'QPS (batch)'
    return 'QPS_seq'  # legacy



def run_search(
    config: dict,
    methods: dict,
    num_threads=None,
    mode=None,
    *,
    fail_fast: bool = False,
    qps_thresh: float | None = None,
    subsample_n: int | None = None,
    subsample_seed: int | None = None,
) -> int:
    """Return the number of failed jobs / parameter evaluations."""
    failures: list[tuple[str, str]] = []
    abort = False

    for ds in config["datasets"]:
        if abort:
            break
        ds_name = ds["name"]
        ds_path = ds["path"]
        index_dir = ds["index_dir"]
        results_dir = ds["results_dir"]
        print(f"\n=== Dataset: {ds_name} ({ds_path}) ===", flush=True)

        points, queries, gt = None, None, None

        for index_details in config["indices"]:
            if abort:
                break
            index_name = index_details["name"]
            metric = index_details.get("metric", "ip")
            method_info = methods[index_name]
            family = method_info["class"]
            variable_param = method_info.get("variable_param")
            labels = method_info.get("labels") or []

            if index_name == "fastplaid":
                try:
                    _run_fastplaid(
                        ds, index_details, method_info, mode,
                        qps_thresh=qps_thresh,
                        subsample_n=subsample_n,
                        subsample_seed=subsample_seed,
                    )
                except Exception as e:
                    jid = f"{ds_name}/fastplaid"
                    print(f"  [FAIL {jid}] {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                    failures.append((jid, str(e)))
                    if fail_fast:
                        abort = True
                continue

            factory = _resolve_factory(index_name)

            for build in index_details["builds"]:
                if abort:
                    break
                build_name = build["build_name"]
                build_params = build.get("build_params") or {}
                variants = build.get("variants") or [
                    {"name": "", "compress": False, "quantizer": "None"}
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
                    points, queries, gt = load_dataset(
                        ds_path, ds_name, is_mmap=bool(ds.get("is_mmap", False))
                    )
                    queries, gt = _maybe_subsample_queries(
                        ds, points, queries, gt, subsample_n, subsample_seed,
                        fastplaid=False,
                    )
                dim = queries[0].get_dims()

                ip = factory(**build_params)

                for variant in variants:
                    if abort:
                        break
                    v_name = variant.get("name", "")
                    v_compress = bool(variant.get("compress", False))
                    v_quantizer = variant.get("quantizer", "None")

                    job_id = f"{ds_name}/{index_name}/{build_name}/{v_name or 'raw'}"

                    try:
                        cls_name, index_cls = _resolve_class(
                            family,
                            v_compress,
                            v_quantizer,
                            metric,
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
                            variant_results_dir = os.path.join(
                                variant_results_dir, v_name
                            )
                        os.makedirs(variant_results_dir, exist_ok=True)

                        for search_config in build.get("search_configs", []):
                            search_name = search_config["name"]
                            search_out_dir = os.path.join(
                                variant_results_dir, search_name
                            )
                            os.makedirs(search_out_dir, exist_ok=True)

                            combos = _expand_search_params(
                                search_config, variable_param
                            )
                            variants_map = {}
                            for p in combos:
                                vn = p.pop("_variant_name")
                                variants_map.setdefault(vn, []).append(p)

                            for sv_name, params_list in variants_map.items():
                                if mode:
                                    prefix = _csv_prefix_for(mode)
                                else:
                                    prefix = "latency_" if num_threads else ""
                                suffix = f"_{sv_name}" if sv_name else ""
                                results_path = os.path.join(
                                    search_out_dir, f"{prefix}results{suffix}.csv"
                                )

                                append = search_config.get("append", True)
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

                                # If the resumed CSV already shows the latest
                                # cached value falling below ``qps_thresh``,
                                # skip computing the next sweep step entirely.
                                # The check uses the per-mode "QPS that matters"
                                # (see _qps_attr_for_mode) and the same
                                # variable_param ordering as the live sweep.
                                cached_below_thresh = False
                                if (
                                    qps_thresh is not None
                                    and existing is not None
                                    and not existing.empty
                                ):
                                    qps_col = _qps_attr_for_mode(mode, num_threads)
                                    if qps_col in existing.columns:
                                        sort_col = (
                                            variable_param
                                            if variable_param
                                            and variable_param in existing.columns
                                            else None
                                        )
                                        ex = (
                                            existing.sort_values(by=sort_col)
                                            if sort_col
                                            else existing
                                        )
                                        try:
                                            last_qps = float(
                                                ex.iloc[-1][qps_col]
                                            )
                                        except (TypeError, ValueError):
                                            last_qps = None
                                        if (
                                            last_qps is not None
                                            and last_qps > 0
                                            and last_qps < qps_thresh
                                        ):
                                            cached_below_thresh = True
                                            print(
                                                f"      cached {qps_col}={last_qps:.2f} "
                                                f"< qps_thresh={qps_thresh}; "
                                                "skipping remaining sweep",
                                                flush=True,
                                            )

                                all_results = []
                                if cached_below_thresh:
                                    params_list = []

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
                                            recall_from_cache = existing[
                                                mask
                                            ].iloc[0]["recall_k_k"]

                                    if skip:

                                        class Mock:
                                            def __init__(self, r):
                                                self.recall_k_k = r

                                        current = Mock(recall_from_cache)
                                    else:
                                        try:
                                            sp = _build_search_params(
                                                index_name, params
                                            )
                                            with suppress_stdout_stderr():
                                                if mode == "latency":
                                                    res = mvsic.compute_stats_latency(
                                                        index,
                                                        points,
                                                        queries,
                                                        gt,
                                                        [sp],
                                                    )
                                                elif mode == "multi_latency":
                                                    res = mvsic.compute_stats_multi_latency(
                                                        index,
                                                        points,
                                                        queries,
                                                        gt,
                                                        [sp],
                                                    )
                                                elif mode == "batch":
                                                    res = mvsic.compute_stats_batch(
                                                        index,
                                                        points,
                                                        queries,
                                                        gt,
                                                        [sp],
                                                    )
                                                elif num_threads:
                                                    res = mvsic.compute_stats_extended_p_threaded(
                                                        index,
                                                        points,
                                                        queries,
                                                        gt,
                                                        [sp],
                                                        num_threads,
                                                    )
                                                else:
                                                    res = mvsic.compute_stats_extended(
                                                        index,
                                                        points,
                                                        queries,
                                                        gt,
                                                        [sp],
                                                    )
                                                current = StatsExtended(
                                                    k=params["k"],
                                                    recall_1_k=min(
                                                        1.0, res[0].recall_1_k
                                                    ),
                                                    recall_k_k=min(
                                                        1.0, res[0].recall_k_k
                                                    ),
                                                    QPS_seq=res[0].QPS_seq,
                                                    QPS_par=res[0].QPS_par,
                                                    avg_cmps=res[0].avg_cmps,
                                                    avg_timings=res[0].avg_timings,
                                                )
                                            qps_attr = _qps_attr_for_mode(
                                                mode, num_threads
                                            )
                                            qps_label = _qps_label_for_mode(
                                                mode, num_threads
                                            )
                                            qps_val = getattr(
                                                current, qps_attr, None
                                            )
                                            qps_str = (
                                                f"{qps_val:.1f}"
                                                if qps_val is not None
                                                else "N/A"
                                            )
                                            print(
                                                f"      {params} | "
                                                f"R@{params['k']}={current.recall_k_k:.3f} "
                                                f"{qps_label}={qps_str}",
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
                                            df = _reorder_columns(
                                                df, variable_param, labels
                                            )
                                            df = df.map(_format_scalar)

                                            if os.path.exists(results_path):
                                                try:
                                                    prev = pd.read_csv(results_path)
                                                    df = pd.concat(
                                                        [prev, df], ignore_index=True
                                                    )
                                                except pd.errors.EmptyDataError:
                                                    pass
                                            if (
                                                variable_param
                                                and variable_param in df.columns
                                            ):
                                                df = df.sort_values(
                                                    by=variable_param
                                                )
                                            df.to_csv(results_path, index=False)
                                        except Exception as e:
                                            pe = f"{job_id}/{search_name}/{sv_name}"
                                            print(
                                                f"  [FAIL {pe}] params={params} | "
                                                f"{type(e).__name__}: {e}",
                                                flush=True,
                                            )
                                            traceback.print_exc()
                                            continue

                                    all_results.append(current)

                                    if current.recall_k_k >= 1.0:
                                        print(
                                            "      recall@k reached 1.0, stopping sweep",
                                            flush=True,
                                        )
                                        break
                                    if (
                                        len(all_results) > 3
                                        and current.recall_k_k
                                        == all_results[-2].recall_k_k
                                        and current.recall_k_k
                                        == all_results[-3].recall_k_k
                                    ):
                                        print(
                                            "      recall plateau, stopping sweep",
                                            flush=True,
                                        )
                                        break
                                    if (
                                        len(all_results) > 1
                                        and current.recall_k_k
                                        < all_results[-2].recall_k_k
                                    ):
                                        print(
                                            "      recall dropped, stopping sweep",
                                            flush=True,
                                        )
                                        break

                                    # QPS-threshold early exit. Use the per-mode
                                    # "QPS that matters" (see _qps_attr_for_mode)
                                    # so latency stops on QPS_seq while
                                    # multi_latency / batch stop on QPS_par.
                                    # Cached / mocked rows lack QPS attrs; only
                                    # check freshly-computed rows.
                                    if qps_thresh is not None and not skip:
                                        qps_attr = _qps_attr_for_mode(
                                            mode, num_threads
                                        )
                                        qps_val = getattr(
                                            current, qps_attr, None
                                        )
                                        if (
                                            qps_val is not None
                                            and qps_val > 0
                                            and qps_val < qps_thresh
                                        ):
                                            print(
                                                f"      {qps_attr}={qps_val:.2f} "
                                                f"< qps_thresh={qps_thresh}; "
                                                "stopping sweep",
                                                flush=True,
                                            )
                                            break
                    except Exception as e:
                        print(
                            f"  [FAIL {job_id}] {type(e).__name__}: {e}",
                            flush=True,
                        )
                        traceback.print_exc()
                        failures.append((job_id, str(e)))
                        if fail_fast:
                            abort = True
                            break
                if abort:
                    break
            if abort:
                break
        if abort:
            break

    if failures:
        print(
            f"\nSearch finished with {len(failures)} failed job(s)/stage(s):",
            flush=True,
        )
        for jid, msg in failures:
            print(f"  {jid}: {msg}", flush=True)
    return len(failures)


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
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed job or parameter evaluation.",
    )
    ap.add_argument(
        "--QPS_thresh",
        type=float,
        default=10.0,
        dest="qps_thresh",
        help=(
            "Stop a sweep once the per-mode QPS falls below this threshold "
            "(latency: QPS_seq, multi_latency/batch: QPS_par). Also applies "
            "to resumed sweeps: if the last cached row already sits below "
            "the threshold, the next param is skipped. Set <=0 to disable. "
            "Default: 10."
        ),
    )
    ap.add_argument(
        "--query_subsample",
        type=int,
        default=_DEFAULT_QUERY_SUBSAMPLE,
        dest="subsample_n",
        help=(
            "If the queries file has more than this many point clouds, "
            "deterministically sample down to this size before running any "
            "sweep, so the time per parameter combo is fixed across "
            "datasets. Set <=0 to disable subsampling. Default: 1000."
        ),
    )
    ap.add_argument(
        "--query_subsample_seed",
        type=int,
        default=_DEFAULT_QUERY_SUBSAMPLE_SEED,
        dest="subsample_seed",
        help=(
            "Seed for --query_subsample. Same seed -> same indices across "
            "runs and across modes, so latency/multi_latency/batch all see "
            "the same subset for a given dataset. Default: 42."
        ),
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

    qps_thresh = args.qps_thresh if args.qps_thresh and args.qps_thresh > 0 else None
    subsample_n = (
        args.subsample_n if args.subsample_n and args.subsample_n > 0 else None
    )
    n_fail = run_search(
        config,
        methods,
        num_threads=num_threads,
        mode=mode,
        fail_fast=args.fail_fast,
        qps_thresh=qps_thresh,
        subsample_n=subsample_n,
        subsample_seed=args.subsample_seed,
    )
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
