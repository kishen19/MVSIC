import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from itertools import product

import pandas as pd
import yaml
from framework_utils import *
from utils import load_point_clouds

import mvsic


# Graceful exit on Ctrl+C
def signal_handler(sig, frame):
    print('\nCtrl+C detected. Exiting gracefully.')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def _get_git_info(repo_dir: str):
    """
    Best-effort git metadata for reproducibility.
    Returns (sha, is_dirty). If unavailable, returns (None, None).
    """
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        dirty = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_dir, stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        return sha, bool(dirty)
    except Exception:
        return None, None


def _format_scalar_for_csv(x):
    """Round floats to <=3 decimals; print whole numbers as ints."""
    # Preserve None/NaN
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return x
    # Preserve booleans as-is
    if isinstance(x, bool):
        return x
    # Numpy scalars / pandas may show up here
    try:
        import numpy as np

        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            x = float(x)
    except Exception:
        pass

    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if x.is_integer():
            return int(x)
        return round(x, 3)
    return x


def _expand_method_timings(df, method_info):
    """
    If methods.yaml provides 'labels', expand avg_timings (list) into separate columns.
    - Error if avg_timings has fewer entries than labels.
    - If avg_timings has extra entries, store the remainder as a list in a 'timings' column.
    """
    labels = method_info.get("labels") or []
    if not labels or "avg_timings" not in df.columns:
        return df

    def _row_expand(row):
        t = row.get("avg_timings", None)
        if not isinstance(t, (list, tuple)):
            raise ValueError(
                f"Expected avg_timings to be a list/tuple for labeled method; got {type(t)}"
            )
        if len(t) < len(labels):
            raise ValueError(
                f"avg_timings has {len(t)} entries but methods.yaml defines {len(labels)} labels: {labels}"
            )
        out = {lab: t[i] for i, lab in enumerate(labels)}
        rest = list(t[len(labels):])
        if rest:
            out["timings"] = rest
        return pd.Series(out)

    expanded = df.apply(_row_expand, axis=1)
    df = pd.concat([df.drop(columns=["avg_timings"]), expanded], axis=1)
    return df


def _reorder_result_columns(df, method_info):
    """Common columns first; method-specific timing columns at the end."""
    variable_param = method_info.get("variable_param")
    labels = method_info.get("labels") or []

    common_front = [
        "k",
        "recall_1_k",
        "recall_k_k",
        "QPS_seq",
        "QPS_par",
    ]
    if variable_param and variable_param in df.columns:
        common_front.append(variable_param)
    # These are commonly present in configs/results
    for c in ["num_rerank", "avg_cmps"]:
        if c in df.columns:
            common_front.append(c)

    # Remaining columns: keep params & other metadata next.
    remaining = [c for c in df.columns if c not in common_front and c not in labels and c != "timings"]
    # Timing label columns go at the end.
    timing_cols = [c for c in labels if c in df.columns]
    if "timings" in df.columns:
        timing_cols.append("timings")

    ordered = common_front + remaining + timing_cols
    # Deduplicate while preserving order
    seen = set()
    ordered = [c for c in ordered if not (c in seen or seen.add(c))]
    return df[ordered]


# Context manager to suppress stdout/stderr
# To silence long errors from c++ backend during index build/search
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


# hash params to get a unique code for index file names
def get_params_hash(params):
    """Creates a stable hash from a dictionary of parameters."""
    sorted_params = json.dumps(params, sort_keys=True)
    return hashlib.md5(sorted_params.encode('utf-8')).hexdigest()[:8]


def _enum_to_name(e):
    # pybind11 enums typically expose .name; fall back to str().
    name = getattr(e, "name", None)
    if isinstance(name, str) and name:
        return name
    s = str(e)
    # Often "QuantizerType.PQ" -> "PQ"
    if "." in s:
        return s.split(".")[-1]
    return s


def index_params_to_dict(p):
    """
    Materialize a full IndexParams -> nested dict (including defaults),
    matching the exposed pybind attributes.
    """
    def pq_to_dict(pq):
        return {
            "method": _enum_to_name(pq.method),
            "block_size": int(pq.block_size),
            "num_clusters_per_block": int(pq.num_clusters_per_block),
            "num_points_per_cluster": int(pq.num_points_per_cluster),
            "rabitq_bits": int(pq.rabitq_bits),
        }

    def fde_to_dict(fde):
        return {
            "num_repetitions": int(fde.num_repetitions),
            "num_simhash_projections": int(fde.num_simhash_projections),
            "seed": int(fde.seed),
            "projection_dimension": int(fde.projection_dimension),
            "fill_empty_partitions": bool(fde.fill_empty_partitions),
            "final_projection_dimension": int(fde.final_projection_dimension),
            "normalize": bool(fde.normalize),
        }

    def ann_to_dict(ann):
        return {
            "R": int(ann.R),
            "L": int(ann.L),
            "alpha": float(ann.alpha),
            "num_pass": int(ann.num_pass),
        }

    def mvclus_to_dict(mv):
        return {
            "niters": int(mv.niters),
            "max_point_clouds_per_cluster": int(mv.max_point_clouds_per_cluster),
            "max_points_per_centroid_inner_kmeans": int(mv.max_points_per_centroid_inner_kmeans),
            "verbose": int(mv.verbose),
            "init": str(mv.init),
            "seed": int(mv.seed),
            "use_weighted_inner_kmeans": bool(mv.use_weighted_inner_kmeans),
        }

    return {
        "method": str(p.method),
        "verbose": int(p.verbose),
        "compress_input": bool(p.compress_input),
        "k_per_level": int(p.k_per_level),
        "max_leaf_size": int(p.max_leaf_size),
        "quantize_centers": bool(p.quantize_centers),
        "mvclus": mvclus_to_dict(p.mvclus),
        "s": int(p.s),
        "fde": fde_to_dict(p.fde),
        "ann": ann_to_dict(p.ann),
        "normalize": bool(p.normalize),
        "R": int(p.R),
        "L": int(p.L),
        "alpha": float(p.alpha),
        "two_pass": bool(p.two_pass),
        "pq": pq_to_dict(p.pq),
        "max_points_per_centroid": int(p.max_points_per_centroid),
    }


def generate_search_params(search_config, method_info):
    """
    Generates search parameter combinations from the config, handling overrides.
    """
    search_params = search_config.get('params') or {}
    if not search_params:
        return []

    variable_param_name = method_info.get('variable_param')
    if not variable_param_name:
        # Fallback or error if not defined in methods.yaml
        # For now, let's try to infer, but a warning would be good.
        print(f'  Warning: \'variable_param\' not defined for method. Inferring...', flush=True)
        inferred = [p for p in ['nprobes', 'L', 'n_ivf_probe'] if p in search_params]
        if not inferred:
            raise ValueError("Could not determine variable parameter.")
        variable_param_name = inferred[0]
        print(f'  Inferred \'{variable_param_name}\'.', flush=True)

    base_variable_params = search_params.get(variable_param_name, [])

    all_param_combos = []

    for variant_config in search_params.get('config', []):
        variant_name = variant_config['name']
        # Check for override of the variable parameter
        variant_variable_params = variant_config.get(variable_param_name, base_variable_params)
        # Create a dictionary of all parameters for this variant
        param_lists = {variable_param_name: variant_variable_params}
        for p, v in search_params.items():
            if p != variable_param_name and p != 'config':
                param_lists[p] = v
        for p, v in variant_config.items():
            if p != 'name' and p != variable_param_name:
                # This allows variants to override other params too, not just the variable one
                if isinstance(v, list):
                    param_lists[p] = v
                else:
                    param_lists[p] = [v]
        # Generate Cartesian product of parameters
        keys, values = zip(*param_lists.items())
        for bundle in product(*values):
            param_dict = dict(zip(keys, bundle))
            # Attach variant name for result tracking
            param_dict['_variant_name'] = variant_name
            all_param_combos.append(param_dict)

    return all_param_combos


def run(config, methods, experiment_name, tasks, num_threads=None):
    """
    Builds indices and runs searches based on the experiment config and specified tasks.
    """
    print(f"--- Running experiment: {experiment_name} for tasks: {tasks} ---", flush=True)

    # Use 'datasets' list if available, otherwise fall back to single 'dataset' for backward compatibility
    dataset_configs = config.get('datasets') or [config.get('dataset')]
    if not dataset_configs or dataset_configs == [None]:
        raise ValueError("No 'datasets' or 'dataset' key found in the config file.")

    for dataset_config in dataset_configs:
        print(f"\n--- Processing dataset: {dataset_config['name']} ---", flush=True)

        # Storage paths.  The cleaner schema is a nested `storage:` block with
        # `results_dir` and `index_dir`; the legacy flat `results` and
        # `index_dir` keys on `dataset_config` are still accepted for
        # back-compat (and override the storage block if both are present).
        storage_cfg = dataset_config.get('storage') or {}
        default_results_dir = os.path.join('results', experiment_name)
        base_results_dir = (
            dataset_config.get('results')
            or storage_cfg.get('results_dir')
            or default_results_dir
        )
        # Index .bin files can go to a separate directory (not synced from cloud).
        # Default is alongside results if neither 'index_dir' nor 'storage.index_dir'
        # is specified.
        base_index_dir = (
            dataset_config.get('index_dir')
            or storage_cfg.get('index_dir')
            or base_results_dir
        )

        for index_details in config['indices']:
            index_name = index_details['name']
            method_info = methods[index_name]

            points, queries, gt = None, None, None

            for build_config in index_details['build_configs']:
                build_name = build_config['name']
                build_params_dict = build_config.get('params') or {}

                print(f"\n-- Processing build: '{build_name}' for index: '{index_name}' --", flush=True)

                if index_name == 'fastplaid':
                    index_class = FastPlaidWrapper
                else:
                    index_class_name = f"Index{method_info['class']}IP"
                    index_class = getattr(mvsic, index_class_name)

                # results_dir: CSVs, build_stats.json, index_params.json (small, synced)
                # index_store_dir: index .bin files (large, not synced)
                results_dir = os.path.join(base_results_dir, index_name, build_name)
                index_store_dir = os.path.join(base_index_dir, index_name, build_name)
                os.makedirs(results_dir, exist_ok=True)
                os.makedirs(index_store_dir, exist_ok=True)

                # Use a hash of fully materialized IndexParams (includes defaults) for stable naming.
                # fastplaid uses its own wrapper params.
                index_params_dict = None
                if index_name == "fastplaid":
                    params_hash = get_params_hash(build_params_dict)
                else:
                    build_params_func = getattr(mvsic.IndexParams, index_name)
                    build_params = build_params_func(**build_params_dict)
                    index_params_dict = index_params_to_dict(build_params)
                    params_hash = get_params_hash(index_params_dict)
                    index_params_path = os.path.join(results_dir, "index_params.json")
                    with open(index_params_path, "w") as f:
                        json.dump(index_params_dict, f, indent=2, sort_keys=True)

                index_filename = f"index_{params_hash}.bin"
                index_path = os.path.join(index_store_dir, index_filename)

                # --- Build Task ---
                if 'build' in tasks:
                    if os.path.exists(index_path) and not build_config.get('rebuild', False):
                        print(f"  Index found at {index_path}. Skipping build.", flush=True)
                    else:
                        print(f"  Building index...", flush=True)
                        if points is None:
                            if index_name == 'fastplaid':
                                points_path = os.path.join(dataset_config['path'], f"{dataset_config['name']}_points.pcs")
                                points = load_point_clouds(points_path)
                            else:
                                points, _, _ = load_dataset(dataset_config['path'], dataset_config['name'])

                        dim = points[0].shape[1] if index_name == 'fastplaid' else points[0].get_dims()

                        if index_name == 'fastplaid':
                            index = index_class(dim, build_params_dict, index_path=index_path)
                        else:
                            index = index_class(dim, build_params)

                        start_time = time.time()
                        index.build(points)
                        build_time = time.time() - start_time

                        print(f"  Saving index to {index_path}", flush=True)
                        index.save(index_path)

                        try:
                            if os.path.isdir(index_path):
                                index_size_bytes = sum(
                                    os.path.getsize(os.path.join(dirpath, f)) for dirpath, _, filenames in os.walk(index_path) for f in filenames
                                )
                            else:
                                index_size_bytes = os.path.getsize(index_path)

                            git_sha, git_dirty = _get_git_info(os.getcwd())
                            build_stats = {
                                'built_at': time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime()),
                                'git_sha': git_sha,
                                'git_dirty': git_dirty,
                                'build_time_sec': build_time,
                                'index_size_mb': index_size_bytes / (1024 * 1024),
                                'build_params': build_params_dict,
                            }
                            if index_name == 'mvivf':
                                build_stats['kmeans_tree_height'] = index.get_height()
                                build_stats.update(mvsic.get_mvivf_tree_stats(index))
                            stats_path = os.path.join(results_dir, 'build_stats.json')
                            with open(stats_path, 'w') as f:
                                json.dump(build_stats, f, indent=2)
                            print(
                                f"  Build time: {build_time:.2f}s, Index size: {build_stats['index_size_mb']:.2f}MB",
                                flush=True,
                            )
                        except OSError as e:
                            print(f"  Warning: Could not get index size. {e}", flush=True)

                # --- Search Task ---
                if 'search' in tasks:
                    if not os.path.exists(index_path):
                        print(
                            f"  Index file not found: {index_path}. Cannot run search. Please run the 'build' task first.",
                            flush=True,
                        )
                        continue

                    print("  Loading index for search...", flush=True)
                    # Ensure all data is loaded for search task if not already present
                    if points is None or queries is None or gt is None:
                        if index_name == 'fastplaid':
                            points_path = os.path.join(dataset_config['path'], f"{dataset_config['name']}_points.pcs")
                            queries_path = os.path.join(dataset_config['path'], f"{dataset_config['name']}_queries.pcs")
                            gt_path = os.path.join(dataset_config['path'], f"{dataset_config['name']}_chamfer_neighbors.gt")
                            if points is None:
                                points = load_point_clouds(points_path)
                            if queries is None:
                                queries = load_point_clouds(queries_path)
                            if gt is None:
                                gt = mvsic.ReadGT(gt_path, len(queries))
                        else:
                            points, queries, gt = load_dataset(dataset_config['path'], dataset_config['name'])

                    dim = queries[0].shape[1] if index_name == 'fastplaid' else queries[0].get_dims()

                    if index_name == 'fastplaid':
                        index = index_class(dim, build_params_dict, index_path=index_path)
                    else:
                        build_params_func = getattr(mvsic.IndexParams, index_name)
                        build_params = build_params_func(**build_params_dict)
                        index = index_class(dim, build_params)

                    index.load(index_path, points)

                    for search_config in build_config.get('search_configs', []):
                        search_name = search_config['name']
                        print(f"    Running search experiment: '{search_name}'", flush=True)

                        search_param_combos = generate_search_params(search_config, method_info)

                        variants = {}
                        for p in search_param_combos:
                            variant_name = p.pop('_variant_name')
                            if variant_name not in variants:
                                variants[variant_name] = []
                            variants[variant_name].append(p)

                        for variant_name, params_list in variants.items():
                            print(
                                f"      Variant: '{variant_name}', {len(params_list)} combinations",
                                flush=True,
                            )

                            search_results_dir = os.path.join(results_dir, search_name)
                            os.makedirs(search_results_dir, exist_ok=True)
                            if num_threads:
                                results_filename = f"latency_{variant_name}_results.csv" if variant_name else "latency_results.csv"
                            else:
                                results_filename = f"{variant_name}_results.csv" if variant_name else "results.csv"
                            results_path = os.path.join(search_results_dir, results_filename)

                            append_results = search_config.get('append', True)
                            if os.path.exists(results_path) and not append_results:
                                print(
                                    f"      'append' is false. Deleting existing results at {results_path}",
                                    flush=True,
                                )
                                os.remove(results_path)

                            # --- Feature 2: Skip Existing Results ---
                            existing_results_df = None
                            if os.path.exists(results_path) and append_results:
                                try:
                                    existing_results_df = pd.read_csv(results_path)
                                except pd.errors.EmptyDataError:
                                    print(f"      Warning: Existing results file is empty: {results_path}", flush=True)
                                    existing_results_df = None  # Treat as if it doesn't exist

                            # Sort params for efficient sweep
                            variable_param_name = method_info.get('variable_param')
                            if variable_param_name:
                                params_list.sort(key=lambda p: p.get(variable_param_name, 0))

                            all_results_for_variant = []

                            for params in params_list:
                                skip_computation = False
                                existing_recall_k_k = None

                                # Check if this parameter combination already exists
                                if existing_results_df is not None:
                                    # Build a boolean mask for matching parameters
                                    mask = pd.Series([True] * len(existing_results_df))
                                    for key, val in params.items():
                                        if key in existing_results_df.columns:
                                            # Use .isin for list-like columns, otherwise direct comparison
                                            if isinstance(val, list):
                                                mask &= existing_results_df[key].isin(val)
                                            else:
                                                mask &= existing_results_df[key] == val
                                    if mask.any():
                                        print(f"      Skipping existing params: {params}", flush=True)
                                        skip_computation = True
                                        existing_recall_k_k = existing_results_df[mask].iloc[0]['recall_k_k']

                                if skip_computation:
                                    class MockResult:
                                        def __init__(self, recall_k_k):
                                            self.recall_k_k = recall_k_k
                                    current_result = MockResult(existing_recall_k_k)
                                else:
                                    if index_name == 'fastplaid':
                                        with suppress_stdout_stderr():
                                            result = index.compute_stats_extended(queries, gt, params)
                                        current_result = result
                                    else:
                                        search_params_func = getattr(mvsic.SearchParams, index_name)
                                        search_params_obj = search_params_func(**params)
                                        with suppress_stdout_stderr():
                                            if num_threads:
                                                result = mvsic.compute_stats_extended_p_threaded(
                                                    index, points, queries, gt, [search_params_obj], num_threads
                                                )
                                            else:
                                                result = mvsic.compute_stats_extended(index, points, queries, gt, [search_params_obj])
                                            result_ = StatsExtended(
                                                k=params['k'],
                                                recall_1_k=min(1.0, result[0].recall_1_k),
                                                recall_k_k=min(1.0, result[0].recall_k_k),
                                                QPS_seq=result[0].QPS_seq,
                                                QPS_par=result[0].QPS_par,
                                                avg_cmps=result[0].avg_cmps,
                                                avg_timings=result[0].avg_timings,
                                            )
                                        current_result = result_

                                    print(
                                        "QPS:",
                                        current_result.QPS_seq,
                                        "QPS_par:",
                                        current_result.QPS_par,
                                        f"Recall 1@{params['k']}:",
                                        current_result.recall_1_k,
                                        f"Recall {params['k']}@{params['k']}",
                                        current_result.recall_k_k,
                                        flush=True,
                                    )

                                    # Update CSV row by row and keep sorted
                                    df = pd.DataFrame([current_result])
                                    params_df = pd.DataFrame([params])
                                    single_df = pd.concat([df, params_df], axis=1)
                                    single_df = single_df.loc[:, ~single_df.columns.duplicated()]
                                    single_df = _expand_method_timings(single_df, method_info)
                                    single_df = _reorder_result_columns(single_df, method_info)
                                    single_df = single_df.map(_format_scalar_for_csv)

                                    if os.path.exists(results_path):
                                        try:
                                            existing_df = pd.read_csv(results_path)
                                            updated_df = pd.concat([existing_df, single_df], ignore_index=True)
                                        except pd.errors.EmptyDataError:
                                            updated_df = single_df
                                    else:
                                        updated_df = single_df

                                    if variable_param_name and variable_param_name in updated_df.columns:
                                        updated_df = updated_df.sort_values(by=variable_param_name)

                                    print(f"      Saving updated results to {results_path}", flush=True)
                                    updated_df.to_csv(results_path, index=False)

                                all_results_for_variant.append(current_result)

                                # Check for early exit
                                if current_result.recall_k_k >= 1.0:
                                    print(
                                        f"      Recall@k reached 1.0. Stopping sweep for this variant.",
                                        flush=True,
                                    )
                                    break
                                elif (
                                    len(all_results_for_variant) > 3
                                    and current_result.recall_k_k == all_results_for_variant[-2].recall_k_k
                                    and current_result.recall_k_k == all_results_for_variant[-3].recall_k_k
                                ):
                                    print(
                                        f"      Recall@k did not improve. Stopping sweep for this variant.",
                                        flush=True,
                                    )
                                    break
                                elif (
                                    len(all_results_for_variant) > 1
                                    and current_result.recall_k_k < all_results_for_variant[-2].recall_k_k
                                ):
                                    print(
                                        f"      Recall@k dropped. Stopping sweep for this variant.",
                                        flush=True,
                                    )
                                    break


def main():
    print("Starting benchmark script...", flush=True)
    parser = argparse.ArgumentParser(description="MVSIC Benchmarking Framework")
    parser.add_argument(
        "--task",
        type=str,
        choices=["build", "search", "latency"],
        help=(
            "The task to perform. If not specified, runs build and search. "
            "'latency' runs the search flow with single-threaded per-query timing "
            "(compute_stats_extended_p_threaded, num_threads=1) and writes results "
            "with a 'latency_' CSV prefix."
        ),
    )

    # Args for build/search
    parser.add_argument("--config", type=str, help="Path to the experiment config file (for build/search).")
    parser.add_argument(
        "--methods",
        type=str,
        default="benchmarks/methods.yaml",
        help="Path to the methods schema file.",
    )
    parser.add_argument("--num_threads", type=int, help="Number of threads for parallel search.")

    args = parser.parse_args()

    # Handle build and/or search tasks
    if not args.config:
        parser.error("--config is required for build or search tasks.")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.methods, 'r') as f:
        methods = yaml.safe_load(f)
    experiment_name = os.path.splitext(os.path.basename(args.config))[0]

    tasks_to_run = []
    num_threads = args.num_threads
    if args.task is None:
        tasks_to_run = ["build", "search"]
    elif args.task == "latency":
        # Latency is the search flow forced to single-threaded per-query timing.
        tasks_to_run = ["search"]
        if num_threads is None:
            num_threads = 1
    else:
        tasks_to_run = [args.task]

    if tasks_to_run:
        run(config, methods, experiment_name, tasks_to_run, num_threads)


if __name__ == "__main__":
    main()
