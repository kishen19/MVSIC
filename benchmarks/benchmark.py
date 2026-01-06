import argparse
import yaml
import os
import hashlib
import json
from itertools import product
import pandas as pd
import signal
import sys
import time

from framework_utils import *
from utils import load_point_clouds
import mvsic


# Graceful exit on Ctrl+C
def signal_handler(sig, frame):
    print('\nCtrl+C detected. Exiting gracefully.')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


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


def run(config, methods, experiment_name, tasks):
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

        base_results_dir = dataset_config.get('results', os.path.join('results', experiment_name))

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

                params_hash = get_params_hash(build_params_dict)
                index_filename = f"index_{params_hash}.bin"
                index_dir = os.path.join(base_results_dir, index_name, build_name)
                index_path = os.path.join(index_dir, index_filename)
                os.makedirs(index_dir, exist_ok=True)

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
                            build_params_func = getattr(mvsic.IndexParams, index_name)
                            build_params = build_params_func(**build_params_dict)
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

                            build_stats = {
                                'build_time_sec': build_time,
                                'index_size_mb': index_size_bytes / (1024 * 1024),
                                'build_params': build_params_dict,
                            }
                            stats_path = os.path.join(index_dir, 'build_stats.json')
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

                            results_dir = os.path.join(index_dir, search_name)
                            os.makedirs(results_dir, exist_ok=True)
                            results_filename = f"{variant_name}_results.csv" if variant_name else "results.csv"
                            results_path = os.path.join(results_dir, results_filename)

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
                            all_params_for_variant = []

                            for params in params_list:
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
                                        continue

                                if index_name == 'fastplaid':
                                    with suppress_stdout_stderr():
                                        result = index.compute_stats_extended(queries, gt, params)
                                    all_results_for_variant.append(result)
                                    all_params_for_variant.append(params)
                                else:
                                    search_params_func = getattr(mvsic.SearchParams, index_name)
                                    search_params_obj = search_params_func(**params)
                                    with suppress_stdout_stderr():
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
                                    all_results_for_variant.append(result_)
                                    all_params_for_variant.append(params)

                                # Check for early exit
                                if all_results_for_variant and all_results_for_variant[-1].recall_k_k >= 1.0:
                                    print(
                                        f"      Recall@k reached 1.0. Stopping sweep for this variant.",
                                        flush=True,
                                    )
                                    break
                                elif (
                                    all_results_for_variant
                                    and len(all_results_for_variant) > 3
                                    and all_results_for_variant[-1].recall_k_k == all_results_for_variant[-2].recall_k_k
                                    and all_results_for_variant[-1].recall_k_k == all_results_for_variant[-3].recall_k_k
                                ):
                                    print(
                                        f"      Recall@k did not improve. Stopping sweep for this variant.",
                                        flush=True,
                                    )
                                    break
                                print(
                                    "QPS:",
                                    all_results_for_variant[-1].QPS_seq,
                                    "QPS_par:",
                                    all_results_for_variant[-1].QPS_par,
                                    f"Recall 1@{params['k']}:",
                                    all_results_for_variant[-1].recall_1_k,
                                    f"Recall {params['k']}@{params['k']}",
                                    all_results_for_variant[-1].recall_k_k,
                                    flush=True,
                                )

                            if not all_results_for_variant:
                                continue

                            df = pd.DataFrame(all_results_for_variant)
                            params_df = pd.DataFrame(all_params_for_variant)
                            # Reset index to ensure correct alignment
                            params_df.reset_index(drop=True, inplace=True)
                            df.reset_index(drop=True, inplace=True)
                            full_df = pd.concat([df, params_df], axis=1)

                            mode = 'a' if os.path.exists(results_path) else 'w'
                            header = not (os.path.exists(results_path) and os.path.getsize(results_path) > 0)

                            print(f"      Saving results to {results_path}", flush=True)
                            full_df.to_csv(results_path, mode=mode, header=header, index=False)


def main():
    print("Starting benchmark script...", flush=True)
    parser = argparse.ArgumentParser(description="MVSIC Benchmarking Framework")
    parser.add_argument(
        "--task",
        type=str,
        choices=["build", "search"],
        help="The task to perform. If not specified, runs build and search.",
    )

    # Args for build/search
    parser.add_argument("--config", type=str, help="Path to the experiment config file (for build/search).")
    parser.add_argument(
        "--methods",
        type=str,
        default="benchmarks/methods.yaml",
        help="Path to the methods schema file.",
    )

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
    if args.task is None:
        tasks_to_run = ["build", "search"]
    else:
        tasks_to_run = [args.task]

    if tasks_to_run:
        run(config, methods, experiment_name, tasks_to_run)


if __name__ == "__main__":
    main()
