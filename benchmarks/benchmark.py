import argparse
import yaml
import os
import hashlib
import json
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import signal
import sys

from framework_utils import *
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


from framework_utils import FastPlaidWrapper


# def compute_recall(gt, result_neighbors):
#     """
#     Computes recall for a single run's results against the ground truth.
#     - gt: An iterable MVSIC ground truth object.
#     - result_neighbors: A list (one item per query) of lists of (id, score) tuples.
#     """
#     total_recall = 0
#     num_queries = len(result_neighbors)
#     if num_queries == 0:
#         return 0.0

#     for i, true_neighbor_tuples in enumerate(gt):
#         true_neighbor_ids = {n[1] for n in true_neighbor_tuples}

#         # Handle cases where a query might have no ground truth neighbors
#         if not true_neighbor_ids:
#             num_queries -= 1
#             continue

#         predicted_neighbor_ids = {n[0] for n in result_neighbors[i]}

#         recall = len(true_neighbor_ids.intersection(predicted_neighbor_ids)) / len(
#             true_neighbor_ids
#         )
#         total_recall += recall

#     return total_recall / num_queries if num_queries > 0 else 0.0


from utils import load_point_clouds
import time


def run(config, methods, experiment_name, tasks):
    """
    Builds indices and runs searches based on the experiment config and specified tasks.
    """
    print(f"--- Running experiment: {experiment_name} for tasks: {tasks} ---", flush=True)

    dataset_config = config['dataset']
    print(f"Dataset: {dataset_config['name']}", flush=True)

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
                            points_path = os.path.join(
                                dataset_config['path'], f"{dataset_config['name']}_points.pcs"
                            )
                            points = load_point_clouds(points_path)
                        else:
                            points, _, _ = load_dataset(
                                dataset_config['path'], dataset_config['name']
                            )

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
                                os.path.getsize(os.path.join(dirpath, f))
                                for dirpath, _, filenames in os.walk(index_path)
                                for f in filenames
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
                        points_path = os.path.join(
                            dataset_config['path'], f"{dataset_config['name']}_points.pcs"
                        )
                        queries_path = os.path.join(
                            dataset_config['path'], f"{dataset_config['name']}_queries.pcs"
                        )
                        gt_path = os.path.join(
                            dataset_config['path'], f"{dataset_config['name']}_chamfer_neighbors.gt"
                        )
                        if points is None:
                            points = load_point_clouds(points_path)
                        if queries is None:
                            queries = load_point_clouds(queries_path)
                        if gt is None:
                            gt = mvsic.ReadGT(gt_path, len(queries))
                    else:
                        points, queries, gt = load_dataset(
                            dataset_config['path'], dataset_config['name']
                        )

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
                        results_filename = (
                            f"{variant_name}_results.csv" if variant_name else "results.csv"
                        )
                        results_path = os.path.join(results_dir, results_filename)

                        append_results = search_config.get('append', True)
                        if os.path.exists(results_path) and not append_results:
                            print(
                                f"      'append' is false. Deleting existing results at {results_path}",
                                flush=True,
                            )
                            os.remove(results_path)

                        # Sort params for efficient sweep
                        variable_param_name = method_info.get('variable_param')
                        if variable_param_name:
                            params_list.sort(key=lambda p: p.get(variable_param_name, 0))

                        all_results_for_variant = []
                        all_params_for_variant = []

                        for params in params_list:
                            if index_name == 'fastplaid':
                                with suppress_stdout_stderr():
                                    result = index.compute_stats_extended(queries, gt, params)
                                all_results_for_variant.append(result)
                                all_params_for_variant.append(params)
                            else:
                                search_params_func = getattr(mvsic.SearchParams, index_name)
                                search_params_obj = search_params_func(**params)
                                with suppress_stdout_stderr():
                                    result = mvsic.compute_stats_extended(
                                        index, points, queries, gt, [search_params_obj]
                                    )
                                all_results_for_variant.extend(result)
                                all_params_for_variant.append(params)

                            # Check for early exit
                            if (
                                all_results_for_variant
                                and all_results_for_variant[-1].recall_k_k >= 1.0
                            ):
                                print(
                                    f"      Recall@k reached 1.0. Stopping sweep for this variant.",
                                    flush=True,
                                )
                                break
                            print(
                                "QPS:",
                                all_results_for_variant[-1].QPS_seq,
                                "Recall 1@10:",
                                all_results_for_variant[-1].recall_1_k,
                                "Recall 10@10",
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
                        header = not os.path.exists(results_path)

                        print(f"      Saving results to {results_path}", flush=True)
                        full_df.to_csv(results_path, mode=mode, header=header, index=False)


import pathlib


def plot(input_dirs, output_dir):
    """


    Aggregates results from multiple experiment directories and generates plots and summaries.


    """

    print("--- Aggregating results and generating plots ---", flush=True)

    all_results = []

    print(f"Searching for results in: {input_dirs}", flush=True)

    for input_dir in input_dirs:

        for path in pathlib.Path(input_dir).rglob("*_results.csv"):

            try:

                df = pd.read_csv(path)

                # Enrich with data from path

                parts = path.parts

                # Expected structure: .../<dataset>/<index>/<build>/<search>/<variant>_results.csv

                df['dataset'] = parts[-5]

                df['index'] = parts[-4]

                df['build'] = parts[-3]

                df['search'] = parts[-2]

                df['variant'] = path.name.replace('_results.csv', '')

                all_results.append(df)

            except Exception as e:

                print(f"Warning: Could not read {path}. Error: {e}", flush=True)

    if not all_results:

        print("No result files found. Exiting plot task.", flush=True)

        return

    master_df = pd.concat(all_results, ignore_index=True)

    # --- Merge Build Stats ---

    all_build_stats = []

    for input_dir in input_dirs:

        for path in pathlib.Path(input_dir).rglob("build_stats.json"):

            try:

                with open(path, 'r') as f:

                    stats = json.load(f)

                parts = path.parts

                stats['dataset'] = parts[-4]

                stats['index'] = parts[-3]

                stats['build'] = parts[-2]

                all_build_stats.append(stats)

            except Exception as e:

                print(f"Warning: Could not read {path}. Error: {e}", flush=True)

    if all_build_stats:

        build_df = pd.DataFrame(all_build_stats)

        master_df = pd.merge(master_df, build_df, on=['dataset', 'index', 'build'], how='left')

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Generating Plots and Tables in {output_dir} ---", flush=True)

    # --- Generate Pareto Plots ---

    def get_pareto_frontier(df, x_col, y_col):

        df = df.sort_values(by=[x_col, y_col], ascending=[True, False]).copy()

        pareto_points = []

        max_y = -1

        for _, row in df.iterrows():

            if row[y_col] > max_y:

                pareto_points.append(row)

                max_y = row[y_col]

        return pd.DataFrame(pareto_points)

    grouped_by_k = master_df.groupby(['dataset', 'k'])

    for (dataset, k), group in grouped_by_k:

        plt.style.use('seaborn-v0_8-whitegrid')

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        fig.suptitle(f'Performance on {dataset} (k={k})', fontsize=16)

        # Plot 1: Latency vs. Recall

        ax1 = axes[0]

        ax1.set_title('Latency vs. Recall')

        ax1.set_xlabel(f'Recall@{k}')

        ax1.set_ylabel('Latency (ms/query)')

        ax1.set_xscale('linear')

        ax1.set_yscale('log')

        # Plot 2: Throughput vs. Recall

        ax2 = axes[1]

        ax2.set_title('Throughput vs. Recall')

        ax2.set_xlabel(f'Recall@{k}')

        ax2.set_ylabel('Throughput (QPS)')

        ax2.set_xscale('linear')

        ax2.set_yscale('log')

        for method, method_group in group.groupby('index'):

            pareto_df = get_pareto_frontier(method_group, f'recall_{k}_k', 'QPS_seq')

            if pareto_df.empty:
                continue

            latency_ms = 1000 / pareto_df['QPS_seq']

            ax1.plot(
                pareto_df[f'recall_{k}_k'], latency_ms, marker='o', linestyle='-', label=method
            )

            ax2.plot(
                pareto_df[f'recall_{k}_k'],
                pareto_df['QPS_par'],
                marker='o',
                linestyle='-',
                label=method,
            )

        ax1.legend()

        ax2.legend()

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        plot_path = os.path.join(output_dir, f'{dataset}_k={k}_performance.pdf')

        print(f"  Saving plot: {plot_path}", flush=True)

        fig.savefig(plot_path)

        plt.close(fig)

    # --- Generate Summary Tables ---

    summary_data = []

    # TODO: Implement interpolation to find QPS at fixed recall points (e.g., 0.90, 0.95)

    # For now, we find the configuration that gives the highest recall.

    for (dataset, index), group in master_df.groupby(['dataset', 'index']):

        best_run = (
            group.loc[group['recall_10_k'].idxmax()] if 'recall_10_k' in group else group.iloc[0]
        )

        summary_data.append(
            {
                'Dataset': dataset,
                'Method': index,
                'Best Recall@10': best_run.get('recall_10_k', 'N/A'),
                'QPS @ Best Recall': best_run.get('QPS_seq', 'N/A'),
                'Build Time (s)': best_run.get('build_time_sec', 'N/A'),
                'Index Size (MB)': best_run.get('index_size_mb', 'N/A'),
            }
        )

    if summary_data:

        summary_df = pd.DataFrame(summary_data)

        csv_path = os.path.join(output_dir, 'summary_best_recall.csv')

        md_path = os.path.join(output_dir, 'summary_best_recall.md')

        print(f"  Saving summary table: {csv_path}", flush=True)

        summary_df.to_csv(csv_path, index=False)

        print(f"  Saving summary table: {md_path}", flush=True)

        summary_df.to_markdown(md_path, index=False)


def main():
    print("Starting benchmark script...", flush=True)
    parser = argparse.ArgumentParser(description="MVSIC Benchmarking Framework")
    parser.add_argument(
        "--task",
        type=str,
        choices=["build", "search", "plot"],
        help="The task to perform. If not specified, runs build and search.",
    )

    # Args for build/search
    parser.add_argument(
        "--config", type=str, help="Path to the experiment config file (for build/search)."
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="benchmarks/methods.yaml",
        help="Path to the methods schema file.",
    )

    # Args for plot
    parser.add_argument(
        "--input-dirs",
        type=str,
        nargs='+',
        help="List of result directories to aggregate for plotting.",
    )
    parser.add_argument("--output-dir", type=str, help="Directory to save plots and summaries.")

    args = parser.parse_args()

    if args.task == "plot":
        if not args.input_dirs or not args.output_dir:
            parser.error("--input-dirs and --output-dir are required for the plot task.")
        plot(args.input_dirs, args.output_dir)
        return

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
    elif args.task != "plot":
        tasks_to_run = [args.task]

    if tasks_to_run:
        run(config, methods, experiment_name, tasks_to_run)


if __name__ == "__main__":
    main()
