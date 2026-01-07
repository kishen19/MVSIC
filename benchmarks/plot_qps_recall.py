import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


def generate_plot_for_dataset(dataset_config, k, results_to_plot, experiment_name, affix=""):
    """
    Generates a QPS vs. Recall plot for a single dataset.
    """
    dataset_name = dataset_config['name']
    base_results_dir = dataset_config['results']

    print(f"--- Generating plots for dataset: {dataset_name} (k={k}) ---")

    # --- Setup Plots ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(f'QPS vs. Recall for {dataset_name} (k={k})', fontsize=16)

    # Define the four plots to generate
    # Use literal column names 'recall_1_k' and 'recall_k_k' as found in the CSV
    plot_configs = [
        {'ax': axes[0, 0], 'x': 'recall_1_k', 'y': 'QPS_seq', 'title': f'QPS_seq vs. Recall 1@{k}'},
        {
            'ax': axes[0, 1],
            'x': 'recall_k_k',
            'y': 'QPS_seq',
            'title': f'QPS_seq vs. Recall {k}@{k}',
        },
        {'ax': axes[1, 0], 'x': 'recall_1_k', 'y': 'QPS_par', 'title': f'QPS_par vs. Recall 1@{k}'},
        {
            'ax': axes[1, 1],
            'x': 'recall_k_k',
            'y': 'QPS_par',
            'title': f'QPS_par vs. Recall {k}@{k}',
        },
    ]

    for p_config in plot_configs:
        p_config['ax'].set_xlabel('Recall')
        p_config['ax'].set_ylabel('QPS')
        p_config['ax'].set_title(p_config['title'])
        p_config['ax'].set_yscale('log')
        p_config['ax'].grid(True, which="both", ls="--")

    # --- Load and Plot Results ---
    all_data = []
    for result_info in results_to_plot:
        method, build_name, search_variant = result_info

        # Construct the specific path to the results file
        search_dir = os.path.join(base_results_dir, method, build_name, f"k={k}")

        if search_variant == "":
            result_filename = "results.csv"
        else:
            result_filename = f"{search_variant}_results.csv"

        result_path = os.path.join(search_dir, result_filename)

        if not os.path.exists(result_path):
            print(f"  Warning: Result file not found at '{result_path}'")
            continue

        print(f"  Loading results from: {result_path}")
        try:
            df = pd.read_csv(result_path)

            # Create a label for the plot legend
            label = f"{method}_{build_name}"
            if search_variant:
                label += f"_{search_variant}"

            # Sort by recall for a clean line plot
            sort_col = 'recall_k_k'
            if sort_col in df.columns:
                df = df.sort_values(by=sort_col).reset_index(drop=True)

            all_data.append({'df': df, 'label': label})

        except Exception as e:
            print(f"  Error reading {result_path}: {e}")

    # --- Plotting ---
    if not all_data:
        print("No data loaded. Exiting.")
        plt.close(fig)
        return

    for data in all_data:
        df = data['df']
        label = data['label']

        for p_config in plot_configs:
            ax = p_config['ax']
            x_col = p_config['x']
            y_col = p_config['y']

            if x_col in df.columns and y_col in df.columns:
                ax.plot(df[x_col], df[y_col], marker='o', linestyle='-', label=label)
            else:
                print(
                    f"  Warning: Columns '{x_col}' or '{y_col}' not found for '{label}'. Skipping plot."
                )

    for p_config in plot_configs:
        p_config['ax'].legend()

    # --- Save Plot ---
    affix_str = "" if len(affix) == 0 else f"{affix}_"
    output_dir = f"./results/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = (
        f"{output_dir}/{affix_str}{dataset_name}_k={k}_qps_vs_recall.pdf"
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    print(f"\nSaving plot to: {output_filename}")
    fig.savefig(output_filename)
    plt.close(fig)


def plot_qps_vs_recall(config_path, affix=""):
    """
    Generates QPS vs. Recall plots from benchmark results based on a specific pathing convention.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    experiment_name = config.get('name', 'plots')

    dataset_configs = config.get('datasets') or [config.get('dataset')]
    if not dataset_configs or dataset_configs == [None]:
        raise ValueError("No 'datasets' or 'dataset' key found in the config file.")

    k = config['k']
    results_to_plot = config['results']

    for dataset_config in dataset_configs:
        generate_plot_for_dataset(dataset_config, k, results_to_plot, experiment_name, affix)


def main():
    parser = argparse.ArgumentParser(description="Plot QPS vs. Recall from benchmark results.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file for plotting.",
    )
    parser.add_argument("--affix", type=str, default="", help="Name of Plot.")
    args = parser.parse_args()
    plot_qps_vs_recall(args.config, args.affix)


if __name__ == "__main__":
    main()
