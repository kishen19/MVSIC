import argparse
import csv
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

# ======================================================================================
# ======================= ALGORITHM DEFINITIONS (Edit Rarely) ==========================
# ======================================================================================
# Define the properties of each algorithm, like how to parse its files and how to style its plots.
ALGORITHM_DEFINITIONS = {
    "mvivf": {
        "regex": r"mvivf_maxsize\d+_k=(?P<k>\d+)\.csv",
        "style": {"marker": "o", "linestyle": "-"},
    },
    "svh": {
        "regex": r"svh_maxsize\d+_k=(?P<k>\d+)_cand=(?P<cand>\d+)\.csv",
        "style": {"marker": "s", "linestyle": "--"},
    },
    "vamana": {
        "regex": r"vamana_R(?P<R>\d+)_k=(?P<k>\d+)\.csv",
        "style": {"marker": "^", "linestyle": "dashdot"},
    },
    "muvera": {
        "regex": r"muvera_norm_fde(?P<fde>\d+)_k=(?P<k>\d+)\.csv",
        "style": {"marker": "*", "linestyle": ":"},
    },
    "mpv": {
        "regex": r"mpv_norm_k=(?P<k>\d+)\.csv",
        "style": {"marker": "P", "linestyle": "-"},
    },
    # To add a new algorithm, define its properties here.
}

# ======================================================================================
# =================== PLOT CONFIGURATION (Editable Section) ============================
# ======================================================================================
# Specify which algorithm configurations to include in the plots.
# This is a list of tuples: (algorithm_name, parameter_filters)
# - algorithm_name: Must be a key from ALGORITHM_DEFINITIONS.
# - parameter_filters: A dictionary mapping a parameter name to a list of values
#                      to plot. An empty dictionary {} means plot all variations.
#                      An empty list for a parameter, e.g. {"R": []}, also plots all values.
PLOTS_TO_GENERATE = [
    ("mvivf", {}),  # Plot all mvivf results
    ("svh", {"cand": [8, 16, 32]}),  # Plot svh, but only for cand=4, 8, 16
    ("muvera", {"fde": [2560, 5120, 10240]}),
    ("vamana", {"R": [256]}),  # Plot all R values for vamana
    ("mpv", {}),
]
# ======================================================================================


def parse_data(directory_path, definitions):
    """
    Scans a directory and parses all relevant CSV files based on the provided definitions.
    Returns a single pandas DataFrame with all collected data.
    """
    all_data = []
    print(f"Scanning for data in: {directory_path}")

    for algo_name, definition in definitions.items():
        pattern = re.compile(definition['regex'])
        for filename in os.listdir(directory_path):
            match = pattern.match(filename)
            if not match:
                continue

            params = {key: int(val) for key, val in match.groupdict().items()}
            params['algorithm'] = algo_name

            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, mode='r', newline='') as infile:
                    reader = csv.reader(infile)
                    next(reader, None)  # Skip header
                    for row in reader:
                        try:
                            if len(row) == 7:
                                data_row = params.copy()
                                data_row.update(
                                    {
                                        "L_or_nprobes": int(row[1]),
                                        "QPS": float(row[2]),
                                        "Avg_Cmps": float(row[4]),
                                        "Recall_1@k": float(row[5]),
                                        "Recall_k@k": float(row[6]),
                                    }
                                )
                                all_data.append(data_row)
                        except (ValueError, IndexError):
                            continue
            except (IOError, StopIteration) as e:
                print(f"    ! Error processing file {filename}: {e}")

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    # Add placeholder for algorithms without certain params to keep dtype int
    for param in ['fde', 'R', 'cand', 'iters']:
        if param not in df.columns:
            df[param] = -1
    df.fillna(-1, inplace=True)
    for param in ['fde', 'R', 'cand', 'iters']:
        if param in df.columns:
            df[param] = df[param].astype(int)

    return df


def generate_plots(df, definitions, plot_selections, dataset_name, k_val):
    """
    Generates and saves a 2x2 grid of plots from the collected data.
    """
    df_k = df[df['k'] == k_val].copy()
    if df_k.empty:
        print(f"No data found for k={k_val} to plot.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(24, 18))
    fig.suptitle(f'{dataset_name.capitalize()} Performance (k={k_val})', fontsize=20)

    plot_configs = {
        (0, 0): ('Recall_1@k', 'Avg_Cmps', True, f'Avg Cmps vs. Recall 1@{k_val}'),
        (0, 1): ('Recall_k@k', 'Avg_Cmps', True, f'Avg Cmps vs. Recall {k_val}@{k_val}'),
        (1, 0): ('Recall_1@k', 'QPS', True, f'QPS vs. Recall 1@{k_val}'),
        (1, 1): ('Recall_k@k', 'QPS', True, f'QPS vs. Recall {k_val}@{k_val}'),
    }

    # First loop: Plot all the data
    for algo_name, param_filters in plot_selections:
        df_algo = df_k[df_k['algorithm'] == algo_name]
        if df_algo.empty:
            continue

        definition = definitions[algo_name]
        df_filtered = df_algo.copy()
        if param_filters:
            for param_name, values in param_filters.items():
                if values:
                    df_filtered = df_filtered[df_filtered[param_name].isin(values)]

        if df_filtered.empty:
            continue
        metric_cols = ['k', 'L_or_nprobes', 'QPS', 'Avg_Cmps', 'Recall_1@k', 'Recall_k@k']
        grouping_cols = [col for col in df_filtered.columns if col not in metric_cols]
        grouped = df_filtered.groupby(grouping_cols, dropna=False)

        for name, group in grouped:
            params_dict = dict(zip(grouping_cols, name if isinstance(name, tuple) else [name]))
            algo = params_dict.pop('algorithm')

            label_params = {k: v for k, v in params_dict.items() if v != -1}
            label = f"{algo}_" + "_".join([f"{k}={v}" for k, v in label_params.items()])

            style = definition.get('style', {})

            for (r, c), (x_metric, y_metric, _, _) in plot_configs.items():
                ax = axs[r, c]
                sorted_group = group.sort_values(by=[x_metric, 'L_or_nprobes'])

                # Truncate data at recall = 1.0
                first_one_recall_idx = sorted_group[sorted_group[x_metric] >= 1.0].index.min()
                if pd.notna(first_one_recall_idx):
                    plot_group = sorted_group.loc[:first_one_recall_idx]
                else:
                    plot_group = sorted_group

                ax.plot(
                    plot_group[x_metric].to_numpy(),
                    plot_group[y_metric].to_numpy(),
                    label=label,
                    **style,
                )

    # Set styles in a separate loop to ensure all legends are populated first
    for (r, c), (x_metric, y_metric, y_log, title) in plot_configs.items():
        ax = axs[r, c]
        ax.set_title(title)
        ax.set_xlabel(x_metric.replace('_', ' '))
        ax.set_ylabel(y_metric.replace('_', ' '))
        ax.grid(True, which="both", ls="--")
        if y_log:
            ax.set_yscale('log')
        ax.legend(fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_filename = f"/home/kishen/MVC/plots/{dataset_name}_QPS_AvgCmps_vs_Recall_k{k_val}.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.close()


def main():
    """
    Main function to parse command-line arguments and run the script.
    """
    parser = argparse.ArgumentParser(
        description="Process benchmark CSV data files and generate plots.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the dataset (e.g., 'sift1m').\nThis script will look for data in '/ssd2/kishen/MVC/{dataset_name}/stats'.",
    )
    args = parser.parse_args()

    data_path = f"/ssd2/kishen/MVC/{args.dataset_name}/stats"
    if not os.path.exists(data_path):
        print(f"Error: Dataset directory '{data_path}' does not exist.")
        return

    full_df = parse_data(data_path, ALGORITHM_DEFINITIONS)
    if full_df.empty:
        print("No data was parsed. Exiting.")
        return

    k_values = sorted(full_df['k'].unique())
    print(f"\nFound data for k values: {k_values}")
    for k in k_values:
        print(f"--- Generating plots for k={k} ---")
        generate_plots(full_df, ALGORITHM_DEFINITIONS, PLOTS_TO_GENERATE, args.dataset_name, k)


if __name__ == "__main__":
    main()
