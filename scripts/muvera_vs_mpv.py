import argparse
import csv
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_data(directory_path, k_val):
    """
    Parses CSV files for a specific k and returns a pandas DataFrame.
    """
    all_data = []
    for filename in os.listdir(directory_path):
        if not filename.endswith(f"_k={k_val}.csv"):
            continue

        parts = filename.split('_')
        algorithm = parts[0]
        if algorithm not in ["muvera", "mpv"]:
            continue

        params = {
            'algorithm': algorithm,
            'k': k_val,
            'norm': 'norm' in parts,
            'norerank': 'norerank' in parts,
            'normq': 'normq' in parts,
            'fde': None,
        }

        if algorithm == 'muvera':
            for part in parts:
                if part.startswith('fde'):
                    params['fde'] = int(part[3:])

        file_path = os.path.join(directory_path, filename)
        try:
            with open(file_path, mode='r', newline='') as infile:
                reader = csv.reader(infile)
                next(reader, None)
                for row in reader:
                    try:
                        if len(row) == 7:
                            data_row = params.copy()
                            data_row.update(
                                {
                                    "L": int(row[1]),
                                    "QPS": float(row[2]),
                                    "Avg_Cmps": float(row[4]),
                                    "Recall_1@k": float(row[5]),
                                    "Recall_k@k": float(row[6]),
                                }
                            )
                            all_data.append(data_row)
                    except ValueError:
                        continue
        except (IOError, StopIteration) as e:
            print(f"    ! Error processing file {filename}: {e}")

    return pd.DataFrame(all_data)


def generate_plots(df, dataset_name, k):
    """
    Generates and saves a 1x2 grid of plots for QPS vs. Recall.
    """
    if df.empty:
        print(f"No data to plot for {dataset_name} with k={k}.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(24, 9))
    title_suffix = f"(k={k}, no-rerank)"
    fig.suptitle(f'{dataset_name.capitalize()} Performance {title_suffix}', fontsize=20)

    plot_configs = {
        0: ('Recall_1@k', 'QPS', f'QPS vs. Recall 1@{k}'),
        1: ('Recall_k@k', 'QPS', f'QPS vs. Recall {k}@{k}'),
    }

    df['fde'] = df['fde'].fillna(-1).astype(int)
    grouped = df.groupby(['algorithm', 'norm', 'fde'])

    for col, (x_metric, y_metric, title) in plot_configs.items():
        ax = axs[col]
        ax.set_title(title)
        ax.set_xlabel(x_metric.replace('_', ' '))
        ax.set_ylabel(y_metric.replace('_', ' '))
        ax.grid(True, which="both", ls="--")
        ax.set_yscale('log')

        for name, group in grouped:
            algo, norm, fde = name
            label = f"{algo}"
            if norm:
                label += "-norm"
            if fde != -1:
                label += f"-fde{fde}"

            sorted_group = group.sort_values(by=[x_metric, 'L'])

            ax.plot(
                sorted_group[x_metric].to_numpy(),
                sorted_group[y_metric].to_numpy(),
                marker='o' if algo == 'muvera' else 's',
                linestyle='-',
                label=label,
            )

        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_filename = f"/home/kishen/MVC/plots/{dataset_name}_muvera_vs_mpv_k{k}.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.close()


def main():
    """
    Main function to parse command-line arguments and run the processing.
    """
    parser = argparse.ArgumentParser(description="Process and plot benchmark CSV data.")
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the dataset (e.g., 'arguana').",
    )
    args = parser.parse_args()

    data_path = f"/ssd2/kishen/MVC/{args.dataset_name}/stats"

    if not os.path.exists(data_path):
        print(f"Error: Dataset directory '{data_path}' does not exist.")
        return

    for k_val in [10, 100]:
        print(f"\nProcessing data for k={k_val}...")
        df = parse_data(data_path, k_val)

        df_norerank = df[df['norerank'] == True].copy()
        df_filtered = df_norerank[df_norerank['normq'] == False].copy()

        if df_filtered.empty:
            print(f"No data found for k={k_val} after filtering.")
            continue

        df_filtered['config'] = df_filtered.apply(
            lambda row: f"{row['algorithm']}{'-norm' if row['norm'] else ''}{'-fde' + str(row['fde']) if not pd.isna(row['fde']) and row['fde'] != -1 else ''}",
            axis=1,
        )

        print(
            f"\n--- Summary Table for {args.dataset_name.capitalize()} (k={k_val}, no-rerank, no-normq) ---"
        )
        table = df_filtered.pivot_table(
            index='L', columns='config', values=['QPS', 'Avg_Cmps', 'Recall_k@k']
        )
        print(table.to_markdown())

        generate_plots(df_filtered, args.dataset_name, k_val)


if __name__ == "__main__":
    main()
