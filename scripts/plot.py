import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse


def plot_dataset(dataset_name):
    results_dir = f'/ssd2/kishen/MVC/grid_search/{dataset_name}/results/'
    csv_files = glob.glob(os.path.join(results_dir, '*.csv'))

    all_dfs = []
    if csv_files:
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            filename = os.path.basename(csv_file)
            parts = filename.split('_')
            # Assuming filename format is like results_mls500_kpl0_pq_nb16_nc256.csv
            label = f"mvivf_pq_{parts[-1].replace('.csv', '')}_{parts[-2]}"
            df['label'] = label
            all_dfs.append(df)
    else:
        print(f"No PQ CSV files found for dataset {dataset_name}")

    non_pq_file = f'/ssd2/kishen/MVC/{dataset_name}/stats/mvivf_maxsize500_iters5_k=10.csv'
    if os.path.exists(non_pq_file):
        print(f"Found non-PQ file for {dataset_name}")
        non_pq_df = pd.read_csv(non_pq_file)
        non_pq_df.rename(
            columns={
                'Recall 1@k': 'recall_1',
                'Recall k@k': 'recall_k',
                'QPS_seq': 'QPS',
                'Avg Cmps': 'Avg_cmps',
            },
            inplace=True,
        )
        non_pq_df['Avg_cmps'] *= 128
        non_pq_df['label'] = 'mvivf'
        if 'num_rerank' not in non_pq_df.columns:
            non_pq_df['num_rerank'] = 80  # so it does not get filtered out
        all_dfs.append(non_pq_df)

    if not all_dfs:
        print("No data found in CSV files.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)

    k_values = full_df['k'].unique()

    for k_val in k_values:
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        fig.suptitle(f'QPS vs Recall for k={k_val} on {dataset_name}')

        df_k = full_df[full_df['k'] == k_val].copy()

        # Apply filtering based on k_val
        if k_val == 10:
            df_k = df_k[df_k['num_rerank'] == 80]
        elif k_val == 100:
            # Assuming user meant k=100 for num_rerank=800 from the provided data
            df_k = df_k[df_k['num_rerank'] == 800]

        if df_k.empty:
            print(f"No data found for k={k_val} after filtering. Skipping plot.")
            continue

        # one plot for each csv file
        for label, group in df_k.groupby('label'):
            # Sort by recall before plotting
            group_sorted_1 = group.sort_values(by='recall_1')
            group_sorted_k = group.sort_values(by='recall_k')

            # Plot for recall_1
            axes[0].plot(group_sorted_1['recall_1'], group_sorted_1['QPS'], marker='o', label=label)

            # Plot for recall_k
            axes[1].plot(group_sorted_k['recall_k'], group_sorted_k['QPS'], marker='o', label=label)

            axes[2].plot(
                group_sorted_1['recall_1'], group_sorted_1['Avg_cmps'], marker='o', label=label
            )

        axes[0].set_xlabel(f'recall 1@{k_val}')
        axes[0].set_ylabel('QPS')
        axes[0].set_title(f'QPS vs recall 1@{k_val}')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].set_xlabel(f'recall {k_val}@{k_val}')
        axes[1].set_ylabel('QPS')
        axes[1].set_title(f'QPS vs recall {k_val}@{k_val}')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].set_xlabel(f'recall 1@{k_val}')
        axes[2].set_ylabel('Avg Cmps')
        axes[2].set_title(f'Avg Cmps vs recall 1@{k_val}')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'/home/kishen/MVSIC/scripts/{dataset_name}_k{k_val}.png')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['arguana', 'scidocs', 'nq'], help='Dataset to plot')
    args = parser.parse_args()
    plot_dataset(args.dataset)
