import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_plot(collection, dataset, method, method_config):
    """
    Generates scatter plots for QPS vs recall.
    """
    base_path = f'./results/{collection}/{dataset}/{method}/{method_config}/k=10/'
    files = glob.glob(os.path.join(base_path, 'nr*_results.csv'))

    if not files:
        print(f"No files found at {base_path}")
        return

    all_data = []
    for f in files:
        try:
            # Extract num_rerank from filename
            match = re.search(r'nr(\d+)_results.csv', os.path.basename(f))
            if match:
                nr = int(match.group(1))
                df = pd.read_csv(f)
                df['num_rerank_file'] = nr
                all_data.append(df)
        except Exception as e:
            print(f"Error reading or processing {f}: {e}")

    if not all_data:
        print("No data to plot.")
        return

    full_df = pd.concat(all_data, ignore_index=True)

    # Use a color map and different markers
    num_nrs = full_df['num_rerank_file'].nunique()
    colors = plt.cm.viridis(np.linspace(0, 1, num_nrs))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot recall_1_k vs QPS_seq
    for i, (nr, group) in enumerate(full_df.groupby('num_rerank_file')):
        ax1.scatter(
            group['recall_1_k'],
            group['QPS_seq'],
            color=colors[i],
            marker=markers[i % len(markers)],
            label=f'nr={nr}',
        )

    ax1.set_xlabel('recall_1_k')
    ax1.set_ylabel('QPS_seq')
    ax1.set_title('Recall@1 vs. QPS_seq')
    ax1.legend()
    ax1.grid(True)

    # Plot recall_k_k vs QPS_seq
    for i, (nr, group) in enumerate(full_df.groupby('num_rerank_file')):
        ax2.scatter(
            group['recall_k_k'],
            group['QPS_seq'],
            color=colors[i],
            marker=markers[i % len(markers)],
            label=f'nr={nr}',
        )

    ax2.set_xlabel('recall_k_k')
    ax2.set_ylabel('QPS_seq')
    ax2.set_title('Recall@k vs. QPS_seq')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(f'Performance for {method_config} on {dataset}')

    # Save the plot
    output_filename = (
        f'./results/plots/num_rerank_ablation/{collection}_{dataset}_{method_config}_plots.png'
    )
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate scatter plots for performance evaluation.'
    )
    parser.add_argument(
        'collection', type=str, help='The dataset collection name (e.g., beir, lotte, vidore)'
    )
    parser.add_argument('dataset', type=str, help='The dataset name (e.g., arguana)')
    parser.add_argument('method', type=str, help='The method name (e.g., mvivf)')
    parser.add_argument(
        'method_config', type=str, help='The method configuration (e.g., mvivf_500_pq_256_8)'
    )
    args = parser.parse_args()

    generate_plot(args.collection, args.dataset, args.method, args.method_config)
