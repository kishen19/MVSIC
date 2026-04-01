import mvsic
import os
import torch
import time
from fast_plaid.search import FastPlaid
from collections import namedtuple

StatsExtended = namedtuple(
    'StatsExtended',
    ['k', 'recall_1_k', 'recall_k_k', 'QPS_seq', 'QPS_par', 'avg_cmps', 'avg_timings'],
)


def load_dataset(path, name):
    """
    Loads a dataset for MVSIC indices.
    """
    points_file = os.path.join(path, f"{name}_points.pcs")
    queries_file = os.path.join(path, f"{name}_queries.pcs")
    gt_file = os.path.join(path, f"{name}_chamfer_neighbors.gt")

    points = mvsic.PointCloudSetIP(points_file)
    queries = mvsic.PointCloudSetIP(queries_file)

    gt = mvsic.ReadGT(gt_file, queries.size())
    return points, queries, gt


class FastPlaidWrapper:
    """
    A wrapper to make the FastPlaid library conform to the MVSIC index interface.
    This wrapper expects data to be passed as lists of PyTorch tensors.
    """

    def __init__(self, dim, build_params, index_path):
        self.dim = dim
        self.build_params = build_params
        self.index = FastPlaid(index=index_path)

    def build(self, documents: list):  # Expects list of tensors
        print("    Building FastPlaid index...")
        self.index.create(documents_embeddings=documents, **self.build_params)

    def compute_stats_extended(self, queries: list, gt: list, params: dict):  # Expects list of tensors and a single param dict
        """
        Runs search for a single parameter combination and computes stats.
        """
        k = params['k']
        search_args = params.copy()
        search_args['top_k'] = search_args.pop('k')
        # Benchmark configs use num_rerank; FastPlaid expects n_full_scores.
        if 'num_rerank' in search_args and 'n_full_scores' not in search_args:
            search_args['n_full_scores'] = search_args.pop('num_rerank')

        # 1. QPS_seq
        # start_time = time.time()
        # # The search method in fast-plaid expects a list of queries and returns a list of results
        # for _query in queries:
        #     _ = self.index.search(queries_embeddings=[_query], **search_args)
        # end_time = time.time()
        # qps_seq = len(queries) / (end_time - start_time) if (end_time - start_time) > 0 else 0

        # 2. QPS_par
        start_time = time.time()
        neighbors = self.index.search(queries_embeddings=queries, **search_args)
        end_time = time.time()
        batch_search_time = end_time - start_time
        qps_par = len(queries) / (end_time - start_time) if (end_time - start_time) > 0 else 0

        # 3. Compute scores
        recall_1_k, recall_k_k = mvsic.compute_scores(neighbors, gt, k)

        return StatsExtended(
            k=k,
            recall_1_k=min(1.0, recall_1_k),
            recall_k_k=min(1.0, recall_k_k),
            QPS_seq=None,
            QPS_par=qps_par,
            avg_cmps=0.0,
            avg_timings=[batch_search_time],
        )

    def save(self, path: str):
        # FastPlaid handles its own saving/loading based on the path in __init__
        pass

    def load(self, path: str, points: list):  # Expects list of tensors for points
        # FastPlaid handles its own saving/loading based on the path in __init__
        pass


def plot_results(csv_filename, plot_filename):
    """
    Generates and saves plots from the results CSV.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not os.path.exists(csv_filename):
        print(f"Results file not found: {csv_filename}")
        return

    df = pd.read_csv(csv_filename)
    k_values = df['k'].unique()

    for k in k_values:
        df_k = df[df['k'] == k]
        plt.figure(figsize=(12, 8))
        sns.set_theme(style="whitegrid")
        # ... (plotting code remains the same)
