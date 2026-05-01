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

    def _to_fp_args(self, params: dict):
        """Translate a single MVSIC search-param dict to FastPlaid kwargs."""
        k = params['k']
        search_args = params.copy()
        search_args['top_k'] = search_args.pop('k')
        # Benchmark configs use num_rerank; FastPlaid expects n_full_scores.
        if 'num_rerank' in search_args and 'n_full_scores' not in search_args:
            search_args['n_full_scores'] = search_args.pop('num_rerank')
        return k, search_args

    def compute_stats_extended(self, queries: list, gt: list, params: dict):
        """Default = batch path. Mirrors the legacy wrapper API."""
        return self.compute_stats_batch(queries, gt, params)

    def compute_stats_batch(self, queries: list, gt: list, params: dict):
        """Single batched search_all call; reports QPS_par."""
        k, search_args = self._to_fp_args(params)

        start_time = time.time()
        neighbors = self.index.search(queries_embeddings=queries, **search_args)
        batch_search_time = time.time() - start_time
        qps_par = len(queries) / batch_search_time if batch_search_time > 0 else 0.0

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

    def compute_stats_latency(self, queries: list, gt: list, params: dict):
        """Single-query loop (mirrors examples/fast_plaid_benchmark.py).

        Reports QPS_seq from the per-query loop and uses the same per-query
        outputs to compute recall@k. No batch search_all is performed.
        """
        k, search_args = self._to_fp_args(params)

        # Warmup with the first ~5 queries.
        for q in queries[: min(5, len(queries))]:
            self.index.search(queries_embeddings=[q], **search_args)

        per_query_latencies = []
        individual_neighbors = []  # list of [doc_id_list], one per query
        for q in queries:
            t0 = time.time()
            results = self.index.search(queries_embeddings=[q], **search_args)
            per_query_latencies.append(time.time() - t0)
            # FastPlaid returns results[0] = [(doc_id, score), ...] for a single query.
            individual_neighbors.append([res[0] for res in results[0]])

        total_seq = sum(per_query_latencies) or 1e-9
        qps_seq = len(queries) / total_seq

        recall_1_k, recall_k_k = mvsic.compute_scores(individual_neighbors, gt, k)

        return StatsExtended(
            k=k,
            recall_1_k=min(1.0, recall_1_k),
            recall_k_k=min(1.0, recall_k_k),
            QPS_seq=qps_seq,
            QPS_par=qps_seq,
            avg_cmps=0.0,
            avg_timings=[total_seq],
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
