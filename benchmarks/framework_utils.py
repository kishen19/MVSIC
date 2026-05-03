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


def load_dataset(path, name, is_mmap: bool = False):
    """
    Loads a dataset for MVSIC indices.

    If `is_mmap` is true, the points (database) file is mmap-loaded. Queries are
    always loaded normally (small + accessed many times).
    """
    points_file = os.path.join(path, f"{name}_points.pcs")
    queries_file = os.path.join(path, f"{name}_queries.pcs")
    gt_file = os.path.join(path, f"{name}_chamfer_neighbors.gt")

    points = mvsic.PointCloudSetIP(points_file, is_mmap=bool(is_mmap))
    queries = mvsic.PointCloudSetIP(queries_file)

    gt = mvsic.ReadGT(gt_file, queries.size())
    return points, queries, gt


class FastPlaidWrapper:
    """A wrapper to make the FastPlaid library conform to the MVSIC index interface.

    Threading / device notes (CPU-only setup -- no GPU)
    ----------------------------------------------------
    Mirroring the MVSIC stats convention:

      * ``compute_stats_latency`` -- per-query, **single-threaded, sequential**
        loop (matches ``parlay::execute_with_scheduler(1, ...)`` in the C++
        ``compute_stats_latency``). Constructed with ``device="cpu"`` so each
        ``index.search([q])`` stays on FastPlaid's single-device fast path
        (avoids the ``joblib.Parallel`` per-call worker spawn). Inside the
        per-query loop we also pin ``torch.set_num_threads(1)`` so the BLAS
        intra-op pool is single-threaded. ``QPS_par = QPS_seq`` is reported
        (matches MVSIC's StatsExtended convention for latency).

      * ``compute_stats_batch`` -- one batched ``index.search(all_queries, ...)``
        call, **uses all CPU threads** by leaving FastPlaid's default device
        selection in place (``["cpu"] * os.cpu_count()`` -- one joblib worker
        per core). Reports ``QPS_par`` only.

      * ``compute_stats_multi_latency`` -- not supported. FastPlaid has no
        per-query multi-thread path (its multi-device CPU mode is *per call*,
        not per-query parallelism). The benchmark runner already skips it.

    For multi-GPU setups, you can pass ``device=["cuda:0", "cuda:1", ...]``;
    FastPlaid will then sharpen via ``torch.multiprocessing``.

    Intra-op parallelism inside FastPlaid's Rust kernel also honors the usual
    env vars (``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` / ``RAYON_NUM_THREADS``);
    set them in the runner script before ``python3`` starts if you want to pin
    kernel thread counts globally (e.g. ``OMP_NUM_THREADS=1`` for the latency
    runner).

    Data contract: documents and queries are lists of ``torch.Tensor`` with
    shape ``(num_tokens, dim)`` per item.
    """

    def __init__(self, dim, build_params, index_path, device=None):
        self.dim = dim
        self.build_params = build_params
        self.index = FastPlaid(index=index_path, device=device)

    def build(self, documents: list):
        print("    Building FastPlaid index...")
        self.index.create(documents_embeddings=documents, **self.build_params)

    def _to_fp_args(self, params: dict):
        """Translate a single MVSIC search-param dict to FastPlaid kwargs.

        We always pass ``show_progress=False`` -- FastPlaid's default ``True``
        would print a tqdm bar per call, which floods stderr and adds overhead
        on the per-query latency loop.
        """
        k = params['k']
        search_args = params.copy()
        search_args['top_k'] = search_args.pop('k')
        if 'num_rerank' in search_args and 'n_full_scores' not in search_args:
            search_args['n_full_scores'] = search_args.pop('num_rerank')
        search_args.setdefault('show_progress', False)
        return k, search_args

    def compute_stats_extended(self, queries: list, gt: list, params: dict):
        """Default = batch path. Mirrors the legacy wrapper API."""
        return self.compute_stats_batch(queries, gt, params)

    def compute_stats_batch(self, queries: list, gt: list, params: dict):
        """Single batched ``search`` call; reports QPS_par.

        The result of ``index.search`` is already
        ``list[list[tuple[doc_id, score]]]`` -- the exact format
        ``mvsic.compute_scores`` wants for ``pred``.
        """
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
        """Single-query, single-thread, sequential loop.

        Mirrors the MVSIC C++ ``compute_stats_latency`` convention (10 warm-up
        queries, single timed pass, ``QPS_par = QPS_seq`` reported). To keep
        the per-query path single-threaded we pin ``torch.set_num_threads(1)``
        for the duration of the loop (and restore it on exit). The wrapper
        should already have been constructed with ``device="cpu"`` so that
        FastPlaid stays on the single-device fast path.

        Per-query predictions are kept as ``[(doc_id, score), ...]`` so they
        match ``mvsic.compute_scores``'s expected
        ``parlay::sequence<std::pair<uint32_t, float>>`` layout.
        """
        k, search_args = self._to_fp_args(params)

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for q in queries[: min(10, len(queries))]:
                self.index.search(queries_embeddings=[q], **search_args)

            per_query_latencies = []
            individual_neighbors = []
            for q in queries:
                t0 = time.time()
                results = self.index.search(queries_embeddings=[q], **search_args)
                per_query_latencies.append(time.time() - t0)
                individual_neighbors.append(results[0])
        finally:
            torch.set_num_threads(old_threads)

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
        pass

    def load(self, path: str, points: list):
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
