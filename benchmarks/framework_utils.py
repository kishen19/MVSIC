import csv
import io
import json
import os
import subprocess
import sys
import time
import ctypes
from collections import namedtuple

import mvsic
import numpy as np
import torch

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
        # Keep fast_plaid import local so non-fastplaid runs (e.g., IGP-only)
        # don't load its native deps at module import time.
        from fast_plaid.search import FastPlaid
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


class IGPWrapper:
    """Adapter that lets the external IGP module run inside the benchmark harness.

    The wrapper mirrors the FastPlaid contract:
      - build() writes index artifacts into ``index_path``.
      - load() reconstructs an in-memory ``IGP.DocRetrieval`` instance.
      - compute_stats_latency()/compute_stats_batch() return StatsExtended.

    Expected build params:
      - n_bit (default: 2)
      - n_centroid (optional; if omitted, a sqrt heuristic is used)
      - n_centroid_factor (default: 16.0, used when n_centroid is omitted)
      - kmeans_niter (default: 20)
      - kmeans_seed (default: 123)
      - kmeans_sample_max_vectors (default: 2_000_000)
      - assign_batch_size (default: 200_000)
      - residual_batch_size (default: 200_000)
      - kmeans_gpu (default: false)
    """

    _CENTROIDS_FNAME = "centroid_l.npy"
    _CODES_FNAME = "vq_code_l.npy"
    _WEIGHTS_FNAME = "weight_l.npy"
    _RESIDUAL_FNAME = "residual_code_l.npy"
    _DOCLENS_FNAME = "doclens.npy"
    _META_FNAME = "metadata.json"

    def __init__(self, dim, build_params, index_path, device=None):
        self.dim = int(dim)
        self.build_params = build_params or {}
        self.index_path = index_path
        self.device = device
        self._igp_module = None
        self._index = None
        self._doclens = None
        self._n_centroid = None
        self._n_bit = int(self.build_params.get("n_bit", 2))

    def _require_igp(self):
        if self._igp_module is None:
            # Some launch paths may resolve system libstdc++ first, which can
            # be too old for the conda-built spdlog dependency pulled by IGP.
            # Preload conda's C++ runtime into the global namespace to avoid
            # CXXABI mismatch errors when importing the pybind module.
            conda_prefix = os.environ.get("CONDA_PREFIX")
            if conda_prefix:
                conda_lib = os.path.join(conda_prefix, "lib")
                for libname in ("libstdc++.so.6", "libgcc_s.so.1"):
                    libpath = os.path.join(conda_lib, libname)
                    if os.path.exists(libpath):
                        try:
                            ctypes.CDLL(libpath, mode=ctypes.RTLD_GLOBAL)
                        except OSError:
                            # Fall through; the subsequent import will raise a
                            # clear error if runtime symbols are still missing.
                            pass
            try:
                import IGP as igp_module
            except ImportError as e:
                raise RuntimeError(
                    "IGP python module is unavailable. This is only required for "
                    "`index.name: igp` runs. Install it with `bash setup_external.sh --igp` "
                    "from the repo root (or build/install upstream `IGP` manually). "
                    f"Original import error: {e}"
                ) from e
            self._igp_module = igp_module
        return self._igp_module

    @staticmethod
    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return x / norms

    @staticmethod
    def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().to(dtype=torch.float32).numpy()

    def _points_to_flat_arrays(self, documents: list):
        if not documents:
            raise ValueError("Cannot build IGP index from an empty documents list.")
        doclens = np.asarray([int(d.shape[0]) for d in documents], dtype=np.uint32)
        vecs = np.concatenate([self._tensor_to_numpy(d) for d in documents], axis=0).astype(
            np.float32, copy=False
        )
        if vecs.shape[1] != self.dim:
            raise ValueError(
                f"IGP dimension mismatch: wrapper dim={self.dim}, input dim={vecs.shape[1]}"
            )
        return doclens, self._normalize_rows(vecs)

    def _resolve_n_centroid(self, n_vec: int) -> int:
        explicit = self.build_params.get("n_centroid")
        if explicit is not None:
            n_centroid = int(explicit)
        else:
            factor = float(self.build_params.get("n_centroid_factor", 16.0))
            # Keep parity with upstream's 2^floor(log2(f * sqrt(n_vec))) heuristic.
            n_centroid = int(2 ** np.floor(np.log2(max(2.0, factor * np.sqrt(max(1, n_vec))))))
        n_centroid = max(1, min(int(n_centroid), int(n_vec)))
        return n_centroid

    def _sample_for_kmeans(self, vecs: np.ndarray) -> np.ndarray:
        max_vectors = int(self.build_params.get("kmeans_sample_max_vectors", 2_000_000))
        if vecs.shape[0] <= max_vectors:
            return vecs
        seed = int(self.build_params.get("kmeans_seed", 123))
        rng = np.random.default_rng(seed)
        idx = rng.choice(vecs.shape[0], size=max_vectors, replace=False)
        idx.sort()
        return vecs[idx]

    def _fit_centroids(self, sample_vecs: np.ndarray, n_centroid: int) -> np.ndarray:
        try:
            import faiss
        except ImportError as e:
            raise RuntimeError(
                "IGP build requires faiss (faiss-cpu or faiss-gpu Python package)."
            ) from e
        niter = int(self.build_params.get("kmeans_niter", 20))
        seed = int(self.build_params.get("kmeans_seed", 123))
        use_gpu = bool(self.build_params.get("kmeans_gpu", False))
        verbose = bool(self.build_params.get("verbose", False))
        kmeans = faiss.Kmeans(
            self.dim, n_centroid, niter=niter, gpu=use_gpu, verbose=verbose, seed=seed
        )
        kmeans.train(sample_vecs.astype(np.float32, copy=False))
        centroids = np.asarray(kmeans.centroids, dtype=np.float32).reshape(n_centroid, self.dim)
        return self._normalize_rows(centroids)

    def _assign_codes(self, vecs: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        batch_size = int(self.build_params.get("assign_batch_size", 200_000))
        n_vec = vecs.shape[0]
        codes = np.empty(n_vec, dtype=np.uint32)
        for start in range(0, n_vec, batch_size):
            end = min(start + batch_size, n_vec)
            sims = vecs[start:end] @ centroids.T
            codes[start:end] = np.argmax(sims, axis=1).astype(np.uint32)
        return codes

    def _build_residual_codes(
        self,
        sq_ins,
        vecs: np.ndarray,
        code_l: np.ndarray,
        n_val_per_vec: int,
    ) -> np.ndarray:
        batch_size = int(self.build_params.get("residual_batch_size", 200_000))
        n_vec = vecs.shape[0]
        residual_code_l = np.empty(n_vec * n_val_per_vec, dtype=np.uint8)
        cursor = 0
        for start in range(0, n_vec, batch_size):
            end = min(start + batch_size, n_vec)
            raw = sq_ins.compute_residual_code(vec_l=vecs[start:end], code_l=code_l[start:end])
            chunk = raw[0] if isinstance(raw, (tuple, list)) else raw
            chunk = np.asarray(chunk, dtype=np.uint8).reshape(-1)
            expected = (end - start) * n_val_per_vec
            if chunk.size != expected:
                raise RuntimeError(
                    f"Unexpected residual chunk length: got {chunk.size}, expected {expected}"
                )
            residual_code_l[cursor : cursor + expected] = chunk
            cursor += expected
        return residual_code_l

    def _artifact_path(self, fname: str) -> str:
        return os.path.join(self.index_path, fname)

    def build(self, documents: list):
        os.makedirs(self.index_path, exist_ok=True)
        igp = self._require_igp()
        doclens, vecs = self._points_to_flat_arrays(documents)
        n_item = int(doclens.shape[0])
        n_vec = int(vecs.shape[0])
        n_centroid = self._resolve_n_centroid(n_vec)
        n_bit = int(self.build_params.get("n_bit", 2))

        sample_vecs = self._sample_for_kmeans(vecs)
        centroids = self._fit_centroids(sample_vecs, n_centroid)
        sample_codes = self._assign_codes(sample_vecs, centroids)
        code_l = self._assign_codes(vecs, centroids)

        cutoff_l, weight_l = igp.compute_quantized_scalar(
            item_vec_l=sample_vecs,
            centroid_l=centroids,
            code_l=sample_codes,
            n_bit=n_bit,
        )
        cutoff_l = np.asarray(cutoff_l, dtype=np.float32)
        weight_l = np.asarray(weight_l, dtype=np.float32)
        sq_ins = igp.CompressResidualCode(
            centroid_l=centroids,
            cutoff_l=cutoff_l,
            weight_l=weight_l,
            n_bit=n_bit,
        )
        n_val_per_vec = int(sq_ins.n_val_per_vec)
        residual_code_l = self._build_residual_codes(sq_ins, vecs, code_l, n_val_per_vec)

        np.save(self._artifact_path(self._CENTROIDS_FNAME), centroids.astype(np.float32))
        np.save(self._artifact_path(self._CODES_FNAME), code_l.astype(np.uint32))
        np.save(self._artifact_path(self._WEIGHTS_FNAME), weight_l.astype(np.float32))
        np.save(self._artifact_path(self._RESIDUAL_FNAME), residual_code_l.astype(np.uint8))
        np.save(self._artifact_path(self._DOCLENS_FNAME), doclens.astype(np.uint32))
        with open(self._artifact_path(self._META_FNAME), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dim": self.dim,
                    "n_item": n_item,
                    "n_vec": n_vec,
                    "n_centroid": n_centroid,
                    "n_bit": n_bit,
                    "n_val_per_vec": n_val_per_vec,
                    "build_params": self.build_params,
                },
                f,
                indent=2,
            )

        self._n_centroid = n_centroid
        self._n_bit = n_bit
        self._doclens = doclens
        self._index = self._build_doc_retrieval(centroids, code_l, weight_l, residual_code_l, doclens)

    def _build_doc_retrieval(
        self,
        centroids: np.ndarray,
        code_l: np.ndarray,
        weight_l: np.ndarray,
        residual_code_l: np.ndarray,
        doclens: np.ndarray,
    ):
        igp = self._require_igp()
        index = igp.DocRetrieval(
            item_n_vec_l=doclens.tolist(),
            n_item=int(doclens.shape[0]),
            vec_dim=int(self.dim),
            n_centroid=int(centroids.shape[0]),
            n_bit=int(self._n_bit),
        )
        index.load_quantization_index(
            centroid_l=centroids,
            vq_code_l=code_l,
            weight_l=weight_l,
            residual_code_l=residual_code_l,
        )
        return index

    def _load_artifacts(self):
        centroids = np.load(self._artifact_path(self._CENTROIDS_FNAME)).astype(np.float32, copy=False)
        code_l = np.load(self._artifact_path(self._CODES_FNAME)).astype(np.uint32, copy=False)
        weight_l = np.load(self._artifact_path(self._WEIGHTS_FNAME)).astype(np.float32, copy=False)
        residual_code_l = np.load(self._artifact_path(self._RESIDUAL_FNAME)).astype(np.uint8, copy=False)
        doclens = np.load(self._artifact_path(self._DOCLENS_FNAME)).astype(np.uint32, copy=False)
        return centroids, code_l, weight_l, residual_code_l, doclens

    def save(self, path: str):
        # Artifacts are persisted during build(). Keep a no-op save() for API parity.
        return

    def load(self, path: str, points: list):
        self.index_path = path
        centroids, code_l, weight_l, residual_code_l, doclens = self._load_artifacts()
        self._doclens = doclens
        self._n_centroid = int(centroids.shape[0])
        self._index = self._build_doc_retrieval(centroids, code_l, weight_l, residual_code_l, doclens)

    @staticmethod
    def _stack_queries(queries: list) -> np.ndarray:
        if not queries:
            raise ValueError("Cannot search IGP with an empty queries list.")
        q_lens = [int(q.shape[0]) for q in queries]
        q_max = max(q_lens)
        dim = int(queries[0].shape[1])
        out = np.zeros((len(queries), q_max, dim), dtype=np.float32)
        for i, q in enumerate(queries):
            q_np = IGPWrapper._tensor_to_numpy(q)
            out[i, : q_np.shape[0], :] = q_np
        return IGPWrapper._normalize_rows(out.reshape(-1, dim)).reshape(len(queries), q_max, dim)

    @staticmethod
    def _pred_from_arrays(est_id_l: np.ndarray, est_dist_l: np.ndarray):
        pred = []
        for row_id, row_dist in zip(est_id_l, est_dist_l):
            pred.append(
                [(int(doc_id), float(score)) for doc_id, score in zip(row_id.tolist(), row_dist.tolist())]
            )
        return pred

    def _search_igp(self, queries_np: np.ndarray, params: dict):
        if self._index is None:
            raise RuntimeError("IGP index is not loaded. Call load() before search.")
        k = int(params["k"])
        nprobe = int(params.get("nprobe", 1))
        if "probe_topk" not in params:
            raise ValueError(
                "IGP search params must include 'probe_topk'. "
                "num_rerank is not used by IGP."
            )
        probe_topk = int(params["probe_topk"])
        probe_topk = max(probe_topk, k)
        n_thread = int(params.get("n_thread", params.get("threads", 1)))
        n_thread = max(1, n_thread)
        return self._index.search(
            query_l=queries_np,
            topk=k,
            nprobe=nprobe,
            probe_topk=probe_topk,
            n_thread=n_thread,
        )

    def compute_stats_extended(self, queries: list, gt: list, params: dict):
        return self.compute_stats_batch(queries, gt, params)

    def compute_stats_batch(self, queries: list, gt: list, params: dict):
        params = dict(params)
        # Batch throughput path mirrors other methods: run one search over all
        # queries using the full available CPU thread budget automatically.
        try:
            n_all_threads = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            n_all_threads = os.cpu_count() or 1
        params["n_thread"] = max(1, int(n_all_threads))

        queries_np = self._stack_queries(queries)
        k = int(params["k"])
        t0 = time.time()
        out = self._search_igp(queries_np, params)
        wall = time.time() - t0
        (
            total_retrieval_time,
            est_dist_l,
            est_id_l,
            retrieval_time_l,
            filter_time_l,
            decode_time_l,
            refine_time_l,
            _n_seen_item_l,
            _incremental_graph_n_compute_l,
            n_vq_score_linear_scan_l,
        ) = out
        total = float(total_retrieval_time) if float(total_retrieval_time) > 0 else wall
        qps_par = len(queries) / total if total > 0 else 0.0
        pred = self._pred_from_arrays(est_id_l, est_dist_l)
        recall_1_k, recall_k_k = mvsic.compute_scores(pred, gt, k)
        avg_timings = [
            float(np.mean(retrieval_time_l)) if len(retrieval_time_l) else total,
            float(np.mean(filter_time_l)) if len(filter_time_l) else 0.0,
            float(np.mean(decode_time_l)) if len(decode_time_l) else 0.0,
            float(np.mean(refine_time_l)) if len(refine_time_l) else 0.0,
        ]
        avg_cmps = float(np.mean(n_vq_score_linear_scan_l)) if len(n_vq_score_linear_scan_l) else 0.0
        return StatsExtended(
            k=k,
            recall_1_k=min(1.0, recall_1_k),
            recall_k_k=min(1.0, recall_k_k),
            QPS_seq=None,
            QPS_par=qps_par,
            avg_cmps=avg_cmps,
            avg_timings=avg_timings,
        )

    def compute_stats_latency(self, queries: list, gt: list, params: dict):
        k = int(params["k"])
        params = dict(params)
        # Latency path mirrors other methods: single-query, single-thread.
        params["n_thread"] = 1
        # Warmup to reduce first-call effects.
        warmup_n = min(10, len(queries))
        for i in range(warmup_n):
            q_np = self._stack_queries([queries[i]])
            self._search_igp(q_np, params)

        per_query_l = []
        pred = []
        filter_l, decode_l, refine_l, cmp_l = [], [], [], []
        for q in queries:
            q_np = self._stack_queries([q])
            t0 = time.time()
            out = self._search_igp(q_np, params)
            wall = time.time() - t0
            (
                _total_retrieval_time,
                est_dist_l,
                est_id_l,
                retrieval_time_l,
                filter_time_l,
                decode_time_l,
                refine_time_l,
                _n_seen_item_l,
                _incremental_graph_n_compute_l,
                n_vq_score_linear_scan_l,
            ) = out
            per_query = (
                float(retrieval_time_l[0]) if len(retrieval_time_l) else wall
            )
            per_query_l.append(per_query)
            pred.append(
                [
                    (int(doc_id), float(score))
                    for doc_id, score in zip(est_id_l[0].tolist(), est_dist_l[0].tolist())
                ]
            )
            filter_l.append(float(filter_time_l[0]) if len(filter_time_l) else 0.0)
            decode_l.append(float(decode_time_l[0]) if len(decode_time_l) else 0.0)
            refine_l.append(float(refine_time_l[0]) if len(refine_time_l) else 0.0)
            cmp_l.append(
                float(n_vq_score_linear_scan_l[0]) if len(n_vq_score_linear_scan_l) else 0.0
            )

        total_seq = sum(per_query_l) if per_query_l else 1e-9
        qps_seq = len(queries) / total_seq
        recall_1_k, recall_k_k = mvsic.compute_scores(pred, gt, k)
        avg_timings = [
            float(np.mean(per_query_l)) if per_query_l else total_seq,
            float(np.mean(filter_l)) if filter_l else 0.0,
            float(np.mean(decode_l)) if decode_l else 0.0,
            float(np.mean(refine_l)) if refine_l else 0.0,
        ]
        avg_cmps = float(np.mean(cmp_l)) if cmp_l else 0.0
        return StatsExtended(
            k=k,
            recall_1_k=min(1.0, recall_1_k),
            recall_k_k=min(1.0, recall_k_k),
            QPS_seq=qps_seq,
            QPS_par=qps_seq,
            avg_cmps=avg_cmps,
            avg_timings=avg_timings,
        )


class GEMWrapper:
    """Subprocess-backed adapter for the GEM (sigmod26gem) baseline.

    Unlike IGP / FastPlaid (Python-importable extensions), GEM ships only as
    a C++ library. We invoke the standalone ``gem_runner`` binary
    (//benchmarks/gem:gem_runner) as a subprocess for both build and search.

    Search-time IO contract differs from the other wrappers because the
    runner amortizes its ~multi-second data-load cost over an internal
    ``ef_search`` x ``rerank_k`` cartesian sweep. The wrapper exposes:

      * ``sweep(...)``     -- preferred: one runner invocation over many
                              (ef, rerank) combos; returns one row per combo.
      * ``compute_stats_latency`` / ``compute_stats_batch`` -- single-combo
                              shims that internally call ``sweep`` with a
                              1x1 list. Kept for API parity, but the
                              ``_run_gem`` dispatcher in benchmark_search.py
                              should call ``sweep`` directly.

    Build-time inputs (the gem_data tree) are produced by
    ``benchmarks/gem/gem_preprocess.py`` (two-stage faiss k-means + TF-IDF coarse
    routing + FP16 doc shards). The wrapper calls the preprocessor lazily
    on the first ``build()`` invocation, keyed on (k1, k2, top_r) via the
    ``.READY`` marker the preprocessor writes.

    Per-build artifacts live entirely under ``index_path``:

      <index_path>/
        gem_data/cdata/{centroids,coarse_centroids}.npy
        gem_data/cdata/coarse_cluster_info.txt
        gem_data/docdata/{encoding,doc_codes,doclens}<i>.npy
        gem_data/qdata/{qembs,qlens}.npy
        0.bin             # HNSW graph
        gem_meta.json
        build_stats.json  # written by benchmark_build.py
        _BUILD_OK         # written by benchmark_build.py
    """

    GEM_DATA_SUBDIR = "gem_data"

    def __init__(self, dim, build_params, index_path, device=None):
        self.dim = int(dim)
        self.build_params = dict(build_params or {})
        self.index_path = index_path
        self._runner_bin: str | None = None
        # Set by the dispatcher (_run_gem) right after construction so the
        # wrapper can locate the source .pcs files for preprocessing and
        # the .gt file for recall evaluation. Without these, build() and
        # search() will refuse to run.
        self.ds_path: str | None = None
        self.ds_name: str | None = None
        self._meta: dict | None = None

    # ----- runner discovery ------------------------------------------------

    def _find_runner(self) -> str:
        if self._runner_bin:
            return self._runner_bin
        env = os.environ.get("MVSIC_GEM_RUNNER")
        if env and os.path.exists(env) and os.access(env, os.X_OK):
            self._runner_bin = env
            return env
        # Repo root = parent of the benchmarks/ folder this file lives in.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cand = os.path.join(repo_root, "bazel-bin", "benchmarks", "gem", "gem_runner")
        if (
            os.path.exists(cand)
            and os.access(cand, os.X_OK)
        ):
            self._runner_bin = cand
            return cand
        raise RuntimeError(
            "gem_runner binary not found at "
            f"{cand}. Run `bash setup_external.sh --gem` from the repo root, "
            "or set $MVSIC_GEM_RUNNER."
        )

    def _repo_root(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ----- gem_data preprocessing -----------------------------------------

    def _ensure_gem_data(self) -> str:
        if not self.ds_path or not self.ds_name:
            raise RuntimeError(
                "GEMWrapper requires ds_path / ds_name to be set before "
                "build/search (normally injected by _run_gem)."
            )
        out_dir = os.path.join(self.index_path, self.GEM_DATA_SUBDIR)
        ready = os.path.join(out_dir, ".READY")
        if os.path.exists(ready) and not self.build_params.get("force_preprocess", False):
            return out_dir
        os.makedirs(out_dir, exist_ok=True)
        preproc = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "gem", "gem_preprocess.py",
        )
        py = os.environ.get("PYTHON", sys.executable)
        cmd = [
            py, preproc,
            "--dataset", str(self.ds_name),
            "--ds-path", str(self.ds_path),
            "--out-dir", out_dir,
            "--num-shards", str(int(self.build_params.get("num_doc_shards", 1))),
            "--niter", str(int(self.build_params.get("kmeans_niter", 20))),
            "--seed", str(int(self.build_params.get("kmeans_seed", 123))),
            "--kmeans-sample-max",
            str(int(self.build_params.get("kmeans_sample_max", 2_000_000))),
            "--k1", str(int(self.build_params.get("num_fine_clusters", 0))),
            "--k2", str(int(self.build_params.get("num_coarse_clusters", 0))),
            "--top-r", str(int(self.build_params.get("top_r", 4))),
        ]
        if self.build_params.get("verbose_preprocess", False):
            cmd.append("--verbose")
        print(f"    [gem] preprocessing -> {out_dir}", flush=True)
        # Inherit stdout/stderr so the preprocessor's progress prints stream
        # into the surrounding build log.
        subprocess.run(cmd, check=True)
        return out_dir

    # ----- build ----------------------------------------------------------

    def build(self, documents=None):
        # ``documents`` is unused by GEM (the runner reads .npy files from
        # the gem_data tree directly). We accept it for API parity with
        # FastPlaid / IGP wrappers in benchmark_build.py.
        del documents
        runner = self._find_runner()
        gem_data = self._ensure_gem_data()
        num_shards = int(self.build_params.get("num_doc_shards", 1))
        m_index = int(self.build_params.get("m_index", 24))
        ef_construction = int(self.build_params.get("ef_construction", 80))
        try:
            n_thr_default = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            n_thr_default = os.cpu_count() or 1
        threads = int(self.build_params.get("build_threads", n_thr_default))
        cmd = [
            runner,
            "--mode", "build",
            "--gem-data", gem_data,
            "--num-doc-shards", str(num_shards),
            "--dim", str(self.dim),
            "--m-index", str(m_index),
            "--ef-construction", str(ef_construction),
            "--out", self.index_path,
            "--threads", str(threads),
        ]
        print(
            f"    [gem] building hnswlib graph -> {self.index_path} "
            f"(m={m_index}, ef={ef_construction}, threads={threads})",
            flush=True,
        )
        subprocess.run(cmd, check=True)

    def save(self, path: str):
        return  # gem_runner writes 0.bin during build().

    def load(self, path, points):
        # No eager load -- the runner reloads gem_data + 0.bin for each
        # search subprocess. We only need to remember where the index lives.
        del points
        self.index_path = path
        meta = os.path.join(path, "gem_meta.json")
        if os.path.exists(meta):
            try:
                with open(meta) as f:
                    self._meta = json.load(f)
            except Exception:
                self._meta = None

    # ----- search ---------------------------------------------------------

    def _gt_path(self) -> str:
        if not self.ds_path or not self.ds_name:
            raise RuntimeError(
                "GEMWrapper requires ds_path / ds_name to be set before search."
            )
        return os.path.join(self.ds_path, f"{self.ds_name}_chamfer_neighbors.gt")

    def sweep(
        self,
        *,
        k: int,
        nprobe: int,
        ef_list,
        rerank_list,
        threads: int,
        qps_thresh: float | None = None,
        warmup: int = 10,
        query_indices=None,
    ):
        """Invoke gem_runner once over the cartesian product (ef x rerank).

        Returns the parsed CSV rows as a list of dicts (one per combo).

        ``qps_thresh`` is forwarded as ``--qps-stop-below``, which makes the
        runner stop each ``rerank_k`` chain at the first ef whose QPS falls
        below the threshold (matches the per-sweep early-termination
        semantics of _run_igp's outer loop).
        """
        runner = self._find_runner()
        gem_data = os.path.join(self.index_path, self.GEM_DATA_SUBDIR)
        if not os.path.exists(os.path.join(gem_data, ".READY")):
            raise RuntimeError(
                f"GEM gem_data tree missing or incomplete at {gem_data}. "
                "Run benchmark_build.py first."
            )
        num_shards = int(self.build_params.get("num_doc_shards", 1))
        m_index = int(self.build_params.get("m_index", 24))
        ef_construction = int(self.build_params.get("ef_construction", 80))
        ef_list = sorted({int(x) for x in ef_list})
        rerank_list = sorted({int(x) for x in rerank_list})
        cmd = [
            runner,
            "--mode", "search",
            "--gem-data", gem_data,
            "--num-doc-shards", str(num_shards),
            "--dim", str(self.dim),
            "--m-index", str(m_index),
            "--ef-construction", str(ef_construction),
            "--index", self.index_path,
            "--gt", self._gt_path(),
            "--k", str(int(k)),
            "--nprobe-t", str(int(nprobe)),
            "--ef-list", ",".join(str(x) for x in ef_list),
            "--rerank-list", ",".join(str(x) for x in rerank_list),
            "--threads", str(int(threads)),
            "--warmup", str(int(warmup)),
            "--output-csv", "-",
        ]
        if qps_thresh is not None and qps_thresh > 0:
            cmd += ["--qps-stop-below", str(float(qps_thresh))]
        if query_indices is not None:
            idx = (
                query_indices.tolist()
                if hasattr(query_indices, "tolist")
                else list(query_indices)
            )
            if idx:
                cmd += ["--query-indices", ",".join(str(int(i)) for i in idx)]
        # capture stdout (CSV rows), forward stderr so progress logs stream
        # into the surrounding search log.
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
            sys.stderr.flush()
        if proc.returncode != 0:
            err_tail = (proc.stderr or "").strip() or (proc.stdout or "").strip()
            raise RuntimeError(
                f"gem_runner failed (exit {proc.returncode}): {err_tail}"
            )
        rows = []
        reader = csv.DictReader(io.StringIO(proc.stdout))
        for row in reader:
            rows.append({key: row[key] for key in row.keys()})
        return rows

    def row_to_stats(self, row) -> StatsExtended:
        """Convert one parsed CSV row from gem_runner to a StatsExtended.

        Labels in methods.yaml must match the order
        ["t_filter", "t_graph_search", "t_rerank"] (the order is the same
        as the runner's CSV columns t_filter, t_graph_search, t_rerank).
        """
        k = int(float(row["k"]))
        qps_seq = float(row["QPS_seq"]) if row.get("QPS_seq") else 0.0
        qps_par = float(row["QPS_par"]) if row.get("QPS_par") else 0.0
        return StatsExtended(
            k=k,
            recall_1_k=min(1.0, float(row["recall_1_k"])),
            recall_k_k=min(1.0, float(row["recall_k_k"])),
            QPS_seq=qps_seq if qps_seq > 0 else None,
            QPS_par=qps_par,
            avg_cmps=float(row.get("avg_cmps", 0.0) or 0.0),
            avg_timings=[
                float(row.get("t_filter", 0.0) or 0.0),
                float(row.get("t_graph_search", 0.0) or 0.0),
                float(row.get("t_rerank", 0.0) or 0.0),
            ],
        )

    def compute_stats_latency(self, queries, gt, params):
        del queries, gt
        rows = self.sweep(
            k=int(params["k"]),
            nprobe=int(params.get("nprobe", 4)),
            ef_list=[int(params["ef_search"])],
            rerank_list=[int(params.get("rerank_k", 256))],
            threads=1,
        )
        if not rows:
            raise RuntimeError("gem_runner returned no CSV rows.")
        return self.row_to_stats(rows[0])

    def compute_stats_batch(self, queries, gt, params):
        del queries, gt
        try:
            n_thr = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            n_thr = os.cpu_count() or 1
        rows = self.sweep(
            k=int(params["k"]),
            nprobe=int(params.get("nprobe", 4)),
            ef_list=[int(params["ef_search"])],
            rerank_list=[int(params.get("rerank_k", 256))],
            threads=int(n_thr),
        )
        if not rows:
            raise RuntimeError("gem_runner returned no CSV rows.")
        return self.row_to_stats(rows[0])


class HnswlibWrapper:
    """Subprocess-backed adapter for upstream nmslib/hnswlib (v0.9.0+).

    Flat token indexing with per-query-token graph probes and exact
    asymmetric chamfer-IP rerank. See //benchmarks/hnswlib:hnswlib_runner and
    benchmarks/hnswlib/hnswlib_preprocess.py.
    """

    HNSWLIB_DATA_SUBDIR = "hnswlib_data"

    def __init__(self, dim, build_params, index_path, device=None):
        del device
        self.dim = int(dim)
        self.build_params = dict(build_params or {})
        self.index_path = index_path
        self._runner_bin: str | None = None
        self.ds_path: str | None = None
        self.ds_name: str | None = None
        self._meta: dict | None = None

    def _find_runner(self) -> str:
        if self._runner_bin:
            return self._runner_bin
        env = os.environ.get("MVSIC_HNSWLIB_RUNNER")
        if env and os.path.exists(env) and os.access(env, os.X_OK):
            self._runner_bin = env
            return env
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cand = os.path.join(repo_root, "bazel-bin", "benchmarks", "hnswlib", "hnswlib_runner")
        if os.path.exists(cand) and os.access(cand, os.X_OK):
            self._runner_bin = cand
            return cand
        raise RuntimeError(
            "hnswlib_runner binary not found at "
            f"{cand}. Run `bash setup_external.sh --hnswlib` from the repo root, "
            "or set $MVSIC_HNSWLIB_RUNNER."
        )

    def _repo_root(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _ensure_hnswlib_data(self) -> str:
        if not self.ds_path or not self.ds_name:
            raise RuntimeError(
                "HnswlibWrapper requires ds_path / ds_name before build/search."
            )
        out_dir = os.path.join(self.index_path, self.HNSWLIB_DATA_SUBDIR)
        ready = os.path.join(out_dir, ".READY")
        if os.path.exists(ready) and not self.build_params.get("force_preprocess", False):
            return out_dir
        os.makedirs(out_dir, exist_ok=True)
        preproc = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "hnswlib", "hnswlib_preprocess.py",
        )
        py = os.environ.get("PYTHON", sys.executable)
        metric = str(self.build_params.get("metric", "ip")).lower()
        cmd = [
            py,
            preproc,
            "--dataset",
            str(self.ds_name),
            "--ds-path",
            str(self.ds_path),
            "--out-dir",
            out_dir,
            "--metric",
            metric,
        ]
        if self.build_params.get("force_preprocess", False):
            cmd.append("--force")
        print(f"    [hnswlib] preprocessing -> {out_dir}", flush=True)
        subprocess.run(cmd, check=True)
        return out_dir

    def build(self, documents=None):
        del documents
        runner = self._find_runner()
        hnswlib_data = self._ensure_hnswlib_data()
        m_index = int(self.build_params.get("m_index", 16))
        ef_construction = int(self.build_params.get("ef_construction", 200))
        metric = str(self.build_params.get("metric", "ip")).lower()
        try:
            n_thr_default = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            n_thr_default = os.cpu_count() or 1
        threads = int(self.build_params.get("build_threads", n_thr_default))
        cmd = [
            runner,
            "--mode",
            "build",
            "--hnswlib-data",
            hnswlib_data,
            "--dim",
            str(self.dim),
            "--metric",
            metric,
            "--m-index",
            str(m_index),
            "--ef-construction",
            str(ef_construction),
            "--out",
            self.index_path,
            "--threads",
            str(threads),
        ]
        print(
            f"    [hnswlib] building graph -> {self.index_path} "
            f"(m={m_index}, ef={ef_construction}, threads={threads})",
            flush=True,
        )
        subprocess.run(cmd, check=True)

    def save(self, path: str):
        return

    def load(self, path, points):
        del points
        self.index_path = path
        meta = os.path.join(path, "hnswlib_meta.json")
        if os.path.exists(meta):
            try:
                with open(meta) as f:
                    self._meta = json.load(f)
            except Exception:
                self._meta = None

    def _gt_path(self) -> str:
        if not self.ds_path or not self.ds_name:
            raise RuntimeError(
                "HnswlibWrapper requires ds_path / ds_name before search."
            )
        return os.path.join(self.ds_path, f"{self.ds_name}_chamfer_neighbors.gt")

    def sweep(
        self,
        *,
        k: int,
        ef_list,
        rerank_list,
        threads: int,
        qps_thresh: float | None = None,
        warmup: int = 10,
        query_indices=None,
    ):
        runner = self._find_runner()
        hnswlib_data = os.path.join(self.index_path, self.HNSWLIB_DATA_SUBDIR)
        if not os.path.exists(os.path.join(hnswlib_data, ".READY")):
            raise RuntimeError(
                f"hnswlib_data tree missing at {hnswlib_data}. Run benchmark_build.py first."
            )
        m_index = int(self.build_params.get("m_index", 16))
        ef_construction = int(self.build_params.get("ef_construction", 200))
        metric = str(self.build_params.get("metric", "ip")).lower()
        ef_list = sorted({int(x) for x in ef_list})
        rerank_list = sorted({int(x) for x in rerank_list})
        cmd = [
            runner,
            "--mode",
            "search",
            "--hnswlib-data",
            hnswlib_data,
            "--dim",
            str(self.dim),
            "--metric",
            metric,
            "--m-index",
            str(m_index),
            "--ef-construction",
            str(ef_construction),
            "--index",
            self.index_path,
            "--gt",
            self._gt_path(),
            "--k",
            str(int(k)),
            "--ef-list",
            ",".join(str(x) for x in ef_list),
            "--rerank-list",
            ",".join(str(x) for x in rerank_list),
            "--threads",
            str(int(threads)),
            "--warmup",
            str(int(warmup)),
            "--output-csv",
            "-",
        ]
        if qps_thresh is not None and qps_thresh > 0:
            cmd += ["--qps-stop-below", str(float(qps_thresh))]
        if query_indices is not None:
            idx = (
                query_indices.tolist()
                if hasattr(query_indices, "tolist")
                else list(query_indices)
            )
            if idx:
                cmd += ["--query-indices", ",".join(str(int(i)) for i in idx)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
            sys.stderr.flush()
        if proc.returncode != 0:
            err_tail = (proc.stderr or "").strip() or (proc.stdout or "").strip()
            raise RuntimeError(
                f"hnswlib_runner failed (exit {proc.returncode}): {err_tail}"
            )
        rows = []
        reader = csv.DictReader(io.StringIO(proc.stdout))
        for row in reader:
            rows.append({key: row[key] for key in row.keys()})
        return rows

    def row_to_stats(self, row) -> StatsExtended:
        k = int(float(row["k"]))
        qps_seq = float(row["QPS_seq"]) if row.get("QPS_seq") else 0.0
        qps_par = float(row["QPS_par"]) if row.get("QPS_par") else 0.0
        return StatsExtended(
            k=k,
            recall_1_k=min(1.0, float(row["recall_1_k"])),
            recall_k_k=min(1.0, float(row["recall_k_k"])),
            QPS_seq=qps_seq if qps_seq > 0 else None,
            QPS_par=qps_par,
            avg_cmps=float(row.get("avg_cmps", 0.0) or 0.0),
            avg_timings=[
                float(row.get("t_graph_search", 0.0) or 0.0),
                float(row.get("t_rerank", 0.0) or 0.0),
            ],
        )

    def compute_stats_latency(self, queries, gt, params):
        del queries, gt
        rows = self.sweep(
            k=int(params["k"]),
            ef_list=[int(params["ef_search"])],
            rerank_list=[int(params.get("rerank_k", 256))],
            threads=1,
        )
        if not rows:
            raise RuntimeError("hnswlib_runner returned no CSV rows.")
        return self.row_to_stats(rows[0])

    def compute_stats_batch(self, queries, gt, params):
        del queries, gt
        try:
            n_thr = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            n_thr = os.cpu_count() or 1
        rows = self.sweep(
            k=int(params["k"]),
            ef_list=[int(params["ef_search"])],
            rerank_list=[int(params.get("rerank_k", 256))],
            threads=int(n_thr),
        )
        if not rows:
            raise RuntimeError("hnswlib_runner returned no CSV rows.")
        return self.row_to_stats(rows[0])


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
