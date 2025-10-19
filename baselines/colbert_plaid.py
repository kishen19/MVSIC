import argparse
import time
import os
import pathlib
import logging
import itertools
import warnings

# Suppress specific future warnings from torch.amp and other libraries for a cleaner output
warnings.filterwarnings(
    "ignore",
    message="`torch.cuda.amp.GradScaler(args...)` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="`torch.cuda.amp.autocast(args...)` is deprecated.*",
    category=FutureWarning,
)

# Configure logging to be clean and informative
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    from datasets import load_dataset
    import colbert
    from colbert import Indexer, Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
except ImportError as e:
    logging.error(f"Import error: {e}. Please ensure you have installed the necessary libraries.")
    logging.error("You can install them using: pip install colbert-ai beir datasets")
    exit(1)


def main(args):
    """
    Main function to drive the benchmarking process based on CLI arguments.
    """
    # Conditionally reduce library verbosity based on the --quiet flag
    if args.quiet:
        # Suppress datasets library's noisy logging and progress bars
        import datasets

        datasets.logging.set_verbosity_error()

    dataset_type = args.dataset[0].lower()
    dataset_name = args.dataset[1]

    # --- 1. Load Dataset (BEIR or LoTTE) ---
    logging.info(f"--- Loading dataset: {dataset_type}/{dataset_name} ---")

    if dataset_type == 'beir':
        data_path = os.path.join(args.dataset_path, dataset_name)
        if not os.path.exists(data_path):
            logging.error(
                f"BEIR dataset not found at '{data_path}'. Please check the path or download it first."
            )
            exit(1)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    elif dataset_type == 'lotte':
        # LoTTE requires constructing config names and transforming data to BEIR format
        logging.info("Loading LoTTE data from Hugging Face Hub...")
        qrels = {}
        corpus = {}
        queries = {}

        # Load corpus
        corpus_dataset = load_dataset('mteb/lotte', f"{dataset_name}-corpus", split='test')
        corpus = {
            str(row['_id']): {'title': row['title'], 'text': row['text']} for row in corpus_dataset
        }

        # Load queries
        queries_dataset = load_dataset('mteb/lotte', f"{dataset_name}-queries", split='test')
        queries = {str(row['_id']): row['text'] for row in queries_dataset}

        # Load and transform qrels
        qrels_dataset = load_dataset('mteb/lotte', f"{dataset_name}-qrels", split='test')
        for row in qrels_dataset:
            query_id = str(row['query-id'])
            doc_id = str(row['corpus-id'])
            score = int(row['score'])
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = score
    else:
        logging.error(f"Unknown dataset type '{dataset_type}'. Please use 'beir' or 'lotte'.")
        exit(1)

    # --- Optional: Subset for Debugging ---
    if args.debug_subset_size:
        logging.warning(
            f"--- Running in debug mode. Using a subset of {args.debug_subset_size} documents and queries. ---"
        )
        # Take the first N queries
        queries = dict(itertools.islice(queries.items(), args.debug_subset_size))

        # Filter qrels to only include the selected queries
        qrels = {qid: qrels[qid] for qid in queries if qid in qrels}

        # Take the first N documents for the corpus subset
        corpus = dict(itertools.islice(corpus.items(), args.debug_subset_size))

    logging.info(f"Loaded {len(corpus)} documents and {len(queries)} queries.")

    # Prepare corpus for ColBERT Searcher and Indexer
    corpus_list = [
        (doc.get('title', '') + " " + doc.get('text', '')).strip() for doc in corpus.values()
    ]
    corpus_ids = list(corpus.keys())

    # --- 2. Configure ColBERT Paths and Settings ---
    checkpoint = 'colbert-ir/colbertv2.0'
    # The index name now includes the nbits value to avoid overwriting experiments
    index_name = f"{dataset_type}-{dataset_name}.{args.nbits}bits"

    # --- 3. Conditional Indexing ---
    if args.build:
        logging.info(f"--- Building index at: {args.index_path} (nbits={args.nbits}) ---")
        start_time_indexing = time.time()

        with Run().context(RunConfig(nranks=1, experiment=dataset_name, root=args.index_path)):
            # Parameters from ColBERTv2 paper (doc_maxlen=180) and library defaults
            config = ColBERTConfig(
                doc_maxlen=180,
                nbits=args.nbits,
                kmeans_niters=4,
            )
            indexer = Indexer(checkpoint=checkpoint, config=config)
            indexer.index(name=index_name, collection=corpus_list, overwrite=True)

        end_time_indexing = time.time()
        indexing_time = end_time_indexing - start_time_indexing
        print("\n--- Indexing Report ---")
        print(f"  - Total Indexing Time: {indexing_time:.2f} seconds")

    # --- 4. Conditional Searching & Evaluation ---
    if args.search:
        logging.info(f"--- Loading index and preparing for search ---")

        with Run().context(RunConfig(experiment=dataset_name, root=args.index_path)):
            searcher = Searcher(index=index_name, collection=corpus_list)
            # Configure searcher with command-line parameters
            searcher.configure(nprobe=args.nprobe)

        # --- Latency Test (One-by-one Search) ---
        logging.info(f"Running latency test (k={args.k}, nprobe={args.nprobe})...")
        results = {}
        latencies = []
        for q_id, q_text in queries.items():
            start_time = time.time()
            pids, _, scores = searcher.search(q_text, k=args.k)
            latencies.append(time.time() - start_time)
            results[q_id] = {corpus_ids[pid]: score for pid, score in zip(pids, scores)}

        if latencies:
            avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
        else:
            avg_latency_ms = 0

        # --- Throughput Test (Batch Search) ---
        logging.info(f"Running throughput test (k={args.k}, nprobe={args.nprobe})...")
        start_time_batch = time.time()
        if queries:
            # search_all expects a dictionary of {qid: q_text}, not a list of texts
            searcher.search_all(queries, k=args.k)
        total_batch_time = time.time() - start_time_batch

        # --- Evaluation ---
        logging.info("Evaluating search results...")
        evaluator = EvaluateRetrieval()
        k_values = [1, 3, 5, 10, 100, 1000]
        # The evaluate method returns a tuple of dictionaries: (ndcg, map, recall, precision)
        _, _, recall, _ = evaluator.evaluate(qrels, results, k_values)

        # --- Reporting ---
        print("\n--- Search & Evaluation Report ---")
        print(f"\n[Dataset: {dataset_type}/{dataset_name}]")
        print(f"\n[Index Parameters: nbits={args.nbits}]")
        print(f"[Search Parameters: k={args.k}, nprobe={args.nprobe}]")
        print("\n[Timing Report]")
        print(f"  - Average Search Latency: {avg_latency_ms:.2f} ms/query")
        if total_batch_time > 0:
            print(f"  - Batch Search Throughput: {len(queries) / total_batch_time:.2f} queries/sec")
        else:
            print("  - Batch Search Throughput: N/A (no queries processed or time was negligible)")
        print(
            f"  - Total Batch Search Time: {total_batch_time:.2f} seconds for {len(queries)} queries"
        )

        print("\n[Performance Report]")
        print("  - Recall@k:")
        for k in [100, 1000]:
            if f'Recall@{k}' in recall and k <= args.k:
                print(f"    - Recall@{k:<4}: {recall[f'Recall@{k}']:.4f}")

    logging.info("--- Benchmark Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A flexible benchmarking script for ColBERT on BEIR and LoTTE datasets."
    )
    parser.add_argument(
        '--dataset',
        type=str,
        nargs=2,
        required=True,
        metavar=('TYPE', 'NAME'),
        help="Type and name of the dataset. E.g., '--dataset beir fiqa' or '--dataset lotte lifestyle_search'.",
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='./datasets',
        help="Root directory where BEIR datasets are stored. Default: './datasets'.",
    )
    parser.add_argument(
        '--index_path',
        type=str,
        required=True,
        help="Root directory to save or load the ColBERT index.",
    )
    parser.add_argument(
        '--build', action='store_true', help="If set, the script will build a new index."
    )
    parser.add_argument(
        '--search',
        action='store_true',
        help="If set, the script will run search and evaluation.",
    )
    parser.add_argument(
        '--k',
        type=int,
        default=1000,
        help="Number of documents to retrieve for each query. Default: 1000.",
    )
    parser.add_argument(
        '--nprobe',
        type=int,
        default=10,
        help="Number of IVF partitions to search. Affects speed/accuracy trade-off. Default: 10.",
    )
    parser.add_argument(
        '--nbits',
        type=int,
        default=2,
        help="Number of bits for vector quantization. Default: 2.",
    )
    parser.add_argument(
        '--debug_subset_size',
        type=int,
        default=None,
        help="Run on a small subset of N documents/queries for debugging purposes.",
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="If set, suppresses detailed logging and progress bars from underlying libraries.",
    )

    args = parser.parse_args()

    if not args.build and not args.search:
        parser.error("No action requested. Please specify at least one of --build or --search.")

    main(args)
