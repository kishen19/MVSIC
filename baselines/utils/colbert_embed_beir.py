import argparse
import os
import struct
import logging
import torch
import numpy as np
import itertools

# Configure logging for clear output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from beir.datasets.data_loader import GenericDataLoader
    import colbert
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint
except ImportError as e:
    logging.error(f"Import error: {e}. Please ensure you have installed the necessary libraries.")
    logging.error("You can install them using: pip install colbert-ai beir")
    exit(1)


def write_embeddings_to_binary(embeddings_list: list, output_file: str):
    """
    Writes a list of multi-vector embeddings to a binary file in the specified C++ format.

    Args:
        embeddings_list: A list where each item is a torch.Tensor representing the embeddings for one document/query.
        output_file: The path to the binary file to be created.
    """
    if not embeddings_list:
        logging.warning(f"Embeddings list is empty. Cannot write to {output_file}.")
        return

    # --- 1. Prepare data for binary packing ---
    num_docs = len(embeddings_list)
    dim = embeddings_list[0].shape[1]

    # Flatten all vectors into a single contiguous numpy array
    all_vectors_flat = np.concatenate([t.cpu().numpy() for t in embeddings_list], axis=0).astype(
        'float32'
    )
    total_num_vectors = all_vectors_flat.shape[0]

    # Calculate offsets
    # offsets[i] is the starting index of vectors for document i
    offsets = [0] * (num_docs + 1)
    current_offset = 0
    for i, tensor in enumerate(embeddings_list):
        offsets[i] = current_offset
        current_offset += tensor.shape[0]
    offsets[num_docs] = current_offset

    # --- 2. Write to binary file ---
    logging.info(f"Writing {num_docs} point clouds to {output_file}...")
    with open(output_file, 'wb') as f:
        # Write metadata: dims, num_docs, total_num_vectors
        # Using 'Q' for unsigned 64-bit integer (size_t in C++)
        f.write(struct.pack('Q', dim))
        f.write(struct.pack('Q', num_docs))
        f.write(struct.pack('Q', total_num_vectors))

        # Write the flat array of float values
        f.write(all_vectors_flat.tobytes())

        # Write offsets
        f.write(struct.pack('Q', len(offsets)))
        f.write(np.array(offsets, dtype=np.uint64).tobytes())

    logging.info("Write operation complete.")


def encode_and_save(
    collection: dict, model: Checkpoint, output_path: str, is_query: bool, batch_size: int = 64
):
    """
    Encodes a collection of texts (corpus or queries) and saves them to a binary file.
    """
    texts = list(collection.values())
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            if is_query:
                embeddings = model.queryFromText(batch)
            else:
                embeddings = model.docFromText(batch)

            # model returns a single tensor for the batch, we need to split it
            # For queries, it's a fixed size tensor.
            # For docs, it's a ragged tensor, so the model returns a list of tensors.
            if is_query:
                # Unbind the batch tensor into a list of tensors
                all_embeddings.extend(list(torch.unbind(embeddings, dim=0)))
            else:
                all_embeddings.extend(embeddings)

            if (i // batch_size + 1) % 10 == 0:
                # Use ceiling division to get total number of batches
                total_batches = -(len(texts) // -batch_size)
                logging.info(f"  - Encoded batch {i // batch_size + 1} / {total_batches}")

    write_embeddings_to_binary(all_embeddings, output_path)


def main(args):
    """
    Main function to load data, generate embeddings, and save them.
    """
    # --- 1. Load BEIR Dataset ---
    logging.info(f"--- Loading BEIR dataset from: {args.dataset_path} ---")
    corpus, queries, _ = GenericDataLoader(data_folder=args.dataset_path).load(split="test")

    # Convert corpus to the required format (title + text)
    corpus_texts = {
        doc_id: doc.get('title', '') + " " + doc.get('text', '') for doc_id, doc in corpus.items()
    }
    query_texts = {qid: q_text for qid, q_text in queries.items()}

    if args.debug_subset_size:
        logging.warning(
            f"--- Running in debug mode. Using a subset of {args.debug_subset_size} documents and queries. ---"
        )
        corpus_texts = dict(itertools.islice(corpus_texts.items(), args.debug_subset_size))
        query_texts = dict(itertools.islice(query_texts.items(), args.debug_subset_size))

    os.makedirs(args.output_path, exist_ok=True)

    # --- 2. Initialize ColBERT Model ---
    logging.info(f"--- Initializing ColBERT model from checkpoint: {args.checkpoint} ---")
    with Run().context(RunConfig(nranks=1, experiment="embedding_generation")):
        config = ColBERTConfig(
            doc_maxlen=args.doc_maxlen,
            query_maxlen=args.query_maxlen,
        )
        model = Checkpoint(args.checkpoint, colbert_config=config)

        # --- 3. Encode and Save Corpus ---
        logging.info(f"--- Encoding Corpus ({len(corpus_texts)} documents) ---")
        corpus_output_file = os.path.join(args.output_path, "corpus_embeddings.bin")
        encode_and_save(
            corpus_texts, model, corpus_output_file, is_query=False, batch_size=args.batch_size
        )

        # --- 4. Encode and Save Queries ---
        logging.info(f"--- Encoding Queries ({len(query_texts)} queries) ---")
        queries_output_file = os.path.join(args.output_path, "query_embeddings.bin")
        encode_and_save(
            query_texts, model, queries_output_file, is_query=True, batch_size=args.batch_size
        )

    logging.info("--- Embedding generation complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ColBERT embeddings for a BEIR dataset and save in a custom binary format."
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help="Path to the BEIR dataset directory (e.g., './datasets/scifact').",
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help="Directory to save the generated binary embedding files.",
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='colbert-ir/colbertv2.0',
        help="The ColBERT model checkpoint to use for encoding.",
    )
    parser.add_argument(
        '--doc_maxlen', type=int, default=300, help="Max sequence length for documents."
    )
    parser.add_argument(
        '--query_maxlen', type=int, default=32, help="Max sequence length for queries."
    )
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for encoding.")
    parser.add_argument(
        '--debug_subset_size',
        type=int,
        default=None,
        help="Run on a small subset of N documents/queries for debugging purposes.",
    )

    args = parser.parse_args()
    main(args)
