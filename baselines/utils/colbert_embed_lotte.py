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
    from datasets import load_dataset
    import colbert
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint
except ImportError as e:
    logging.error(f"Import error: {e}. Please ensure you have installed the necessary libraries.")
    logging.error("You can install them using: pip install colbert-ai datasets torch numpy")
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

    # Calculate offsets: offsets[i] is the starting index of the float data for document i.
    # The C++ format expects offsets in terms of the number of floats, not the number of vectors.
    offsets = [0] * (num_docs + 1)
    current_offset = 0
    for i, tensor in enumerate(embeddings_list):
        offsets[i] = current_offset
        current_offset += (
            tensor.numel()
        )  # numel() gives the total number of floats (vectors * dims)
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
    Main function to load LoTTE data, generate embeddings, and save them.
    """
    # --- 0. Set number of threads ---
    if args.num_threads:
        logging.info(f"--- Setting number of threads to {args.num_threads} ---")
        torch.set_num_threads(args.num_threads)

    os.makedirs(args.output_path, exist_ok=True)

    # --- Initialize ColBERT Model ---
    logging.info(f"--- Initializing ColBERT model from checkpoint: {args.checkpoint} ---")
    with Run().context(RunConfig(nranks=1, experiment="embedding_generation")):
        config = ColBERTConfig(
            doc_maxlen=args.doc_maxlen,
            query_maxlen=args.query_maxlen,
        )
        model = Checkpoint(args.checkpoint, colbert_config=config)

        # --- Encode and Save Corpus (once for the domain) ---
        corpus_config_name = f"{args.dataset_config}_forum-corpus"
        logging.info(f"\n--- Processing Corpus for LoTTE config: '{corpus_config_name}' (split: {args.split}) ---")

        try:
            # Load the corpus for the specified split ('dev' or 'test')
            corpus_dataset = load_dataset(args.dataset_name, corpus_config_name, split=args.split)
        except Exception as e:
            logging.error(f"Failed to load corpus dataset '{corpus_config_name}' for split '{args.split}'. Error: {e}")
            return  # Exit if corpus can't be loaded

        corpus_texts = {str(row['_id']): row['title'] + " " + row['text'] for row in corpus_dataset}
        logging.info(f"Loaded {len(corpus_texts)} documents from the '{args.split}' corpus.")

        if args.debug_subset_size:
            logging.warning(
                f"--- Running in debug mode. Using a subset of {args.debug_subset_size} documents. ---"
            )
            corpus_texts = dict(itertools.islice(corpus_texts.items(), args.debug_subset_size))

        corpus_output_file = os.path.join(args.output_path, f"corpus_embeddings_{args.split}.bin")
        logging.info(f"--- Encoding Corpus ({len(corpus_texts)} documents) ---")
        encode_and_save(
            corpus_texts, model, corpus_output_file, is_query=False, batch_size=args.batch_size
        )

        # --- Encode and Save Queries (for each subset) ---
        for subset in ['forum', 'search']:
            logging.info(
                f"\n--- Processing Queries for subset: '{subset}' for LoTTE config: '{args.dataset_config}' (split: {args.split}) ---"
            )
            queries_config_name = f"{args.dataset_config}_{subset}-queries"

            try:
                queries_dataset = load_dataset(
                    args.dataset_name, queries_config_name, split=args.split
                )
            except Exception as e:
                logging.error(f"Failed to load queries for subset '{subset}' on split '{args.split}'. Error: {e}")
                continue

            query_texts = {str(row['_id']): row['text'] for row in queries_dataset}
            logging.info(
                f"Loaded {len(query_texts)} queries from the '{args.split}' split for subset '{subset}'."
            )

            if args.debug_subset_size:
                query_texts = dict(itertools.islice(query_texts.items(), args.debug_subset_size))

            # Save query embeddings to the main output path with a descriptive name
            queries_output_file = os.path.join(
                args.output_path, f"{subset}_query_embeddings_{args.split}.bin"
            )
            logging.info(f"--- Encoding Queries ({len(query_texts)} queries) ---")
            encode_and_save(
                query_texts, model, queries_output_file, is_query=True, batch_size=args.batch_size
            )

    logging.info(f"\n--- All embedding generation for split '{args.split}' complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ColBERT embeddings for a specific split ('dev' or 'test') of a LoTTE dataset."
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='mteb/lotte',
        help="Name of the dataset on Hugging Face Hub (e.g., 'mteb/lotte').",
    )
    parser.add_argument(
        '--dataset_config',
        type=str,
        required=True,
        help="Base configuration of the LoTTE dataset to use, e.g., 'lifestyle' or 'writing'.",
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['dev', 'test'],
        required=True,
        help="The dataset split to process ('dev' or 'test'). This affects both corpus and queries.",
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help="Base directory to save the generated binary embedding files. Subdirectories for 'forum' and 'search' will be created here.",
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
    # For climate-FEVER: set this to 64
    parser.add_argument(
        '--query_maxlen', type=int, default=32, help="Max sequence length for queries."
    )
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for encoding.")
    parser.add_argument(
        '--debug_subset_size',
        type=int,
        default=None,
        help="Run on a small subset of N documents/queries for debugging purposes.",
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=None,
        help="Number of threads to use for Torch. Defaults to all available.",
    )

    args = parser.parse_args()
    main(args)
