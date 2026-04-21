"""
Script to efficiently filter zero vectors (padding) from Multi-Vector PointCloud datasets.
Designed for high-performance and low-memory overhead on large-scale datasets (e.g., NQ, HotpotQA).

Uses memory mapping (mmap) and multiprocessing to parallelize the identification 
of zero vectors without loading the entire dataset into RAM.
"""

import argparse
import multiprocessing as mp
import os
import numpy as np
from tqdm import tqdm


def get_file_layout(filename):
    """
    Parses the binary header to determine the byte offsets and shapes 
    required to memory-map the large arrays later.
    """
    with open(filename, "rb") as f:
        dims = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        n = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        num_vectors = np.frombuffer(f.read(8), dtype=np.uint64)[0]

    header_size = 24
    values_offset = header_size
    values_shape = (int(num_vectors), int(dims))
    values_bytes = num_vectors * dims * 4

    # Seek past the values array to get the offsets header
    with open(filename, "rb") as f:
        f.seek(values_offset + values_bytes)
        num_offsets = np.frombuffer(f.read(8), dtype=np.uint64)[0]

    offsets_offset = values_offset + values_bytes + 8
    offsets_shape = (int(num_offsets),)

    return dims, n, num_vectors, values_offset, values_shape, offsets_offset, offsets_shape


def worker_process_chunk(args):
    """
    Worker function executed by the multiprocessing Pool.
    Reads a subset of vectors, identifies non-zero vectors, and computes new document sizes.
    """
    (filename, val_off, val_shape, dims, start_doc, end_doc, start_vec, end_vec, doc_sizes) = args

    # Memory map only the required slice (read-only)
    values = np.memmap(filename, dtype=np.float32, mode='r', offset=val_off, shape=val_shape)
    chunk_vecs = values[start_vec:end_vec]

    # Fast vectorized squared L2 norm computation. 
    # If the squared norm is 0, the vector is a zero vector.
    sq_norms = np.einsum('ij,ij->i', chunk_vecs, chunk_vecs)
    mask = ~np.isclose(sq_norms, 0.0, atol=1e-9)

    # Calculate the new number of vectors for each document in this chunk
    new_doc_sizes = []
    current_idx = 0
    for size in doc_sizes:
        new_doc_sizes.append(np.sum(mask[current_idx : current_idx + size]))
        current_idx += size

    return start_doc, mask, np.array(new_doc_sizes, dtype=np.uint64)


def main(input_file, output_file, num_cores):
    print(f"Analyzing file structure for: {input_file}")
    dims, n, num_vectors, val_off, val_shape, off_off, off_shape = get_file_layout(input_file)
    
    print(f"Dataset Stats: {n} queries/docs, {num_vectors} total vectors, Dim: {dims}")
    
    # Map the offsets array to determine chunk boundaries
    offsets = np.memmap(input_file, dtype=np.uint64, mode='r', offset=off_off, shape=off_shape)
    
    # Calculate how many float-vectors belong to each document
    # Note: The original format stores float counts (vector_idx * dims), so we divide by dims
    vector_indices = offsets // dims
    doc_sizes = np.diff(vector_indices)

    # Prepare chunks for multiprocessing
    num_chunks = num_cores * 4  # Oversubscribe slightly to balance uneven doc lengths
    docs_per_chunk = int(np.ceil(n / num_chunks))
    
    tasks = []
    for i in range(0, int(n), docs_per_chunk):
        start_doc = i
        end_doc = min(i + docs_per_chunk, int(n))
        
        start_vec = int(vector_indices[start_doc])
        end_vec = int(vector_indices[end_doc])
        chunk_doc_sizes = doc_sizes[start_doc:end_doc]
        
        tasks.append((
            input_file, val_off, val_shape, dims, 
            start_doc, end_doc, start_vec, end_vec, chunk_doc_sizes
        ))

    print(f"Launching pool with {num_cores} cores across {len(tasks)} tasks...")
    
    # Execute worker pool
    results = []
    with mp.Pool(num_cores) as pool:
        for res in tqdm(pool.imap_unordered(worker_process_chunk, tasks), total=len(tasks), desc="Filtering zero vectors"):
            results.append(res)
            
    # Sort results to maintain original document order
    results.sort(key=lambda x: x[0])
    
    # Reconstruct the global mask and new offsets
    full_mask = np.concatenate([res[1] for res in results])
    all_new_doc_sizes = np.concatenate([res[2] for res in results])
    
    new_num_vectors = np.sum(all_new_doc_sizes)
    print(f"\nFiltered! Kept {new_num_vectors} vectors (Dropped {num_vectors - new_num_vectors} padding vectors).")

    # Rebuild offsets (stored as cumulative float-counts to match utils.py logic)
    new_offsets = np.zeros(n + 1, dtype=np.uint64)
    new_offsets[1:] = np.cumsum(all_new_doc_sizes) * dims

    # Write output to disk efficiently
    print(f"Writing optimized data to {output_file}...")
    values = np.memmap(input_file, dtype=np.float32, mode='r', offset=val_off, shape=val_shape)
    
    with open(output_file, "wb") as f_out:
        # 1. Write Header
        f_out.write(np.array([dims], dtype=np.uint64).tobytes())
        f_out.write(np.array([n], dtype=np.uint64).tobytes())
        f_out.write(np.array([new_num_vectors], dtype=np.uint64).tobytes())

        # 2. Write filtered vectors in chunks to respect RAM constraints
        chunk_size = 5_000_000  # Process 5 million vectors at a time
        for i in tqdm(range(0, val_shape[0], chunk_size), desc="Writing binary chunks"):
            end = min(i + chunk_size, val_shape[0])
            
            v_chunk = values[i:end]
            m_chunk = full_mask[i:end]
            
            valid_vectors = v_chunk[m_chunk]
            f_out.write(valid_vectors.tobytes())

        # 3. Write new offsets header & data
        f_out.write(np.array([len(new_offsets)], dtype=np.uint64).tobytes())
        f_out.write(new_offsets.tobytes())
        
    print("\nFile written successfully. Ready for index generation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drop zero-vectors from a PointCloud file efficiently.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input pointcloud file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output cleaned file.")
    parser.add_argument("-c", "--cores", type=int, default=32, help="Number of CPU cores to use (default: 32).")
    
    args = parser.parse_args()
    main(args.input, args.output, args.cores)