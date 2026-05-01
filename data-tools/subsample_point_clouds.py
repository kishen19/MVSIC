
import argparse
import numpy as np
import struct
import random

def read_point_cloud_data(filename):
    """Reads a .pcs file (see mvsic/core/types/point_cloud_set.h and
    data-tools/compute_ground_truth.py): uint64 dim, n, num_vectors; then
    float32[num_vectors * dim]; then uint64 num_offsets; then uint64 offsets
    where each offset is a cumulative *float* count (same as C++), not a
    row index into the (num_vectors, dim) view.
    """
    with open(filename, 'rb') as f:
        # Read header
        dim, num_clouds, num_vectors = struct.unpack('QQQ', f.read(3 * 8))

        # Read vectors
        vectors = np.fromfile(f, dtype=np.float32, count=num_vectors * dim)
        vectors = vectors.reshape((num_vectors, dim))

        # Read num_offsets
        num_offsets, = struct.unpack('Q', f.read(8))

        # Read offsets
        offsets = np.fromfile(f, dtype=np.uint64, count=num_offsets)

    return dim, num_clouds, vectors, offsets

def write_point_cloud_data(filename, dim, vectors, offsets):
    """Writes a .pcs file; offsets are cumulative float counts (n+1 entries)."""
    num_clouds = len(offsets) - 1
    num_vectors = vectors.shape[0]
    num_offsets = len(offsets)

    with open(filename, 'wb') as f:
        # Write header
        f.write(struct.pack('QQQ', dim, num_clouds, num_vectors))

        # Write vectors
        vectors.astype(np.float32).tofile(f)

        # Write num_offsets
        f.write(struct.pack('Q', num_offsets))

        # Write offsets
        offsets.astype(np.uint64).tofile(f)

def main():
    parser = argparse.ArgumentParser(description="Subsample a point cloud dataset.")
    parser.add_argument("-i", "--input", help="Path to the input point cloud data file.")
    parser.add_argument("-o", "--output", help="Path to save the subsampled data file.")
    parser.add_argument("-s", "--samples", type=int, help="Number of point clouds to subsample.")
    args = parser.parse_args()

    print(f"Reading data from {args.input}...")
    dim, num_clouds, vectors, offsets = read_point_cloud_data(args.input)
    print(f"Original dataset contains {num_clouds} point clouds.")

    if args.samples > num_clouds:
        print(f"Warning: Requested number of samples ({args.samples}) is larger than the number of point clouds ({num_clouds}). Using all point clouds.")
        args.samples = num_clouds

    print(f"Subsampling to {args.samples} point clouds...")
    sampled_indices = sorted(random.sample(range(num_clouds), args.samples))

    new_vectors_list = []
    new_offsets = [0]
    current_offset = 0

    for i in sampled_indices:
        start_f = int(offsets[i])
        end_f = int(offsets[i + 1])
        # Offsets are cumulative float32 counts; row slice is // dim (see
        # compute_ground_truth._read_offsets_vec).
        if start_f % dim != 0 or end_f % dim != 0:
            raise ValueError(
                f"Offsets not aligned to dim={dim}: "
                f"offsets[{i}]={start_f}, offsets[{i+1}]={end_f}"
            )
        start_row = start_f // dim
        end_row = end_f // dim
        float_span = end_f - start_f  # number of float32 values for this cloud
        new_vectors_list.append(vectors[start_row:end_row])
        current_offset += float_span
        new_offsets.append(current_offset)

    if not new_vectors_list:
        new_vectors = np.array([], dtype=np.float32).reshape(0, dim)
    else:
        new_vectors = np.concatenate(new_vectors_list, axis=0)
    new_offsets = np.array(new_offsets, dtype=np.uint64)

    print(f"Writing subsampled data to {args.output}...")
    write_point_cloud_data(args.output, dim, new_vectors, new_offsets)
    print("Done.")

if __name__ == "__main__":
    main()
