
import argparse
import numpy as np
import struct
import random

def read_point_cloud_data(filename):
    """Reads a point cloud dataset from a file in the specified binary format."""
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
    """Writes a point cloud dataset to a file in the specified binary format."""
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
    parser.add_argument("input_file", help="Path to the input point cloud data file.")
    parser.add_argument("output_file", help="Path to save the subsampled data file.")
    parser.add_argument("num_samples", type=int, help="Number of point clouds to subsample.")
    args = parser.parse_args()

    print(f"Reading data from {args.input_file}...")
    dim, num_clouds, vectors, offsets = read_point_cloud_data(args.input_file)
    print(f"Original dataset contains {num_clouds} point clouds.")

    if args.num_samples > num_clouds:
        print(f"Warning: Requested number of samples ({args.num_samples}) is larger than the number of point clouds ({num_clouds}). Using all point clouds.")
        args.num_samples = num_clouds

    print(f"Subsampling to {args.num_samples} point clouds...")
    sampled_indices = sorted(random.sample(range(num_clouds), args.num_samples))

    new_vectors_list = []
    new_offsets = [0]
    current_offset = 0

    for i in sampled_indices:
        start_offset = offsets[i]
        end_offset = offsets[i+1]
        # The number of vectors in a point cloud is (end_offset - start_offset) / dim
        # The offsets are already scaled by dim in the C++ code.
        # The offsets refer to the start of the float values.
        num_vectors_in_cloud = (end_offset - start_offset)
        new_vectors_list.append(vectors[start_offset:end_offset])
        current_offset += num_vectors_in_cloud
        new_offsets.append(current_offset)

    if not new_vectors_list:
        new_vectors = np.array([], dtype=np.float32).reshape(0, dim)
    else:
        new_vectors = np.concatenate(new_vectors_list, axis=0)
    new_offsets = np.array(new_offsets, dtype=np.uint64)

    print(f"Writing subsampled data to {args.output_file}...")
    write_point_cloud_data(args.output_file, dim, new_vectors, new_offsets)
    print("Done.")

if __name__ == "__main__":
    main()
