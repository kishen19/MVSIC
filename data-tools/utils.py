import numpy as np

pair_dtype = np.dtype(
    [("distance", np.float32), ("id", np.uint32)]
)  # Each pair is 4 (float) + 4 (uint32) = 8 bytes


def ReadQueries(filename):
    with open(filename, "rb") as f:
        dims = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        n = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        num_vectors = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        coordinate_size = num_vectors * dims
        values = np.fromfile(f, dtype=np.float32, count=int(coordinate_size))
        values = values.reshape(int(num_vectors), int(dims))
        num_offsets = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        offsets = np.fromfile(f, dtype=np.uint64, count=int(num_offsets))
        pc = []
        for i in range(int(n)):
            start = int(offsets[i] / dims)
            end = int(offsets[i + 1] / dims)
            pc.append(values[start:end])
        return pc


def ReadGT(file_path, num_points):
    with open(file_path, "rb") as f:
        num_neighbors_bytes = f.read(4)
        num_neighbors = np.frombuffer(num_neighbors_bytes, dtype=np.int32)[0]
        total_pairs_to_read = num_points * num_neighbors
        data_array = np.fromfile(f, dtype=pair_dtype, count=total_pairs_to_read)
        result_array = data_array.reshape((num_points, num_neighbors))
        return result_array


def ReadGoldGT(file_path, num_points):
    with open(file_path, "rb") as f:
        num_offsets_bytes = f.read(8)
        num_gt_entries_bytes = f.read(8)
        num_offsets = np.frombuffer(num_offsets_bytes, dtype=np.uint64)[0]
        num_gt_entries = np.frombuffer(num_gt_entries_bytes, dtype=np.uint64)[0]
        offsets = np.fromfile(f, dtype=np.uint64, count=int(num_offsets))
        ground_truth = np.fromfile(f, dtype=np.uint32, count=int(num_gt_entries))
        result_array = []
        for i in range(num_points):
            start_index = int(offsets[i])
            end_index = int(offsets[i + 1])
            neighbors = [
                (float(j), ground_truth[start_index + j]) for j in range(end_index - start_index)
            ]
            result_array.append(neighbors)
        return result_array
