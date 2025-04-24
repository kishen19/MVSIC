import numpy as np
from numba import njit, prange
from multiprocessing import shared_memory

def shared_array_init(X):
	shm = shared_memory.SharedMemory(create=True, size=X.nbytes)
	shared_X = np.ndarray(X.shape, dtype=X.dtype, buffer=shm.buf)
	shared_X[:] = X[:]
	# shm.close()
	return shm

def shared_array_load(shm, shape, dtype):
	# shm = shared_memory.SharedMemory(name=name)
	shared_X = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
	return shared_X

def shared_array_unlink(shm):
	# shm = shared_memory.SharedMemory(name=name)
	shm.close()
	shm.unlink()

# Chamfer Distance
@njit(parallel=True, fastmath=True)
def chamfer_distance(A, B):
	sA, d = A.shape
	sB = B.shape[0]

	cost_A = 0.0
	for i in prange(sA):
		min_dist = np.float32(np.inf)
		for j in range(sB):
			dist = np.float32(0.0)
			for k in range(d):
				diff = A[i, k] - B[j, k]
				dist += diff * diff
			if dist < min_dist:
				min_dist = dist
		cost_A += min_dist

	cost_B = 0.0
	for j in prange(sB):
		min_dist = np.float32(np.inf)
		for i in range(sA):
			dist = np.float32(0.0)
			for k in range(d):
				diff = B[j, k] - A[i, k]
				dist += diff * diff
			if dist < min_dist:
				min_dist = dist
		cost_B += min_dist

	return (cost_A / sA) + (cost_B / sB)

# Batching Helper
def split_batches(n, batch_size):
	return [range(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]

# Read Multi-Vector Input
def read_pointcloud(filename, dtype=np.float32):
	with open(filename, "rb") as f:
		# Step 1: Read num_points and dims
		header = f.read(8)
		if len(header) < 8:
			raise ValueError("File too small to contain header.")
		n, d = struct.unpack("II", header)

		# Step 2: Get file size
		f.seek(0, 2)
		file_size = f.tell()

		# Step 3: Compute sizes of various sections
		size_header = 8
		size_perm = n * np.dtype(np.uint32).itemsize
		size_offsets = (n + 1) * np.dtype(np.uint64).itemsize
		size_coords = file_size - (size_header + size_offsets + size_perm)

		# Step 4: Read coordinate values
		f.seek(size_header)
		coords = np.frombuffer(f.read(size_coords), dtype=dtype)

		# Step 5: Read offsets
		offsets = np.frombuffer(f.read(size_offsets), dtype=np.uint64)

		# Step 6: Read perm
		perm = np.frombuffer(f.read(size_perm), dtype=np.uint32)

	return n, d, offsets, coords, perm

if __name__ == "__main__":
	n, d, offsets, coords, perm = read_pointcloud("/ssd2/laxman/multivector/arguana/data.pointcloud")
	print(f"n = {n}, d = {d}")
	print("offsets:", offsets)
	sizes = np.array([(offsets[i+1] - offsets[i])/d for i in range(n)])
	print(sizes)
	print(np.mean(sizes))
	print("perm:", perm[:5])
	print("first point coords:", coords[offsets[perm[0]]:offsets[perm[0]+1]].reshape(-1, d))
	print(offsets[-1])
	assert len(set(perm)) == n
	for i in range(n):
		assert perm[i] < n and perm[i] >= 0