import numpy as np
from numba import njit
from multiprocessing import shared_memory

# Shared Memory Setup
def shared_array_init(X):
    shm = shared_memory.SharedMemory(create=True, size=X.nbytes)
    shared_X = np.ndarray(X.shape, dtype=X.dtype, buffer=shm.buf)
    shared_X[:] = X[:]
    return shm, shared_X

# Chamfer Distance (Vectorized)
# Assuming both |A|=|B|=s
@njit(fastmath=True)
def chamfer_distance_uniform(A, B):
    assert A.shape[0] == B.shape[0]
    s, d = A.shape
    cost_A = 0.0
    for i in range(s):
        min_dist = np.inf
        for j in range(s):
            dist = 0.0
            for k in range(d):
                diff = A[i, k] - B[j, k]
                dist += diff * diff
            dist = dist ** 0.5
            if dist < min_dist:
                min_dist = dist
        cost_A += min_dist
    cost_A /= s
    cost_B = 0.0
    for j in range(s):
        min_dist = np.inf
        for i in range(s):
            dist = 0.0
            for k in range(d):
                diff = B[j, k] - A[i, k]
                dist += diff * diff
            dist = dist ** 0.5
            if dist < min_dist:
                min_dist = dist
        cost_B += min_dist
    cost_B /= s
    return cost_A + cost_B

@njit
def chamfer_distance(A, B):
    if A.shape[0] == B.shape[0]:
        return chamfer_distance_uniform(A, B)
    sA, d = A.shape
    sB = B.shape[0]

    cost_A = 0.0
    for i in range(sA):
        min_dist = 1e10
        for j in range(sB):
            dist = 0.0
            for k in range(d):
                diff = A[i, k] - B[j, k]
                dist += diff * diff
            dist = dist ** 0.5
            if dist < min_dist:
                min_dist = dist
        cost_A += min_dist
    cost_A /= sA
    cost_B = 0.0
    for j in range(sB):
        min_dist = 1e10
        for i in range(sA):
            dist = 0.0
            for k in range(d):
                diff = B[j, k] - A[i, k]
                dist += diff * diff
            dist = dist ** 0.5
            if dist < min_dist:
                min_dist = dist
        cost_B += min_dist
    cost_B /= sB
    return cost_A + cost_B

# Batching Helper
def split_batches(n, batch_size):
    return [range(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]