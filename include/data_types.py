import numpy as np
from utils import shared_array_init, shared_array_load, shared_array_unlink, chamfer_distance

class ChamferPoint:
  def __init__(self, id, coords, dim):
    self.id = id
    self.coords = coords
    self.dim = dim

  def distance(self, other):
    return chamfer_distance(self.coords, other.coords)

class PointCloud:
  def __init__(self, *args):
    if type(args[0]) == str: # File name and dtype
      self.read_input(*args)
    else: # Given a list of 2D arrays
      self.load_input(*args)

  def num_embeddings(self, i):
    pi = self.perm[i]
    return (self.offsets[pi + 1] - self.offsets[pi])/self.d

  def coords(self, i):
    pi = self.perm[i]
    values = shared_array_load(self.shm, self.shape, self.dtype)
    out = values[self.offsets[pi]:self.offsets[pi + 1]].reshape(-1, self.d)
    # shm.close()
    return out

  def __getitem__(self, i):
    return ChamferPoint(i, self.coords(i), self.d)
  
  def dims(self):
    return self.d

  def size(self):
    return self.n

  def read_input(self, filename, dtype=np.float32):
    with open(filename, "rb") as f:
      # Step 1: Read num_points and dims
      header = np.frombuffer(f.read(8), dtype=np.uint32)
      if header.size < 2:
        raise ValueError("Invalid Input Format!")
      self.n, self.d = header
      self.dtype = dtype
      n = self.n

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
      values = np.frombuffer(f.read(size_coords), dtype=dtype)
      self.shape = values.shape
      self.shm = shared_array_init(values)

      # Step 5: Read offset values
      self.offsets = np.frombuffer(f.read(size_offsets), dtype=np.uint64)
      # self.offsets_shm, self.offsets = shared_array_init(offsets, dtype=np.uint64)

      # Step 6: Read permutation of points
      self.perm = np.frombuffer(f.read(size_perm), dtype=np.uint32)
      # self.perm_shm, self.perm = shared_array_init(perm, dtype=np.uint32)

  def load_input(self, points):
    self.n = len(points)
    self.d = points[0].shape[1]
    values = np.concatenate(points).flatten()
    self.shape = values.shape
    self.dtype = values.dtype
    self.shm = shared_array_init(values)
    offsets = np.zeros(self.n + 1, dtype=np.uint64)
    for i in range(1, self.n + 1):
      offsets[i] = offsets[i - 1] + points[i - 1].shape[0] * self.d
    self.offsets = offsets
    # self.offsets_shm, self.offsets = shared_array_init(offsets)
    self.perm = np.arange(self.n, dtype=np.uint32)
    # self.perm_shm, self.perm = shared_array_init(perm)

  def delete(self):
    shared_array_unlink(self.shm)