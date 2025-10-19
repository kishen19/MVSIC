
import numpy as np
import torch
import struct

def load_point_clouds(file_path):
    point_clouds = []
    with open(file_path, 'rb') as f:
        dim = struct.unpack('Q', f.read(8))[0]
        n = struct.unpack('Q', f.read(8))[0]
        num_vectors = struct.unpack('Q', f.read(8))[0]
        
        data = np.fromfile(f, dtype=np.float32, count=num_vectors * dim)
        data = data.reshape(num_vectors, dim)
        
        num_offsets = struct.unpack('Q', f.read(8))[0]
        offsets = np.fromfile(f, dtype=np.uint64, count=num_offsets)
        
        for i in range(n):
            start = offsets[i] // dim
            end = offsets[i+1] // dim
            point_clouds.append(torch.from_numpy(data[start:end]))
            
    return point_clouds

def load_gt(file_path, num_points):
    with open(file_path, 'rb') as f:
        num_neighbors = struct.unpack('i', f.read(4))[0]
        gt = []
        for _ in range(num_points):
            neighbors = []
            for _ in range(num_neighbors):
                dist = struct.unpack('f', f.read(4))[0]
                doc_id = struct.unpack('I', f.read(4))[0]
                neighbors.append((dist, doc_id))
            gt.append(neighbors)
    return gt
