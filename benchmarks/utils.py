import torch
import numpy as np
import struct
import csv
import os

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
            end = offsets[i + 1] // dim
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
                neighbors.append((doc_id, dist))
            gt.append(neighbors)
    return gt

def update_index_time_csv(dataset, method, index_time):
    file_path = './results1/index_times.csv'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    data = []
    header = ['dataset', 'method', 'index_time']
    
    if os.path.exists(file_path):
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            data = list(reader)

    entry_found = False
    for row in data:
        if row['dataset'] == dataset and row['method'] == method:
            row['index_time'] = index_time
            entry_found = True
            break
    
    if not entry_found:
        data.append({'dataset': dataset, 'method': method, 'index_time': index_time})

    data.sort(key=lambda x: (x['dataset'], x['method']))

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(data)
