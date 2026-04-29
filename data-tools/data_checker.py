"""
Script to check if the pointcloud file contains valid inputs.
- Doesn't contain zero vectors
- Contains only unit vectors
"""

import argparse

import numpy as np
from tqdm import tqdm
from utils import ReadQueries


def check_pointcloud_validity(pointcloud_file):
    queries = ReadQueries(pointcloud_file)
    print(f"PointCloudSet loaded. Number of point clouds: {len(queries)}", flush=True)
    print(f"Number of vectors in pointcloud file: {sum(len(q) for q in queries)}", flush=True)
    num_zero = 0
    num_non_unit = 0
    for query in tqdm(queries):
        for emb in query:
            if np.isclose(np.linalg.norm(emb), 0.0):
                num_zero += 1
            if not np.isclose(np.linalg.norm(emb), 1.0):
                num_non_unit += 1
    if num_zero > 0:
        print(f"Found {num_zero} zero vectors in the pointcloud file.", flush=True)
    if num_non_unit > 0:
        print(f"Found {num_non_unit} non-unit vectors in the pointcloud file.", flush=True)
    return num_zero == 0 and num_non_unit == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check pointcloud validity.")
    parser.add_argument("-i", "--input", type=str, help="Path to the pointcloud file.")
    args = parser.parse_args()

    if not check_pointcloud_validity(args.input):
        print("Pointcloud file contains invalid inputs.", flush=True)
    else:
        print("Pointcloud file is valid.", flush=True)
