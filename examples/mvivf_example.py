"""Build and search MVIVF using the installed package."""

import tempfile
from pathlib import Path

import numpy as np
import mvsic


def main():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(64, 4, 128)).astype(np.float32)
    values /= np.linalg.norm(values, axis=-1, keepdims=True)
    points = mvsic.PointCloudSet(values)

    params = mvsic.IndexParams.mvivf(k_per_level=4, max_leaf_size=16, niters=3, s=4)
    params.quantizer = mvsic.QuantizerType.OneBitTQ
    params.mvclus.assignment_quantizer = mvsic.ClusteringQuantizer.TQ8
    index = mvsic.IndexMVIVF(points.dims, mvsic.Metric.IP, params)
    index.build(points)

    search = mvsic.SearchParams.mvivf(k=5, nprobes=16, num_rerank=20, tq8_rerank=True)
    ids, distances = index.search(values[0], search)
    print("Nearest document IDs:", ids)
    print("Negative mean MaxSim distances:", distances)

    queries = mvsic.PointCloudSet(values[:3])
    batch_ids, _ = index.search_all(queries, search)
    print("Batch IDs:\n", batch_ids)

    with tempfile.TemporaryDirectory() as directory:
        filename = str(Path(directory) / "example.mvivf")
        index.save(filename)
        loaded = mvsic.IndexMVIVF(points.dims, mvsic.Metric.IP, params)
        loaded.load(filename, points)
        np.testing.assert_array_equal(loaded.search(values[0], search)[0], ids)


if __name__ == "__main__":
    main()
